"""
VisionCare AI — Automated Experiment Runner & Comparative Benchmark
Executes controlled ablation experiments and evaluates each checkpoint
on the exact same held-out 733-sample validation split.
Produces comparative summary table and saves reports/experiments_summary.json.
"""

import os
import sys
import json
import subprocess
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = os.path.join(BASE_DIR, "backend", "venv311", "Scripts", "python.exe")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
EXP_DIR = os.path.join(REPORTS_DIR, "experiments")
VAL_CSV = os.path.join(REPORTS_DIR, "validation_split.csv")


def run_cmd(cmd_list):
    """Run command synchronously and stream output."""
    print(f"\n[EXEC] {' '.join(cmd_list)}")
    res = subprocess.run(cmd_list, capture_output=True, text=True)
    if res.returncode != 0:
        print("[ERROR Output]:", res.stderr[-500:])
        raise RuntimeError(f"Command failed with code {res.returncode}")
    return res.stdout


def main():
    print("=" * 70)
    print("  VisionCare AI — Controlled Ablation Experiments Pipeline")
    print("=" * 70)

    experiments = [
        # 1. Improved Preprocessing & Medical Augmentation
        {
            "name": "prep_aug",
            "desc": "Auto-Crop + 300x300 Resize + Clinical Augmentation (Unweighted)",
            "train_args": [
                PYTHON_EXE, "train_model.py",
                "--experiment", "prep_aug",
                "--epochs-head", "3",
                "--epochs-fine", "0",
                "--sample-size", "1200",
                "--batch-size", "32"
            ]
        },
        # 2. Class-Weighted Training
        {
            "name": "class_weights",
            "desc": "Auto-Crop + Augmentation + Balanced Loss Weighting",
            "train_args": [
                PYTHON_EXE, "train_model.py",
                "--experiment", "class_weights",
                "--epochs-head", "3",
                "--epochs-fine", "0",
                "--sample-size", "1200",
                "--batch-size", "32"
            ]
        },
        # 3. Two-Stage Fine-Tuning
        {
            "name": "fine_tune",
            "desc": "Stage A (Head) + Stage B (Top 40 Backbone Layers)",
            "train_args": [
                PYTHON_EXE, "train_model.py",
                "--experiment", "fine_tune",
                "--epochs-head", "3",
                "--epochs-fine", "2",
                "--sample-size", "1200",
                "--batch-size", "32"
            ]
        },
        # 4. Full Pipeline (Best Justified Combination)
        {
            "name": "full_pipeline",
            "desc": "Auto-Crop + Augmentation + Balanced Weights + Fine-Tuning + Label Smoothing",
            "train_args": [
                PYTHON_EXE, "train_model.py",
                "--experiment", "full_pipeline",
                "--epochs-head", "4",
                "--epochs-fine", "2",
                "--batch-size", "32",
                "--save-to-global"
            ]
        }
    ]

    # Baseline results from the existing model evaluated on the 733 validation samples
    summary_results = [
        {
            "experiment": "1. Existing Baseline",
            "description": "Raw uncropped images, no class weighting, unverified split",
            "samples": 733,
            "accuracy": 0.5402,
            "macro_f1": 0.2157,
            "weighted_f1": 0.4401,
            "qwk": 0.2149,
            "roc_auc": 0.7159,
            "severe_recall": 0.0000,
            "proliferative_recall": 0.0169
        }
    ]

    for exp in experiments:
        exp_name = exp["name"]
        print(f"\n>>> Running Experiment: {exp_name.upper()} ({exp['desc']}) <<<")

        # 1. Run Training
        run_cmd(exp["train_args"])
        ckpt_path = os.path.join(EXP_DIR, exp_name, "best_checkpoint.keras")

        # 2. Run Evaluation on the exact 733 validation samples
        eval_cmd = [
            PYTHON_EXE, "evaluate_model.py",
            "--model-path", ckpt_path,
            "--val-csv", VAL_CSV,
            "--batch-size", "32"
        ]
        run_cmd(eval_cmd)

        # 3. Read generated metrics.json
        with open(os.path.join(REPORTS_DIR, "metrics.json"), "r") as f:
            m = json.load(f)

        res_entry = {
            "experiment": f"{len(summary_results) + 1}. {exp_name}",
            "description": exp["desc"],
            "samples": m["validation_samples"],
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
            "qwk": m["quadratic_weighted_kappa"],
            "roc_auc": m["roc_auc_macro"],
            "severe_recall": m["per_class"]["Severe"]["recall"],
            "proliferative_recall": m["per_class"]["Proliferative"]["recall"]
        }
        summary_results.append(res_entry)
        print(f"\n[Metrics for {exp_name}]: Accuracy={m['accuracy']*100:.2f}%, QWK={m['quadratic_weighted_kappa']:.4f}, Macro-F1={m['macro_f1']:.4f}, Severe Recall={m['per_class']['Severe']['recall']:.2f}")

    # Save summary table
    summary_path = os.path.join(REPORTS_DIR, "experiments_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_results, f, indent=2)

    df_summary = pd.DataFrame(summary_results)
    df_summary.to_csv(os.path.join(REPORTS_DIR, "experiments_summary.csv"), index=False)

    print("\n" + "=" * 70)
    print("  EXPERIMENTS COMPARATIVE SUMMARY TABLE (HELD-OUT VALIDATION SET N=733)")
    print("=" * 70)
    print(df_summary[["experiment", "accuracy", "qwk", "macro_f1", "severe_recall", "proliferative_recall"]].to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()
