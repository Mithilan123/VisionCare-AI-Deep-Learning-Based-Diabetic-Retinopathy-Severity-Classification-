"""
VisionCare AI — Model Evaluation & Performance Analysis
Task: Diabetic Retinopathy Severity Grading (APTOS 2019 Blindness Detection)
Evaluates on the permanent 733-sample held-out validation set.
Metrics: Accuracy, Macro/Weighted F1, Per-Class Sensitivity & Precision,
         Quadratic Weighted Kappa (QWK), Multiclass ROC-AUC, Confusion Matrix.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "backend", "saved_model", "model.keras")
DEFAULT_VAL_CSV = os.path.join(BASE_DIR, "reports", "validation_split.csv")
CACHE_DIR = os.path.join(BASE_DIR, "preprocessed_images")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

CLASSES = [
    "No Diabetic Retinopathy",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative",
]
NUM_CLASSES = len(CLASSES)
IMAGE_SIZE = (300, 300)


def load_validation_records(val_csv_path):
    """Load the held-out validation split records."""
    if not os.path.exists(val_csv_path):
        raise FileNotFoundError(
            f"Validation split CSV not found at {val_csv_path}. "
            "Please run 'python preprocess_dataset.py' to generate the 80/20 split."
        )

    df = pd.read_csv(val_csv_path)
    print(f"Loaded validation split from {val_csv_path} with {len(df)} samples.")

    # Determine file paths: prefer cached preprocessed images, fallback to raw
    paths = []
    for cid in df["id_code"]:
        cached_p = os.path.join(CACHE_DIR, f"{cid}.png")
        raw_p = os.path.join(TRAIN_IMG_DIR, f"{cid}.png")
        if os.path.exists(cached_p):
            paths.append(cached_p)
        elif os.path.exists(raw_p):
            paths.append(raw_p)
        else:
            raise FileNotFoundError(f"Image for {cid} not found in cache or train_images.")

    df["resolved_path"] = paths

    # Print class counts
    print("Validation Class Distribution:")
    for cls_idx, count in df["diagnosis"].value_counts().sort_index().items():
        pct = (count / len(df)) * 100
        print(f"  Grade {cls_idx} [{CLASSES[cls_idx]}]: {count:4d} images ({pct:5.2f}%)")

    return df


def build_val_pipeline(paths, batch_size=32):
    """Batched data pipeline for deterministic validation evaluation."""
    def gen():
        for p in paths:
            img_bytes = tf.io.read_file(p)
            img = tf.image.decode_png(img_bytes, channels=3)
            if img.shape[0] != IMAGE_SIZE[0] or img.shape[1] != IMAGE_SIZE[1]:
                img = tf.image.resize(img, IMAGE_SIZE)
            yield tf.cast(img, tf.float32)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=tf.TensorSpec(shape=(*IMAGE_SIZE, 3), dtype=tf.float32)
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def plot_confusion_matrix(cm, classes, output_path):
    """Draw and save high-resolution confusion matrix heatmap."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    short_labels = [
        "Grade 0\n(No DR)",
        "Grade 1\n(Mild)",
        "Grade 2\n(Moderate)",
        "Grade 3\n(Severe)",
        "Grade 4\n(Proliferative)"
    ]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=short_labels,
        yticklabels=short_labels,
        title="Confusion Matrix — Diabetic Retinopathy Severity Grading (N=733)",
        ylabel="True Grade (Ground Truth)",
        xlabel="Predicted Grade (Model Output)"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            # Flag severe grading errors (e.g. Grade 0 predicted as Grade 4, or Grade 4 predicted as Grade 0)
            is_severe_error = abs(i - j) >= 3 and val > 0
            color = "yellow" if is_severe_error else ("white" if val > thresh else "black")
            ax.text(j, i, format(val, "d"),
                    ha="center", va="center",
                    color=color,
                    fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix plot to {output_path}")


def plot_roc_curves(y_true, y_probs, output_path):
    """Plot multi-class ROC curves (One-vs-Rest) with AUC per class."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    y_true_oh = tf.keras.utils.to_categorical(y_true, NUM_CLASSES)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#2563eb", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]

    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_true_oh[:, i], y_probs[:, i])
        try:
            auc = roc_auc_score(y_true_oh[:, i], y_probs[:, i])
            label = f"Grade {i} [{CLASSES[i]}] (AUC = {auc:.3f})"
        except Exception:
            label = f"Grade {i} [{CLASSES[i]}]"
        ax.plot(fpr, tpr, color=colors[i], lw=2, label=label)

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance Level (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title("Multiclass ROC Curves — Diabetic Retinopathy (One-vs-Rest, N=733)", fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved ROC curves plot to {output_path}")


def evaluate(args):
    print("=" * 65)
    print("  VisionCare AI — Authentic Model Evaluation")
    print("=" * 65)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Trained model not found at {args.model_path}.\n"
            "Please train or provide the model checkpoint before evaluating."
        )

    # 1. Load Model
    print(f"Loading model checkpoint: {args.model_path} ...")
    model = tf.keras.models.load_model(args.model_path)
    print("Model loaded successfully!")
    print(f"Input shape: {model.input_shape} | Output shape: {model.output_shape}")

    # 2. Load Validation Data
    val_df = load_validation_records(args.val_csv)
    y_true = val_df["diagnosis"].values.astype(int)
    paths = val_df["resolved_path"].values

    # 3. Predict on Validation Data
    print(f"\nRunning batch inference on {len(paths)} held-out validation samples...")
    val_ds = build_val_pipeline(paths, batch_size=args.batch_size)
    y_probs = model.predict(val_ds, verbose=1)
    y_pred = np.argmax(y_probs, axis=1)

    # Verify predictions count matches ground truth exactly
    if len(y_pred) != len(y_true):
        raise ValueError(f"Prediction count ({len(y_pred)}) does not match ground truth count ({len(y_true)})!")

    # 4. Compute Metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    try:
        y_true_oh = tf.keras.utils.to_categorical(y_true, NUM_CLASSES)
        roc_auc_macro = roc_auc_score(y_true_oh, y_probs, multi_class="ovr", average="macro")
    except Exception as e:
        roc_auc_macro = None
        print(f"Notice: Multiclass ROC-AUC calculation: {e}")

    # Per-Class Metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    clf_report_dict = classification_report(
        y_true, y_pred,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0
    )
    clf_report_text = classification_report(
        y_true, y_pred,
        target_names=CLASSES,
        zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 65)
    print("  EVALUATION RESULTS (HELD-OUT VALIDATION SET, N=733)")
    print("=" * 65)
    print(f"Sample Count:                      {len(y_true)} images")
    print(f"Overall Accuracy:                  {acc * 100:.2f}%")
    print(f"Quadratic Weighted Kappa (QWK):    {qwk:.4f}")
    print(f"Macro-Averaged F1-Score:           {macro_f1:.4f}")
    print(f"Weighted F1-Score:                 {weighted_f1:.4f}")
    if roc_auc_macro is not None:
        print(f"Macro Multiclass ROC-AUC (OvR):   {roc_auc_macro:.4f}")
    print("\nDetailed Classification Report:")
    print(clf_report_text)
    print("\nConfusion Matrix:")
    print(cm)

    # 5. Save Artifacts
    os.makedirs(REPORTS_DIR, exist_ok=True)

    metrics_summary = {
        "model_path": os.path.abspath(args.model_path),
        "validation_samples": int(len(y_true)),
        "accuracy": float(round(acc, 4)),
        "quadratic_weighted_kappa": float(round(qwk, 4)),
        "macro_f1": float(round(macro_f1, 4)),
        "weighted_f1": float(round(weighted_f1, 4)),
        "roc_auc_macro": float(round(roc_auc_macro, 4)) if roc_auc_macro is not None else None,
        "per_class": {
            CLASSES[i]: {
                "precision": float(round(precision_per_class[i], 4)),
                "recall": float(round(recall_per_class[i], 4)),
                "f1_score": float(round(f1_per_class[i], 4)),
                "support": int((y_true == i).sum())
            }
            for i in range(NUM_CLASSES)
        },
        "confusion_matrix": cm.tolist()
    }

    metrics_json_path = os.path.join(REPORTS_DIR, "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"\nSaved metrics summary to {metrics_json_path}")

    # Copy to backend/reports as well
    backend_metrics_path = os.path.join(BASE_DIR, "backend", "reports", "metrics.json")
    os.makedirs(os.path.dirname(backend_metrics_path), exist_ok=True)
    with open(backend_metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)

    # Save CSV Report
    report_df = pd.DataFrame(clf_report_dict).transpose()
    report_csv_path = os.path.join(REPORTS_DIR, "classification_report.csv")
    report_df.to_csv(report_csv_path)
    print(f"Saved classification report CSV to {report_csv_path}")

    # Save Plots
    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(cm, CLASSES, cm_path)

    roc_path = os.path.join(REPORTS_DIR, "roc_curves.png")
    plot_roc_curves(y_true, y_probs, roc_path)

    print("=" * 65)
    print("  Evaluation successfully finished!")
    print("=" * 65)
    return metrics_summary


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="VisionCare AI — Model Evaluation & Metric Generation"
    )
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH,
                        help=f"Path to saved model checkpoint. Default: {DEFAULT_MODEL_PATH}")
    parser.add_argument("--val-csv", type=str, default=DEFAULT_VAL_CSV,
                        help=f"Path to validation split CSV. Default: {DEFAULT_VAL_CSV}")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for evaluation. Default: 32")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    evaluate(args)
