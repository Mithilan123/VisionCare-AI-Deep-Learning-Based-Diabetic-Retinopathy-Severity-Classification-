"""
VisionCare AI — Advanced Deep Learning Training Pipeline
Task: Diabetic Retinopathy Severity Classification (APTOS 2019 Blindness Detection)
Model: EfficientNetB3 with ImageNet Transfer Learning + Two-Stage Fine-Tuning
Supports controlled ablation experiments:
  1. baseline
  2. prep_aug (Auto-crop + Medical Augmentation)
  3. class_weights (Balanced loss weighting for minority grades)
  4. fine_tune (Two-stage transfer learning + unfreezing top conv blocks)
  5. full_pipeline (Best combination: Preprocessing + Augmentation + Balanced Weights + Fine-Tuning)
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
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.utils.class_weight import compute_class_weight

from preprocess_dataset import prepare_dataset_splits_and_cache, crop_retina_border

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
CACHE_DIR = os.path.join(BASE_DIR, "preprocessed_images")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
EXPERIMENTS_DIR = os.path.join(REPORTS_DIR, "experiments")
MODEL_DIR = os.path.join(BASE_DIR, "backend", "saved_model")
DEFAULT_MODEL_SAVE = os.path.join(MODEL_DIR, "model.keras")

CLASSES = [
    "No Diabetic Retinopathy",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative",
]
NUM_CLASSES = len(CLASSES)
IMAGE_SIZE = (300, 300)


def setup_hardware():
    """Detect available hardware and configure optimal CPU threading / GPU memory."""
    print("=" * 65)
    print("  VisionCare AI — Hardware & Environment Diagnostics")
    print("=" * 65)
    print(f"Python: {sys.version.split()[0]} | TensorFlow: {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            print(f"CUDA GPUs Configured: {gpus}")
        except Exception as e:
            print(f"GPU config note: {e}")
    else:
        print("TensorFlow Native Windows CPU Execution (AVX2/FMA multi-threading active)")
    print("=" * 65)


def load_dataset(use_cache=True, sample_size=None, random_seed=42):
    """
    Load verified train and validation splits.
    Ensures the permanent 733-sample validation split is strictly held out.
    """
    train_split_csv = os.path.join(REPORTS_DIR, "train_split.csv")
    val_split_csv = os.path.join(REPORTS_DIR, "validation_split.csv")

    if not (os.path.exists(train_split_csv) and os.path.exists(val_split_csv)):
        print("Splits not found. Generating permanent 80/20 train/val splits...")
        train_df, val_df = prepare_dataset_splits_and_cache(force_recompute=False)
    else:
        train_df = pd.read_csv(train_split_csv)
        val_df = pd.read_csv(val_split_csv)

    # Resolve image paths (cached 300x300 or raw PNGs)
    def resolve_path(row):
        cached_p = os.path.join(CACHE_DIR, f"{row['id_code']}.png")
        if use_cache and os.path.exists(cached_p):
            return cached_p
        return os.path.join(TRAIN_IMG_DIR, f"{row['id_code']}.png")

    train_df["img_path"] = train_df.apply(resolve_path, axis=1)
    val_df["img_path"] = val_df.apply(resolve_path, axis=1)

    if sample_size and sample_size < len(train_df):
        print(f"Subsampling {sample_size} training records for accelerated experiment...")
        from sklearn.model_selection import train_test_split
        train_df, _ = train_test_split(
            train_df,
            train_size=sample_size,
            stratify=train_df["diagnosis"],
            random_state=random_seed
        )
        train_df = train_df.reset_index(drop=True)

    print(f"\nActive Training Samples:   {len(train_df)}")
    print(f"Held-Out Validation Samples: {len(val_df)}")
    return train_df, val_df


def compute_class_weights(y_train, mode="balanced"):
    """
    Compute class weights for imbalanced diabetic retinopathy severity distributions.
    Modes:
      - 'none': Equal weight 1.0 for all classes
      - 'balanced': Inverse frequency weighting
      - 'smoothed': Mild square-root damped weighting to avoid gradient spikes
    """
    if mode == "none":
        return {i: 1.0 for i in range(NUM_CLASSES)}

    classes = np.arange(NUM_CLASSES)
    raw_weights = compute_class_weight("balanced", classes=classes, y=y_train)

    if mode == "smoothed":
        # Square-root dampening for training stability
        smoothed = np.sqrt(raw_weights)
        smoothed = smoothed / smoothed.mean()
        return {int(cls): float(round(w, 4)) for cls, w in zip(classes, smoothed)}

    return {int(cls): float(round(w, 4)) for cls, w in zip(classes, raw_weights)}


def build_tf_data_pipelines(train_df, val_df, batch_size=32, augment=True):
    """
    Create optimized tf.data.Dataset pipelines for training and validation.
    Applies medical image augmentation ONLY during training.
    """
    train_paths = list(train_df["img_path"].values)
    train_labels = tf.keras.utils.to_categorical(train_df["diagnosis"].values, NUM_CLASSES)

    val_paths = list(val_df["img_path"].values)
    val_labels = tf.keras.utils.to_categorical(val_df["diagnosis"].values, NUM_CLASSES)

    def parse_img(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_png(img_bytes, channels=3)
        if img.shape[0] != IMAGE_SIZE[0] or img.shape[1] != IMAGE_SIZE[1]:
            img = tf.image.resize(img, IMAGE_SIZE)
        return tf.cast(img, tf.float32), label

    def apply_medical_augmentation(img, label):
        # 1. Random horizontal & vertical flips (clinically invariant)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)

        # 2. Random 90/180/270 degree rotation
        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        img = tf.image.rot90(img, k)

        # 3. Mild contrast & brightness jitter
        img = tf.image.random_brightness(img, max_delta=10.0)
        img = tf.image.random_contrast(img, lower=0.92, upper=1.08)

        img = tf.clip_by_value(img, 0.0, 255.0)
        return img, label

    # Train pipeline
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = train_ds.map(parse_img, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        train_ds = train_ds.map(apply_medical_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(buffer_size=min(len(train_paths), 1024), reshuffle_each_iteration=True)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Validation pipeline (deterministic, no augmentation)
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = val_ds.map(parse_img, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def build_model(pretrained=True):
    """
    Construct functional EfficientNetB3 architecture:
    Inputs (300x300x3) -> EfficientNetB3 -> GAP2D -> BatchNorm ->
    Dense(256, ReLU) -> Dropout(0.4) -> Dense(128, ReLU) -> Dropout(0.3) -> Dense(5, Softmax).
    """
    weights = "imagenet" if pretrained else None
    base = EfficientNetB3(
        include_top=False,
        weights=weights,
        input_shape=(*IMAGE_SIZE, 3)
    )

    x = base.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dense(256, activation="relu", name="head_dense_256")(x)
    x = layers.Dropout(0.4, name="head_dropout_0.4")(x)
    x = layers.Dense(128, activation="relu", name="head_dense_128")(x)
    x = layers.Dropout(0.3, name="head_dropout_0.3")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = Model(inputs=base.input, outputs=outputs, name="VisionCare_EfficientNetB3")
    return model


def build_visioncare_model(weights="imagenet"):
    """Compatibility wrapper for build_model."""
    pretrained = (weights == "imagenet" or weights is True)
    return build_model(pretrained=pretrained)


def load_dataset_metadata(csv_path=TRAIN_CSV, img_dir=TRAIN_IMG_DIR, sample_size=None):
    """Load dataset metadata DataFrame with optional sampling."""
    df = pd.read_csv(csv_path)
    if sample_size and sample_size < len(df):
        df = df.head(sample_size).copy()
    return df



def plot_curves(history, output_path, title_prefix=""):
    """Plot and save training and validation metrics curves."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    epochs = range(1, len(history["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["accuracy"], "b-o", label="Training Accuracy", markersize=4)
    if "val_accuracy" in history:
        ax1.plot(epochs, history["val_accuracy"], "r--s", label="Validation Accuracy", markersize=4)
    ax1.set_title(f"{title_prefix} Accuracy Trajectory", fontsize=12)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["loss"], "b-o", label="Training Loss", markersize=4)
    if "val_loss" in history:
        ax2.plot(epochs, history["val_loss"], "r--s", label="Validation Loss", markersize=4)
    ax2.set_title(f"{title_prefix} Loss Trajectory (Categorical Cross-Entropy)", fontsize=12)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_experiment(exp_name, args):
    """
    Execute a controlled experiment.
    Tracks hyperparameters, history, checkpoint, and evaluation outputs.
    """
    setup_hardware()
    exp_dir = os.path.join(EXPERIMENTS_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\n" + "=" * 65)
    print(f"  RUNNING CONTROLLED EXPERIMENT: {exp_name.upper()}")
    print("=" * 65)

    # 1. Load data
    train_df, val_df = load_dataset(
        use_cache=args.use_cache,
        sample_size=args.sample_size,
        random_seed=args.random_seed
    )

    # 2. Configure experiment parameters
    augment = (exp_name != "baseline")
    weight_mode = "balanced" if exp_name in ["class_weights", "full_pipeline"] else "none"
    fine_tune = exp_name in ["fine_tune", "full_pipeline"]
    label_smoothing = 0.05 if exp_name == "full_pipeline" else 0.0

    class_weights = compute_class_weights(train_df["diagnosis"].values, mode=weight_mode)
    print(f"Augmentation active:     {augment}")
    print(f"Class weighting mode:    {weight_mode} -> {class_weights}")
    print(f"Label smoothing:         {label_smoothing}")
    print(f"Fine-tuning Stage B:     {fine_tune}")

    # 3. Pipelines
    train_ds, val_ds = build_tf_data_pipelines(
        train_df, val_df,
        batch_size=args.batch_size,
        augment=augment
    )

    # 4. Model construction
    model = build_model(pretrained=True)

    # Stage A: Freeze backbone, train head
    head_layers = {"gap", "head_bn", "head_dense_256", "head_dropout_0.4",
                   "head_dense_128", "head_dropout_0.3", "predictions"}
    for l in model.layers:
        l.trainable = (l.name in head_layers)

    checkpoint_path = os.path.join(exp_dir, "best_checkpoint.keras")
    checkpoint_cb = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr_cb = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
    csv_log_path = os.path.join(exp_dir, "training_log.csv")
    csv_logger = CSVLogger(csv_log_path)

    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr_head),
        loss=loss_fn,
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    print("\n[Stage A] Training Classification Head...")
    hist_a = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_head,
        class_weight=class_weights,
        callbacks=[checkpoint_cb, early_stop_cb, reduce_lr_cb, csv_logger],
        verbose=1
    )

    hist_combined = {k: list(v) for k, v in hist_a.history.items()}

    # Stage B: Fine-tuning if enabled
    if fine_tune and args.epochs_fine > 0:
        print("\n[Stage B] Fine-Tuning Top 40 Convolutional Layers...")
        # Unfreeze top 40 layers, keep BatchNorm frozen
        for layer in model.layers[:-40]:
            layer.trainable = False
        for layer in model.layers[-40:]:
            layer.trainable = not isinstance(layer, layers.BatchNormalization)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr_fine),
            loss=loss_fn,
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall")
            ]
        )

        stage_b_csv = CSVLogger(csv_log_path, append=True)
        hist_b = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs_head + args.epochs_fine,
            initial_epoch=len(hist_a.history["loss"]),
            class_weight=class_weights,
            callbacks=[checkpoint_cb, early_stop_cb, reduce_lr_cb, stage_b_csv],
            verbose=1
        )

        for k in hist_b.history:
            hist_combined[k] = hist_combined.get(k, []) + list(hist_b.history[k])

    # Save training curves & history JSON
    plot_path = os.path.join(exp_dir, "training_curves.png")
    plot_curves(hist_combined, plot_path, title_prefix=exp_name.upper())

    clean_hist = {k: [float(val) for val in vals] for k, vals in hist_combined.items()}
    hist_json_path = os.path.join(exp_dir, "history.json")
    with open(hist_json_path, "w") as f:
        json.dump(clean_hist, f, indent=2)

    # Save config
    config_dict = {
        "experiment_name": exp_name,
        "sample_size": len(train_df),
        "validation_size": len(val_df),
        "batch_size": args.batch_size,
        "lr_head": args.lr_head,
        "lr_fine": args.lr_fine,
        "epochs_head": args.epochs_head,
        "epochs_fine": args.epochs_fine,
        "augment": augment,
        "class_weight_mode": weight_mode,
        "class_weights": class_weights,
        "label_smoothing": label_smoothing,
        "fine_tune": fine_tune,
        "checkpoint": os.path.abspath(checkpoint_path)
    }
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # Save to global model path if this is full_pipeline or default model save requested
    if args.save_to_global or exp_name == "full_pipeline":
        import shutil
        shutil.copyfile(checkpoint_path, DEFAULT_MODEL_SAVE)
        shutil.copyfile(plot_path, os.path.join(REPORTS_DIR, "training_curves.png"))
        shutil.copyfile(hist_json_path, os.path.join(REPORTS_DIR, "training_history.json"))
        shutil.copyfile(csv_log_path, os.path.join(REPORTS_DIR, "training_history.csv"))
        meta_save = os.path.join(MODEL_DIR, "model_config.json")
        with open(meta_save, "w") as f:
            json.dump(config_dict, f, indent=2)
        print(f"\nSaved best model checkpoint to primary path: {DEFAULT_MODEL_SAVE}")

    print("=" * 65)
    print(f"  Experiment '{exp_name}' completed! Checkpoint: {checkpoint_path}")
    print("=" * 65)
    return checkpoint_path


def parse_args():
    parser = argparse.ArgumentParser(description="VisionCare AI — Model Training & Experiment Runner")
    parser.add_argument("--experiment", type=str, default="full_pipeline",
                        choices=["baseline", "prep_aug", "class_weights", "fine_tune", "full_pipeline"],
                        help="Experiment preset to execute.")
    parser.add_argument("--epochs-head", type=int, default=5,
                        help="Number of epochs for Stage A. Default: 5")
    parser.add_argument("--epochs-fine", type=int, default=3,
                        help="Number of epochs for Stage B. Default: 3")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size. Default: 32")
    parser.add_argument("--lr-head", type=float, default=5e-4,
                        help="Learning rate for Stage A. Default: 5e-4")
    parser.add_argument("--lr-fine", type=float, default=2e-5,
                        help="Learning rate for Stage B. Default: 2e-5")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Subsample N training records for rapid iteration.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable preprocessed_images cache and read raw PNGs.")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed. Default: 42")
    parser.add_argument("--save-to-global", action="store_true",
                        help="Save resulting model as default backend/saved_model/model.keras")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.use_cache = not args.no_cache
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)
    run_experiment(args.experiment, args)
