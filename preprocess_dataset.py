"""
VisionCare AI — Preprocessing Engine and Dataset Caching
Performs reproducible stratified train/validation splitting (80/20)
and applies circular retinal border auto-cropping and 300x300 resizing.
Uses multi-core parallel processing to prepare clean image caches.
"""

import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
CACHE_DIR = os.path.join(BASE_DIR, "preprocessed_images")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

TARGET_SIZE = (300, 300)
RANDOM_SEED = 42


def crop_retina_border(img, tol=7):
    """
    Auto-detect and crop uninformative black background margins around the retina.
    Preserves all active ocular tissue while ensuring the retina fills the canvas.
    """
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > tol
    if not np.any(mask):
        return img

    row_sums = np.sum(mask, axis=1)
    col_sums = np.sum(mask, axis=0)

    y_indices = np.where(row_sums > 0)[0]
    x_indices = np.where(col_sums > 0)[0]

    if len(y_indices) == 0 or len(x_indices) == 0:
        return img

    y_min, y_max = y_indices[0], y_indices[-1]
    x_min, x_max = x_indices[0], x_indices[-1]

    # Add small safety padding (1% of dimension)
    h, w = img.shape[:2]
    pad_y = int(0.01 * h)
    pad_x = int(0.01 * w)

    y_min = max(0, y_min - pad_y)
    y_max = min(h, y_max + pad_y)
    x_min = max(0, x_min - pad_x)
    x_max = min(w, x_max + pad_x)

    return img[y_min:y_max, x_min:x_max]


def process_single_image(args):
    """Worker task to crop and resize a single fundus image."""
    in_path, out_path, target_size = args
    if os.path.exists(out_path):
        return True  # Already cached

    img = cv2.imread(in_path)
    if img is None:
        return False

    cropped = crop_retina_border(img)
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_path, resized, [cv2.IMWRITE_PNG_COMPRESSION, 4])
    return True


def prepare_dataset_splits_and_cache(force_recompute=False):
    """
    1. Loads train.csv (3,662 records)
    2. Generates reproducible 80% train (2,929) and 20% validation (733) splits
    3. Saves reports/train_split.csv and reports/validation_split.csv
    4. Parallel pre-processes all 3,662 images into preprocessed_images/ (300x300 PNG)
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("=" * 65)
    print("  VisionCare AI — Dataset Stratification and Cache Generation")
    print("=" * 65)

    df = pd.read_csv(TRAIN_CSV)
    df = df.dropna(subset=["id_code", "diagnosis"]).copy()
    df["diagnosis"] = df["diagnosis"].astype(int)

    # Verify all images exist
    df["raw_path"] = df["id_code"].apply(lambda c: os.path.join(TRAIN_IMG_DIR, f"{c}.png"))
    df["cached_path"] = df["id_code"].apply(lambda c: os.path.join(CACHE_DIR, f"{c}.png"))

    missing = ~df["raw_path"].apply(os.path.exists)
    if missing.any():
        print(f"Warning: {missing.sum()} images missing from {TRAIN_IMG_DIR}")
        df = df[~missing].copy()

    # Stratified 80/20 train/validation split
    train_df, val_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df["diagnosis"],
        random_state=RANDOM_SEED
    )

    train_split_csv = os.path.join(REPORTS_DIR, "train_split.csv")
    val_split_csv = os.path.join(REPORTS_DIR, "validation_split.csv")

    train_df.to_csv(train_split_csv, index=False)
    val_df.to_csv(val_split_csv, index=False)

    print(f"Saved Train Split ({len(train_df)} samples) to: {train_split_csv}")
    print(f"Saved Validation Split ({len(val_df)} samples) to: {val_split_csv}")

    print("\nTrain Split Class Counts:")
    for cls, cnt in train_df["diagnosis"].value_counts().sort_index().items():
        print(f"  Grade {cls}: {cnt:4d} samples ({cnt / len(train_df) * 100:.1f}%)")

    print("\nValidation Split Class Counts:")
    for cls, cnt in val_df["diagnosis"].value_counts().sort_index().items():
        print(f"  Grade {cls}: {cnt:4d} samples ({cnt / len(val_df) * 100:.1f}%)")

    # Multi-core batch image preprocessing
    tasks = [
        (row["raw_path"], row["cached_path"], TARGET_SIZE)
        for _, row in df.iterrows()
    ]
    to_process = [t for t in tasks if force_recompute or not os.path.exists(t[1])]

    print(f"\nProcessing {len(to_process)} / {len(tasks)} images with multi-core auto-cropping...")
    if to_process:
        t0 = time.time()
        max_workers = max(1, os.cpu_count() - 1)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_single_image, to_process))
        elapsed = time.time() - t0
        success_count = sum(results)
        print(f"Successfully processed and cached {success_count} images in {elapsed:.1f}s ({elapsed/len(to_process)*1000:.1f} ms/img)")
    else:
        print("All 3,662 images are already cached in preprocessed_images/.")

    print("=" * 65)
    return train_df, val_df


if __name__ == "__main__":
    prepare_dataset_splits_and_cache(force_recompute=False)
