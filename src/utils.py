"""
Utility functions for the Copy-Move Forgery Detection system.

Provides timing, metrics computation, file I/O helpers, and logging setup.
"""

import os
import time
import logging
import csv
from contextlib import contextmanager

import numpy as np


# ==============================================================================
# Logging
# ==============================================================================

def setup_logging(debug: bool = False) -> logging.Logger:
    """Configure and return the project logger."""
    logger = logging.getLogger("cmfd")
    if logger.handlers:
        return logger

    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)

    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


# ==============================================================================
# Timing
# ==============================================================================

@contextmanager
def timer(label: str, logger: logging.Logger = None):
    """Context manager that measures and logs elapsed time for a block.

    Usage:
        with timer("DCT extraction"):
            extract_dct_features(...)
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    msg = f"{label}: {elapsed:.3f}s"
    if logger:
        logger.info(msg)
    else:
        print(msg)


# ==============================================================================
# Metrics
# ==============================================================================

def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """Compute pixel-level detection metrics.

    Args:
        pred_mask: Binary prediction mask (0/255 or 0/1).
        gt_mask: Binary ground truth mask (0/255 or 0/1).

    Returns:
        Dictionary with keys: accuracy, precision, recall, f1, tp, fp, fn, tn.
    """
    # Normalize to binary
    pred = (pred_mask > 0).astype(np.uint8).ravel()
    gt = (gt_mask > 0).astype(np.uint8).ravel()

    tp = int(np.sum((pred == 1) & (gt == 1)))
    fp = int(np.sum((pred == 1) & (gt == 0)))
    fn = int(np.sum((pred == 0) & (gt == 1)))
    tn = int(np.sum((pred == 0) & (gt == 0)))

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
           if (precision + recall) > 0 else 0.0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# ==============================================================================
# File I/O
# ==============================================================================

def list_images(directory: str, extensions: tuple = (".png", ".jpg", ".jpeg",
                                                      ".bmp", ".tif", ".tiff")
                ) -> list:
    """Recursively list all image files in a directory.

    Args:
        directory: Root directory to search.
        extensions: Tuple of valid file extensions (case-insensitive).

    Returns:
        Sorted list of absolute image file paths.
    """
    images = []
    if not os.path.isdir(directory):
        return images
    for root, _, files in os.walk(directory):
        for f in files:
            name = f.lower()
            if not name.endswith(extensions):
                continue
            # Skip ground-truth and mask files in batch input discovery.
            if "_gt" in name or "_mask" in name or "groundtruth" in name:
                continue
            images.append(os.path.join(root, f))
    return sorted(images)


def find_ground_truth(image_path: str) -> str | None:
    """Attempt to find a ground truth mask for the given image.

    Supports multiple dataset naming conventions:
        - Standard: <name>_gt.png, <name>_mask.png
        - CoMoFoD:  <id>_F_<variant>.png → <id>_B_<variant>.png (binary mask)
        - MICC:     sibling 'gt' / 'mask' / 'groundtruth' directories

    Returns:
        Path to ground truth file, or None if not found.
    """
    base_dir = os.path.dirname(image_path)
    name, ext = os.path.splitext(os.path.basename(image_path))

    candidates = []

    # --- Pattern 1: Standard naming (same directory) ---------------------
    for suffix in ("_gt", "_mask", "_groundtruth"):
        candidates.append(os.path.join(base_dir, f"{name}{suffix}{ext}"))
        candidates.append(os.path.join(base_dir, f"{name}{suffix}.png"))

    # --- Pattern 2: CoMoFoD convention -----------------------------------
    # Forged: <id>_F_<variant>.ext → Mask: <id>_B_<variant>.ext
    if "_F_" in name:
        mask_name = name.replace("_F_", "_B_", 1)
        candidates.append(os.path.join(base_dir, f"{mask_name}{ext}"))
        candidates.append(os.path.join(base_dir, f"{mask_name}.png"))
        # Check parent and sibling directories
        parent_dir = os.path.dirname(base_dir)
        for subdir in ("mask", "binary", "gt", "masks"):
            d = os.path.join(parent_dir, subdir)
            if os.path.isdir(d):
                candidates.append(os.path.join(d, f"{mask_name}{ext}"))
                candidates.append(os.path.join(d, f"{mask_name}.png"))

    # --- Pattern 3: MICC convention (tamp suffix) -------------------------
    # e.g., DSC_0535tamp133.jpg → look for DSC_0535tamp133_gt.png
    # Already covered by Pattern 1 above.

    # --- Pattern 4: Sibling gt/mask directories --------------------------
    parent_dir = os.path.dirname(base_dir)
    for subdir in ("gt", "mask", "groundtruth", "GT", "Mask",
                    "GroundTruth", "ground_truth", "masks", "binary"):
        gt_dir = os.path.join(parent_dir, subdir)
        if os.path.isdir(gt_dir):
            candidates.append(os.path.join(gt_dir, f"{name}{ext}"))
            candidates.append(os.path.join(gt_dir, f"{name}.png"))
            candidates.append(os.path.join(gt_dir, f"{name}_mask.png"))

    # --- Pattern 5: Same directory, numeric mask (001.png → 001_mask.png) -
    # Try stripping known suffixes and re-appending mask suffix
    import re
    numeric_match = re.match(r"^(\d+)", name)
    if numeric_match:
        num_id = numeric_match.group(1)
        parent_dir = os.path.dirname(base_dir)
        for subdir in ("mask", "gt", "binary", "masks"):
            d = os.path.join(parent_dir, subdir)
            if os.path.isdir(d):
                candidates.append(os.path.join(d, f"{num_id}.png"))
                candidates.append(os.path.join(d, f"{num_id}_mask.png"))

    for path in candidates:
        if os.path.isfile(path):
            return path

    return None


def save_results_csv(results: list, output_path: str):
    """Save batch processing results to CSV.

    Args:
        results: List of dicts with keys like 'image', 'forgery_detected',
                 'num_clusters', 'processing_time', and optional metrics.
        output_path: Path for the output CSV file.
    """
    if not results:
        return

    # Build a stable schema from all rows to avoid key mismatch crashes.
    all_keys = set()
    for row in results:
        all_keys.update(row.keys())

    preferred_order = [
        "image",
        "forgery_detected",
        "confidence",
        "num_sift_clusters",
        "num_dct_clusters",
        "num_regions",
        "processing_time",
        "output_path",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "tn",
        "error",
    ]
    ordered_present = [k for k in preferred_order if k in all_keys]
    remaining = sorted(k for k in all_keys if k not in ordered_present)
    fieldnames = ordered_present + remaining

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            normalized_row = {key: row.get(key, "") for key in fieldnames}
            writer.writerow(normalized_row)


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
