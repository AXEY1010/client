"""Error Level Analysis (ELA) feature extraction.

Computes recompression residuals and extracts localized high-error regions
that can indicate post-compression tampering artifacts.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger("cmfd")


def _odd_kernel_size(k: int) -> int:
    """Return a positive odd kernel size."""
    k = max(1, int(k))
    return k if k % 2 == 1 else k + 1


def compute_ela_residual(image_bgr: np.ndarray,
                         jpeg_quality: int = 90,
                         blur_kernel_size: int = 5) -> np.ndarray:
    """Compute grayscale ELA residual map from an input BGR image.

    Args:
        image_bgr: Input image in BGR format.
        jpeg_quality: JPEG quality used for synthetic recompression.
        blur_kernel_size: Smoothing kernel size applied to residual map.

    Returns:
        Grayscale residual map (uint8).
    """
    if image_bgr is None or image_bgr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    quality = int(np.clip(jpeg_quality, 40, 99))
    ok, enc = cv2.imencode(
        ".jpg",
        image_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not ok:
        logger.warning("ELA: JPEG re-encoding failed.")
        return np.zeros(image_bgr.shape[:2], dtype=np.uint8)

    recompressed = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if recompressed is None:
        logger.warning("ELA: JPEG decode failed.")
        return np.zeros(image_bgr.shape[:2], dtype=np.uint8)

    if recompressed.shape[:2] != image_bgr.shape[:2]:
        h, w = image_bgr.shape[:2]
        recompressed = cv2.resize(recompressed, (w, h), interpolation=cv2.INTER_LINEAR)

    diff = cv2.absdiff(image_bgr, recompressed)
    residual = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    k = _odd_kernel_size(blur_kernel_size)
    if k > 1:
        residual = cv2.GaussianBlur(residual, (k, k), 0)

    # Normalize residual dynamic range so subtle JPEG artifacts remain usable.
    max_val = float(np.max(residual))
    if max_val > 0:
        residual = cv2.convertScaleAbs(residual, alpha=(255.0 / max_val))

    return residual


def extract_ela_evidence(image_bgr: np.ndarray,
                         jpeg_quality: int = 90,
                         threshold_percentile: float = 98.0,
                         blur_kernel_size: int = 5,
                         min_area: int = 120,
                         min_region_count: int = 2,
                         max_region_fraction: float = 0.30) -> tuple:
    """Extract ELA mask, regions, and aggregate confidence score.

    Args:
        image_bgr: Input BGR image.
        jpeg_quality: JPEG quality for synthetic recompression.
        threshold_percentile: Percentile threshold for selecting strong residuals.
        blur_kernel_size: Smoothing kernel size for residual map.
        min_area: Minimum contour area for ELA regions.
        min_region_count: Preferred minimum number of ELA regions.
        max_region_fraction: Fractional area upper bound for valid ELA support.

    Returns:
        Tuple of:
            ela_mask: Binary ELA mask (uint8).
            regions: List of region dicts with contour, bbox, area.
            ela_score: Score in [0.0, 1.0].
            region_fraction: Total ELA region area / image area.
    """
    h, w = image_bgr.shape[:2]
    image_area = max(1, h * w)

    residual = compute_ela_residual(image_bgr, jpeg_quality, blur_kernel_size)
    if residual.shape[:2] != (h, w):
        residual = cv2.resize(residual, (w, h), interpolation=cv2.INTER_LINEAR)

    if np.max(residual) <= 0:
        return np.zeros((h, w), dtype=np.uint8), [], 0.0, 0.0

    p = float(np.percentile(residual, np.clip(threshold_percentile, 90.0, 99.9)))
    threshold = max(5.0, 0.85 * p)
    _, raw_mask = cv2.threshold(residual, threshold, 255, cv2.THRESH_BINARY)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, open_kernel)
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    ela_mask = np.zeros((h, w), dtype=np.uint8)
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        regions.append({"contour": cnt, "bbox": (x, y, bw, bh), "area": area})
        cv2.drawContours(ela_mask, [cnt], -1, 255, thickness=-1)

    regions.sort(key=lambda r: r["area"], reverse=True)

    total_region_area = float(sum(r["area"] for r in regions))
    region_fraction = total_region_area / float(image_area)

    mean_level = float(np.mean(residual)) / 255.0
    peak_level = float(np.percentile(residual, 99.0)) / 255.0
    contrast = max(0.0, peak_level - mean_level)

    contrast_score = np.clip((contrast - 0.08) / 0.40, 0.0, 1.0)
    region_score = min(1.0, len(regions) / 6.0)
    area_score = min(1.0, total_region_area / max(1.0, 0.12 * image_area))

    score = 0.55 * contrast_score + 0.25 * region_score + 0.20 * area_score

    if len(regions) < int(max(1, min_region_count)):
        score *= 0.70
    if region_fraction > max_region_fraction:
        score *= 0.55

    ela_score = float(np.clip(score, 0.0, 1.0))
    return ela_mask, regions, round(ela_score, 3), float(region_fraction)
