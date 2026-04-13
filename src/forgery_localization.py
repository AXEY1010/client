"""
Forgery Localization Module.

Converts matched keypoints and block positions into spatial forgery masks.
Applies morphological operations and contour analysis to produce clean
detection regions with bounding boxes.

Implements Improvement #7 — Refined forgery confirmation rule:
    Forgery is confirmed only when displacement clusters exist.
    Isolated detections are ignored.

Improvement #10 — Confidence scoring:
    Returns a continuous confidence score [0.0–1.0] instead of a
    binary flag, based on cluster strength, agreement, and region quality.
"""

import cv2
import numpy as np
import logging
from src.ela_features import extract_ela_evidence

logger = logging.getLogger("cmfd")


def create_sift_mask(image_shape: tuple,
                     points1: np.ndarray,
                     points2: np.ndarray,
                     dilation_kernel_size: int = 15,
                     dilation_iterations: int = 3) -> np.ndarray:
    """Create a SIFT binary mask.

    Args:
        image_shape: (height, width) of the image.
        points1: Source keypoint positions (K, 2) as (x, y).
        points2: Destination keypoint positions (K, 2) as (x, y).
        dilation_kernel_size: Size of the dilation kernel.
        dilation_iterations: Number of dilation passes.

    Returns:
        Binary mask (uint8, 0 or 255).
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if len(points1) == 0:
        return mask

    # mark keypoints
    for pt in points1:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(mask, (x, y), 3, 255, -1)

    for pt in points2:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(mask, (x, y), 3, 255, -1)

    # form regions
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (dilation_kernel_size, dilation_kernel_size)
    )
    mask = cv2.dilate(mask, kernel, iterations=dilation_iterations)

    return mask


def create_dct_mask(image_shape: tuple,
                    matched_pairs: list,
                    block_size: int = 16,
                    dilation_kernel_size: int = 15,
                    dilation_iterations: int = 2) -> np.ndarray:
    """Create a DCT binary mask.

    Args:
        image_shape: (height, width) of the image.
        matched_pairs: List of ((r1,c1), (r2,c2)) block position tuples.
        block_size: Size of each block.
        dilation_kernel_size: Size of the dilation kernel.
        dilation_iterations: Number of dilation passes.

    Returns:
        Binary mask (uint8, 0 or 255).
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if not matched_pairs:
        return mask

    for (r1, c1), (r2, c2) in matched_pairs:
        # mark source
        r1e = min(r1 + block_size, h)
        c1e = min(c1 + block_size, w)
        mask[r1:r1e, c1:c1e] = 255

        # mark destination
        r2e = min(r2 + block_size, h)
        c2e = min(c2 + block_size, w)
        mask[r2:r2e, c2:c2e] = 255

    # apply morphology
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (dilation_kernel_size, dilation_kernel_size)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=dilation_iterations)

    return mask


def dct_spatially_consistent(dct_mask: np.ndarray,
                             min_area: int = 50,
                             min_regions: int = 2,
                             max_region_fraction: float = 0.35,
                             min_secondary_ratio: float = 0.10,
                             max_regions: int = 8,
                             min_primary_fraction: float = 0.002,
                             min_top2_share: float = 0.65) -> bool:
    """Validate DCT-only detections using spatial checks."""
    h, w = dct_mask.shape[:2]
    image_area = max(1, h * w)

    contours, _ = cv2.findContours(
        dct_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    areas = sorted(
        [cv2.contourArea(cnt) for cnt in contours if cv2.contourArea(cnt) >= min_area],
        reverse=True,
    )

    if len(areas) < min_regions:
        logger.info(
            "DCT standalone rejected: insufficient disconnected regions "
            f"({len(areas)} < {min_regions})"
        )
        return False

    if len(areas) > max_regions:
        logger.info(
            "DCT standalone rejected: too many disconnected regions "
            f"({len(areas)} > {max_regions})"
        )
        return False

    primary_ratio = areas[0] / image_area
    if primary_ratio < min_primary_fraction:
        logger.info(
            "DCT standalone rejected: dominant region too small "
            f"({primary_ratio:.4f} < {min_primary_fraction:.4f})"
        )
        return False

    if primary_ratio > max_region_fraction:
        logger.info(
            "DCT standalone rejected: dominant region covers too much of image "
            f"({primary_ratio:.2f} > {max_region_fraction:.2f})"
        )
        return False

    total_area = max(1.0, float(sum(areas)))
    top2_share = (areas[0] + areas[1]) / total_area
    if top2_share < min_top2_share:
        logger.info(
            "DCT standalone rejected: top-2 region concentration too low "
            f"({top2_share:.2f} < {min_top2_share:.2f})"
        )
        return False

    secondary_ratio = areas[1] / max(1.0, areas[0])
    if secondary_ratio < min_secondary_ratio:
        logger.info(
            "DCT standalone rejected: secondary region too small "
            f"({secondary_ratio:.2f} < {min_secondary_ratio:.2f})"
        )
        return False

    return True


def keep_dominant_components(mask: np.ndarray,
                             min_area: int = 50,
                             max_components: int = 8) -> np.ndarray:
    """Keep only the largest connected components from a binary mask."""
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(
        [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area],
        key=cv2.contourArea,
        reverse=True,
    )

    filtered = np.zeros_like(mask)
    for cnt in contours[:max_components]:
        cv2.drawContours(filtered, [cnt], -1, 255, thickness=-1)

    return filtered


def merge_masks(sift_mask: np.ndarray, dct_mask: np.ndarray,
                sift_largest_cluster_size: int,
                dct_largest_cluster_size: int,
                min_cluster_matches: int = 10,
                min_dct_standalone: int = 80,
                min_area: int = 50,
                dct_max_regions: int = 8,
                dct_min_primary_fraction: float = 0.002,
                dct_min_top2_share: float = 0.65) -> np.ndarray:
    """Merge SIFT and DCT masks.

    Args:
        sift_mask: SIFT detection mask.
        dct_mask: DCT detection mask.
        sift_largest_cluster_size: Largest accepted SIFT cluster size.
        dct_largest_cluster_size: Largest accepted DCT cluster size.
        min_cluster_matches: Min cluster size for baseline confirmation.
        min_dct_standalone: Min DCT matches for standalone confirmation.
        min_area: Minimum region area used for spatial consistency checks.
        dct_max_regions: Max disconnected regions allowed for DCT-only path.
        dct_min_primary_fraction: Min dominant-region area/image ratio.
        dct_min_top2_share: Min top-2 region area share.

    Returns:
        Merged binary mask.
    """
    merged = np.zeros_like(sift_mask)
    sift_confirmed = sift_largest_cluster_size >= min_cluster_matches
    dct_has_cluster = dct_largest_cluster_size >= min_cluster_matches
    dct_standalone_ok = dct_largest_cluster_size >= min_dct_standalone

    if sift_confirmed:
        merged = cv2.bitwise_or(merged, sift_mask)
        # include DCT
        if dct_has_cluster:
            # constrain DCT
            support_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (21, 21)
            )
            sift_support = cv2.dilate(sift_mask, support_kernel, iterations=1)
            guided_dct_mask = cv2.bitwise_and(dct_mask, sift_support)
            if np.any(guided_dct_mask > 0):
                merged = cv2.bitwise_or(merged, guided_dct_mask)
    elif dct_standalone_ok:
        if dct_spatially_consistent(
            dct_mask,
            min_area=min_area,
            max_regions=dct_max_regions,
            min_primary_fraction=dct_min_primary_fraction,
            min_top2_share=dct_min_top2_share,
        ):
            dominant_dct_mask = keep_dominant_components(
                dct_mask,
                min_area=min_area,
                max_components=8,
            )
            logger.info(
                "DCT standalone confirmation: largest cluster "
                f"{dct_largest_cluster_size} (threshold={min_dct_standalone})"
            )
            merged = cv2.bitwise_or(merged, dominant_dct_mask)
        else:
            logger.info("DCT standalone rejected by spatial consistency checks.")
    elif dct_has_cluster:
        logger.info(
            "DCT clusters found but insufficient for standalone confirmation "
            f"({dct_largest_cluster_size} < {min_dct_standalone}). Ignoring."
        )

    return merged


def refine_forgery_mask(mask: np.ndarray,
                        close_kernel_size: int = 5) -> np.ndarray:
    """Refine merged forgery mask with morphological closing."""
    if not np.any(mask > 0):
        return mask

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (close_kernel_size, close_kernel_size)
    )
    refined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, open_kernel)
    return refined


def find_forged_regions(mask: np.ndarray, min_area: int = 50) -> list:
    """Find contours and bounding boxes of forged regions.

    Args:
        mask: Binary detection mask.
        min_area: Minimum contour area to keep.

    Returns:
        List of dicts with keys: contour, bbox (x, y, w, h), area.
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        regions.append({
            "contour": cnt,
            "bbox": (x, y, w, h),
            "area": area,
        })

    regions.sort(key=lambda r: r["area"], reverse=True)

    logger.info(f"Forgery localization: {len(regions)} regions found "
                f"(min area = {min_area})")

    return regions


def compute_confidence_score(sift_valid_clusters: list,
                             sift_largest_cluster_size: int,
                             dct_valid_clusters: list,
                             dct_largest_cluster_size: int,
                             regions: list,
                             image_shape: tuple,
                             min_cluster_matches: int = 5,
                             ela_score: float = 0.0,
                             ela_weight: float = 0.22) -> float:
    """Compute confidence score.

    Args:
        sift_valid_clusters: Valid SIFT cluster IDs.
        sift_largest_cluster_size: Largest SIFT cluster match count.
        dct_valid_clusters: Valid DCT cluster IDs.
        dct_largest_cluster_size: Largest DCT cluster match count.
        regions: List of detected region dicts.
        image_shape: (height, width) of the image.
        min_cluster_matches: Min cluster size for baseline confirmation.
        ela_score: ELA evidence score in [0.0, 1.0].
        ela_weight: ELA confidence contribution weight.

    Returns:
        Confidence score between 0.0 (no evidence) and 1.0 (strong evidence).
    """
    score = 0.0

    n_sift = len(sift_valid_clusters)
    n_dct = len(dct_valid_clusters)
    n_regions = len(regions)

    if n_regions == 0:
        return 0.0

    # compute SIFT score
    if n_sift > 0:
        # apply base score
        sift_base = min(0.20, 0.10 * n_sift)
        # apply size bonus
        sift_size_bonus = min(0.20, 0.20 * (sift_largest_cluster_size /
                                             max(40, min_cluster_matches)))
        score += sift_base + sift_size_bonus

    # compute DCT score
    if n_dct > 0:
        dct_base = min(0.10, 0.05 * n_dct)
        dct_size_bonus = min(0.15, 0.15 * (dct_largest_cluster_size /
                                            max(80, min_cluster_matches)))
        score += dct_base + dct_size_bonus

    # multi-method bonus
    if n_sift > 0 and n_dct > 0:
        score += 0.15

    # check region quality
    h, w = image_shape[:2]
    image_area = max(1, h * w)
    total_region_area = sum(r["area"] for r in regions)
    region_fraction = total_region_area / image_area

    if 0.001 < region_fraction < 0.40:
        # add region score
        region_score = min(0.15, 0.05 * n_regions)
        score += region_score
    elif region_fraction >= 0.40:
        # penalize large region
        score *= 0.5

    # check pairs
    if n_regions == 2:
        score += 0.05

    # add ELA bonus
    if ela_score > 0:
        score += min(0.20, max(0.0, ela_weight) * ela_score)

    return round(min(1.0, max(0.0, score)), 3)


def localize_forgery(image_shape: tuple,
                     sift_pts1: np.ndarray,
                     sift_pts2: np.ndarray,
                     sift_valid_clusters: list,
                     sift_largest_cluster_size: int,
                     dct_matched_pairs: list,
                     dct_valid_clusters: list,
                     dct_largest_cluster_size: int,
                     block_size: int = 16,
                     min_area: int = 50,
                     min_confirmed_regions: int = 1,
                     dilation_kernel_size: int = 15,
                     dilation_iterations: int = 3,
                     min_cluster_matches: int = 10,
                     min_dct_standalone: int = 80,
                     dct_max_regions: int = 8,
                     dct_min_primary_fraction: float = 0.002,
                     dct_min_top2_share: float = 0.65,
                     image_bgr: np.ndarray | None = None,
                     enable_ela: bool = False,
                     ela_jpeg_quality: int = 90,
                     ela_threshold_percentile: float = 98.0,
                     ela_blur_kernel_size: int = 5,
                     ela_min_area: int = 120,
                     ela_min_regions: int = 2,
                     ela_max_region_fraction: float = 0.30,
                     ela_detection_threshold: float = 0.68,
                     ela_confidence_weight: float = 0.22,
                     ela_require_dct_overlap: bool = True,
                     ela_min_dct_overlap_ratio: float = 0.35,
                     ela_min_dct_jaccard: float = 0.06) -> tuple:
    """Full forgery localization pipeline.

    Args:
        image_shape: (height, width) of the image.
        sift_pts1: SIFT source points from valid clusters.
        sift_pts2: SIFT destination points from valid clusters.
        sift_valid_clusters: List of valid SIFT cluster IDs.
        sift_largest_cluster_size: Largest accepted SIFT cluster size.
        dct_matched_pairs: DCT matched pairs from valid clusters.
        dct_valid_clusters: List of valid DCT cluster IDs.
        dct_largest_cluster_size: Largest accepted DCT cluster size.
        block_size: DCT block size.
        min_area: Minimum region area.
        min_confirmed_regions: Minimum number of localized regions required
            to classify as confirmed forgery.
        dilation_kernel_size: Dilation kernel size.
        dilation_iterations: Number of dilation passes.
        min_cluster_matches: Min cluster size for baseline confirmation.
        min_dct_standalone: Min DCT matches for standalone confirmation.
        dct_max_regions: Max disconnected regions allowed for DCT-only path.
        dct_min_primary_fraction: Min dominant-region area/image ratio.
        dct_min_top2_share: Min top-2 region area share.
        image_bgr: Original/preprocessed BGR image for ELA analysis.
        enable_ela: Whether to run ELA evidence extraction.
        ela_jpeg_quality: JPEG quality for synthetic ELA recompression.
        ela_threshold_percentile: Percentile threshold for ELA masking.
        ela_blur_kernel_size: ELA residual smoothing kernel size.
        ela_min_area: Minimum ELA contour area.
        ela_min_regions: Minimum ELA region count for standalone use.
        ela_max_region_fraction: Maximum allowed ELA area/image ratio.
        ela_detection_threshold: ELA score threshold for standalone detect.
        ela_confidence_weight: ELA contribution weight to confidence.
        ela_require_dct_overlap: Require DCT/ELA spatial agreement for fallback.
        ela_min_dct_overlap_ratio: Min overlap area ratio over ELA mask area.
        ela_min_dct_jaccard: Min Jaccard overlap between ELA and DCT masks.

    Returns:
        Tuple of:
            merged_mask: Combined binary detection mask.
            regions: List of detected region dicts.
            forgery_detected: Boolean.
            sift_mask: SIFT-only mask.
            dct_mask: DCT-only mask.
    """
    sift_mask = create_sift_mask(
        image_shape, sift_pts1, sift_pts2,
        dilation_kernel_size, dilation_iterations
    )

    # adjust footprint
    dct_kernel_size = max(5, dilation_kernel_size // 2)
    dct_dilation_iterations = max(1, dilation_iterations - 2)

    dct_mask = create_dct_mask(
        image_shape, dct_matched_pairs, block_size,
        dct_kernel_size, dct_dilation_iterations
    )

    # fallback handling
    if sift_largest_cluster_size <= 0 and sift_valid_clusters:
        sift_largest_cluster_size = min_cluster_matches
    if dct_largest_cluster_size <= 0 and dct_valid_clusters:
        dct_largest_cluster_size = min_cluster_matches

    merged_mask = merge_masks(
        sift_mask, dct_mask,
        sift_largest_cluster_size=sift_largest_cluster_size,
        dct_largest_cluster_size=dct_largest_cluster_size,
        min_cluster_matches=min_cluster_matches,
        min_dct_standalone=min_dct_standalone,
        min_area=min_area,
        dct_max_regions=dct_max_regions,
        dct_min_primary_fraction=dct_min_primary_fraction,
        dct_min_top2_share=dct_min_top2_share,
    )

    # apply close
    merged_mask = refine_forgery_mask(merged_mask, close_kernel_size=5)

    regions = find_forged_regions(merged_mask, min_area)

    # handle ELA cues
    ela_score = 0.0
    if enable_ela and image_bgr is not None and image_bgr.size > 0:
        ela_mask, ela_regions, ela_score, ela_region_fraction = extract_ela_evidence(
            image_bgr,
            jpeg_quality=ela_jpeg_quality,
            threshold_percentile=ela_threshold_percentile,
            blur_kernel_size=ela_blur_kernel_size,
            min_area=ela_min_area,
            min_region_count=ela_min_regions,
            max_region_fraction=ela_max_region_fraction,
        )

        overlap_ratio = 0.0
        jaccard = 0.0
        overlap_ok = True
        if ela_require_dct_overlap:
            ela_area_px = int(np.count_nonzero(ela_mask))
            dct_area_px = int(np.count_nonzero(dct_mask))
            if ela_area_px <= 0 or dct_area_px <= 0:
                overlap_ok = False
            else:
                overlap_mask = cv2.bitwise_and(ela_mask, dct_mask)
                overlap_px = int(np.count_nonzero(overlap_mask))
                overlap_ratio = overlap_px / float(ela_area_px)

                union_mask = cv2.bitwise_or(ela_mask, dct_mask)
                union_px = int(np.count_nonzero(union_mask))
                jaccard = overlap_px / float(max(1, union_px))

                overlap_ok = (
                    overlap_ratio >= ela_min_dct_overlap_ratio and
                    jaccard >= ela_min_dct_jaccard
                )

        # ELA fallback
        if (not regions and ela_score >= ela_detection_threshold and
                len(ela_regions) >= max(1, ela_min_regions) and
                ela_region_fraction <= ela_max_region_fraction and
                overlap_ok):
            merged_mask = refine_forgery_mask(ela_mask, close_kernel_size=5)
            regions = find_forged_regions(merged_mask, max(min_area, ela_min_area))
            if regions:
                logger.info(
                    "ELA standalone confirmation: "
                    f"score={ela_score:.3f}, regions={len(regions)}, "
                    f"area_frac={ela_region_fraction:.3f}, "
                    f"ovr={overlap_ratio:.3f}, jac={jaccard:.3f}"
                )

    # get confidence
    confidence = compute_confidence_score(
        sift_valid_clusters, sift_largest_cluster_size,
        dct_valid_clusters, dct_largest_cluster_size,
        regions, image_shape, min_cluster_matches,
        ela_score=ela_score,
        ela_weight=ela_confidence_weight,
    )

    forgery_detected = len(regions) >= max(1, min_confirmed_regions)

    if not forgery_detected and len(regions) > 0:
        logger.info(
            "Detection rejected: insufficient confirmed regions "
            f"({len(regions)} < {max(1, min_confirmed_regions)})"
        )
        regions = []
        merged_mask = np.zeros_like(merged_mask)
        confidence = 0.0

    return merged_mask, regions, forgery_detected, sift_mask, dct_mask, confidence
