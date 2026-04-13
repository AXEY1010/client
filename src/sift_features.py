"""
Local Feature Extraction and Matching Module.

Builds a hybrid sparse matching set by combining SIFT and ORB self-matches.
Each branch is filtered with Lowe-style ratio test, spatial separation checks,
and optional RANSAC inlier filtering before matches are merged.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger("cmfd")


def detect_sift_features(gray: np.ndarray,
                         n_features: int = 3000,
                         contrast_threshold: float = 0.02,
                         edge_threshold: float = 10.0,
                         sigma: float = 1.6) -> tuple:
    """Detect SIFT keypoints and compute descriptors.

    Args:
        gray: Grayscale image (uint8).
        n_features: Maximum number of keypoints.
        contrast_threshold: SIFT contrast threshold.
        edge_threshold: SIFT edge threshold.
        sigma: SIFT Gaussian sigma.

    Returns:
        Tuple of (keypoints, descriptors). Descriptors is ndarray of
        shape (K, 128) or None if no keypoints found.
    """
    sift = cv2.SIFT_create(
        nfeatures=n_features,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
        sigma=sigma,
    )

    keypoints, descriptors = sift.detectAndCompute(gray, None)

    n_kps = len(keypoints) if keypoints else 0
    logger.info(f"SIFT: detected {n_kps} keypoints")

    if n_kps == 0:
        return [], None

    return keypoints, descriptors


def detect_orb_features(gray: np.ndarray,
                        n_features: int = 2500,
                        scale_factor: float = 1.2,
                        n_levels: int = 8,
                        fast_threshold: int = 20) -> tuple:
    """Detect ORB keypoints and compute binary descriptors.

    Args:
        gray: Grayscale image (uint8).
        n_features: Maximum number of ORB keypoints.
        scale_factor: ORB pyramid decimation ratio.
        n_levels: Number of ORB pyramid levels.
        fast_threshold: FAST detector threshold in ORB.

    Returns:
        Tuple of (keypoints, descriptors).
    """
    orb = cv2.ORB_create(
        nfeatures=n_features,
        scaleFactor=scale_factor,
        nlevels=n_levels,
        fastThreshold=fast_threshold,
    )

    keypoints, descriptors = orb.detectAndCompute(gray, None)
    n_kps = len(keypoints) if keypoints else 0
    logger.info(f"ORB: detected {n_kps} keypoints")

    if n_kps == 0:
        return [], None

    return keypoints, descriptors


def match_descriptors(descriptors: np.ndarray,
                      ratio_threshold: float = 0.75,
                      min_keypoint_distance: float = 10.0,
                      keypoints: list = None,
                      norm_type: int = cv2.NORM_L2) -> list:
    """Self-match descriptors using BFMatcher + KNN + Lowe's ratio test.

    Matches each descriptor against all others in the same image.
    Filters out self-matches (same or nearby keypoints).

    Args:
        descriptors: Descriptor matrix (K, D).
        ratio_threshold: Lowe's ratio test threshold.
        min_keypoint_distance: Minimum spatial distance between matched points.
        keypoints: List of cv2.KeyPoint objects for spatial filtering.

    Returns:
        List of (idx1, idx2) tuples representing valid match pairs.
    """
    if descriptors is None or len(descriptors) < 2:
        logger.warning("Not enough descriptors for matching")
        return []

    bf = cv2.BFMatcher(norm_type)
    matches = bf.knnMatch(descriptors, descriptors, k=3)
    # k=3 because first match is always self-match (distance=0)

    valid_pairs = []
    seen = set()

    for match_group in matches:
        # Skip the self-match (first result)
        candidates = match_group[1:]  # Skip index 0 (self)
        if len(candidates) < 2:
            continue

        m, n = candidates[0], candidates[1]

        # Lowe's ratio test
        if m.distance >= ratio_threshold * n.distance:
            continue

        idx1, idx2 = m.queryIdx, m.trainIdx

        # Avoid duplicate pairs
        pair_key = (min(idx1, idx2), max(idx1, idx2))
        if pair_key in seen:
            continue

        # Spatial distance check
        if keypoints is not None:
            pt1 = np.array(keypoints[idx1].pt)
            pt2 = np.array(keypoints[idx2].pt)
            dist = np.linalg.norm(pt1 - pt2)
            if dist < min_keypoint_distance:
                continue

        seen.add(pair_key)
        valid_pairs.append((idx1, idx2))

    logger.info(f"Descriptor matching: {len(valid_pairs)} valid pairs "
                f"(from {len(matches)} raw matches)")

    return valid_pairs


def ransac_filter(keypoints: list, match_pairs: list,
                  reproj_threshold: float = 5.0,
                  min_matches: int = 8) -> list:
    """Filter matches using RANSAC homography estimation (Improvement #5).

    Fits a homography to the matched keypoint positions and retains
    only inlier matches.

    Args:
        keypoints: List of cv2.KeyPoint objects.
        match_pairs: List of (idx1, idx2) tuples.
        reproj_threshold: RANSAC reprojection error threshold.
        min_matches: Minimum number of matches to attempt RANSAC.

    Returns:
        Filtered list of (idx1, idx2) tuples (inliers only).
    """
    min_required = max(min_matches, 4)
    if len(match_pairs) < min_required:
        logger.debug(f"Too few matches ({len(match_pairs)}) for RANSAC "
                     f"(need {min_required}). Skipping.")
        return match_pairs

    pts1 = np.float32([keypoints[i].pt for i, _ in match_pairs])
    pts2 = np.float32([keypoints[j].pt for _, j in match_pairs])

    _, inlier_mask = cv2.findHomography(
        pts1, pts2, cv2.RANSAC, reproj_threshold
    )

    if inlier_mask is None:
        logger.debug("RANSAC failed to find homography. Rejecting matches.")
        return []

    inlier_mask = inlier_mask.ravel().astype(bool)
    filtered = [p for p, m in zip(match_pairs, inlier_mask) if m]

    logger.info(f"RANSAC: {len(filtered)}/{len(match_pairs)} inliers "
                f"retained")

    return filtered


def compute_displacement_vectors(keypoints: list, match_pairs: list
                                  ) -> tuple:
    """Compute displacement vectors for matched keypoint pairs.

    Args:
        keypoints: List of cv2.KeyPoint objects.
        match_pairs: List of (idx1, idx2) tuples.

    Returns:
        Tuple of:
            points1: ndarray (M, 2) — source points
            points2: ndarray (M, 2) — destination points
            displacements: ndarray (M, 2) — (dx, dy) vectors
    """
    if not match_pairs:
        empty = np.array([]).reshape(0, 2)
        return empty, empty, empty

    points1 = np.array([keypoints[i].pt for i, _ in match_pairs])
    points2 = np.array([keypoints[j].pt for _, j in match_pairs])
    displacements = points2 - points1

    return points1, points2, displacements


def merge_feature_matches(sift_pts1: np.ndarray,
                          sift_pts2: np.ndarray,
                          orb_pts1: np.ndarray,
                          orb_pts2: np.ndarray) -> tuple:
    """Merge SIFT and ORB matched point pairs, removing duplicates."""
    merged1 = []
    merged2 = []
    seen = set()

    def add_pairs(pts1: np.ndarray, pts2: np.ndarray):
        if pts1 is None or len(pts1) == 0:
            return
        for pt1, pt2 in zip(pts1, pts2):
            key = (
                int(round(float(pt1[0]))),
                int(round(float(pt1[1]))),
                int(round(float(pt2[0]))),
                int(round(float(pt2[1]))),
            )
            if key in seen:
                continue
            seen.add(key)
            merged1.append([float(pt1[0]), float(pt1[1])])
            merged2.append([float(pt2[0]), float(pt2[1])])

    add_pairs(sift_pts1, sift_pts2)
    add_pairs(orb_pts1, orb_pts2)

    if not merged1:
        empty = np.array([]).reshape(0, 2)
        return empty, empty, empty

    points1 = np.asarray(merged1, dtype=np.float32)
    points2 = np.asarray(merged2, dtype=np.float32)
    displacements = points2 - points1
    return points1, points2, displacements


def extract_orb_matches(gray: np.ndarray,
                        n_features: int = 2500,
                        scale_factor: float = 1.2,
                        n_levels: int = 8,
                        fast_threshold: int = 20,
                        ratio_threshold: float = 0.85,
                        min_keypoint_distance: float = 10.0,
                        apply_ransac: bool = True,
                        ransac_reproj: float = 5.0,
                        ransac_min: int = 8) -> tuple:
    """Extract ORB keypoints and robust self-matches."""
    keypoints, descriptors = detect_orb_features(
        gray, n_features, scale_factor, n_levels, fast_threshold
    )

    if descriptors is None:
        empty = np.array([]).reshape(0, 2)
        return keypoints, [], empty, empty, empty

    match_pairs = match_descriptors(
        descriptors,
        ratio_threshold,
        min_keypoint_distance,
        keypoints,
        norm_type=cv2.NORM_HAMMING,
    )

    if apply_ransac and match_pairs:
        match_pairs = ransac_filter(
            keypoints, match_pairs, ransac_reproj, ransac_min
        )

    points1, points2, displacements = compute_displacement_vectors(
        keypoints, match_pairs
    )

    return keypoints, match_pairs, points1, points2, displacements


def extract_sift_matches(gray: np.ndarray,
                         n_features: int = 3000,
                         contrast_threshold: float = 0.02,
                         edge_threshold: float = 10.0,
                         sigma: float = 1.6,
                         ratio_threshold: float = 0.75,
                         min_keypoint_distance: float = 10.0,
                         apply_ransac: bool = True,
                         ransac_reproj: float = 5.0,
                         ransac_min: int = 8,
                         enable_orb: bool = True,
                         orb_features: int = 2500,
                         orb_scale_factor: float = 1.2,
                         orb_n_levels: int = 8,
                         orb_fast_threshold: int = 20,
                         orb_ratio_threshold: float = 0.85) -> tuple:
    """Hybrid SIFT+ORB extraction and matching pipeline.

    Args:
        gray: Grayscale input image.
        n_features: Max SIFT keypoints.
        contrast_threshold: SIFT contrast threshold.
        edge_threshold: SIFT edge threshold.
        sigma: SIFT sigma.
        ratio_threshold: Lowe's ratio test threshold.
        min_keypoint_distance: Min distance to reject near-matches.
        apply_ransac: Whether to apply RANSAC filtering.
        ransac_reproj: RANSAC reprojection threshold.
        ransac_min: Min matches for RANSAC.

    Returns:
        Tuple of:
            keypoints: SIFT keypoints (for debug visualization)
            match_pairs: SIFT inlier pairs (for debug visualization)
            points1: merged SIFT+ORB source points for clustering
            points2: merged SIFT+ORB destination points for clustering
            displacements: merged displacement vectors
    """
    keypoints, descriptors = detect_sift_features(
        gray, n_features, contrast_threshold, edge_threshold, sigma
    )

    if descriptors is None:
        empty = np.array([]).reshape(0, 2)
        return keypoints, [], empty, empty, empty

    match_pairs = match_descriptors(
        descriptors,
        ratio_threshold,
        min_keypoint_distance,
        keypoints,
        norm_type=cv2.NORM_L2,
    )

    # RANSAC filtering (Improvement #5)
    if apply_ransac and match_pairs:
        match_pairs = ransac_filter(
            keypoints, match_pairs, ransac_reproj, ransac_min
        )

    sift_points1, sift_points2, _ = compute_displacement_vectors(
        keypoints, match_pairs
    )

    orb_points1 = np.array([]).reshape(0, 2)
    orb_points2 = np.array([]).reshape(0, 2)
    orb_match_count = 0

    if enable_orb:
        (_, orb_pairs,
         orb_points1, orb_points2,
         _) = extract_orb_matches(
            gray,
            n_features=orb_features,
            scale_factor=orb_scale_factor,
            n_levels=orb_n_levels,
            fast_threshold=orb_fast_threshold,
            ratio_threshold=orb_ratio_threshold,
            min_keypoint_distance=min_keypoint_distance,
            apply_ransac=apply_ransac,
            ransac_reproj=ransac_reproj,
            ransac_min=ransac_min,
        )
        orb_match_count = len(orb_pairs)

    points1, points2, displacements = merge_feature_matches(
        sift_points1,
        sift_points2,
        orb_points1,
        orb_points2,
    )

    logger.info(
        "Merged sparse matches: "
        f"SIFT={len(match_pairs)}, ORB={orb_match_count}, total={len(points1)}"
    )

    return keypoints, match_pairs, points1, points2, displacements
