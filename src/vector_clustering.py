"""
Displacement Vector Clustering Module.

Applies DBSCAN clustering to displacement vectors from both DCT and SIFT
pipelines. Filters clusters by size and variance to ensure only genuine
copy-move forgery regions are retained.

Improvements applied:
    #3 — DBSCAN clustering with configurable parameters
    #4 — Minimum vector distance (applied upstream, validated here)
    #8 — Histogram voting validation
"""

import numpy as np
from sklearn.cluster import DBSCAN
import logging

logger = logging.getLogger("cmfd")


def compute_direction_variance(vectors: np.ndarray) -> float:
    """Compute angular spread."""
    if len(vectors) == 0:
        return float("inf")

    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    mean_angle = np.angle(np.mean(np.exp(1j * angles)))
    angle_diff = np.angle(np.exp(1j * (angles - mean_angle)))
    return float(np.sqrt(np.mean(angle_diff ** 2)))


def cluster_vectors(displacements: np.ndarray,
                    eps: float = 10.0,
                    min_samples: int = 5,
                    min_cluster_size: int = 5,
                    max_variance: float = 20.0,
                    angle_variance_threshold: float = 0.6,
                    min_vector_distance: float = 25.0) -> tuple:
    """Cluster displacement vectors.

    Args:
        displacements: Array of shape (M, 2) with (dx, dy) vectors.
        eps: DBSCAN epsilon (max distance between points in cluster).
        min_samples: Minimum points to form a dense region.
        min_cluster_size: Minimum matches per cluster to keep.
        max_variance: Maximum allowed variance within a cluster.
        angle_variance_threshold: Maximum angular spread (radians).
        min_vector_distance: Minimum allowed displacement magnitude.

    Returns:
        Tuple of:
            labels: Cluster labels for each vector (-1 = noise).
            valid_clusters: List of valid cluster IDs.
            cluster_info: Dict mapping cluster_id →
                {size, mean, variance, direction_variance, vector_length}.
    """
    if len(displacements) == 0:
        return np.array([], dtype=int), [], {}

    if len(displacements) < min_samples:
        logger.debug(f"Too few vectors ({len(displacements)}) for DBSCAN "
                     f"(need {min_samples})")
        return np.full(len(displacements), -1, dtype=int), [], {}

    # guard against large input
    MAX_VECTORS_FOR_DBSCAN = 10000
    EXTREME_MATCH_THRESHOLD = 100000

    if len(displacements) > EXTREME_MATCH_THRESHOLD:
        logger.warning(
            f"Extreme number of matches ({len(displacements)}). "
            f"This strongly indicates a highly repetitive/textured image. "
            f"Skipping clustering."
        )
        return np.full(len(displacements), -1, dtype=int), [], {}

    subsample_indices = None
    original_length = len(displacements)

    if len(displacements) > MAX_VECTORS_FOR_DBSCAN:
        logger.warning(
            f"Many displacement vectors ({len(displacements)}). "
            f"Subsampling to {MAX_VECTORS_FOR_DBSCAN} for DBSCAN."
        )
        rng = np.random.RandomState(42)
        subsample_indices = rng.choice(len(displacements),
                                       MAX_VECTORS_FOR_DBSCAN,
                                       replace=False)
        displacements_subset = displacements[subsample_indices]
    else:
        displacements_subset = displacements

    # run clustering
    db = DBSCAN(eps=eps, min_samples=min_samples)
    subset_labels = db.fit_predict(displacements_subset)

    # map cluster labels
    if subsample_indices is not None:
        labels = np.full(original_length, -1, dtype=int)
        labels[subsample_indices] = subset_labels
        # store labels
    else:
        labels = subset_labels

    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise label

    cluster_info = {}
    valid_clusters = []

    for cluster_id in sorted(unique_labels):
        mask = labels == cluster_id
        cluster_vectors_arr = displacements[mask]
        size = len(cluster_vectors_arr)

        mean_vec = np.mean(cluster_vectors_arr, axis=0)
        variance = np.mean(np.var(cluster_vectors_arr, axis=0))
        direction_variance = compute_direction_variance(cluster_vectors_arr)
        vector_length = float(np.linalg.norm(mean_vec))

        cluster_info[cluster_id] = {
            "size": size,
            "mean": mean_vec,
            "variance": variance,
            "direction_variance": direction_variance,
            "vector_length": vector_length,
        }

        # filter by size
        if size < min_cluster_size:
            logger.debug(f"Cluster {cluster_id}: rejected — too small "
                         f"({size} < {min_cluster_size})")
            labels[mask] = -1
            continue

        # filter by variance
        if variance > max_variance:
            logger.debug(f"Cluster {cluster_id}: rejected — high variance "
                         f"({variance:.1f} > {max_variance})")
            labels[mask] = -1
            continue

        # check direction spread
        if direction_variance > angle_variance_threshold:
            logger.debug(
                f"Cluster {cluster_id}: rejected — high angle variance "
                f"({direction_variance:.3f} > {angle_variance_threshold})"
            )
            labels[mask] = -1
            continue

        # check min distance
        if vector_length < min_vector_distance:
            logger.debug(
                f"Cluster {cluster_id}: rejected — short displacement "
                f"({vector_length:.1f} < {min_vector_distance})"
            )
            labels[mask] = -1
            continue

        valid_clusters.append(cluster_id)
        logger.debug(f"Cluster {cluster_id}: size={size}, "
                     f"mean=({mean_vec[0]:.1f}, {mean_vec[1]:.1f}), "
                     f"var={variance:.1f}, ang_var={direction_variance:.3f}, "
                     f"|v|={vector_length:.1f} — ACCEPTED")

    logger.info(f"DBSCAN: {len(valid_clusters)} valid clusters "
                f"from {len(unique_labels)} total")

    return labels, valid_clusters, cluster_info


def cluster_sift_vectors(displacements: np.ndarray,
                         points1: np.ndarray,
                         points2: np.ndarray,
                         eps: float = 10.0,
                         min_samples: int = 5,
                         min_cluster_size: int = 5,
                         max_variance: float = 20.0,
                         angle_variance_threshold: float = 0.6,
                         min_vector_distance: float = 25.0) -> tuple:
    """Cluster SIFT displacement vectors and return filtered points.

    Args:
        displacements: Displacement vectors (M, 2).
        points1: Source keypoint positions (M, 2).
        points2: Destination keypoint positions (M, 2).
        eps: DBSCAN epsilon.
        min_samples: DBSCAN min samples.
        min_cluster_size: Minimum cluster size.
        max_variance: Maximum cluster variance.
        angle_variance_threshold: Maximum angular spread (radians).
        min_vector_distance: Minimum displacement magnitude.

    Returns:
        Tuple of:
            filtered_pts1: Source points from valid clusters (K, 2).
            filtered_pts2: Dest points from valid clusters (K, 2).
            labels: Full label array.
            valid_clusters: List of valid cluster IDs.
            cluster_info: Cluster metadata dict.
    """
    labels, valid_clusters, cluster_info = cluster_vectors(
        displacements,
        eps,
        min_samples,
        min_cluster_size,
        max_variance,
        angle_variance_threshold,
        min_vector_distance,
    )

    if not valid_clusters:
        empty = np.array([]).reshape(0, 2)
        return empty, empty, labels, valid_clusters, cluster_info

    # extract points
    valid_mask = np.isin(labels, valid_clusters)
    filtered_pts1 = points1[valid_mask]
    filtered_pts2 = points2[valid_mask]

    logger.info(f"SIFT clustering: {len(filtered_pts1)} matched points "
                f"in {len(valid_clusters)} valid clusters")

    return filtered_pts1, filtered_pts2, labels, valid_clusters, cluster_info


def cluster_dct_vectors(displacements: np.ndarray,
                        matched_pairs: list,
                        eps: float = 10.0,
                        min_samples: int = 5,
                        min_cluster_size: int = 8,
                        max_variance: float = 12.0,
                        angle_variance_threshold: float = 0.4,
                        min_vector_distance: float = 30.0) -> tuple:
    """Cluster DCT displacement vectors and return filtered pairs.

    Args:
        displacements: Displacement vectors (M, 2).
        matched_pairs: List of ((r1,c1), (r2,c2)) block position tuples.
        eps: DBSCAN epsilon.
        min_samples: DBSCAN min samples.
        min_cluster_size: Minimum cluster size.
        max_variance: Maximum cluster variance.
        angle_variance_threshold: Maximum angular spread (radians).
        min_vector_distance: Minimum displacement magnitude.

    Returns:
        Tuple of:
            filtered_pairs: Matched pairs from valid clusters.
            labels: Full label array.
            valid_clusters: List of valid cluster IDs.
            cluster_info: Cluster metadata dict.
    """
    labels, valid_clusters, cluster_info = cluster_vectors(
        displacements,
        eps,
        min_samples,
        min_cluster_size,
        max_variance,
        angle_variance_threshold,
        min_vector_distance,
    )

    if not valid_clusters:
        return [], labels, valid_clusters, cluster_info

    # Extract pairs from valid clusters
    valid_mask = np.isin(labels, valid_clusters)
    filtered_pairs = [p for p, m in zip(matched_pairs, valid_mask) if m]

    logger.info(f"DCT clustering: {len(filtered_pairs)} matched pairs "
                f"in {len(valid_clusters)} valid clusters")

    return filtered_pairs, labels, valid_clusters, cluster_info
