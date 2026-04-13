"""
DCT Block Matching Module.

Sorts DCT feature vectors lexicographically, then uses a sliding window
comparison to find matching blocks. Computes displacement vectors and
filters by minimum distance. Uses histogram voting to identify dominant
displacement vectors.

Improvements applied:
    #1 — Sliding window comparison (DCT_COMPARE_WINDOW)
    #4 — Minimum vector distance filtering
    #8 — Vector histogram voting
"""

import numpy as np
import logging

logger = logging.getLogger("cmfd")


def lexicographic_sort(features: np.ndarray, positions: np.ndarray
                       ) -> tuple:
    """Sort feature vectors lexicographically.

    Sorts by all columns left-to-right, so similar vectors end up adjacent.

    Args:
        features: Feature matrix of shape (N, D).
        positions: Position array of shape (N, 2).

    Returns:
        Tuple of (sorted_features, sorted_positions, sort_indices).
    """
    # np.lexsort sorts by last key first, so reverse column order
    keys = [features[:, i] for i in range(features.shape[1] - 1, -1, -1)]
    sort_idx = np.lexsort(keys)
    return features[sort_idx], positions[sort_idx], sort_idx


def find_matching_blocks(features: np.ndarray, positions: np.ndarray,
                         compare_window: int = 4,
                         match_threshold: float = 0.0,
                         min_vector_distance: float = 20.0) -> tuple:
    """Find matching blocks using vectorized sliding window comparison.

    After lexicographic sorting, compares each row with the next
    `compare_window` rows using fully vectorized NumPy operations
    (Improvement #1 + #11). No Python for-loops over blocks.

    Args:
        features: Feature matrix (N, D) — already quantized.
        positions: Block positions (N, 2) as (row, col).
        compare_window: Number of adjacent rows to compare.
        match_threshold: Maximum L1 distance for a match (0 = exact).
        min_vector_distance: Minimum displacement magnitude.

    Returns:
        Tuple of:
            matched_pairs: list of ((r1,c1), (r2,c2)) tuples
            displacement_vectors: ndarray of shape (M, 2)
    """
    if len(features) == 0:
        return [], np.array([]).reshape(0, 2)

    n = len(features)
    sorted_features, sorted_positions, _ = lexicographic_sort(
        features, positions
    )

    # Cap on total matches to prevent memory/time issues on repetitive images.
    # DBSCAN downstream is O(n²), so this must be conservative.
    MAX_MATCHES = 5000

    all_pos_i = []
    all_pos_j = []
    all_displacements = []

    for offset in range(1, compare_window + 1):
        if offset >= n:
            break

        # Vectorized difference: compare row[i] with row[i+offset]
        feat_a = sorted_features[:n - offset]
        feat_b = sorted_features[offset:]

        # L1 distance per row (sum of absolute differences)
        diffs = np.sum(np.abs(feat_a - feat_b), axis=1)

        # Mask: which pairs match?
        match_mask = diffs <= match_threshold

        if not np.any(match_mask):
            continue

        # Extract matched positions
        pos_a = sorted_positions[:n - offset][match_mask]
        pos_b = sorted_positions[offset:][match_mask]

        # Compute displacement vectors
        dvec = (pos_b - pos_a).astype(np.int64)

        # Normalize direction: ensure consistent sign
        flip_mask = (dvec[:, 0] < 0) | ((dvec[:, 0] == 0) & (dvec[:, 1] < 0))
        dvec[flip_mask] = -dvec[flip_mask]

        # Filter by minimum displacement distance (Improvement #4)
        dist_sq = dvec[:, 0] ** 2 + dvec[:, 1] ** 2
        dist_mask = dist_sq >= (min_vector_distance ** 2)

        if not np.any(dist_mask):
            continue

        valid_pos_a = pos_a[dist_mask]
        valid_pos_b = pos_b[dist_mask]
        valid_dvec = dvec[dist_mask]

        all_pos_i.append(valid_pos_a)
        all_pos_j.append(valid_pos_b)
        all_displacements.append(valid_dvec)

        # Check cumulative match count
        total = sum(len(d) for d in all_displacements)
        if total > MAX_MATCHES:
            logger.warning(
                f"DCT block matching: exceeded {MAX_MATCHES} matches "
                f"at offset {offset}/{compare_window}. Likely a highly "
                f"textured/patterned image. Capping results."
            )
            break

    # Combine results
    if not all_displacements:
        logger.info(f"DCT block matching: 0 matches found (from {n} blocks)")
        return [], np.array([]).reshape(0, 2)

    combined_pos_i = np.concatenate(all_pos_i, axis=0)
    combined_pos_j = np.concatenate(all_pos_j, axis=0)
    displacements = np.concatenate(all_displacements, axis=0)

    # Truncate if over cap
    if len(displacements) > MAX_MATCHES:
        combined_pos_i = combined_pos_i[:MAX_MATCHES]
        combined_pos_j = combined_pos_j[:MAX_MATCHES]
        displacements = displacements[:MAX_MATCHES]

    # Build matched_pairs list
    matched_pairs = [
        (tuple(a), tuple(b))
        for a, b in zip(combined_pos_i.tolist(), combined_pos_j.tolist())
    ]

    logger.info(f"DCT block matching: {len(matched_pairs)} matches found "
                f"(from {n} blocks)")

    return matched_pairs, displacements


def histogram_voting(displacements: np.ndarray, n_bins: int = 50,
                     min_votes: int = 5) -> tuple:
    """Identify dominant displacement vectors via histogram voting.

    Bins displacement vectors into a 2D histogram and returns vectors
    belonging to bins with vote count >= min_votes (Improvement #8).

    Args:
        displacements: Array of shape (M, 2) with (dx, dy) vectors.
        n_bins: Number of bins per axis.
        min_votes: Minimum votes for a bin to be considered dominant.

    Returns:
        Tuple of:
            dominant_mask: Boolean mask of length M indicating dominant vectors
            histogram: 2D histogram array
            x_edges: Bin edges for x-axis
            y_edges: Bin edges for y-axis
    """
    if len(displacements) == 0:
        return np.array([], dtype=bool), None, None, None

    hist, x_edges, y_edges = np.histogram2d(
        displacements[:, 0], displacements[:, 1], bins=n_bins
    )

    # Find which bin each displacement belongs to
    x_bin = np.clip(
        np.digitize(displacements[:, 0], x_edges) - 1, 0, n_bins - 1
    )
    y_bin = np.clip(
        np.digitize(displacements[:, 1], y_edges) - 1, 0, n_bins - 1
    )

    # Mark vectors in dominant bins
    dominant_mask = hist[x_bin, y_bin] >= min_votes

    n_dominant = np.sum(dominant_mask)
    logger.debug(f"Histogram voting: {n_dominant}/{len(displacements)} "
                 f"vectors in dominant bins (threshold={min_votes})")

    return dominant_mask, hist, x_edges, y_edges


def match_blocks(features: np.ndarray, positions: np.ndarray,
                 compare_window: int = 4,
                 match_threshold: float = 0.0,
                 min_vector_distance: float = 20.0,
                 histogram_bins: int = 50,
                 histogram_min_votes: int = 5) -> tuple:
    """Full block matching pipeline with histogram voting.

    Args:
        features: Quantized DCT feature matrix (N, D).
        positions: Block positions (N, 2).
        compare_window: Sliding window size.
        match_threshold: Maximum feature distance for a match.
        min_vector_distance: Minimum displacement magnitude.
        histogram_bins: Bins for histogram voting.
        histogram_min_votes: Minimum votes for dominant vector.

    Returns:
        Tuple of:
            matched_pairs: Filtered list of ((r1,c1), (r2,c2)) tuples
            displacements: Filtered displacement vectors (M, 2)
            histogram_data: Tuple of (hist, x_edges, y_edges) or None
    """
    matched_pairs, displacements = find_matching_blocks(
        features, positions, compare_window, match_threshold,
        min_vector_distance
    )

    if len(displacements) == 0:
        return [], np.array([]).reshape(0, 2), None

    # Apply histogram voting (Improvement #8)
    dominant_mask, hist, x_edges, y_edges = histogram_voting(
        displacements, histogram_bins, histogram_min_votes
    )

    if np.sum(dominant_mask) > 0:
        filtered_pairs = [p for p, m in zip(matched_pairs, dominant_mask) if m]
        filtered_displacements = displacements[dominant_mask]
    elif len(matched_pairs) <= histogram_min_votes * 2:
        # Very few matches — skip voting, keep all
        filtered_pairs = matched_pairs
        filtered_displacements = displacements
    else:
        # Many matches but no dominant bin — likely noise, discard
        logger.info("Histogram voting: no dominant bins found with many "
                     f"matches ({len(matched_pairs)}). Discarding.")
        filtered_pairs = []
        filtered_displacements = np.array([]).reshape(0, 2)

    logger.info(f"After histogram voting: {len(filtered_pairs)} matches "
                f"retained")

    hist_data = (hist, x_edges, y_edges) if hist is not None else None
    return filtered_pairs, filtered_displacements, hist_data
