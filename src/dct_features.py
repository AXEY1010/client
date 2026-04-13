"""
DCT Block Feature Extraction Module.

Extracts overlapping blocks from a grayscale image, computes the 2D Discrete
Cosine Transform for each block, quantizes coefficients, and returns a compact
feature matrix for block matching.

Uses NumPy stride tricks for vectorized block extraction to avoid nested loops.
"""

import numpy as np
from scipy.fftpack import dct
import logging

logger = logging.getLogger("cmfd")


def _zigzag_indices(n: int, count: int) -> list:
    """Generate zigzag scan indices.

    Args:
        n: Matrix dimension.
        count: Number of indices to return.

    Returns:
        List of (row, col) tuples.
    """
    indices = []
    for s in range(2 * n - 1):
        if s % 2 == 0:
            # even block
            for i in range(min(s, n - 1), max(0, s - n + 1) - 1, -1):
                j = s - i
                if 0 <= j < n:
                    indices.append((i, j))
                    if len(indices) == count:
                        return indices
        else:
            # odd block
            for i in range(max(0, s - n + 1), min(s, n - 1) + 1):
                j = s - i
                if 0 <= j < n:
                    indices.append((i, j))
                    if len(indices) == count:
                        return indices
    return indices


def extract_blocks(gray: np.ndarray, block_size: int, block_step: int
                   ) -> tuple:
    """Extract overlapping blocks.

    Args:
        gray: Grayscale image as 2D numpy array.
        block_size: Size of each square block.
        block_step: Step between adjacent blocks.

    Returns:
        Tuple of (blocks, positions) where:
            blocks: ndarray of shape (N, block_size, block_size)
            positions: ndarray of shape (N, 2) with (row, col) of each block
    """
    h, w = gray.shape
    if h < block_size or w < block_size:
        logger.warning(f"Image ({w}x{h}) smaller than block size "
                       f"({block_size}). Skipping DCT extraction.")
        return np.array([]), np.array([])

    # compute grid size
    n_rows = (h - block_size) // block_step + 1
    n_cols = (w - block_size) // block_step + 1

    logger.debug(f"Extracting {n_rows * n_cols} blocks "
                 f"({n_rows} rows × {n_cols} cols)")

    # extract blocks
    img = gray.astype(np.float64)
    strides = img.strides
    shape = (n_rows, n_cols, block_size, block_size)
    new_strides = (strides[0] * block_step, strides[1] * block_step,
                   strides[0], strides[1])

    blocks_view = np.lib.stride_tricks.as_strided(img, shape=shape,
                                                   strides=new_strides)
    # reshape
    blocks = blocks_view.reshape(-1, block_size, block_size).copy()

    # build positions
    rows = np.arange(n_rows) * block_step
    cols = np.arange(n_cols) * block_step
    row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")
    positions = np.column_stack([row_grid.ravel(), col_grid.ravel()])

    return blocks, positions


def compute_dct_features(blocks: np.ndarray, n_coeffs: int,
                         block_size: int,
                         quantization_factor: int = 10) -> np.ndarray:
    """Compute quantized DCT feature vectors.

    Args:
        blocks: ndarray of shape (N, block_size, block_size).
        n_coeffs: Number of DCT coefficients to keep.
        block_size: Block dimension (for zigzag index computation).
        quantization_factor: Quantization divisor (Improvement #2).

    Returns:
        Feature matrix of shape (N, n_coeffs) with quantized integer features.
    """
    if len(blocks) == 0:
        return np.array([])

    # get zigzag indices
    zz_indices = _zigzag_indices(block_size, n_coeffs)
    zz_rows = np.array([idx[0] for idx in zz_indices])
    zz_cols = np.array([idx[1] for idx in zz_indices])

    # compute 2D DCT
    dct_blocks = dct(dct(blocks, type=2, norm="ortho", axis=2),
                     type=2, norm="ortho", axis=1)

    # extract zigzag coeffs
    features = dct_blocks[:, zz_rows, zz_cols]

    # quantize
    features = np.round(features / quantization_factor).astype(np.int32)

    logger.debug(f"DCT features: {features.shape[0]} blocks × "
                 f"{features.shape[1]} coefficients")

    return features


def extract_dct_features(gray: np.ndarray, block_size: int = 16,
                         block_step: int = 2, n_coeffs: int = 15,
                         quantization_factor: int = 10,
                         min_block_std: float = 0.0) -> tuple:
    """Extract DCT features.

    Args:
        gray: Preprocessed grayscale image.
        block_size: Block dimension.
        block_step: Block overlap step.
        n_coeffs: Number of DCT coefficients.
        quantization_factor: Quantization divisor.
        min_block_std: Minimum block standard deviation.

    Returns:
        Tuple of (features, positions) where:
            features: ndarray of shape (N, n_coeffs)
            positions: ndarray of shape (N, 2) with (row, col)
    """
    blocks, positions = extract_blocks(gray, block_size, block_step)
    if len(blocks) == 0:
        return np.array([]), np.array([])

    if min_block_std > 0:
        texture_std = np.std(blocks.reshape(blocks.shape[0], -1), axis=1)
        keep_mask = texture_std >= float(min_block_std)
        kept = int(np.count_nonzero(keep_mask))

        if kept == 0:
            logger.warning(
                "DCT extraction: all blocks filtered by texture threshold "
                f"(std >= {min_block_std})."
            )
            return np.array([]), np.array([])

        removed = len(blocks) - kept
        if removed > 0:
            logger.info(
                "DCT extraction: filtered low-texture blocks "
                f"{removed}/{len(blocks)} (std < {min_block_std})"
            )

        blocks = blocks[keep_mask]
        positions = positions[keep_mask]

    features = compute_dct_features(blocks, n_coeffs, block_size,
                                    quantization_factor)
    return features, positions
