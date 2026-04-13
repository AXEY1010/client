"""
Unit tests for the DCT feature extraction module.
"""
import numpy as np
import pytest
from src.dct_features import _zigzag_indices, extract_blocks, compute_dct_features, extract_dct_features


class TestZigzagIndices:
    def test_returns_correct_count(self):
        indices = _zigzag_indices(8, 15)
        assert len(indices) == 15

    def test_starts_at_origin(self):
        indices = _zigzag_indices(8, 1)
        assert indices[0] == (0, 0)

    def test_full_matrix(self):
        indices = _zigzag_indices(4, 16)
        assert len(indices) == 16
        # All positions should be unique
        assert len(set(indices)) == 16


class TestExtractBlocks:
    def test_output_shape(self, gray_image):
        blocks, positions = extract_blocks(gray_image, block_size=8, block_step=4)
        assert blocks.ndim == 3
        assert blocks.shape[1] == 8
        assert blocks.shape[2] == 8
        assert positions.shape[0] == blocks.shape[0]
        assert positions.shape[1] == 2

    def test_step_1_gives_most_blocks(self, gray_image):
        blocks_s1, _ = extract_blocks(gray_image, block_size=8, block_step=1)
        blocks_s4, _ = extract_blocks(gray_image, block_size=8, block_step=4)
        assert len(blocks_s1) > len(blocks_s4)

    def test_image_too_small(self):
        tiny = np.zeros((4, 4), dtype=np.uint8)
        blocks, positions = extract_blocks(tiny, block_size=8, block_step=2)
        assert len(blocks) == 0
        assert len(positions) == 0


class TestComputeDCTFeatures:
    def test_feature_shape(self, gray_image):
        blocks, _ = extract_blocks(gray_image, 8, 4)
        features = compute_dct_features(blocks, n_coeffs=15, block_size=8)
        assert features.shape == (len(blocks), 15)

    def test_quantization_reduces_values(self, gray_image):
        blocks, _ = extract_blocks(gray_image, 8, 4)
        feat_q1 = compute_dct_features(blocks, n_coeffs=15, block_size=8, quantization_factor=1)
        feat_q10 = compute_dct_features(blocks, n_coeffs=15, block_size=8, quantization_factor=10)
        # Higher quantization → smaller absolute values
        assert np.mean(np.abs(feat_q10)) <= np.mean(np.abs(feat_q1))

    def test_empty_blocks(self):
        result = compute_dct_features(np.array([]), n_coeffs=15, block_size=8)
        assert len(result) == 0


class TestExtractDCTFeatures:
    def test_end_to_end(self, gray_image):
        features, positions = extract_dct_features(gray_image)
        assert len(features) > 0
        assert features.shape[1] == 15  # default n_coeffs
        assert positions.shape[1] == 2

    def test_different_block_sizes(self, forged_gray):
        f8, p8 = extract_dct_features(forged_gray, block_size=8, block_step=4)
        f16, p16 = extract_dct_features(forged_gray, block_size=16, block_step=4)
        assert len(f8) > len(f16)  # Smaller blocks → more blocks
