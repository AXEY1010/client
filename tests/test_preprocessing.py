"""
Unit tests for the preprocessing module.
"""
import numpy as np
import cv2
import pytest
from src.preprocessing import load_image, resize_image, preprocess


class TestLoadImage:
    def test_load_valid_image(self, tmp_clean_path):
        img = load_image(tmp_clean_path)
        assert img is not None
        assert img.ndim == 3
        assert img.shape[2] == 3

    def test_load_nonexistent_returns_none(self):
        img = load_image("/nonexistent/path/image.png")
        assert img is None

    def test_load_corrupt_file(self, tmp_path):
        bad_file = tmp_path / "corrupt.png"
        bad_file.write_text("not an image")
        img = load_image(str(bad_file))
        assert img is None


class TestResizeImage:
    def test_no_resize_when_small(self, clean_image):
        result = resize_image(clean_image, max_size=512)
        assert result.shape == clean_image.shape

    def test_resize_large_image(self):
        big = np.zeros((2000, 3000, 3), dtype=np.uint8)
        result = resize_image(big, max_size=1024)
        assert max(result.shape[:2]) == 1024

    def test_aspect_ratio_preserved(self):
        img = np.zeros((400, 800, 3), dtype=np.uint8)
        result = resize_image(img, max_size=400)
        h, w = result.shape[:2]
        assert w == 400
        assert 190 <= h <= 210  # ~200


class TestPreprocess:
    def test_returns_tuple(self, clean_image):
        original, gray = preprocess(clean_image, max_size=None)
        assert original.ndim == 3
        assert gray.ndim == 2

    def test_gray_same_size_as_original(self, clean_image):
        original, gray = preprocess(clean_image, max_size=None)
        assert gray.shape[:2] == original.shape[:2]

    def test_resize_applied(self):
        big = np.zeros((2000, 2000, 3), dtype=np.uint8)
        original, gray = preprocess(big, max_size=512)
        assert max(original.shape[:2]) == 512

    def test_none_input_raises(self):
        with pytest.raises(ValueError):
            preprocess(None)

    def test_histogram_eq_changes_values(self, clean_image):
        _, gray_eq = preprocess(clean_image, apply_hist_eq=True, max_size=None)
        _, gray_no = preprocess(clean_image, apply_hist_eq=False, max_size=None)
        assert not np.array_equal(gray_eq, gray_no)
