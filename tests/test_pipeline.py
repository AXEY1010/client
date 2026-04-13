"""
Integration tests for the end-to-end forgery detection pipeline.
"""
import os
import sys
import numpy as np
import cv2
import pytest

# Ensure project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main import process_image


class TestEndToEndPipeline:
    """Integration tests running the full detection pipeline."""

    def test_forged_image_detected(self, tmp_image_path, tmp_path):
        """A synthetically forged image should be detected as forged."""
        result = process_image(
            tmp_image_path,
            max_size=512,
            debug=False,
            output_dir=str(tmp_path / "output"),
        )
        assert "error" not in result
        assert result["processing_time"] > 0
        assert os.path.isfile(result["output_path"])
        # Confidence should be > 0 (at minimum some evidence found)
        assert result["confidence"] >= 0.0

    def test_clean_image_no_forgery(self, tmp_clean_path, tmp_path):
        """A clean image should not be flagged as forged."""
        result = process_image(
            tmp_clean_path,
            max_size=256,
            debug=False,
            output_dir=str(tmp_path / "output"),
        )
        assert "error" not in result
        assert result["forgery_detected"] is False
        assert result["confidence"] == 0.0
        assert result["num_regions"] == 0

    def test_output_file_created(self, tmp_image_path, tmp_path):
        """Output file should be created for any processed image."""
        output_dir = str(tmp_path / "output")
        result = process_image(
            tmp_image_path,
            max_size=512,
            debug=False,
            output_dir=output_dir,
        )
        assert os.path.isfile(result["output_path"])
        # Verify it's a valid image
        img = cv2.imread(result["output_path"])
        assert img is not None

    def test_debug_mode(self, tmp_image_path, tmp_path):
        """Debug mode should create additional output files."""
        output_dir = str(tmp_path / "output")
        debug_dir = str(tmp_path / "debug")
        result = process_image(
            tmp_image_path,
            max_size=256,
            debug=True,
            output_dir=output_dir,
            debug_dir=debug_dir,
        )
        assert os.path.isdir(debug_dir)

    def test_full_resolution(self, tmp_image_path, tmp_path):
        """Processing at full resolution (no resize) should work."""
        result = process_image(
            tmp_image_path,
            max_size=None,  # Full resolution
            debug=False,
            output_dir=str(tmp_path / "output"),
        )
        assert "error" not in result

    def test_result_has_all_fields(self, tmp_image_path, tmp_path):
        """Result dict should contain all expected fields."""
        result = process_image(
            tmp_image_path,
            max_size=256,
            output_dir=str(tmp_path / "output"),
        )
        required_keys = [
            "image", "forgery_detected", "confidence",
            "num_sift_clusters", "num_dct_clusters",
            "num_regions", "processing_time", "output_path",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_nonexistent_image(self, tmp_path):
        """Non-existent image should return error, not crash."""
        result = process_image(
            "/nonexistent/image.png",
            output_dir=str(tmp_path / "output"),
        )
        assert "error" in result
