"""
Unit tests for utility functions.
"""
import os
import numpy as np
import pytest
from src.utils import (
    compute_metrics, list_images, find_ground_truth,
    save_results_csv, ensure_dir, timer
)


class TestComputeMetrics:
    def test_perfect_prediction(self):
        gt = np.array([0, 0, 0, 255, 255, 255], dtype=np.uint8)
        pred = np.array([0, 0, 0, 255, 255, 255], dtype=np.uint8)
        m = compute_metrics(pred, gt)
        assert m["accuracy"] == 1.0
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0

    def test_all_wrong(self):
        gt = np.array([0, 0, 255, 255], dtype=np.uint8)
        pred = np.array([255, 255, 0, 0], dtype=np.uint8)
        m = compute_metrics(pred, gt)
        assert m["accuracy"] == 0.0
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0

    def test_partial_overlap(self):
        gt = np.array([0, 255, 255, 0], dtype=np.uint8)
        pred = np.array([0, 255, 0, 0], dtype=np.uint8)
        m = compute_metrics(pred, gt)
        assert m["tp"] == 1
        assert m["fn"] == 1
        assert m["fp"] == 0
        assert m["tn"] == 2
        assert m["recall"] == 0.5
        assert m["precision"] == 1.0

    def test_2d_masks(self):
        gt = np.zeros((10, 10), dtype=np.uint8)
        pred = np.zeros((10, 10), dtype=np.uint8)
        gt[2:5, 2:5] = 255
        pred[3:6, 3:6] = 255
        m = compute_metrics(pred, gt)
        assert 0 < m["f1"] < 1.0


class TestListImages:
    def test_finds_images(self, tmp_path):
        (tmp_path / "a.png").write_bytes(b"x")
        (tmp_path / "b.jpg").write_bytes(b"x")
        (tmp_path / "c.txt").write_bytes(b"x")
        result = list_images(str(tmp_path))
        assert len(result) == 2

    def test_skips_ground_truth(self, tmp_path):
        (tmp_path / "photo.png").write_bytes(b"x")
        (tmp_path / "photo_gt.png").write_bytes(b"x")
        (tmp_path / "photo_mask.png").write_bytes(b"x")
        result = list_images(str(tmp_path))
        assert len(result) == 1

    def test_empty_directory(self, tmp_path):
        result = list_images(str(tmp_path))
        assert len(result) == 0

    def test_nonexistent(self):
        result = list_images("/no/such/dir")
        assert len(result) == 0


class TestFindGroundTruth:
    def test_standard_naming(self, tmp_path):
        img = tmp_path / "photo.png"
        gt = tmp_path / "photo_gt.png"
        img.write_bytes(b"x")
        gt.write_bytes(b"x")
        result = find_ground_truth(str(img))
        assert result is not None
        assert "photo_gt" in result

    def test_comofod_naming(self, tmp_path):
        """CoMoFoD: 187_F_JC4.jpg → 187_B_JC4.jpg"""
        img = tmp_path / "187_F_JC4.jpg"
        mask = tmp_path / "187_B_JC4.jpg"
        img.write_bytes(b"x")
        mask.write_bytes(b"x")
        result = find_ground_truth(str(img))
        assert result is not None
        assert "187_B_JC4" in result

    def test_sibling_directory(self, tmp_path):
        forged_dir = tmp_path / "forged"
        gt_dir = tmp_path / "gt"
        forged_dir.mkdir()
        gt_dir.mkdir()
        (forged_dir / "image.png").write_bytes(b"x")
        (gt_dir / "image.png").write_bytes(b"x")
        result = find_ground_truth(str(forged_dir / "image.png"))
        assert result is not None

    def test_no_ground_truth(self, tmp_path):
        img = tmp_path / "photo.png"
        img.write_bytes(b"x")
        result = find_ground_truth(str(img))
        assert result is None


class TestSaveResultsCsv:
    def test_creates_file(self, tmp_path):
        results = [
            {"image": "a.png", "forgery_detected": True, "processing_time": 1.0},
            {"image": "b.png", "forgery_detected": False, "processing_time": 0.5},
        ]
        csv_path = str(tmp_path / "results.csv")
        save_results_csv(results, csv_path)
        assert os.path.isfile(csv_path)

    def test_handles_empty(self, tmp_path):
        csv_path = str(tmp_path / "empty.csv")
        save_results_csv([], csv_path)
        assert not os.path.isfile(csv_path)  # No file for empty results


class TestEnsureDir:
    def test_creates_directory(self, tmp_path):
        new_dir = str(tmp_path / "sub" / "dir")
        ensure_dir(new_dir)
        assert os.path.isdir(new_dir)

    def test_existing_dir_ok(self, tmp_path):
        ensure_dir(str(tmp_path))  # Should not raise


class TestTimer:
    def test_measures_time(self, capsys):
        with timer("test block"):
            pass
        captured = capsys.readouterr()
        assert "test block" in captured.out
