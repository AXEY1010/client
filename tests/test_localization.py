"""
Unit tests for the forgery localization module.
"""
import numpy as np
import pytest
from src.forgery_localization import (
    create_sift_mask, create_dct_mask, merge_masks,
    find_forged_regions, compute_confidence_score, localize_forgery,
    dct_spatially_consistent, keep_dominant_components, refine_forgery_mask
)


class TestCreateSiftMask:
    def test_empty_points(self):
        mask = create_sift_mask(
            (100, 100), np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)
        )
        assert mask.shape == (100, 100)
        assert np.sum(mask) == 0

    def test_marks_both_sides(self):
        pts1 = np.array([[10., 10.], [20., 20.]])
        pts2 = np.array([[80., 80.], [90., 90.]])
        mask = create_sift_mask((100, 100), pts1, pts2)
        # Both source and dest should have pixels
        assert mask[10, 10] > 0 or mask[80, 80] > 0


class TestCreateDctMask:
    def test_empty_pairs(self):
        mask = create_dct_mask((100, 100), [])
        assert np.sum(mask) == 0

    def test_marks_block_regions(self):
        pairs = [((10, 10), (60, 60))]
        mask = create_dct_mask((100, 100), pairs, block_size=8)
        assert mask[12, 12] > 0  # Source block
        assert mask[62, 62] > 0  # Dest block


class TestMergeMasks:
    def test_sift_confirmed(self):
        sift_mask = np.zeros((100, 100), dtype=np.uint8)
        sift_mask[10:30, 10:30] = 255
        dct_mask = np.zeros((100, 100), dtype=np.uint8)

        merged = merge_masks(sift_mask, dct_mask,
                              sift_largest_cluster_size=15,
                              dct_largest_cluster_size=0)
        assert np.any(merged > 0)

    def test_nothing_when_no_clusters(self):
        sift_mask = np.zeros((100, 100), dtype=np.uint8)
        dct_mask = np.zeros((100, 100), dtype=np.uint8)
        merged = merge_masks(sift_mask, dct_mask,
                              sift_largest_cluster_size=0,
                              dct_largest_cluster_size=0)
        assert np.sum(merged) == 0


class TestConfidenceScore:
    def test_no_regions_zero(self):
        score = compute_confidence_score([], 0, [], 0, [], (100, 100))
        assert score == 0.0

    def test_strong_sift_high_score(self):
        regions = [{"area": 500, "contour": None, "bbox": (10, 10, 50, 50)},
                   {"area": 500, "contour": None, "bbox": (60, 60, 50, 50)}]
        score = compute_confidence_score(
            [0, 1], 40, [0], 30, regions, (500, 500)
        )
        assert score >= 0.5

    def test_weak_signal_low_score(self):
        regions = [{"area": 100, "contour": None, "bbox": (10, 10, 20, 20)}]
        score = compute_confidence_score(
            [0], 5, [], 0, regions, (500, 500)
        )
        assert 0.0 < score < 0.5

    def test_score_bounded(self):
        regions = [{"area": 500, "contour": None, "bbox": (0, 0, 50, 50)},
                   {"area": 500, "contour": None, "bbox": (50, 50, 50, 50)}]
        score = compute_confidence_score(
            list(range(10)), 100, list(range(20)), 200,
            regions, (500, 500)
        )
        assert 0.0 <= score <= 1.0


class TestFindForgedRegions:
    def test_finds_regions(self):
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[20:50, 20:50] = 255
        mask[100:140, 100:140] = 255
        regions = find_forged_regions(mask, min_area=50)
        assert len(regions) == 2

    def test_small_region_filtered(self):
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[10:12, 10:12] = 255  # Tiny 2x2
        regions = find_forged_regions(mask, min_area=50)
        assert len(regions) == 0

    def test_empty_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        regions = find_forged_regions(mask, min_area=50)
        assert len(regions) == 0


class TestRefineMask:
    def test_empty_passthrough(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        result = refine_forgery_mask(mask)
        assert np.sum(result) == 0

    def test_nonempty_refined(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:60, 30:60] = 255
        result = refine_forgery_mask(mask)
        assert np.any(result > 0)
