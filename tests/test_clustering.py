"""
Unit tests for the vector clustering module.
"""
import numpy as np
import pytest
from src.vector_clustering import (
    compute_direction_variance, cluster_vectors,
    cluster_sift_vectors, cluster_dct_vectors
)


class TestDirectionVariance:
    def test_identical_vectors(self):
        vectors = np.array([[10, 0], [10, 0], [10, 0]])
        var = compute_direction_variance(vectors)
        assert var == 0.0

    def test_spread_vectors(self):
        vectors = np.array([[10, 0], [0, 10], [-10, 0], [0, -10]])
        var = compute_direction_variance(vectors)
        assert var > 0.5

    def test_empty_returns_inf(self):
        var = compute_direction_variance(np.array([]).reshape(0, 2))
        assert var == float("inf")


class TestClusterVectors:
    def test_single_tight_cluster(self):
        # 20 vectors all near (100, 50)
        np.random.seed(42)
        vecs = np.random.normal([100, 50], [1, 1], size=(20, 2))
        labels, valid, info = cluster_vectors(vecs, eps=6, min_samples=5,
                                               min_cluster_size=5)
        assert len(valid) >= 1
        assert len(labels) == 20

    def test_no_cluster_when_spread(self):
        np.random.seed(42)
        vecs = np.random.uniform(-1000, 1000, size=(30, 2))
        labels, valid, info = cluster_vectors(vecs, eps=2, min_samples=5,
                                               min_cluster_size=5)
        assert len(valid) == 0

    def test_empty_input(self):
        labels, valid, info = cluster_vectors(np.array([]).reshape(0, 2))
        assert len(labels) == 0
        assert len(valid) == 0

    def test_too_few_vectors(self):
        vecs = np.array([[10, 10], [11, 11]])
        labels, valid, info = cluster_vectors(vecs, min_samples=5)
        assert len(valid) == 0

    def test_short_displacement_rejected(self):
        # All vectors very short — should be rejected by min_vector_distance
        vecs = np.random.normal([2, 2], [0.5, 0.5], size=(30, 2))
        labels, valid, info = cluster_vectors(vecs, eps=6, min_samples=5,
                                               min_vector_distance=25)
        assert len(valid) == 0

    def test_subsampling_preserves_label_length(self):
        """Regression test for the DBSCAN subsampling bug."""
        np.random.seed(42)
        # Create many vectors across two clusters
        c1 = np.random.normal([100, 200], [2, 2], size=(6000, 2))
        c2 = np.random.normal([300, 400], [2, 2], size=(6000, 2))
        vecs = np.concatenate([c1, c2], axis=0)

        labels, valid, info = cluster_vectors(
            vecs, eps=10, min_samples=5, min_cluster_size=5,
            min_vector_distance=10
        )
        # The returned labels MUST have the same length as input
        assert len(labels) == len(vecs)


class TestClusterSiftVectors:
    def test_filtering(self):
        np.random.seed(42)
        vecs = np.random.normal([100, 50], [1, 1], size=(20, 2))
        pts1 = np.random.rand(20, 2) * 100
        pts2 = pts1 + vecs

        f1, f2, labels, valid, info = cluster_sift_vectors(
            vecs, pts1, pts2, eps=6, min_samples=5, min_cluster_size=5,
            min_vector_distance=10
        )
        assert len(f1) == len(f2)
        assert len(labels) == 20

    def test_empty_input(self):
        empty = np.array([]).reshape(0, 2)
        f1, f2, labels, valid, info = cluster_sift_vectors(
            empty, empty, empty
        )
        assert len(f1) == 0
        assert len(valid) == 0


class TestClusterDctVectors:
    def test_filtering(self):
        np.random.seed(42)
        vecs = np.random.normal([150, 80], [1, 1], size=(20, 2))
        pairs = [((i, 0), (i+150, 80)) for i in range(20)]

        filtered, labels, valid, info = cluster_dct_vectors(
            vecs, pairs, eps=6, min_samples=5, min_cluster_size=5,
            min_vector_distance=10
        )
        assert len(labels) == 20

    def test_empty_input(self):
        filtered, labels, valid, info = cluster_dct_vectors(
            np.array([]).reshape(0, 2), []
        )
        assert len(filtered) == 0
        assert len(valid) == 0
