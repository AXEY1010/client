"""
Shared test fixtures for Copy-Move Forgery Detection test suite.
"""

import os
import sys
import numpy as np
import cv2
import pytest

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ==============================================================================
# Synthetic Image Fixtures
# ==============================================================================

@pytest.fixture
def clean_image():
    """Generate a 256x256 clean image with random shapes (no forgery)."""
    np.random.seed(42)
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    for _ in range(30):
        x, y = np.random.randint(0, 256, 2)
        r = np.random.randint(5, 25)
        color = tuple(np.random.randint(40, 220, 3).tolist())
        cv2.circle(img, (int(x), int(y)), int(r), color, -1)
    noise = np.random.normal(0, 8, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


@pytest.fixture
def forged_image():
    """Generate a 512x512 image with a clear copy-move forgery."""
    np.random.seed(99)
    img = np.zeros((512, 512, 3), dtype=np.uint8)

    # Rich textured background
    for _ in range(50):
        x, y = np.random.randint(0, 512, 2)
        r = np.random.randint(10, 40)
        color = tuple(np.random.randint(50, 220, 3).tolist())
        cv2.circle(img, (int(x), int(y)), int(r), color, -1)
    for _ in range(30):
        x1, y1 = np.random.randint(0, 400, 2)
        x2 = int(x1) + np.random.randint(20, 80)
        y2 = int(y1) + np.random.randint(20, 80)
        color = tuple(np.random.randint(50, 220, 3).tolist())
        cv2.rectangle(img, (int(x1), int(y1)), (x2, y2), color, -1)

    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Copy-move: copy 120x120 region from (50,50) to (300,300)
    source = img[50:170, 50:170].copy()
    img[300:420, 300:420] = source
    return img


@pytest.fixture
def forged_image_with_gt(forged_image):
    """Return forged image + ground truth mask."""
    gt = np.zeros((512, 512), dtype=np.uint8)
    gt[50:170, 50:170] = 255    # Source region
    gt[300:420, 300:420] = 255   # Copy region
    return forged_image, gt


@pytest.fixture
def gray_image(clean_image):
    """Grayscale version of clean_image."""
    return cv2.cvtColor(clean_image, cv2.COLOR_BGR2GRAY)


@pytest.fixture
def forged_gray(forged_image):
    """Grayscale version of forged_image."""
    return cv2.cvtColor(forged_image, cv2.COLOR_BGR2GRAY)


@pytest.fixture
def tiny_image():
    """Very small 16x16 image for edge case testing."""
    return np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)


@pytest.fixture
def tiled_image():
    """Tiled/patterned image that should NOT trigger forgery detection."""
    tile = np.random.randint(50, 200, (32, 32, 3), dtype=np.uint8)
    return np.tile(tile, (8, 8, 1))


@pytest.fixture
def tmp_image_path(forged_image, tmp_path):
    """Save forged image to a temp file and return path."""
    path = str(tmp_path / "test_forged.png")
    cv2.imwrite(path, forged_image)
    return path


@pytest.fixture
def tmp_clean_path(clean_image, tmp_path):
    """Save clean image to a temp file and return path."""
    path = str(tmp_path / "test_clean.png")
    cv2.imwrite(path, clean_image)
    return path
