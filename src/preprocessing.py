"""
Image Preprocessing Module.

Handles image loading, grayscale conversion, normalization, Gaussian blur,
histogram equalization, and optional resizing for the detection pipeline.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger("cmfd")


def load_image(path: str) -> np.ndarray | None:
    """Load an image from disk.

    Args:
        path: Path to the image file.

    Returns:
        BGR image as numpy array, or None if loading fails.
    """
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        logger.error(f"Failed to load image: {path}")
    return image


def resize_image(image: np.ndarray, max_size: int) -> np.ndarray:
    """Resize image so its largest dimension equals max_size.

    Preserves aspect ratio. If both dimensions are already <= max_size,
    the image is returned unchanged.

    Args:
        image: Input image (BGR or grayscale).
        max_size: Maximum allowed dimension (width or height).

    Returns:
        Resized image.
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image

    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    logger.info(f"Resizing image from {w}x{h} to {new_w}x{new_h} "
                f"(scale={scale:.3f})")
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def preprocess(image: np.ndarray,
               blur_kernel: int = 5,
               blur_sigma: float = 1.0,
               apply_hist_eq: bool = True,
               enable_normalization: bool = True,
               max_size: int | None = 1024) -> tuple:
    """Full preprocessing pipeline.

    Steps:
        1. Optional resize (if max_size is set)
        2. Convert to grayscale
        3. Optional histogram equalization
        4. Optional intensity normalization
        5. Gaussian blur

    Args:
        image: Input BGR image.
        blur_kernel: Gaussian blur kernel size (must be odd).
        blur_sigma: Gaussian blur sigma.
        apply_hist_eq: Whether to apply histogram equalization.
        enable_normalization: Whether to normalize intensity range.
        max_size: Max image dimension. None to skip resizing.

    Returns:
        Tuple of (resized_original_bgr, processed_grayscale).
    """
    if image is None:
        raise ValueError("Input image is None")

    h, w = image.shape[:2]
    if min(h, w) < 32:
        logger.warning(f"Image is very small ({w}x{h}). "
                       "Detection may be unreliable.")

    # Step 1: Resize
    if max_size is not None:
        image = resize_image(image, max_size)

    original = image.copy()

    # Step 2: Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Histogram equalization
    if apply_hist_eq:
        gray = cv2.equalizeHist(gray)

    # Step 4: Intensity normalization
    if enable_normalization:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # Step 5: Gaussian blur
    ksize = (blur_kernel, blur_kernel)
    gray = cv2.GaussianBlur(gray, ksize, blur_sigma)

    return original, gray
