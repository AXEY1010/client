"""
Visualization Module.

Renders detection results on the original image with colored overlays,
bounding boxes, match lines, and optional debug visualizations.

Debug outputs (Improvement #9):
    1. SIFT keypoints overlay
    2. Match lines between keypoints
    3. Displacement vector plot
    4. Vector histogram heatmap
    5. Cluster visualization
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger("cmfd")


def draw_detection_overlay(image: np.ndarray,
                           merged_mask: np.ndarray,
                           sift_mask: np.ndarray,
                           dct_mask: np.ndarray,
                           regions: list,
                           sift_pts1: np.ndarray = None,
                           sift_pts2: np.ndarray = None,
                           color_dct=(0, 0, 255),
                           color_sift=(0, 255, 255),
                           color_bbox=(0, 255, 0),
                           alpha: float = 0.4) -> np.ndarray:
    """Draw detection results on the original image.

    Args:
        image: Original BGR image.
        merged_mask: Combined detection mask.
        sift_mask: SIFT-only mask.
        dct_mask: DCT-only mask.
        regions: List of detected region dicts with 'bbox' keys.
        sift_pts1: SIFT source points for drawing match lines.
        sift_pts2: SIFT destination points for drawing match lines.
        color_dct: BGR color for DCT detections.
        color_sift: BGR color for SIFT match lines.
        color_bbox: BGR color for bounding boxes.
        alpha: Overlay transparency.

    Returns:
        Annotated image.
    """
    output = image.copy()

    # Semi-transparent heatmap overlay for merged forgery regions.
    if np.any(merged_mask > 0):
        heat_seed = cv2.GaussianBlur(merged_mask, (5, 5), 0)
        heatmap = cv2.applyColorMap(heat_seed, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(output, 0.7, heatmap, 0.3, 0)
        mask_idx = merged_mask > 0
        output[mask_idx] = blended[mask_idx]

    # DCT mask overlay (red)
    if np.any(dct_mask > 0):
        dct_overlay = output.copy()
        dct_overlay[dct_mask > 0] = color_dct
        output = cv2.addWeighted(dct_overlay, alpha, output, 1 - alpha, 0)

    # SIFT mask overlay (slightly different shade)
    if np.any(sift_mask > 0):
        sift_overlay = output.copy()
        sift_color = (0, 200, 255)  # Orange-ish
        sift_overlay[sift_mask > 0] = sift_color
        output = cv2.addWeighted(sift_overlay, alpha * 0.7, output,
                                  1 - alpha * 0.7, 0)

    # Draw SIFT match lines (yellow)
    if sift_pts1 is not None and len(sift_pts1) > 0:
        for pt1, pt2 in zip(sift_pts1, sift_pts2):
            p1 = (int(round(pt1[0])), int(round(pt1[1])))
            p2 = (int(round(pt2[0])), int(round(pt2[1])))
            cv2.line(output, p1, p2, color_sift, 1, cv2.LINE_AA)
            cv2.circle(output, p1, 3, color_sift, -1)
            cv2.circle(output, p2, 3, color_sift, -1)

    # Draw bounding boxes (green)
    for region in regions:
        x, y, w, h = region["bbox"]
        cv2.rectangle(output, (x, y), (x + w, y + h), color_bbox, 2)

    # Add text label
    forgery_text = "FORGERY DETECTED" if regions else "NO FORGERY"
    text_color = (0, 0, 255) if regions else (0, 255, 0)
    cv2.putText(output, forgery_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2,
                cv2.LINE_AA)

    return output


def save_output(image: np.ndarray, output_path: str):
    """Save result image to disk.

    Args:
        image: Image to save.
        output_path: Output file path.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, image)
    logger.info(f"Result saved: {output_path}")


# ==============================================================================
# Debug Visualizations (Improvement #9)
# ==============================================================================

def debug_save_keypoints(gray: np.ndarray, keypoints: list,
                         output_dir: str, prefix: str = ""):
    """Save SIFT keypoints visualization."""
    img_kp = cv2.drawKeypoints(
        gray, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    path = os.path.join(output_dir, f"{prefix}sift_keypoints.png")
    cv2.imwrite(path, img_kp)
    logger.debug(f"Debug: keypoints saved to {path}")


def debug_save_matches(image: np.ndarray, keypoints: list,
                       match_pairs: list, output_dir: str,
                       prefix: str = ""):
    """Save match lines visualization."""
    output = image.copy()
    for idx1, idx2 in match_pairs:
        pt1 = tuple(map(int, keypoints[idx1].pt))
        pt2 = tuple(map(int, keypoints[idx2].pt))
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.line(output, pt1, pt2, color, 1, cv2.LINE_AA)
        cv2.circle(output, pt1, 3, color, -1)
        cv2.circle(output, pt2, 3, color, -1)

    path = os.path.join(output_dir, f"{prefix}match_lines.png")
    cv2.imwrite(path, output)
    logger.debug(f"Debug: match lines saved to {path}")


def debug_save_displacement_plot(displacements: np.ndarray,
                                  labels: np.ndarray,
                                  output_dir: str,
                                  prefix: str = "",
                                  title: str = "Displacement Vectors"):
    """Save displacement vector scatter plot with cluster coloring."""
    if len(displacements) == 0:
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    if labels is not None and len(labels) == len(displacements):
        unique = set(labels)
        for lbl in sorted(unique):
            mask = labels == lbl
            color = "gray" if lbl == -1 else None
            label = "noise" if lbl == -1 else f"cluster {lbl}"
            ax.scatter(displacements[mask, 0], displacements[mask, 1],
                       s=10, alpha=0.6, label=label, c=color)
    else:
        ax.scatter(displacements[:, 0], displacements[:, 1],
                   s=10, alpha=0.6)

    ax.set_xlabel("dx (pixels)")
    ax.set_ylabel("dy (pixels)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    path = os.path.join(output_dir, f"{prefix}displacement_vectors.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.debug(f"Debug: displacement plot saved to {path}")


def debug_save_histogram(hist: np.ndarray,
                          x_edges: np.ndarray,
                          y_edges: np.ndarray,
                          output_dir: str,
                          prefix: str = ""):
    """Save displacement vector histogram heatmap."""
    if hist is None:
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # Transpose for correct orientation
    im = ax.imshow(hist.T, origin="lower",
                    extent=[x_edges[0], x_edges[-1],
                            y_edges[0], y_edges[-1]],
                    cmap="hot", aspect="auto")
    plt.colorbar(im, ax=ax, label="Vote count")
    ax.set_xlabel("dx (pixels)")
    ax.set_ylabel("dy (pixels)")
    ax.set_title("Displacement Vector Histogram")

    path = os.path.join(output_dir, f"{prefix}vector_histogram.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.debug(f"Debug: histogram saved to {path}")


def debug_save_cluster_heatmap(image_shape: tuple,
                                points1: np.ndarray,
                                points2: np.ndarray,
                                labels: np.ndarray,
                                output_dir: str,
                                prefix: str = ""):
    """Save cluster heatmap overlaid on image dimensions."""
    if len(points1) == 0:
        return

    h, w = image_shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    for pt in points1:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < w and 0 <= y < h:
            heatmap[max(0, y-5):min(h, y+5),
                    max(0, x-5):min(w, x+5)] += 1

    for pt in points2:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < w and 0 <= y < h:
            heatmap[max(0, y-5):min(h, y+5),
                    max(0, x-5):min(w, x+5)] += 1

    if np.max(heatmap) > 0:
        heatmap = (heatmap / np.max(heatmap) * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        path = os.path.join(output_dir, f"{prefix}cluster_heatmap.png")
        cv2.imwrite(path, heatmap_colored)
        logger.debug(f"Debug: cluster heatmap saved to {path}")


def save_debug_outputs(gray: np.ndarray,
                       image: np.ndarray,
                       keypoints: list,
                       match_pairs: list,
                       sift_displacements: np.ndarray,
                       sift_labels: np.ndarray,
                       dct_displacements: np.ndarray,
                       dct_labels: np.ndarray,
                       dct_hist_data: tuple,
                       sift_pts1: np.ndarray,
                       sift_pts2: np.ndarray,
                       output_dir: str,
                       prefix: str = ""):
    """Generate and save all debug visualizations.

    Args:
        gray: Grayscale image.
        image: Original BGR image.
        keypoints: SIFT keypoints.
        match_pairs: SIFT match pairs.
        sift_displacements: SIFT displacement vectors.
        sift_labels: SIFT cluster labels.
        dct_displacements: DCT displacement vectors.
        dct_labels: DCT cluster labels.
        dct_hist_data: Tuple of (hist, x_edges, y_edges) from histogram voting.
        sift_pts1: SIFT source points.
        sift_pts2: SIFT destination points.
        output_dir: Debug output directory.
        prefix: Filename prefix.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. SIFT keypoints
    if keypoints:
        debug_save_keypoints(gray, keypoints, output_dir, prefix)

    # 2. Match lines
    if match_pairs:
        debug_save_matches(image, keypoints, match_pairs, output_dir, prefix)

    # 3. SIFT displacement vector plot
    if len(sift_displacements) > 0:
        debug_save_displacement_plot(
            sift_displacements, sift_labels, output_dir,
            prefix + "sift_", "SIFT Displacement Vectors"
        )

    # 4. DCT displacement vector plot
    if len(dct_displacements) > 0:
        debug_save_displacement_plot(
            dct_displacements, dct_labels, output_dir,
            prefix + "dct_", "DCT Displacement Vectors"
        )

    # 5. DCT histogram
    if dct_hist_data is not None:
        hist, x_edges, y_edges = dct_hist_data
        debug_save_histogram(hist, x_edges, y_edges, output_dir,
                              prefix + "dct_")

    # 6. Cluster heatmap
    if len(sift_pts1) > 0:
        debug_save_cluster_heatmap(
            image.shape, sift_pts1, sift_pts2, sift_labels,
            output_dir, prefix
        )
