"""
Copy-Move Forgery Detection — Main Entry Point.

Supports single image processing and batch dataset evaluation.

Usage:
    Single image:
        python main.py --image path/to/image.png

    Single image (full resolution, no resize):
        python main.py --image path/to/image.png --full-resolution

    Batch dataset:
        python main.py --dataset dataset/MICC_F220

    Debug mode:
        python main.py --image path/to/image.png --debug
"""

import argparse
import os
import sys
import time
import numpy as np

import config as cfg
from src.preprocessing import load_image, preprocess
from src.dct_features import extract_dct_features
from src.block_matching import match_blocks
from src.sift_features import extract_sift_matches
from src.vector_clustering import cluster_sift_vectors, cluster_dct_vectors
from src.forgery_localization import localize_forgery
from src.visualization import (
    draw_detection_overlay, save_output, save_debug_outputs
)
from src.utils import (
    setup_logging, timer, compute_metrics, list_images,
    find_ground_truth, save_results_csv, ensure_dir
)


def process_image(image_path: str,
                  max_size: int | None = None,
                  debug: bool = False,
                  output_dir: str = None,
                  debug_dir: str = None) -> dict:
    """Run the full forgery detection pipeline on a single image.

    Args:
        image_path: Path to the input image.
        max_size: Max image dimension (None to skip resizing).
        debug: Whether to save debug visualizations.
        output_dir: Directory for output images.
        debug_dir: Directory for debug images.

    Returns:
        Result dict with keys: forgery_detected, num_clusters,
        processing_time, output_path, and optional metrics.
    """
    logger = setup_logging(debug)
    start_time = time.perf_counter()

    basename = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = output_dir or cfg.OUTPUT_DIR
    debug_dir = debug_dir or cfg.DEBUG_OUTPUT_DIR
    ensure_dir(output_dir)

    result = {
        "image": image_path,
        "forgery_detected": False,
        "confidence": 0.0,
        "num_sift_clusters": 0,
        "num_dct_clusters": 0,
        "num_regions": 0,
        "processing_time": 0.0,
        "output_path": "",
    }

    # =========================================================================
    # Step 1: Load & Preprocess
    # =========================================================================
    logger.info(f"Processing: {image_path}")

    raw_image = load_image(image_path)
    if raw_image is None:
        result["error"] = "Failed to load image"
        return result

    with timer("Preprocessing", logger):
        image, gray = preprocess(
            raw_image,
            blur_kernel=cfg.BLUR_KERNEL_SIZE,
            blur_sigma=cfg.BLUR_SIGMA,
            apply_hist_eq=cfg.APPLY_HISTOGRAM_EQ,
            enable_normalization=cfg.ENABLE_NORMALIZATION,
            max_size=max_size,
        )

    h, w = gray.shape
    logger.info(f"Image size: {w}x{h}")

    # =========================================================================
    # Step 2–4: DCT Feature Extraction, Matching & Clustering
    #           (optionally multi-scale)
    # =========================================================================
    all_dct_matched_pairs = []
    all_dct_displacements = []
    all_dct_filtered_pairs = []

    block_sizes = (cfg.MULTISCALE_BLOCK_SIZES
                   if cfg.ENABLE_MULTISCALE_DCT else [cfg.BLOCK_SIZE])

    for bs in block_sizes:
        with timer(f"DCT extraction (block={bs})", logger):
            dct_features, dct_positions = extract_dct_features(
                gray,
                block_size=bs,
                block_step=cfg.BLOCK_STEP,
                n_coeffs=cfg.DCT_COEFFS,
                quantization_factor=cfg.QUANTIZATION_FACTOR,
                min_block_std=cfg.DCT_MIN_BLOCK_STD,
            )

        with timer(f"DCT matching (block={bs})", logger):
            _pairs, _displ, _hist = match_blocks(
                dct_features, dct_positions,
                compare_window=cfg.DCT_COMPARE_WINDOW,
                match_threshold=cfg.DCT_MATCH_THRESHOLD,
                min_vector_distance=cfg.MIN_VECTOR_DISTANCE,
                histogram_bins=cfg.HISTOGRAM_BINS,
                histogram_min_votes=cfg.HISTOGRAM_MIN_VOTES,
            )

        all_dct_matched_pairs.extend(_pairs)
        if len(_displ) > 0:
            all_dct_displacements.append(_displ)

    # Combine multi-scale displacements
    if all_dct_displacements:
        dct_displacements = np.concatenate(all_dct_displacements, axis=0)
    else:
        dct_displacements = np.array([]).reshape(0, 2)
    dct_matched_pairs = all_dct_matched_pairs
    dct_hist_data = _hist if len(block_sizes) == 1 else None

    with timer("DCT vector clustering", logger):
        (dct_filtered_pairs, dct_labels,
         dct_valid_clusters, dct_cluster_info) = cluster_dct_vectors(
            dct_displacements, dct_matched_pairs,
            eps=cfg.DBSCAN_EPS,
            min_samples=cfg.DBSCAN_MIN_SAMPLES,
            min_cluster_size=cfg.DCT_MIN_CLUSTER_MATCHES,
            max_variance=cfg.DCT_VECTOR_VARIANCE_THRESHOLD,
            angle_variance_threshold=cfg.DCT_ANGLE_VARIANCE_THRESHOLD,
            min_vector_distance=cfg.DCT_MIN_VECTOR_DISTANCE,
        )

    # =========================================================================
    # Step 5: SIFT Feature Extraction & Matching
    # =========================================================================
    with timer("SIFT extraction & matching", logger):
        (keypoints, match_pairs,
         sift_pts1, sift_pts2, sift_displacements) = extract_sift_matches(
            gray,
            n_features=cfg.SIFT_FEATURES,
            contrast_threshold=cfg.SIFT_CONTRAST_THRESHOLD,
            edge_threshold=cfg.SIFT_EDGE_THRESHOLD,
            sigma=cfg.SIFT_SIGMA,
            ratio_threshold=cfg.RATIO_TEST,
            min_keypoint_distance=cfg.MIN_KEYPOINT_DISTANCE,
            apply_ransac=cfg.APPLY_RANSAC,
            ransac_reproj=cfg.RANSAC_REPROJ_THRESHOLD,
            ransac_min=cfg.RANSAC_MIN_MATCHES,
            enable_orb=cfg.ENABLE_ORB,
            orb_features=cfg.ORB_FEATURES,
            orb_scale_factor=cfg.ORB_SCALE_FACTOR,
            orb_n_levels=cfg.ORB_N_LEVELS,
            orb_fast_threshold=cfg.ORB_FAST_THRESHOLD,
            orb_ratio_threshold=cfg.ORB_RATIO_TEST,
        )

    # =========================================================================
    # Step 6: SIFT Vector Clustering
    # =========================================================================
    with timer("SIFT vector clustering", logger):
        (sift_filtered_pts1, sift_filtered_pts2,
         sift_labels, sift_valid_clusters,
         sift_cluster_info) = cluster_sift_vectors(
            sift_displacements, sift_pts1, sift_pts2,
            eps=cfg.DBSCAN_EPS,
            min_samples=cfg.DBSCAN_MIN_SAMPLES,
            min_cluster_size=cfg.SIFT_MIN_CLUSTER_MATCHES,
            max_variance=cfg.SIFT_VECTOR_VARIANCE_THRESHOLD,
            angle_variance_threshold=cfg.SIFT_ANGLE_VARIANCE_THRESHOLD,
            min_vector_distance=cfg.SIFT_MIN_VECTOR_DISTANCE,
        )

    sift_largest_cluster_size = max(
        (int(sift_cluster_info[c]["size"]) for c in sift_valid_clusters),
        default=0,
    )
    dct_largest_cluster_size = max(
        (int(dct_cluster_info[c]["size"]) for c in dct_valid_clusters),
        default=0,
    )

    # =========================================================================
    # Step 7: Forgery Localization
    # =========================================================================
    with timer("Forgery localization", logger):
        (merged_mask, regions, forgery_detected,
         sift_mask, dct_mask, confidence) = localize_forgery(
            image.shape,
            sift_filtered_pts1, sift_filtered_pts2,
            sift_valid_clusters, sift_largest_cluster_size,
            dct_filtered_pairs, dct_valid_clusters, dct_largest_cluster_size,
            block_size=cfg.BLOCK_SIZE,
            min_area=cfg.MIN_REGION_AREA,
            min_confirmed_regions=cfg.MIN_CONFIRMED_REGIONS,
            dilation_kernel_size=cfg.DILATION_KERNEL_SIZE,
            dilation_iterations=cfg.DILATION_ITERATIONS,
            min_cluster_matches=cfg.SIFT_MIN_CLUSTER_MATCHES,
            min_dct_standalone=cfg.MIN_DCT_STANDALONE_MATCHES,
            dct_max_regions=cfg.DCT_STANDALONE_MAX_REGIONS,
            dct_min_primary_fraction=cfg.DCT_STANDALONE_MIN_PRIMARY_FRACTION,
            dct_min_top2_share=cfg.DCT_STANDALONE_MIN_TOP2_SHARE,
            image_bgr=image,
            enable_ela=cfg.ENABLE_ELA,
            ela_jpeg_quality=cfg.ELA_JPEG_QUALITY,
            ela_threshold_percentile=cfg.ELA_THRESHOLD_PERCENTILE,
            ela_blur_kernel_size=cfg.ELA_BLUR_KERNEL_SIZE,
            ela_min_area=cfg.ELA_MIN_REGION_AREA,
            ela_min_regions=cfg.ELA_MIN_REGION_COUNT,
            ela_max_region_fraction=cfg.ELA_MAX_REGION_FRACTION,
            ela_detection_threshold=cfg.ELA_DETECTION_THRESHOLD,
            ela_confidence_weight=cfg.ELA_CONFIDENCE_WEIGHT,
            ela_require_dct_overlap=cfg.ELA_REQUIRE_DCT_OVERLAP,
            ela_min_dct_overlap_ratio=cfg.ELA_MIN_DCT_OVERLAP_RATIO,
            ela_min_dct_jaccard=cfg.ELA_MIN_DCT_JACCARD,
        )

    # =========================================================================
    # Step 8: Visualization
    # =========================================================================
    with timer("Visualization", logger):
        output_image = draw_detection_overlay(
            image, merged_mask, sift_mask, dct_mask, regions,
            sift_filtered_pts1, sift_filtered_pts2,
            forgery_detected=forgery_detected,
            valid_cluster_count=(
                len(sift_valid_clusters) + len(dct_valid_clusters)
            ),
            color_dct=cfg.COLOR_DCT_MATCH,
            color_sift=cfg.COLOR_SIFT_MATCH,
            color_bbox=cfg.COLOR_BBOX,
            alpha=cfg.OVERLAY_ALPHA,
        )

        output_path = os.path.join(output_dir,
                                    f"{basename}_detected.png")
        save_output(output_image, output_path)

    # =========================================================================
    # Step 9: Debug Outputs
    # =========================================================================
    if debug:
        with timer("Debug outputs", logger):
            save_debug_outputs(
                gray, image, keypoints, match_pairs,
                sift_displacements, sift_labels,
                dct_displacements, dct_labels,
                dct_hist_data,
                sift_filtered_pts1, sift_filtered_pts2,
                debug_dir, prefix=f"{basename}_",
            )

    # =========================================================================
    # Step 10: Results
    # =========================================================================
    elapsed = time.perf_counter() - start_time

    result.update({
        "forgery_detected": forgery_detected,
        "confidence": confidence,
        "num_sift_clusters": len(sift_valid_clusters),
        "num_dct_clusters": len(dct_valid_clusters),
        "num_regions": len(regions),
        "processing_time": round(elapsed, 3),
        "output_path": output_path,
    })

    # Check for ground truth
    gt_path = find_ground_truth(image_path)
    if gt_path is not None:
        import cv2
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is not None:
            # Resize ground truth to match processed image size
            if gt_mask.shape != gray.shape:
                gt_mask = cv2.resize(gt_mask, (w, h))
            metrics = compute_metrics(merged_mask, gt_mask)
            result.update(metrics)
            logger.info(f"Metrics — Precision: {metrics['precision']:.4f}, "
                        f"Recall: {metrics['recall']:.4f}, "
                        f"F1: {metrics['f1']:.4f}")

    # Print summary
    status = "YES" if forgery_detected else "NO"
    print(f"\n{'='*60}")
    print(f"  Image: {os.path.basename(image_path)}")
    print(f"  Forgery detected: {status}")
    print(f"  Confidence score: {confidence:.1%}")
    print(f"  SIFT clusters: {len(sift_valid_clusters)}")
    print(f"  DCT clusters:  {len(dct_valid_clusters)}")
    print(f"  Regions found: {len(regions)}")
    print(f"  Processing time: {elapsed:.3f}s")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")

    return result


def batch_process(dataset_dir: str,
                  max_size: int | None = None,
                  debug: bool = False) -> list:
    """Process all images in a dataset directory.

    Args:
        dataset_dir: Path to the dataset directory.
        max_size: Max image dimension.
        debug: Whether to save debug outputs.

    Returns:
        List of result dicts.
    """
    logger = setup_logging(debug)
    images = list_images(dataset_dir)

    if not images:
        logger.error(f"No images found in {dataset_dir}")
        return []

    logger.info(f"Found {len(images)} images in {dataset_dir}")

    output_dir = os.path.join(cfg.OUTPUT_DIR, os.path.basename(dataset_dir))
    debug_dir = os.path.join(cfg.DEBUG_OUTPUT_DIR,
                              os.path.basename(dataset_dir))

    results = []
    for i, img_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] Processing {os.path.basename(img_path)}")
        try:
            result = process_image(
                img_path, max_size=max_size, debug=debug,
                output_dir=output_dir, debug_dir=debug_dir,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            results.append({
                "image": img_path,
                "forgery_detected": False,
                "error": str(e),
                "processing_time": 0.0,
            })

    # Save summary CSV
    csv_path = os.path.join(output_dir, "results_summary.csv")
    save_results_csv(results, csv_path)
    logger.info(f"Batch results saved to {csv_path}")

    # Print aggregate summary
    detected = sum(1 for r in results if r.get("forgery_detected"))
    total_time = sum(r.get("processing_time", 0) for r in results)
    errors = sum(1 for r in results if "error" in r)
    avg_confidence = np.mean([r.get("confidence", 0) for r in results
                              if "error" not in r]) if results else 0

    print(f"\n{'='*60}")
    print(f"  BATCH SUMMARY")
    print(f"  Total images:       {len(results)}")
    print(f"  Forgeries detected: {detected}")
    print(f"  Avg confidence:     {avg_confidence:.1%}")
    print(f"  Errors:             {errors}")
    print(f"  Total time:         {total_time:.1f}s")
    print(f"  Avg time/image:     {total_time/max(len(results),1):.1f}s")

    # Aggregate metrics if available
    metric_results = [r for r in results if "accuracy" in r]
    if metric_results:
        avg_accuracy = np.mean([r["accuracy"] for r in metric_results])
        avg_precision = np.mean([r["precision"] for r in metric_results])
        avg_recall = np.mean([r["recall"] for r in metric_results])
        avg_f1 = np.mean([r["f1"] for r in metric_results])
        print(f"  Avg Accuracy:       {avg_accuracy:.4f}")
        print(f"  Avg Precision:      {avg_precision:.4f}")
        print(f"  Avg Recall:         {avg_recall:.4f}")
        print(f"  Avg F1 Score:       {avg_f1:.4f}")

    print(f"{'='*60}\n")

    return results


def main():
    """Parse arguments and run the detection pipeline."""
    parser = argparse.ArgumentParser(
        description="Copy-Move Forgery Detection using DCT + SIFT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --image photo.jpg
  python main.py --image photo.jpg --full-resolution
  python main.py --image photo.jpg --debug
  python main.py --dataset dataset/MICC_F220
    python main.py --dataset dataset/MICC_F600
  python main.py --dataset dataset/MICC_F220 --debug
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str,
                       help="Path to a single image to analyze")
    group.add_argument("--dataset", type=str,
                       help="Path to a dataset directory for batch processing")

    parser.add_argument("--full-resolution", action="store_true",
                        help="Process at full resolution without resizing "
                             "(default: resize to max 1024px)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with intermediate "
                             "visualizations")
    parser.add_argument("--output-dir", type=str, default=None,
                        help=f"Output directory (default: {cfg.OUTPUT_DIR})")

    args = parser.parse_args()

    # Determine max image size
    max_size = None if args.full_resolution else cfg.MAX_IMAGE_SIZE

    # Override debug mode and output dir
    if args.debug:
        cfg.DEBUG_MODE = True
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.image:
        if not os.path.isfile(args.image):
            print(f"Error: Image file not found: {args.image}")
            sys.exit(1)
        process_image(args.image, max_size=max_size, debug=args.debug)
    else:
        if not os.path.isdir(args.dataset):
            print(f"Error: Dataset directory not found: {args.dataset}")
            sys.exit(1)
        batch_process(args.dataset, max_size=max_size, debug=args.debug)


if __name__ == "__main__":
    main()
