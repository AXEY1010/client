"""
Configuration file for Copy-Move Forgery Detection System.

All tunable parameters are defined here for easy experimentation.
Modify values below to adjust detection sensitivity, performance, and behavior.
"""

# ==============================================================================
# Image Preprocessing
# ==============================================================================

# Maximum image dimension (width or height). Images exceeding this are resized
# while preserving aspect ratio before processing. Set to None to disable.
MAX_IMAGE_SIZE = 1024

# Gaussian blur kernel size and sigma for preprocessing
BLUR_KERNEL_SIZE = 5
BLUR_SIGMA = 1.0

# Whether to apply histogram equalization for contrast enhancement
APPLY_HISTOGRAM_EQ = True

# Whether to normalize grayscale intensity to full [0, 255] range
ENABLE_NORMALIZATION = True


# ==============================================================================
# DCT Block Feature Extraction
# ==============================================================================

# Block size for DCT computation (pixels)
BLOCK_SIZE = 8

# Step size for overlapping block extraction (smaller = more overlap = slower)
BLOCK_STEP = 2

# Number of DCT coefficients to retain (zigzag order, top-left region)
DCT_COEFFS = 15

# Quantization factor applied to DCT coefficients before matching.
# Higher values increase robustness to compression but reduce precision.
QUANTIZATION_FACTOR = 10

# Minimum per-block grayscale standard deviation required for DCT processing.
# Suppresses low-texture regions (e.g., clear sky) that often produce
# coincidental block matches.
DCT_MIN_BLOCK_STD = 6.0


# ==============================================================================
# DCT Block Matching
# ==============================================================================

# Number of adjacent rows to compare after lexicographic sorting.
# Larger window handles more variation but increases computation.
DCT_COMPARE_WINDOW = 4

# Threshold for considering two DCT feature vectors as matching.
# Two vectors match if their L2 distance after quantization is below this.
DCT_MATCH_THRESHOLD = 0.0

# Minimum displacement magnitude to accept a block match.
# Matches below this distance are likely repeated textures, not forgery.
MIN_VECTOR_DISTANCE = 22


# ==============================================================================
# SIFT Feature Extraction
# ==============================================================================

# Maximum number of SIFT keypoints to detect
SIFT_FEATURES = 5000

# SIFT contrast threshold (lower = more keypoints in low-contrast regions)
SIFT_CONTRAST_THRESHOLD = 0.02

# SIFT edge threshold
SIFT_EDGE_THRESHOLD = 10

# SIFT Gaussian sigma
SIFT_SIGMA = 1.6

# Whether to add ORB keypoint matches alongside SIFT for clustering
ENABLE_ORB = True

# ORB configuration
ORB_FEATURES = 3000
ORB_SCALE_FACTOR = 1.2
ORB_N_LEVELS = 8
ORB_FAST_THRESHOLD = 20

# Lowe-style ratio test threshold for ORB descriptor matching
ORB_RATIO_TEST = 0.85

# Lowe's ratio test threshold for descriptor matching
RATIO_TEST = 0.75

# Minimum spatial distance (pixels) between matched keypoints.
# Prevents matching a keypoint to itself or near-neighbors.
MIN_KEYPOINT_DISTANCE = 15


# ==============================================================================
# RANSAC Filtering
# ==============================================================================

# Whether to apply RANSAC to filter SIFT matches before clustering
APPLY_RANSAC = True

# RANSAC reprojection threshold (pixels)
RANSAC_REPROJ_THRESHOLD = 4.0

# Minimum number of matches required to attempt RANSAC
RANSAC_MIN_MATCHES = 6


# ==============================================================================
# Displacement Vector Clustering (DBSCAN)
# ==============================================================================

# DBSCAN epsilon — max distance between vectors in the same cluster
DBSCAN_EPS = 6

# DBSCAN minimum samples in a cluster
DBSCAN_MIN_SAMPLES = 6

# Minimum number of matches in a cluster to confirm forgery
MIN_CLUSTER_MATCHES = 4

# Maximum variance allowed within a cluster's displacement vectors.
# Clusters with higher variance are likely false positives.
VECTOR_VARIANCE_THRESHOLD = 20

# Maximum allowed angular spread (radians) within a cluster.
ANGLE_VARIANCE_THRESHOLD = 0.5

# Backward-compatible alias for modules still using old naming.
MAX_VECTOR_VARIANCE = 20

# Source-specific geometric consistency rules:
# - SIFT/ORB clusters: relaxed (recover recall)
# - DCT-only clusters: strict (suppress false positives)
SIFT_MIN_CLUSTER_MATCHES = 4
SIFT_VECTOR_VARIANCE_THRESHOLD = 20
SIFT_ANGLE_VARIANCE_THRESHOLD = 0.45
SIFT_MIN_VECTOR_DISTANCE = 35

DCT_MIN_CLUSTER_MATCHES = 6
DCT_VECTOR_VARIANCE_THRESHOLD = 10
DCT_ANGLE_VARIANCE_THRESHOLD = 0.35
DCT_MIN_VECTOR_DISTANCE = 25

# Minimum number of matched pairs in a DCT cluster for standalone
# DCT-only forgery confirmation (without SIFT corroboration).
# Must be higher than MIN_CLUSTER_MATCHES to avoid false positives.
MIN_DCT_STANDALONE_MATCHES = 45


# ==============================================================================
# Forgery Localization
# ==============================================================================

# Minimum contour area (pixels) for a detected region to be kept
MIN_REGION_AREA = 40

# Minimum number of localized regions required to confirm forgery.
# Copy-move manipulations usually produce at least a source and destination
# region; this helps reject single-blob false positives.
MIN_CONFIRMED_REGIONS = 2

# Dilation kernel size for expanding matched point regions into masks
DILATION_KERNEL_SIZE = 13

# Number of dilation iterations
DILATION_ITERATIONS = 2

# Additional standalone DCT spatial-consistency safeguards.
# DCT-only detections should look like a small set of compact paired regions,
# not many scattered fragments over repetitive textures.
DCT_STANDALONE_MAX_REGIONS = 6
DCT_STANDALONE_MIN_PRIMARY_FRACTION = 0.003
DCT_STANDALONE_MIN_TOP2_SHARE = 0.70


# ==============================================================================
# Multi-Scale DCT (Improvement #11)
# ==============================================================================

# Enable multi-scale DCT analysis at multiple block sizes.
# When enabled, the pipeline runs DCT extraction at each listed block size
# and merges results. Increases computation but improves detection of
# copy-move forgery at varied scales.
ENABLE_MULTISCALE_DCT = True
MULTISCALE_BLOCK_SIZES = [8,16]


# ==============================================================================
# Error Level Analysis (ELA)
# ==============================================================================

# Enable ELA branch for JPEG artifact-based forgery evidence.
ENABLE_ELA = True

# JPEG quality used for synthetic recompression in ELA.
ELA_JPEG_QUALITY = 90

# Percentile used to threshold high-error ELA responses.
ELA_THRESHOLD_PERCENTILE = 95

# Blur kernel size for ELA residual smoothing.
ELA_BLUR_KERNEL_SIZE = 5

# Minimum contour area for ELA regions.
ELA_MIN_REGION_AREA = 100

# Minimum count of ELA regions to consider standalone ELA detection.
ELA_MIN_REGION_COUNT = 2

# Reject ELA masks that occupy too much of the image.
ELA_MAX_REGION_FRACTION = 0.30

# Minimum ELA score required to trigger ELA-only detection when clusters fail.
ELA_DETECTION_THRESHOLD = 0.60

# Contribution weight of ELA score into final confidence.
ELA_CONFIDENCE_WEIGHT = 0.18

# Require ELA support to overlap DCT evidence before ELA-only confirmation.
ELA_REQUIRE_DCT_OVERLAP = True
ELA_MIN_DCT_OVERLAP_RATIO = 0.30
ELA_MIN_DCT_JACCARD = 0.05


# ==============================================================================
# Vector Histogram Voting
# ==============================================================================

# Number of bins per axis for the displacement vector histogram
HISTOGRAM_BINS = 50

# Minimum vote count in histogram bin to consider it a dominant vector
HISTOGRAM_MIN_VOTES = 5


# ==============================================================================
# Debug & Visualization
# ==============================================================================

# Enable debug mode — saves intermediate visualizations
DEBUG_MODE = False

# Output directory for results
OUTPUT_DIR = "output"

# Output directory for debug visualizations
DEBUG_OUTPUT_DIR = "debug_output"

# Colors (BGR format for OpenCV)
COLOR_DCT_MATCH = (0, 0, 255)       # Red — DCT detections
COLOR_SIFT_MATCH = (0, 255, 255)    # Yellow — SIFT match lines
COLOR_BBOX = (0, 255, 0)            # Green — bounding boxes
COLOR_MASK_OVERLAY = (0, 0, 200)    # Dark red — mask overlay

# Overlay transparency
OVERLAY_ALPHA = 0.4
