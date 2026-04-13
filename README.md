# Copy-Move Forgery Detection using DCT + SIFT

A robust Python system for detecting **copy-move image forgery** — where a region of an image is copied and pasted elsewhere in the same image. The system combines two complementary methods:

- **DCT (Discrete Cosine Transform)** — Block-based feature extraction that excels at detecting duplicated textures and smooth regions
- **SIFT (Scale-Invariant Feature Transform)** — Keypoint-based feature extraction robust to rotation, scaling, noise, and JPEG compression

The final pipeline merges results from both techniques using **displacement vector clustering** to maximize detection accuracy while minimizing false positives.

---

## Features

- **Dual-method detection**: DCT block matching + SIFT/ORB keypoint matching
- **Multi-scale DCT**: Runs detection at block sizes 8 and 16 for varied forgery scales
- **Confidence scoring**: Continuous 0–100% confidence instead of binary yes/no
- **Robust to attacks**: Handles rotation, scaling, Gaussian noise, JPEG compression
- **False positive reduction**: DBSCAN clustering, RANSAC filtering, vector histogram voting
- **Automatic image resizing**: Process large images efficiently (default max 1024px)
- **Batch processing**: Evaluate entire datasets with aggregate metrics
- **Debug visualizations**: Keypoints, match lines, displacement plots, cluster heatmaps
- **Ground truth evaluation**: Automatic precision/recall/F1 computation
- **Web interface**: Flask-based upload and visualization with comparison slider
- **Error Level Analysis (ELA)**: Detects JPEG compression inconsistencies

---

## Installation

### Requirements

- Python 3.10+
- OpenCV with SIFT support (opencv-contrib-python)

### Setup

```bash
# Clone or download the project
cd copy_move_forgery

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

```bash
# Automatic download and extraction
python scripts/download_datasets.py
```

Supported datasets:
- **MICC-F220** — 220 images (110 original + 110 tampered)
- **MICC-F600** — 600 images from the MICC benchmark family
- **CoMoFoD** — 260 image sets with various post-processing

If automatic download fails, the script provides official URLs for manual download. Place archives in the `dataset/` directory and re-run the script.

---

## Usage

### Single Image Analysis

```bash
# Default (auto-resize to 1024px max)
python main.py --image path/to/image.jpg

# Full resolution (no resizing)
python main.py --image path/to/image.jpg --full-resolution

# With debug visualizations
python main.py --image path/to/image.jpg --debug
```

### Batch Dataset Processing

```bash
# Process entire MICC-F220 dataset
python main.py --dataset dataset/MICC_F220

# Process entire MICC-F600 dataset
python main.py --dataset dataset/MICC_F600

# With debug mode
python main.py --dataset dataset/MICC_F220 --debug
```

### Web Interface

```bash
python app.py
# → http://localhost:5000
```

### Output

- **Detection result**: `output/<image_name>_detected.png`
- **Batch summary**: `output/<dataset_name>/results_summary.csv`
- **Debug images**: `debug_output/` (when `--debug` is enabled)

---

## Pipeline

```
Input Image
    ↓
Preprocessing (grayscale, histogram eq, normalization, Gaussian blur, resize)
    ↓
┌──────────────────────────────┐    ┌────────────────────────────┐
│  Multi-Scale DCT Features    │    │  SIFT + ORB Detection      │
│  (block sizes: 8, 16)       │    │  (3000 + 2500 features)    │
│        ↓                     │    │        ↓                   │
│  Quantized DCT coefficients  │    │  Self-matching + Lowe's    │
│        ↓                     │    │        ↓                   │
│  Lexicographic sort           │    │  RANSAC outlier removal    │
│  + sliding window match       │    │        ↓                   │
│        ↓                     │    │  Merged displacement       │
│  Histogram voting             │    │  vectors                   │
│        ↓                     │    │        ↓                   │
│  DBSCAN clustering            │    │  DBSCAN clustering         │
└────────┬─────────────────────┘    └────────────┬───────────────┘
         │                                        │
         └──────────┬─────────────────────────────┘
                    ↓
         Refined Forgery Confirmation
         (cluster-based, no isolated detections)
                    ↓
         Confidence Scoring (0.0 – 1.0)
         (incorporates ELA evidence & multi-factor validation)
                    ↓
         Morphological processing + contour detection
                    ↓
         Output Visualization
```

---

## Configuration

All parameters are in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_IMAGE_SIZE` | 1024 | Auto-resize threshold |
| `ENABLE_NORMALIZATION` | True | Normalize grayscale intensity to [0,255] |
| `BLOCK_SIZE` | 8 | DCT block dimension |
| `BLOCK_STEP` | 2 | Block overlap step |
| `DCT_COEFFS` | 15 | DCT coefficients to keep |
| `QUANTIZATION_FACTOR` | 10 | DCT coefficient quantization |
| `DCT_COMPARE_WINDOW` | 4 | Sliding window for matching |
| `ENABLE_MULTISCALE_DCT` | True | Multi-scale DCT at multiple block sizes |
| `MULTISCALE_BLOCK_SIZES` | [8, 16] | Block sizes for multi-scale DCT |
| `SIFT_FEATURES` | 3000 | Max SIFT keypoints |
| `ENABLE_ORB` | True | Add ORB features alongside SIFT |
| `ORB_FEATURES` | 2500 | Max ORB keypoints |
| `RATIO_TEST` | 0.75 | Lowe's ratio test threshold |
| `MIN_KEYPOINT_DISTANCE` | 20 | Min distance to avoid self-matching |
| `APPLY_RANSAC` | True | RANSAC filtering of SIFT matches |
| `DBSCAN_EPS` | 6 | Clustering epsilon |
| `DBSCAN_MIN_SAMPLES` | 8 | Min samples for DBSCAN cluster |
| `MIN_CLUSTER_MATCHES` | 5 | Min matches per cluster |
| `SIFT_MIN_VECTOR_DISTANCE` | 40 | Min SIFT displacement magnitude |
| `MIN_DCT_STANDALONE_MATCHES` | 80 | Min DCT cluster for standalone confirmation |
| `ENABLE_ELA` | True | Enable Error Level Analysis branch |
| `ELA_JPEG_QUALITY` | 90 | Quality for synthetic ELA recompression |
| `ELA_CONFIDENCE_WEIGHT`| 0.18 | Weight of ELA within confidence score |

---

## Confidence Scoring

Instead of just binary detection, the system outputs a **confidence score (0.0 – 1.0)** based on:

| Factor | Weight | Description |
|--------|--------|-------------|
| SIFT cluster count & size | 0.40 | Strong SIFT clusters = high evidence |
| DCT cluster count & size | 0.25 | Supporting block-level evidence |
| SIFT + DCT agreement | 0.15 | Both methods agree = very strong signal |
| Region quality | 0.20 | Reasonable region size, 2-region bonus |
| ELA Evidence Bonus | 0.18 | Highlights compression inconsistencies |

**Interpretation:**
- **≥ 70%**: High confidence of tampering
- **40–70%**: Moderate evidence — review recommended
- **< 40%**: Low evidence — likely authentic

---

## Debug Outputs

Enable with `--debug` flag. Generates:

1. **SIFT keypoints** — All detected keypoints rendered on the image
2. **Match lines** — Lines connecting matched keypoint pairs
3. **Displacement vectors** — Scatter plot colored by cluster
4. **Vector histogram** — Heatmap of displacement vector frequency
5. **Cluster heatmap** — Spatial density of matched points

---

## Sample Output

When forgery is detected, the output image shows:
- **Heatmap overlay** — JET colormap on detected regions
- **Red overlay** — DCT-detected regions
- **Yellow lines** — SIFT matched keypoint pairs
- **Green boxes** — Bounding boxes around forged regions
- **Text label** — "FORGERY DETECTED" or "NO FORGERY"

---

## Testing

```bash
# Run the full test suite
pytest

# Run with verbose output
pytest -v

# Run a specific test module
pytest tests/test_clustering.py -v
```

---

## Project Structure

```
copy_move_forgery/
├── config.py                 # All tunable parameters
├── main.py                   # CLI entry point
├── app.py                    # Flask web interface + REST API
├── requirements.txt          # Python dependencies
├── pytest.ini                # Test configuration
├── README.md                 # This file
├── dataset/                  # Image datasets
│   ├── MICC_F220/
│   ├── MICC_F600/
│   └── CoMoFoD/
├── scripts/
│   ├── create_test_images.py # Synthetic test image generator
│   └── download_datasets.py  # Dataset downloader
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Image loading & preprocessing
│   ├── dct_features.py       # DCT block feature extraction
│   ├── block_matching.py     # Block matching & histogram voting
│   ├── sift_features.py      # SIFT/ORB detection, matching, RANSAC
│   ├── vector_clustering.py  # DBSCAN displacement clustering
│   ├── forgery_localization.py  # Mask creation, confidence scoring
│   ├── visualization.py      # Result rendering & debug outputs
│   └── utils.py              # Timing, metrics, I/O helpers
├── tests/
│   ├── conftest.py           # Shared test fixtures
│   ├── test_preprocessing.py
│   ├── test_dct_features.py
│   ├── test_clustering.py
│   ├── test_localization.py
│   ├── test_utils.py
│   └── test_pipeline.py      # End-to-end integration tests
├── static/                   # Web assets
│   └── js/comparison-slider.js
├── templates/                # Jinja2 HTML templates
│   ├── index.html
│   └── result.html
├── output/                   # Detection results
└── debug_output/             # Debug visualizations
```

---

## Accuracy Improvement Techniques

1. **DCT coefficient quantization** — Robust to JPEG compression artifacts
2. **Multi-scale DCT** — Detects forgeries at multiple block sizes (8, 16)
3. **Sliding window matching** — Handles small variations in block features
4. **Histogram voting** — Identifies dominant displacement vectors
5. **RANSAC filtering** — Removes outlier SIFT matches
6. **Hybrid SIFT + ORB** — Combines floating-point and binary descriptors
7. **DBSCAN clustering** — Groups consistent displacement vectors
8. **Cluster size/variance/angle filtering** — Rejects weak or noisy clusters
9. **Minimum displacement distance** — Ignores repeated texture matches
10. **Refined confirmation rule** — Requires cluster evidence, not isolated matches
11. **Confidence scoring** — Continuous 0–100% based on multi-factor analysis
12. **Morphological filtering** — Clean region boundaries
13. **Error Level Analysis (ELA)** — Supplements geometric detection with compression cues

---

## License

This project is for research and educational purposes. The datasets have their own licenses — please refer to the official dataset pages.
