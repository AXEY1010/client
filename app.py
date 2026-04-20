"""
Flask Web Application for Copy-Move Forgery Detection.

Provides a simple web interface for uploading images
and viewing forgery detection results.

Usage:
    python app.py
    → http://localhost:5000
"""

import os
import uuid
import time
import csv
from flask import (
    Flask, render_template, request,
    redirect, url_for, flash, send_file
)

# ---------------------------------------------------------------------------
# Detection pipeline import
# ---------------------------------------------------------------------------
from main import process_image

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = os.path.join("static", "uploads")
OUTPUT_FOLDER = os.path.join("static", "outputs")
METRICS_OUTPUT_ROOT = "output"
METRICS_CSV_NAME = "results_summary.csv"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload


def allowed_file(filename: str) -> bool:
    """Return True if the file extension is allowed."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def is_safe_output_file(path: str) -> bool:
    """Return True if path exists and is inside OUTPUT_FOLDER."""
    if not path:
        return False

    output_root = os.path.abspath(OUTPUT_FOLDER)
    candidate = os.path.abspath(os.path.normpath(path))

    try:
        common = os.path.commonpath([output_root, candidate])
    except ValueError:
        return False

    return common == output_root and os.path.isfile(candidate)


def _safe_float(value: str | None) -> float | None:
    """Convert a raw value to float, returning None if conversion fails."""
    if value is None:
        return None

    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _safe_bool(value: str | None) -> bool | None:
    """Convert common text representations to bool."""
    if value is None:
        return None

    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    return None


def _mean(values: list[float]) -> float | None:
    """Return average for a list or None when empty."""
    return (sum(values) / len(values)) if values else None


def _infer_ground_truth_label(image_path: str) -> bool | None:
    """Infer expected label from known dataset naming conventions."""
    if not image_path:
        return None

    normalized = image_path.replace("\\", "/").lower()
    filename = os.path.basename(normalized)

    if "/forged/" in normalized:
        return True
    if "/original/" in normalized:
        return False

    # Fallback heuristics for common benchmark naming.
    if "tamp" in filename or "_f_" in filename:
        return True
    if "_o_" in filename or "_orig" in filename or "scale" in filename:
        return False

    return None


def _compute_binary_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    """Compute standard binary classification metrics."""
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _discover_results_csv_files() -> list[str]:
    """Return dataset-level results_summary.csv files under output/ directories."""
    output_root = os.path.abspath(METRICS_OUTPUT_ROOT)
    if not os.path.isdir(output_root):
        return []

    csv_paths = []
    for entry in sorted(os.listdir(output_root)):
        dataset_dir = os.path.join(output_root, entry)
        if not os.path.isdir(dataset_dir):
            continue

        csv_path = os.path.join(dataset_dir, METRICS_CSV_NAME)
        if os.path.isfile(csv_path):
            csv_paths.append(csv_path)

    return csv_paths


def _summarize_results_csv(csv_path: str) -> dict | None:
    """Build per-dataset summary from one results_summary.csv file."""
    rows_total = 0
    rows_labeled = 0

    tp = fp = fn = tn = 0
    detected_count = 0

    confidence_values: list[float] = []
    processing_times: list[float] = []
    pixel_metric_values = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }

    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return None

            for row in reader:
                rows_total += 1

                predicted = _safe_bool(row.get("forgery_detected"))
                expected = _infer_ground_truth_label(row.get("image", ""))

                if predicted is True:
                    detected_count += 1

                if predicted is not None and expected is not None:
                    rows_labeled += 1
                    if predicted and expected:
                        tp += 1
                    elif predicted and not expected:
                        fp += 1
                    elif (not predicted) and expected:
                        fn += 1
                    else:
                        tn += 1

                confidence = _safe_float(row.get("confidence"))
                if confidence is not None:
                    confidence_values.append(confidence)

                processing_time = _safe_float(row.get("processing_time"))
                if processing_time is not None:
                    processing_times.append(processing_time)

                for metric_name in pixel_metric_values:
                    metric_value = _safe_float(row.get(metric_name))
                    if metric_value is not None:
                        pixel_metric_values[metric_name].append(metric_value)
    except OSError:
        return None

    if rows_total == 0:
        return None

    classification_metrics = None
    if rows_labeled > 0:
        classification_metrics = _compute_binary_metrics(tp, fp, fn, tn)

    has_pixel_metrics = any(pixel_metric_values.values())
    pixel_metrics = None
    pixel_metric_count = 0
    if has_pixel_metrics:
        pixel_metrics = {
            "accuracy": _mean(pixel_metric_values["accuracy"]),
            "precision": _mean(pixel_metric_values["precision"]),
            "recall": _mean(pixel_metric_values["recall"]),
            "f1": _mean(pixel_metric_values["f1"]),
        }
        pixel_metric_count = len(pixel_metric_values["accuracy"])

    dataset_name = os.path.basename(os.path.dirname(csv_path))

    return {
        "dataset_name": dataset_name,
        "csv_path": csv_path.replace("\\", "/"),
        "rows_total": rows_total,
        "rows_labeled": rows_labeled,
        "detected_count": detected_count,
        "detection_rate": detected_count / rows_total,
        "avg_confidence": _mean(confidence_values),
        "avg_processing_time": _mean(processing_times),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "classification_metrics": classification_metrics,
        "pixel_metrics": pixel_metrics,
        "pixel_metric_count": pixel_metric_count,
    }


def _build_dashboard_payload(dataset_summaries: list[dict]) -> dict:
    """Create chart-ready payload and global aggregates for the template."""
    labels = []
    accuracy_values = []
    precision_values = []
    recall_values = []
    f1_values = []
    detection_rate_values = []
    avg_confidence_values = []

    total_images = 0
    total_labeled = 0
    tp_total = fp_total = fn_total = tn_total = 0

    best_dataset = None

    for summary in dataset_summaries:
        labels.append(summary["dataset_name"])
        total_images += summary["rows_total"]
        total_labeled += summary["rows_labeled"]

        detection_rate_values.append(summary["detection_rate"])
        avg_confidence_values.append(summary["avg_confidence"])

        metrics = summary.get("classification_metrics")
        if metrics is None:
            accuracy_values.append(None)
            precision_values.append(None)
            recall_values.append(None)
            f1_values.append(None)
            continue

        accuracy_values.append(metrics["accuracy"])
        precision_values.append(metrics["precision"])
        recall_values.append(metrics["recall"])
        f1_values.append(metrics["f1"])

        tp_total += summary["tp"]
        fp_total += summary["fp"]
        fn_total += summary["fn"]
        tn_total += summary["tn"]

        if best_dataset is None or metrics["f1"] > best_dataset["f1"]:
            best_dataset = {
                "name": summary["dataset_name"],
                "f1": metrics["f1"],
                "accuracy": metrics["accuracy"],
            }

    overall_classification = None
    if total_labeled > 0:
        overall_classification = _compute_binary_metrics(
            tp=tp_total,
            fp=fp_total,
            fn=fn_total,
            tn=tn_total,
        )

    return {
        "chart_data": {
            "labels": labels,
            "accuracy": accuracy_values,
            "precision": precision_values,
            "recall": recall_values,
            "f1": f1_values,
            "detection_rate": detection_rate_values,
            "avg_confidence": avg_confidence_values,
        },
        "summary": {
            "dataset_count": len(dataset_summaries),
            "total_images": total_images,
            "total_labeled": total_labeled,
            "overall_classification": overall_classification,
            "best_dataset": best_dataset,
        },
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Render the homepage with the upload form."""
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    """Accept an uploaded image, run the detection pipeline, show results."""

    # --- validate upload ---------------------------------------------------
    if "image" not in request.files:
        flash("No file uploaded. Please select an image.", "error")
        return redirect(url_for("index"))

    file = request.files["image"]

    if file.filename == "":
        flash("No file selected. Please choose an image.", "error")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Please upload a valid image file (JPG, JPEG, or PNG).", "error")
        return redirect(url_for("index"))

    # --- save uploaded file ------------------------------------------------
    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex[:12]}.{ext}"
    upload_path = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(upload_path)

    # --- run detection pipeline --------------------------------------------
    try:
        result = process_image(
            image_path=upload_path,
            max_size=1024,
            debug=False,
            output_dir=OUTPUT_FOLDER,
        )
    except Exception:
        flash("Processing error. Please try another image.", "error")
        return redirect(url_for("index"))

    if result.get("error"):
        flash("Processing failed. Uploaded file is not a readable image.", "error")
        return redirect(url_for("index"))

    # --- prepare template data ---------------------------------------------
    forgery_detected = result.get("forgery_detected", False)
    processing_time = result.get("processing_time", 0.0)
    num_regions = result.get("num_regions", 0)
    confidence = result.get("confidence", 0.0)
    output_path = result.get("output_path", "")

    if not is_safe_output_file(output_path):
        flash("Processing error. Could not generate output image.", "error")
        return redirect(url_for("index"))

    # Make paths relative for use in templates
    original_url = upload_path.replace("\\", "/")
    output_url = output_path.replace("\\", "/")

    return render_template(
        "result.html",
        forgery_detected=forgery_detected,
        processing_time=processing_time,
        num_regions=num_regions,
        confidence=confidence,
        original_url=original_url,
        output_url=output_url,
        output_path=output_path,
    )


@app.route("/metrics")
def metrics_dashboard():
    """Show dataset-level metrics from output/*/results_summary.csv files."""
    csv_paths = _discover_results_csv_files()

    summaries = []
    for csv_path in csv_paths:
        summary = _summarize_results_csv(csv_path)
        if summary is not None:
            summaries.append(summary)

    payload = _build_dashboard_payload(summaries)

    return render_template(
        "metrics.html",
        dataset_summaries=summaries,
        dashboard_payload=payload,
    )


@app.route("/download/<path:filepath>")
def download(filepath):
    """Download the detection output image."""
    requested = os.path.abspath(os.path.normpath(filepath))
    output_root = os.path.abspath(OUTPUT_FOLDER)

    try:
        common = os.path.commonpath([output_root, requested])
    except ValueError:
        flash("Invalid download path.", "error")
        return redirect(url_for("index"))

    if common != output_root:
        flash("Invalid download path.", "error")
        return redirect(url_for("index"))

    if os.path.isfile(requested):
        return send_file(requested, as_attachment=True)

    flash("File not found.", "error")
    return redirect(url_for("index"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Copy-Move Forgery Detection — Web Interface")
    print("  http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
