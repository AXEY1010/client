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
from flask import (
    Flask, render_template, request,
    redirect, url_for, flash, send_file
)
from werkzeug.utils import secure_filename

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
