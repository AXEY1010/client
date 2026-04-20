"""
Tests for Flask dataset metrics dashboard and CSV aggregation.
"""

import csv
import os
import sys
import pytest


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import app as app_module


def _write_results_csv(path: str, rows: list[dict]):
    fieldnames = [
        "image",
        "forgery_detected",
        "confidence",
        "processing_time",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_summarize_results_csv_computes_classification_metrics(tmp_path):
    dataset_dir = tmp_path / "DemoSet"
    dataset_dir.mkdir(parents=True)
    csv_path = dataset_dir / "results_summary.csv"

    # 4 labeled rows:
    # TP=1, TN=1, FP=1, FN=1 -> all metrics should be 0.5
    rows = [
        {
            "image": "dataset/DemoSet/forged/a.jpg",
            "forgery_detected": "True",
            "confidence": "0.91",
            "processing_time": "1.1",
        },
        {
            "image": "dataset/DemoSet/original/b.jpg",
            "forgery_detected": "False",
            "confidence": "0.02",
            "processing_time": "1.3",
        },
        {
            "image": "dataset/DemoSet/forged/c.jpg",
            "forgery_detected": "False",
            "confidence": "0.11",
            "processing_time": "1.2",
        },
        {
            "image": "dataset/DemoSet/original/d.jpg",
            "forgery_detected": "True",
            "confidence": "0.82",
            "processing_time": "1.4",
        },
    ]
    _write_results_csv(str(csv_path), rows)

    summary = app_module._summarize_results_csv(str(csv_path))
    assert summary is not None
    assert summary["rows_total"] == 4
    assert summary["rows_labeled"] == 4
    assert summary["tp"] == 1
    assert summary["fp"] == 1
    assert summary["fn"] == 1
    assert summary["tn"] == 1

    metrics = summary["classification_metrics"]
    assert metrics is not None
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)


def test_metrics_dashboard_route_renders(tmp_path, monkeypatch):
    output_root = tmp_path / "output"
    dataset_dir = output_root / "DemoSet"
    dataset_dir.mkdir(parents=True)
    csv_path = dataset_dir / "results_summary.csv"

    rows = [
        {
            "image": "dataset/DemoSet/forged/a.jpg",
            "forgery_detected": "True",
            "confidence": "0.9",
            "processing_time": "1.0",
        },
        {
            "image": "dataset/DemoSet/original/b.jpg",
            "forgery_detected": "False",
            "confidence": "0.1",
            "processing_time": "1.2",
        },
    ]
    _write_results_csv(str(csv_path), rows)

    monkeypatch.setattr(app_module, "METRICS_OUTPUT_ROOT", str(output_root))

    client = app_module.app.test_client()
    response = client.get("/metrics")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Dataset Metrics Dashboard" in body
    assert "DemoSet" in body
    assert "100.0%" in body


def test_metrics_dashboard_handles_missing_csv(tmp_path, monkeypatch):
    output_root = tmp_path / "output"
    output_root.mkdir(parents=True)

    monkeypatch.setattr(app_module, "METRICS_OUTPUT_ROOT", str(output_root))

    client = app_module.app.test_client()
    response = client.get("/metrics")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "No dataset summaries were found" in body
