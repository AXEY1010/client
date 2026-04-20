/**
 * Dataset metrics dashboard charts.
 * Renders per-dataset classification charts plus overview visualizations.
 */
(function () {
    'use strict';

    const payloadEl = document.getElementById('chartPayload');
    if (!payloadEl || typeof window.Chart === 'undefined') {
        return;
    }

    let payload;
    try {
        payload = JSON.parse(payloadEl.textContent || '{}');
    } catch (_) {
        return;
    }

    const chartData = payload.chart_data || {};
    const labels = Array.isArray(chartData.labels) ? chartData.labels : [];
    const chartCanvases = document.querySelectorAll('.dataset-classification-chart');

    if (labels.length === 0) {
        return;
    }

    const toPercentValue = (value) => {
        if (typeof value !== 'number' || !Number.isFinite(value)) {
            return null;
        }
        return Number((value * 100).toFixed(2));
    };

    const toSeries = (values) => labels.map((_, index) => toPercentValue(values?.[index]));

    const accuracySeries = toSeries(chartData.accuracy);
    const precisionSeries = toSeries(chartData.precision);
    const recallSeries = toSeries(chartData.recall);
    const f1Series = toSeries(chartData.f1);
    const detectionRateSeries = toSeries(chartData.detection_rate);
    const confidenceSeries = toSeries(chartData.avg_confidence);

    const metricLabels = ['Accuracy', 'Precision', 'Recall', 'F1'];

    const palette = [
        '#0f6cbd',
        '#0ea5a3',
        '#f59e0b',
        '#dc2626',
        '#7c3aed',
        '#0284c7',
    ];

    const hexToRgba = (hex, alpha) => {
        const normalized = hex.replace('#', '');
        const value = normalized.length === 3
            ? normalized.split('').map((char) => char + char).join('')
            : normalized;

        const intVal = Number.parseInt(value, 16);
        const r = (intVal >> 16) & 255;
        const g = (intVal >> 8) & 255;
        const b = intVal & 255;
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    };

    const noData = (canvas, message) => {
        const parent = canvas.parentElement;
        if (!parent) {
            return;
        }

        parent.innerHTML = `<div style="height:100%;display:flex;align-items:center;justify-content:center;color:#4a5d85;font-weight:600;font-size:.86rem;">${message}</div>`;
    };

    const baseMetricConfig = {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            mode: 'index',
            intersect: false,
        },
        animation: {
            duration: 900,
            easing: 'easeOutQuart',
        },
        plugins: {
            legend: {
                display: false,
            },
            tooltip: {
                backgroundColor: 'rgba(12, 26, 48, 0.95)',
                titleColor: '#f8fbff',
                bodyColor: '#dce8ff',
                borderColor: 'rgba(143, 176, 229, 0.35)',
                borderWidth: 1,
                callbacks: {
                    label(context) {
                        const value = context.parsed.y;
                        if (value === null || Number.isNaN(value)) {
                            return `${context.dataset.label}: N/A`;
                        }
                        return `${context.dataset.label}: ${value.toFixed(1)}%`;
                    },
                },
            },
        },
        scales: {
            y: {
                beginAtZero: true,
                max: 100,
                ticks: {
                    callback(value) {
                        return `${value}%`;
                    },
                    color: '#35557f',
                    font: {
                        size: 11,
                    },
                },
                grid: {
                    color: 'rgba(97, 134, 183, 0.2)',
                },
            },
            x: {
                ticks: {
                    color: '#1f3d68',
                    font: {
                        size: 11,
                        weight: '600',
                    },
                },
                grid: {
                    display: false,
                },
            },
        },
    };

    Chart.defaults.font.family = '"Plus Jakarta Sans", "Segoe UI", sans-serif';
    Chart.defaults.color = '#27466f';

    chartCanvases.forEach((canvas, canvasIndex) => {
        const index = Number(canvas.dataset.datasetIndex);
        if (!Number.isInteger(index) || index < 0 || index >= labels.length) {
            return;
        }

        const metrics = [
            accuracySeries[index],
            precisionSeries[index],
            recallSeries[index],
            f1Series[index],
        ];

        if (metrics.every((value) => value === null)) {
            noData(canvas, 'No classification metrics for this dataset.');
            return;
        }

        const datasetName = canvas.dataset.datasetName || labels[index];
        const baseColor = palette[canvasIndex % palette.length];

        new Chart(canvas, {
            type: 'bar',
            data: {
                labels: metricLabels,
                datasets: [
                    {
                        label: datasetName,
                        data: metrics,
                        backgroundColor: [
                            hexToRgba(baseColor, 0.82),
                            hexToRgba(baseColor, 0.72),
                            hexToRgba(baseColor, 0.62),
                            hexToRgba(baseColor, 0.9),
                        ],
                        borderColor: [
                            hexToRgba(baseColor, 1),
                            hexToRgba(baseColor, 1),
                            hexToRgba(baseColor, 1),
                            hexToRgba(baseColor, 1),
                        ],
                        borderWidth: 1,
                        borderRadius: 8,
                    },
                ],
            },
            options: baseMetricConfig,
        });
    });

    const radarCanvas = document.getElementById('overviewRadarChart');
    if (radarCanvas) {
        const validIndexes = labels
            .map((_, index) => index)
            .filter((index) => {
                const metricValues = [
                    accuracySeries[index],
                    precisionSeries[index],
                    recallSeries[index],
                    f1Series[index],
                ];
                return metricValues.every((value) => value !== null);
            });

        if (validIndexes.length === 0) {
            noData(radarCanvas, 'No complete classification profiles available.');
        } else {
            const radarDatasets = validIndexes.map((index, datasetIdx) => {
                const color = palette[datasetIdx % palette.length];
                return {
                    label: labels[index],
                    data: [
                        accuracySeries[index],
                        precisionSeries[index],
                        recallSeries[index],
                        f1Series[index],
                    ],
                    borderColor: hexToRgba(color, 0.9),
                    backgroundColor: hexToRgba(color, 0.14),
                    pointBackgroundColor: hexToRgba(color, 1),
                    pointRadius: 3,
                    borderWidth: 2,
                    fill: true,
                };
            });

            new Chart(radarCanvas, {
                type: 'radar',
                data: {
                    labels: metricLabels,
                    datasets: radarDatasets,
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                boxWidth: 10,
                                boxHeight: 10,
                                usePointStyle: true,
                                color: '#25456f',
                            },
                        },
                        tooltip: {
                            callbacks: {
                                label(context) {
                                    const value = context.parsed.r;
                                    return `${context.dataset.label}: ${value.toFixed(1)}%`;
                                },
                            },
                        },
                    },
                    scales: {
                        r: {
                            beginAtZero: true,
                            max: 100,
                            ticks: {
                                stepSize: 20,
                                color: '#37557e',
                                backdropColor: 'rgba(255,255,255,0.72)',
                                callback(value) {
                                    return `${value}%`;
                                },
                            },
                            angleLines: {
                                color: 'rgba(68, 107, 160, 0.2)',
                            },
                            grid: {
                                color: 'rgba(68, 107, 160, 0.2)',
                            },
                            pointLabels: {
                                color: '#1e3b64',
                                font: {
                                    weight: '600',
                                },
                            },
                        },
                    },
                },
            });
        }
    }

    const detectionConfidenceCanvas = document.getElementById('detectionConfidenceChart');
    if (detectionConfidenceCanvas) {
        const hasDetectionOrConfidence = detectionRateSeries.some((value) => value !== null)
            || confidenceSeries.some((value) => value !== null);

        if (!hasDetectionOrConfidence) {
            noData(detectionConfidenceCanvas, 'No detection rate or confidence trend available.');
        } else {
            new Chart(detectionConfidenceCanvas, {
                type: 'bar',
                data: {
                    labels,
                    datasets: [
                        {
                            type: 'bar',
                            label: 'Detection Rate',
                            data: detectionRateSeries,
                            backgroundColor: 'rgba(14, 165, 163, 0.7)',
                            borderColor: 'rgba(13, 148, 136, 1)',
                            borderWidth: 1,
                            borderRadius: 6,
                            maxBarThickness: 42,
                        },
                        {
                            type: 'line',
                            label: 'Avg Confidence',
                            data: confidenceSeries,
                            borderColor: 'rgba(245, 158, 11, 0.95)',
                            backgroundColor: 'rgba(245, 158, 11, 0.2)',
                            borderWidth: 2,
                            pointRadius: 3,
                            pointHoverRadius: 4,
                            tension: 0.28,
                            yAxisID: 'y',
                        },
                    ],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false,
                    },
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                color: '#234165',
                                usePointStyle: true,
                            },
                        },
                        tooltip: {
                            callbacks: {
                                label(context) {
                                    const value = context.parsed.y;
                                    if (value === null || Number.isNaN(value)) {
                                        return `${context.dataset.label}: N/A`;
                                    }
                                    return `${context.dataset.label}: ${value.toFixed(1)}%`;
                                },
                            },
                        },
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            ticks: {
                                callback(value) {
                                    return `${value}%`;
                                },
                                color: '#35557f',
                            },
                            grid: {
                                color: 'rgba(97, 134, 183, 0.2)',
                            },
                        },
                        x: {
                            ticks: {
                                color: '#1f3d68',
                            },
                            grid: {
                                display: false,
                            },
                        },
                    },
                },
            });
        }
    }
})();
