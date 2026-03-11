"""
Model Evaluation & Benchmarking
=================================
Comprehensive evaluation of sentiment models with:
- Classification metrics (F1, precision, recall, confusion matrix)
- Confidence calibration analysis
- Per-aspect accuracy
- Cross-source generalisation
- Latency benchmarking
"""

import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class EvaluationReport:
    model_name: str
    n_test_samples: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_metrics: dict
    confusion_matrix: np.ndarray
    avg_latency_ms: float
    p95_latency_ms: float
    calibration_error: float


def evaluate_model(
    pipeline,
    test_df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    source_col: str = "source",
    model_name: str = "SentimentPipeline",
    verbose: bool = True,
) -> EvaluationReport:
    """
    Run full evaluation suite on a fitted sentiment pipeline.

    Args:
        pipeline: SentimentIntelligencePipeline instance
        test_df: Test DataFrame
        text_col: Column with input texts
        label_col: Column with ground truth labels
        source_col: Column with data source
        model_name: Name for the report
        verbose: Print detailed results

    Returns:
        EvaluationReport dataclass
    """
    texts = test_df[text_col].tolist()
    true_labels = test_df[label_col].tolist()
    sources = test_df[source_col].tolist() if source_col in test_df else ["general"] * len(texts)

    pred_labels = []
    pred_confidences = []
    latencies = []

    for text, source in zip(texts, sources):
        t0 = time.time()
        result = pipeline.analyse_single(text, source=source)
        latencies.append((time.time() - t0) * 1000)
        pred_labels.append(result.sentiment)
        pred_confidences.append(result.sentiment_confidence)

    # Core metrics
    accuracy = np.mean([p == t for p, t in zip(pred_labels, true_labels)])
    macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)

    report = classification_report(
        true_labels, pred_labels, output_dict=True, zero_division=0
    )

    cm = confusion_matrix(
        true_labels, pred_labels,
        labels=["positive", "neutral", "negative"]
    )

    # Calibration error (Expected Calibration Error approximation)
    ece = _calibration_error(true_labels, pred_labels, pred_confidences)

    eval_report = EvaluationReport(
        model_name=model_name,
        n_test_samples=len(texts),
        accuracy=round(accuracy, 4),
        macro_f1=round(macro_f1, 4),
        weighted_f1=round(weighted_f1, 4),
        per_class_metrics={
            k: v for k, v in report.items()
            if k in ("positive", "neutral", "negative")
        },
        confusion_matrix=cm,
        avg_latency_ms=round(np.mean(latencies), 2),
        p95_latency_ms=round(np.percentile(latencies, 95), 2),
        calibration_error=round(ece, 4),
    )

    if verbose:
        _print_report(eval_report, pred_labels, true_labels, sources)

    return eval_report


def _calibration_error(
    true_labels: list,
    pred_labels: list,
    confidences: list,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE)."""
    correct = [p == t for p, t in zip(pred_labels, true_labels)]
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(true_labels)

    for i in range(n_bins):
        mask = [(bins[i] <= c < bins[i + 1]) for c in confidences]
        if not any(mask):
            continue
        bin_acc = np.mean([correct[j] for j in range(n) if mask[j]])
        bin_conf = np.mean([confidences[j] for j in range(n) if mask[j]])
        bin_size = sum(mask)
        ece += (bin_size / n) * abs(bin_acc - bin_conf)

    return ece


def _print_report(
    report: EvaluationReport,
    pred_labels: list,
    true_labels: list,
    sources: list,
):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 60)
    print(f"  Evaluation Report: {report.model_name}")
    print("=" * 60)
    print(f"  Samples:         {report.n_test_samples}")
    print(f"  Accuracy:        {report.accuracy:.2%}")
    print(f"  Macro F1:        {report.macro_f1:.4f}")
    print(f"  Weighted F1:     {report.weighted_f1:.4f}")
    print(f"  Calib. Error:    {report.calibration_error:.4f}")
    print(f"  Avg Latency:     {report.avg_latency_ms:.1f}ms")
    print(f"  P95 Latency:     {report.p95_latency_ms:.1f}ms")
    print("-" * 60)

    # Per-class breakdown
    print("\n  Per-Class Metrics:")
    for label in ["positive", "neutral", "negative"]:
        if label in report.per_class_metrics:
            m = report.per_class_metrics[label]
            print(f"    {label:12s}  P={m['precision']:.2f}  R={m['recall']:.2f}  F1={m['f1-score']:.2f}")

    # Source breakdown
    source_results = defaultdict(lambda: {"correct": 0, "total": 0})
    for p, t, s in zip(pred_labels, true_labels, sources):
        source_results[s]["total"] += 1
        if p == t:
            source_results[s]["correct"] += 1

    print("\n  Accuracy by Source:")
    for src, counts in source_results.items():
        acc = counts["correct"] / counts["total"]
        print(f"    {src:12s}  {acc:.2%} ({counts['total']} samples)")

    print("=" * 60)


def compare_models(reports: list[EvaluationReport]) -> pd.DataFrame:
    """Create a comparison table for multiple model reports."""
    rows = []
    for r in reports:
        rows.append({
            "Model": r.model_name,
            "Accuracy": f"{r.accuracy:.2%}",
            "Macro F1": f"{r.macro_f1:.4f}",
            "Weighted F1": f"{r.weighted_f1:.4f}",
            "ECE": f"{r.calibration_error:.4f}",
            "Avg Latency (ms)": r.avg_latency_ms,
            "P95 Latency (ms)": r.p95_latency_ms,
        })
    return pd.DataFrame(rows).set_index("Model")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.sentiment_pipeline import SentimentIntelligencePipeline
    from src.data_generator import generate_sentiment_dataset

    print("Generating test data...")
    df = generate_sentiment_dataset(n_samples=500, seed=99)
    _, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

    print("Evaluating lexicon pipeline...")
    pipeline = SentimentIntelligencePipeline(use_transformer=False)
    report = evaluate_model(pipeline, test_df)

    print(f"\nFinal Macro F1: {report.macro_f1:.4f}")
