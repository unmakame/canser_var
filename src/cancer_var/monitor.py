import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np

from cancer_var.model import evaluate, malignant_probability
from cancer_var.train import ARTIFACT_DIR


def monitor(artifact, rows, target=None, shift_threshold=0.5):
    rows = np.asarray(rows, dtype=float)
    names = artifact["feature_names"]
    if rows.ndim != 2 or rows.shape[1] != len(names) or len(rows) == 0:
        raise ValueError("Expected a nonempty matrix matching feature_names")
    if not np.isfinite(rows).all():
        raise ValueError("Features must be finite")
    if not np.isfinite(shift_threshold) or shift_threshold <= 0:
        raise ValueError("shift_threshold must be positive and finite")
    reference = artifact["reference"]
    scale = np.maximum(np.asarray(reference["std"]), 1e-12)
    shifts = np.abs(rows.mean(axis=0) - np.asarray(reference["mean"])) / scale
    scores = malignant_probability(artifact["model"], rows)
    report = {
        "sample_count": len(rows),
        "shift_threshold": shift_threshold,
        "mean_probability_malignant": float(scores.mean()),
        "standardized_mean_shifts": dict(zip(names, shifts.tolist())),
        "shifted_features": [name for name, shift in zip(names, shifts) if shift > shift_threshold],
        "quality": None,
    }
    if target is not None:
        target = np.asarray(target)
        if target.shape != (len(rows),) or not np.isin(target, [0, 1]).all():
            raise ValueError("Targets must have one 0 or 1 label per row")
        report["quality"] = evaluate(target, scores)
        baseline = artifact.get("test_metrics", {})
        report["quality_delta_from_test"] = {
            key: value - baseline[key]
            for key, value in report["quality"].items()
            if value is not None and baseline.get(key) is not None
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--model", type=Path, default=ARTIFACT_DIR / "breast_cancer_pipeline.joblib")
    parser.add_argument("--output", type=Path, default=Path("monitoring_report.json"))
    parser.add_argument("--shift-threshold", type=float, default=0.5)
    args = parser.parse_args()
    artifact = joblib.load(args.model)
    with args.csv_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        names = artifact["feature_names"]
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not set(names).issubset(fields):
            parser.error("CSV must have unique headers including all feature names")
        records = list(reader)
    try:
        rows = [[float(row[name]) for name in names] for row in records]
        target = [float(row["target"]) for row in records] if "target" in fields else None
        report = monitor(artifact, rows, target, args.shift_threshold)
    except (ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
