"""Обучает модель и сохраняет артефакт локально."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from cancer_var.model import train_model

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"


def main() -> None:
    result = train_model()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = ARTIFACT_DIR / "breast_cancer_pipeline.joblib"
    joblib.dump(
        {"model": result.model, "feature_names": result.feature_names},
        artifact_path,
    )
    (ARTIFACT_DIR / "metrics.json").write_text(
        json.dumps(result.metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Модель сохранена: {artifact_path}")
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
