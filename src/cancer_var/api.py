from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from typing import Annotated

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, FiniteFloat

ARTIFACT_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "breast_cancer_pipeline.joblib"
app = FastAPI(
    title="Cancer Var — учебный API",
    description=(
        "Демонстрация на табличном наборе scikit-learn. "
        "Не предназначено для медицинской диагностики или принятия решений."
    ),
    version="0.1.0",
)


class PredictionRequest(BaseModel):
    features: Annotated[list[FiniteFloat], Field(min_length=30, max_length=30)]


@app.get("/health")
def health() -> dict[str, str]:
    if not ARTIFACT_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Модель не найдена. Сначала выполните: python -m cancer_var.train",
        )
    return {"status": "ok", "model": "trained"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, float | str]:
    if not ARTIFACT_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Модель не найдена. Сначала выполните: python -m cancer_var.train",
        )

    artifact = load_artifact()
    model = artifact["model"]
    expected_features = len(artifact["feature_names"])
    if len(request.features) != expected_features:
        raise HTTPException(status_code=422, detail=f"Ожидается {expected_features} признаков")

    row = np.asarray(request.features, dtype=float).reshape(1, -1)
    if not np.isfinite(row).all():
        raise HTTPException(status_code=422, detail="Все признаки должны быть конечными числами")

    probabilities = model.predict_proba(row)[0]
    malignant_index = int(np.flatnonzero(model.classes_ == 0)[0])
    score = float(probabilities[malignant_index])
    return {
        "model_probability_malignant": score,
        "model_label": "malignant" if score >= 0.5 else "benign",
        "notice": "Учебная оценка модели; не является медицинским диагнозом или риском пациента.",
    }


@lru_cache(maxsize=1)
def load_artifact() -> dict:
    return joblib.load(ARTIFACT_PATH)


@app.get("/features")
def features() -> dict:
    health()
    artifact = load_artifact()
    return {"feature_names": artifact["feature_names"], "count": 30}
