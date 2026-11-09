"""Обучение и оценка базовой модели на встроенном наборе scikit-learn."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


@dataclass(frozen=True)
class TrainingResult:
    model: Pipeline
    feature_names: list[str]
    metrics: dict[str, float]


def train_model() -> TrainingResult:
    """Обучает воспроизводимый baseline и считает метрики на отложенной выборке."""
    dataset = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=dataset.target,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2_000, random_state=RANDOM_STATE)),
        ]
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)
    malignant_index = int(np.flatnonzero(model.classes_ == 0)[0])
    malignant_scores = probabilities[:, malignant_index]
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_malignant": float(precision_score(y_test, predictions, pos_label=0)),
        "recall_malignant": float(recall_score(y_test, predictions, pos_label=0)),
        "f1_malignant": float(f1_score(y_test, predictions, pos_label=0)),
        "roc_auc_malignant": float(roc_auc_score(y_test == 0, malignant_scores)),
    }
    return TrainingResult(model=model, feature_names=list(dataset.feature_names), metrics=metrics)
