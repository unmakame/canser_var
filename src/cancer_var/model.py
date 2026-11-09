from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, f1_score, log_loss,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def malignant_probability(model, rows):
    index = int(np.flatnonzero(model.classes_ == 0)[0])
    return model.predict_proba(rows)[:, index]


def evaluate(target, scores):
    positive = np.asarray(target) == 0
    predicted = np.asarray(scores) >= 0.5
    return {
        "accuracy": float(accuracy_score(positive, predicted)),
        "precision_malignant": float(precision_score(positive, predicted, zero_division=0)),
        "recall_malignant": float(recall_score(positive, predicted, zero_division=0)),
        "f1_malignant": float(f1_score(positive, predicted, zero_division=0)),
        "brier_score": float(brier_score_loss(positive, scores)),
        "log_loss": float(log_loss(positive, scores, labels=[False, True])),
        "roc_auc_malignant": (
            float(roc_auc_score(positive, scores)) if len(np.unique(positive)) == 2 else None
        ),
    }


@dataclass(frozen=True)
class TrainingResult:
    model: object
    feature_names: list[str]
    metrics: dict
    report: dict
    reference: dict


def train_model():
    dataset = load_breast_cancer()
    train_ids, test_ids = train_test_split(
        np.arange(len(dataset.target)), test_size=0.2,
        random_state=RANDOM_STATE, stratify=dataset.target,
    )
    X_train, X_test = dataset.data[train_ids], dataset.data[test_ids]
    y_train, y_test = dataset.target[train_ids], dataset.target[test_ids]
    candidates = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=1,
        ),
    }
    candidates.update({
        f"{name}_calibrated": CalibratedClassifierCV(
            estimator=model, method="sigmoid",
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        )
        for name, model in list(candidates.items())
    })
    comparisons = {}
    for name, model in candidates.items():
        scores = cross_validate(
            model, X_train, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            scoring={"roc_auc": "roc_auc", "log_loss": "neg_log_loss"},
            error_score="raise", n_jobs=1,
        )
        comparisons[name] = {
            "roc_auc_mean": float(scores["test_roc_auc"].mean()),
            "roc_auc_std": float(scores["test_roc_auc"].std()),
            "log_loss_mean": float(-scores["test_log_loss"].mean()),
        }
    selected = min(comparisons, key=lambda name: comparisons[name]["log_loss_mean"])
    model = candidates[selected].fit(X_train, y_train)
    scores = malignant_probability(model, X_test)
    metrics = evaluate(y_test, scores)
    fraction, mean_probability = calibration_curve(y_test == 0, scores, n_bins=5)
    return TrainingResult(
        model=model, feature_names=list(dataset.feature_names), metrics=metrics,
        reference={"mean": X_train.mean(axis=0).tolist(), "std": X_train.std(axis=0).tolist()},
        report={
            "seed": RANDOM_STATE, "selected_model": selected,
            "selection_metric": "lowest_cv_log_loss", "cross_validation": comparisons,
            "test_metrics": metrics, "train_indices": train_ids.tolist(),
            "test_indices": test_ids.tolist(),
            "calibration_curve": {
                "mean_probability": mean_probability.tolist(),
                "fraction_malignant": fraction.tolist(),
            },
        },
    )
