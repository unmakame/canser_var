import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer
from cancer_var import api
from cancer_var.model import train_model


@pytest.fixture(scope="module")
def trained():
    return train_model()


@pytest.fixture
def client(tmp_path, monkeypatch, trained):
    path = tmp_path / "model.joblib"
    joblib.dump({"model": trained.model, "feature_names": trained.feature_names}, path)
    monkeypatch.setattr(api, "ARTIFACT_PATH", path)
    api.load_artifact.cache_clear()
    with TestClient(api.app) as session:
        yield session
    api.load_artifact.cache_clear()


def test_prediction_matches_model(client, trained):
    row = load_breast_cancer().data[0]
    response = client.post("/predict", json={"features": row.tolist()})
    assert response.status_code == 200
    index = int(np.flatnonzero(trained.model.classes_ == 0)[0])
    expected = trained.model.predict_proba([row])[0, index]
    assert response.json()["model_probability_malignant"] == pytest.approx(expected)


@pytest.mark.parametrize("values", [[], [1.0] * 29, [1.0] * 31, ["invalid"] * 30])
def test_invalid_input(client, values):
    assert client.post("/predict", json={"features": values}).status_code == 422


def test_feature_order(client, trained):
    assert client.get("/features").json()["feature_names"] == trained.feature_names


def test_missing_model(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api, "ARTIFACT_PATH", tmp_path / "missing.joblib")
    assert client.get("/health").status_code == 503
    assert client.post("/predict", json={"features": [1.0] * 30}).status_code == 503


def test_training_metrics(trained):
    assert all(value >= 0 for value in trained.metrics.values())
    assert trained.metrics["roc_auc_malignant"] > 0.9


def test_training_split_and_selection(trained):
    report = trained.report
    assert not set(report["train_indices"]) & set(report["test_indices"])
    assert len(report["train_indices"]) + len(report["test_indices"]) == 569
    comparisons = report["cross_validation"]
    assert len(comparisons) == 4
    assert report["selected_model"] == min(
        comparisons, key=lambda name: comparisons[name]["log_loss_mean"]
    )
