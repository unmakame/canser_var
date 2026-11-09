import numpy as np
import pytest

from cancer_var.monitor import monitor


class Model:
    classes_ = np.array([0, 1])

    def predict_proba(self, rows):
        return np.tile([0.8, 0.2], (len(rows), 1))


@pytest.fixture
def artifact():
    return {
        "model": Model(), "feature_names": ["a", "b"],
        "reference": {"mean": [0, 0], "std": [1, 1]},
        "test_metrics": {"brier_score": 0.02},
    }


def test_detects_shift_without_claiming_quality(artifact):
    report = monitor(artifact, [[0, 3], [0, 3]])
    assert report["shifted_features"] == ["b"]
    assert report["quality"] is None


def test_single_class_quality_has_no_auc(artifact):
    report = monitor(artifact, [[0, 0], [0, 0]], [0, 0])
    assert report["quality"]["roc_auc_malignant"] is None
    assert report["quality"]["brier_score"] == pytest.approx(0.04)
    assert report["quality_delta_from_test"]["brier_score"] == pytest.approx(0.02)


@pytest.mark.parametrize("rows", [[], [[1]], [[float("nan"), 0]], [[float("inf"), 0]]])
def test_rejects_invalid_rows(artifact, rows):
    with pytest.raises(ValueError):
        monitor(artifact, rows)


def test_rejects_invalid_labels(artifact):
    with pytest.raises(ValueError):
        monitor(artifact, [[0, 0]], [2])
