import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn

from cancer_var.model import train_model

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"


def main():
    result = train_model()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "selected_model": result.report["selected_model"],
    }
    joblib.dump({
        "model": result.model, "feature_names": result.feature_names,
        "reference": result.reference, "metadata": metadata,
        "test_metrics": result.metrics,
    }, ARTIFACT_DIR / "breast_cancer_pipeline.joblib")
    for filename, payload in (
        ("metrics.json", result.metrics),
        ("training_report.json", {**result.report, **metadata}),
    ):
        (ARTIFACT_DIR / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
        )
    print(json.dumps({**metadata, "metrics": result.metrics}, indent=2))


if __name__ == "__main__":
    main()
