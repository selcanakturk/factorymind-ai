"""Train, evaluate, and serialize the FactoryMind failure-risk model."""

from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .features import ENGINEERED_FEATURES, RAW_FEATURES
from .pipeline import build_production_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "factorymind_failure_model_v1.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "factorymind_failure_model_v1.metadata.json"
TARGET = "Machine failure"
MODEL_VERSION = "1.0.0"


def evaluate_probabilities(y_true: pd.Series, risk_estimates) -> dict[str, float]:
    """Calculate holdout ranking and calibration metrics."""
    return {
        "roc_auc": float(roc_auc_score(y_true, risk_estimates)),
        "average_precision": float(average_precision_score(y_true, risk_estimates)),
        "brier_score": float(brier_score_loss(y_true, risk_estimates)),
        "log_loss": float(log_loss(y_true, risk_estimates)),
    }


def train_and_save() -> tuple[Path, Path, dict[str, float]]:
    """Fit on the training partition and persist the model plus metadata."""
    data = pd.read_csv(DATA_PATH)
    X = data.loc[:, RAW_FEATURES].copy()
    y = data[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = build_production_model()
    model.fit(X_train, y_train)
    holdout_risk_estimates = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_probabilities(y_test, holdout_risk_estimates)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "model_version": MODEL_VERSION,
        "model_family": "GradientBoostingClassifier",
        "calibration_method": "sigmoid",
        "calibration_cv": "5-fold stratified cross-validation",
        "raw_input_features": RAW_FEATURES,
        "engineered_features": ENGINEERED_FEATURES,
        "target": TARGET,
        "training_dataset": "AI4I 2020 Predictive Maintenance Dataset",
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "training_observations": int(len(y_train)),
        "holdout_observations": int(len(y_test)),
        "holdout_failures": int(y_test.sum()),
        "evaluation_metrics": metrics,
        "output_interpretation": (
            "Sigmoid-calibrated model-derived risk estimate between 0 and 1; "
            "not a guaranteed real-world failure probability."
        ),
        "methodological_warning": (
            "The fixed holdout is development-stage and was examined in previous "
            "notebooks. Validate on an independent lockbox before production claims."
        ),
        "categorical_risk_thresholds": None,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    return MODEL_PATH, METADATA_PATH, metrics


def main() -> None:
    model_path, metadata_path, metrics = train_and_save()
    print("Holdout evaluation")
    for name, value in metrics.items():
        print(f"  {name}: {value:.6f}")
    print(f"Model: {model_path.relative_to(PROJECT_ROOT)}")
    print(f"Metadata: {metadata_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
