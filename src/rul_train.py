"""Train and serialize the frozen FactoryMind RUL v1 development model."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn

from .rul_features import (
    EXCLUDED_SENSORS,
    MINIMUM_FULL_CONTEXT_CYCLES,
    RAW_INPUT_COLUMNS,
    RAW_PREDICTOR_COLUMNS,
    RETAINED_SENSORS,
    RUL_CAP,
    RUL_PREDICTOR_COLUMNS,
    TEMPORAL_BASE_SENSORS,
    TEMPORAL_FEATURE_DEFINITIONS,
    TEMPORAL_FEATURE_COLUMNS,
    engineer_temporal_features,
)
from .rul_pipeline import (
    FROZEN_RANDOM_FOREST_HYPERPARAMETERS,
    MODEL_VERSION,
    RUL_DISCLAIMER,
    RUL_WARNING,
    build_rul_pipeline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train_FD001.txt"
MODEL_PATH = PROJECT_ROOT / "models" / "factorymind_rul_model_v1.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "factorymind_rul_model_v1.metadata.json"

CANONICAL_COLUMNS = (
    ["unit_id", "cycle"]
    + [f"operational_setting_{index}" for index in range(1, 4)]
    + [f"sensor_{index}" for index in range(1, 22)]
)


def load_training_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load FD001 training trajectories and validate the frozen dataset shape."""
    data = pd.read_csv(path, sep=r"\s+", header=None)
    if data.shape[1] != len(CANONICAL_COLUMNS):
        raise ValueError(
            f"Expected {len(CANONICAL_COLUMNS)} FD001 columns, found {data.shape[1]}."
        )
    data.columns = CANONICAL_COLUMNS
    if len(data) != 20_631 or data["unit_id"].nunique() != 100:
        raise ValueError("FD001 training data does not match the frozen 20,631-row/100-unit contract.")
    if data.isna().any().any():
        raise ValueError("FD001 raw training data contains missing values.")
    return data


def prepare_training_matrix(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Create the frozen predictors and capped target from run-to-failure data."""
    prepared = data.sort_values(["unit_id", "cycle"]).copy()
    prepared["raw_rul"] = (
        prepared.groupby("unit_id")["cycle"].transform("max") - prepared["cycle"]
    )
    prepared["capped_rul"] = prepared["raw_rul"].clip(upper=RUL_CAP)
    featured = engineer_temporal_features(prepared)

    if (featured["raw_rul"] < 0).any():
        raise RuntimeError("Derived training RUL cannot be negative.")
    final_rows = featured.loc[featured.groupby("unit_id")["cycle"].idxmax()]
    if not (final_rows["raw_rul"] == 0).all():
        raise RuntimeError("Every training unit must end at RUL zero.")

    X = featured.loc[:, RUL_PREDICTOR_COLUMNS]
    y = featured["capped_rul"]
    if X.shape != (20_631, 47):
        raise RuntimeError(f"Unexpected frozen training matrix shape: {X.shape}")
    return X, y, featured


def build_metadata(training_predictions: np.ndarray) -> dict:
    """Build complete, path-free metadata for the future backend loader."""
    return {
        "model_name": "FactoryMind RUL FD001 Random Forest",
        "model_version": MODEL_VERSION,
        "model_family": "RandomForestRegressor",
        "dataset": "NASA C-MAPSS",
        "dataset_subset": "FD001",
        "target": "capped_rul",
        "rul_cap": RUL_CAP,
        "raw_input_columns": RAW_INPUT_COLUMNS,
        "raw_predictor_columns": RAW_PREDICTOR_COLUMNS,
        "predictor_columns": RUL_PREDICTOR_COLUMNS,
        "predictor_count": len(RUL_PREDICTOR_COLUMNS),
        "excluded_sensors": EXCLUDED_SENSORS,
        "retained_sensors": RETAINED_SENSORS,
        "temporal_base_sensors": TEMPORAL_BASE_SENSORS,
        "temporal_feature_definitions": TEMPORAL_FEATURE_DEFINITIONS,
        "minimum_full_context_cycles": MINIMUM_FULL_CONTEXT_CYCLES,
        "short_history_behavior": (
            "Trajectories with 1-5 cycles are scoreable using trained median "
            "imputation and are labeled limited_history; no history is fabricated."
        ),
        "frozen_hyperparameters": FROZEN_RANDOM_FOREST_HYPERPARAMETERS,
        "training_unit_count": 100,
        "training_row_count": 20_631,
        "groupkfold_metrics_notebook_08": {
            "rmse_mean": 16.3588,
            "rmse_std": 0.8122,
            "mae_mean": 11.1010,
            "mae_std": 0.6563,
            "r2_mean": 0.8456,
            "r2_std": 0.0149,
        },
        "official_fd001_endpoint_metrics_notebook_09": {
            "capped_rmse": 17.5413,
            "capped_mae": 13.0300,
            "capped_r2": 0.8084,
            "nasa_total_score": 522.9481,
            "nasa_mean_penalty": 5.2295,
        },
        "near_failure_metrics": {
            "definition": "official raw RUL <= 30 cycles",
            "rmse": 14.1842,
            "mae": 8.8359,
            "mean_signed_error_prediction_minus_actual": 7.0016,
        },
        "training_diagnostics": {
            "label": "training-fit diagnostics; not generalization metrics",
            "prediction_min": float(np.min(training_predictions)),
            "prediction_max": float(np.max(training_predictions)),
        },
        "known_limitations": [
            "Simulated FD001 data with one operating condition and one fault mode.",
            "The 125-cycle target cap is provisional.",
            "Point estimates only; no calibrated uncertainty is available.",
            "The model may overestimate RUL, particularly near failure.",
            "The official FD001 test set is development-exposed after notebook 09.",
            "No external real-fleet validation exists.",
        ],
        "output_interpretation": (
            "Development-stage capped-RUL point estimate for the latest supplied "
            "trajectory observation; values near the cap represent a long-horizon region."
        ),
        "warning": RUL_WARNING,
        "disclaimer": RUL_DISCLAIMER,
        "package_versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def train_and_save() -> tuple[Path, Path, dict]:
    """Fit on FD001 train only, serialize the model, and write metadata."""
    data = load_training_data()
    X, y, _ = prepare_training_matrix(data)
    model = build_rul_pipeline().fit(X, y)
    training_predictions = model.predict(X)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    metadata = build_metadata(training_predictions)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return MODEL_PATH, METADATA_PATH, metadata


def main() -> None:
    model_path, metadata_path, metadata = train_and_save()
    diagnostics = metadata["training_diagnostics"]
    print("Training-fit diagnostics only (not generalization metrics)")
    print(f"  prediction_min: {diagnostics['prediction_min']:.6f}")
    print(f"  prediction_max: {diagnostics['prediction_max']:.6f}")
    print(f"Model: {model_path.relative_to(PROJECT_ROOT)}")
    print(f"Metadata: {metadata_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
