import json
from pathlib import Path

import joblib

from src.rul_features import RUL_CAP, RUL_PREDICTOR_COLUMNS
from src.rul_pipeline import FROZEN_RANDOM_FOREST_HYPERPARAMETERS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "factorymind_rul_model_v1.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "factorymind_rul_model_v1.metadata.json"


def test_production_artifacts_exist_and_model_reloads():
    assert MODEL_PATH.exists()
    assert METADATA_PATH.exists()
    model = joblib.load(MODEL_PATH)
    fitted_rf = model.named_steps["model"]
    assert all(
        fitted_rf.get_params()[name] == value
        for name, value in FROZEN_RANDOM_FOREST_HYPERPARAMETERS.items()
    )


def test_metadata_contract_is_complete_and_path_free():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    required = {
        "model_name", "model_version", "model_family", "dataset",
        "dataset_subset", "target", "rul_cap", "raw_input_columns",
        "raw_predictor_columns", "predictor_columns", "predictor_count",
        "excluded_sensors", "temporal_base_sensors",
        "temporal_feature_definitions", "minimum_full_context_cycles",
        "short_history_behavior", "frozen_hyperparameters",
        "training_unit_count", "training_row_count",
        "groupkfold_metrics_notebook_08",
        "official_fd001_endpoint_metrics_notebook_09",
        "near_failure_metrics", "known_limitations", "output_interpretation",
        "warning", "disclaimer", "package_versions", "training_timestamp_utc",
    }
    assert required.issubset(metadata)
    assert metadata["rul_cap"] == RUL_CAP
    assert metadata["predictor_count"] == 47
    assert metadata["predictor_columns"] == RUL_PREDICTOR_COLUMNS
    assert metadata["frozen_hyperparameters"] == FROZEN_RANDOM_FOREST_HYPERPARAMETERS
    assert metadata["training_unit_count"] == 100
    assert metadata["training_row_count"] == 20_631

    serialized = json.dumps(metadata)
    assert str(PROJECT_ROOT) not in serialized
    assert not serialized.startswith("/")
