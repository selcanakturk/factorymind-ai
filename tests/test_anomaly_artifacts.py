import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.anomaly_features import (
    ANOMALY_SENSOR_COLUMNS,
    PERSISTENCE_REQUIRED_COUNT,
    PERSISTENCE_WINDOW,
    THRESHOLD_QUANTILE,
)
from src.anomaly_pipeline import (
    AnomalyModelBundle,
    FROZEN_ISOLATION_FOREST_HYPERPARAMETERS,
    evaluate_anomaly_trajectory,
    score_anomaly_observation,
)


PROJECT_ROOT=Path(__file__).resolve().parents[1]
MODEL_PATH=PROJECT_ROOT/"models"/"factorymind_anomaly_model_v1.joblib"
METADATA_PATH=PROJECT_ROOT/"models"/"factorymind_anomaly_model_v1.metadata.json"


def test_production_artifacts_exist_and_reload():
    assert MODEL_PATH.exists() and METADATA_PATH.exists()
    bundle=joblib.load(MODEL_PATH)
    assert isinstance(bundle,AnomalyModelBundle)
    assert all(bundle.detector.get_params()[key]==value for key,value in FROZEN_ISOLATION_FOREST_HYPERPARAMETERS.items())
    assert bundle.raw_threshold==pytest.approx(np.quantile(bundle.sorted_normal_scores,THRESHOLD_QUANTILE))
    assert bundle.raw_threshold==pytest.approx(0.51971655, abs=1e-8)


def test_loaded_artifact_reproduces_observation_and_persistence():
    bundle=joblib.load(MODEL_PATH)
    values=pd.DataFrame([bundle.scaler.mean_],columns=ANOMALY_SENSOR_COLUMNS)
    first=score_anomaly_observation(bundle,values)
    second=score_anomaly_observation(joblib.load(MODEL_PATH),values)
    assert first==second
    trajectory=pd.concat([values]*5,ignore_index=True); trajectory.insert(0,"cycle",range(1,6))
    assert evaluate_anomaly_trajectory(bundle,trajectory)==evaluate_anomaly_trajectory(joblib.load(MODEL_PATH),trajectory)


def test_metadata_is_complete_consistent_and_path_free():
    metadata=json.loads(METADATA_PATH.read_text())
    required={"model_name","model_version","model_family","dataset","dataset_subset","predictor_columns","predictor_count",
      "normal_reference_definition","normal_reference_rul_threshold","normal_reference_row_count","normal_reference_unit_count",
      "scaler_type","isolation_forest_hyperparameters","score_direction","score_distribution_summary","threshold_quantile",
      "threshold_raw_score","percentile_mapping_method","persistence_window","persistence_required_count","minimum_persistence_history",
      "sensor_deviation_method","repeated_split_stability_notebook_12","alert_rate_stability_notebook_12",
      "persistence_diagnostics_notebook_12","lead_time_findings_notebook_12","known_limitations","output_interpretation",
      "warning","disclaimer","training_row_count","training_unit_count","package_versions","training_timestamp_utc"}
    assert required.issubset(metadata)
    assert metadata["predictor_columns"]==ANOMALY_SENSOR_COLUMNS and metadata["predictor_count"]==14
    assert metadata["threshold_quantile"]==THRESHOLD_QUANTILE
    assert metadata["persistence_window"]==PERSISTENCE_WINDOW
    assert metadata["persistence_required_count"]==PERSISTENCE_REQUIRED_COUNT
    assert metadata["normal_reference_row_count"]==8031 and metadata["normal_reference_unit_count"]==100
    assert metadata["isolation_forest_hyperparameters"]==FROZEN_ISOLATION_FOREST_HYPERPARAMETERS
    serialized=json.dumps(metadata)
    assert str(PROJECT_ROOT) not in serialized
    assert "RUL_FD001.txt" not in serialized
