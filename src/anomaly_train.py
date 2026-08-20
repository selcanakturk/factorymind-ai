"""Train and serialize the frozen FactoryMind Anomaly Detection v1 bundle."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn

from .anomaly_features import (
    ANOMALY_PREDICTOR_COUNT,
    ANOMALY_SENSOR_COLUMNS,
    NORMAL_REFERENCE_RUL_THRESHOLD,
    PERSISTENCE_REQUIRED_COUNT,
    PERSISTENCE_WINDOW,
    THRESHOLD_QUANTILE,
)
from .anomaly_pipeline import (
    ANOMALY_DISCLAIMER,
    ANOMALY_WARNING,
    FROZEN_ISOLATION_FOREST_HYPERPARAMETERS,
    MODEL_VERSION,
    fit_anomaly_bundle,
    native_to_anomaly_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train_FD001.txt"
MODEL_PATH = PROJECT_ROOT / "models" / "factorymind_anomaly_model_v1.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "factorymind_anomaly_model_v1.metadata.json"
CANONICAL_COLUMNS = (
    ["unit_id", "cycle"]
    + [f"operational_setting_{index}" for index in range(1, 4)]
    + [f"sensor_{index}" for index in range(1, 22)]
)


def load_training_data(path: Path = DATA_PATH) -> pd.DataFrame:
    data = pd.read_csv(path, sep=r"\s+", header=None)
    if data.shape != (20_631, len(CANONICAL_COLUMNS)):
        raise ValueError("FD001 data does not match the frozen 20,631-row schema.")
    data.columns = CANONICAL_COLUMNS
    if data["unit_id"].nunique() != 100 or data.isna().any().any():
        raise ValueError("FD001 data does not match the frozen 100-unit complete-data contract.")
    return data


def prepare_normal_reference(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared = data.copy()
    prepared["raw_rul"] = prepared.groupby("unit_id")["cycle"].transform("max") - prepared["cycle"]
    reference_rows = prepared[prepared["raw_rul"] > NORMAL_REFERENCE_RUL_THRESHOLD]
    sensors = reference_rows.loc[:, ANOMALY_SENSOR_COLUMNS]
    if sensors.shape != (8_031, ANOMALY_PREDICTOR_COUNT):
        raise RuntimeError(f"Unexpected normal-reference shape: {sensors.shape}")
    if reference_rows["unit_id"].nunique() != 100:
        raise RuntimeError("Final normal reference must contain all 100 development units.")
    return sensors, prepared


def _score_summary(scores: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(scores)), "median": float(np.median(scores)),
        "mean": float(np.mean(scores)), "std": float(np.std(scores, ddof=1)),
        "p90": float(np.quantile(scores, .90)), "p95": float(np.quantile(scores, .95)),
        "p97_5": float(np.quantile(scores, .975)), "p99": float(np.quantile(scores, .99)),
        "max": float(np.max(scores)),
    }


def development_diagnostics(bundle, prepared: pd.DataFrame) -> dict[str, float | int | str]:
    standardized = bundle.scaler.transform(prepared[ANOMALY_SENSOR_COLUMNS])
    scores = native_to_anomaly_score(bundle.detector.score_samples(standardized))
    diagnostics = prepared[["unit_id", "cycle", "raw_rul"]].copy()
    diagnostics["exceeded"] = scores > bundle.raw_threshold
    diagnostics["persistent"] = diagnostics.groupby("unit_id")["exceeded"].transform(
        lambda values: values.rolling(PERSISTENCE_WINDOW, min_periods=PERSISTENCE_WINDOW)
        .sum().ge(PERSISTENCE_REQUIRED_COUNT)
    )
    healthy = diagnostics["raw_rul"] > NORMAL_REFERENCE_RUL_THRESHOLD
    critical = diagnostics["raw_rul"] <= 30
    unit_index = prepared["unit_id"].drop_duplicates().sort_values()
    premature = diagnostics[healthy].groupby("unit_id")["persistent"].any().reindex(unit_index, fill_value=False)
    critical_coverage = diagnostics[critical].groupby("unit_id")["persistent"].any().reindex(unit_index, fill_value=False)
    first_alerts = diagnostics[diagnostics["persistent"]].groupby("unit_id", sort=False).first()
    lead_times = first_alerts["raw_rul"].reindex(unit_index)
    return {
        "label": "development diagnostics; not independent validation",
        "healthy_observation_exceedance_pct": float(100 * diagnostics.loc[healthy, "exceeded"].mean()),
        "critical_observation_exceedance_pct": float(100 * diagnostics.loc[critical, "exceeded"].mean()),
        "premature_healthy_persistent_alert_engines_pct": float(100 * premature.mean()),
        "critical_persistent_alert_engine_coverage_pct": float(100 * critical_coverage.mean()),
        "engines_never_persistently_alerted": int(lead_times.isna().sum()),
        "lead_time_median": float(lead_times.median()),
        "lead_time_q25": float(lead_times.quantile(.25)),
        "lead_time_q75": float(lead_times.quantile(.75)),
        "lead_time_min": float(lead_times.min()), "lead_time_max": float(lead_times.max()),
    }


def build_metadata(bundle, diagnostics: dict) -> dict:
    scores = bundle.sorted_normal_scores
    return {
        "model_name": "FactoryMind Anomaly FD001 Isolation Forest",
        "model_version": MODEL_VERSION, "model_family": "IsolationForest",
        "dataset": "NASA C-MAPSS", "dataset_subset": "FD001",
        "predictor_columns": ANOMALY_SENSOR_COLUMNS, "predictor_count": ANOMALY_PREDICTOR_COUNT,
        "normal_reference_definition": "FD001 training observations with retrospective raw_rul > 125",
        "normal_reference_rul_threshold": NORMAL_REFERENCE_RUL_THRESHOLD,
        "normal_reference_row_count": 8_031, "normal_reference_unit_count": 100,
        "scaler_type": "StandardScaler(with_mean=True, with_std=True)",
        "isolation_forest_hyperparameters": FROZEN_ISOLATION_FOREST_HYPERPARAMETERS,
        "score_direction": "higher_is_more_anomalous via negative score_samples",
        "score_distribution_summary": _score_summary(scores),
        "threshold_quantile": THRESHOLD_QUANTILE, "threshold_raw_score": bundle.raw_threshold,
        "percentile_mapping_method": "100 * count(training_normal_score <= score) / reference_count; right-inclusive ties",
        "persistence_window": PERSISTENCE_WINDOW, "persistence_required_count": PERSISTENCE_REQUIRED_COUNT,
        "minimum_persistence_history": PERSISTENCE_WINDOW,
        "sensor_deviation_method": "StandardScaler z-score; top 5 sorted by absolute deviation",
        "repeated_split_stability_notebook_12": {"spearman_mean": -0.6882, "spearman_std": 0.0288, "spearman_min": -0.7293, "spearman_max": -0.6543},
        "alert_rate_stability_notebook_12": {"quantile": .975, "healthy_observation_mean_pct": 2.85, "healthy_observation_std_pct": 1.03, "critical_observation_mean_pct": 99.65, "critical_observation_std_pct": .42, "critical_engine_coverage_pct": 100.0},
        "persistence_diagnostics_notebook_12": {"policy": "3 of latest 5", "premature_healthy_engines_mean_pct": 7.0, "critical_engine_coverage_pct": 100.0},
        "lead_time_findings_notebook_12": {"mean_median_retrospective_cycles": 59.8},
        "development_diagnostics": diagnostics,
        "known_limitations": ["FD001 is simulated with one operating condition and one fault mode.", "Early-life RUL is a retrospective healthy-state proxy.", "No true or externally validated anomaly labels exist.", "Quantile alerts and lead times are development heuristics.", "No external fleet validation or drift evidence exists."],
        "output_interpretation": "Non-probabilistic unusualness relative to the fitted development normal-reference distribution.",
        "warning": ANOMALY_WARNING, "disclaimer": ANOMALY_DISCLAIMER,
        "training_row_count": 20_631, "training_unit_count": 100,
        "package_versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__},
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def train_and_save() -> tuple[Path, Path, dict]:
    data = load_training_data()
    normal_sensors, prepared = prepare_normal_reference(data)
    bundle = fit_anomaly_bundle(normal_sensors)
    diagnostics = development_diagnostics(bundle, prepared)
    metadata = build_metadata(bundle, diagnostics)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return MODEL_PATH, METADATA_PATH, metadata


def main() -> None:
    model_path, metadata_path, metadata = train_and_save()
    print("Frozen anomaly v1 development diagnostics (not independent validation)")
    print(f"  threshold_raw_score: {metadata['threshold_raw_score']:.8f}")
    for key, value in metadata["development_diagnostics"].items():
        if key != "label": print(f"  {key}: {value}")
    print(f"Model: {model_path.relative_to(PROJECT_ROOT)}")
    print(f"Metadata: {metadata_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
