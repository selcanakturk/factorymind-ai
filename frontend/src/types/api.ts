export type RiskCategory =
  | "Low Risk"
  | "Medium Risk"
  | "High Risk"
  | "Critical Risk";

export interface HealthResponse {
  status: "ok";
  service: string;
  model_loaded: boolean;
  failure_model_loaded: boolean;
  rul_model_loaded: boolean;
  anomaly_model_loaded: boolean;
}

export interface AnomalyObservation {
  cycle: number;
  sensor_2: number;
  sensor_3: number;
  sensor_4: number;
  sensor_7: number;
  sensor_8: number;
  sensor_9: number;
  sensor_11: number;
  sensor_12: number;
  sensor_13: number;
  sensor_14: number;
  sensor_15: number;
  sensor_17: number;
  sensor_20: number;
  sensor_21: number;
}

export interface AnomalyPredictionRequest {
  unit_id?: string | number | null;
  observations: AnomalyObservation[];
}

export interface SensorDeviationItem {
  sensor: string;
  current_value: number;
  reference_mean: number;
  standardized_deviation: number;
  direction: "above_normal" | "below_normal";
}

export interface AnomalyPredictionResponse {
  unit_id: string | number | null;
  current_anomaly_score: number;
  anomaly_percentile: number;
  threshold_percentile: number;
  raw_threshold: number;
  current_threshold_exceeded: boolean;
  recent_window_size: number;
  recent_exceedance_pattern: boolean[];
  recent_exceedance_count: number;
  persistence_required_count: number;
  persistence_window_size: number;
  persistence_status: "available" | "insufficient_history";
  alert_active: boolean;
  history_cycle_count: number;
  top_sensor_deviations: SensorDeviationItem[];
  sensor_context_label: string;
  model_version: string;
  dataset: string;
  development_stage: true;
  warning: string;
  disclaimer: string;
}

export interface AnomalyModelInfoResponse {
  model_name: string;
  model_version: string;
  model_family: string;
  dataset: string;
  dataset_subset: string;
  predictor_count: number;
  normal_reference_definition: string;
  threshold_percentile: number;
  raw_threshold: number;
  persistence_window: number;
  persistence_required_count: number;
  minimum_persistence_history: number;
  repeated_split_stability: Record<string, number>;
  healthy_alert_burden: Record<string, number>;
  critical_alert_coverage: Record<string, number>;
  lead_time_findings: Record<string, number>;
  known_limitations: string[];
  output_interpretation: string;
  warning: string;
  disclaimer: string;
}

export interface RULObservation {
  cycle: number;
  operational_setting_1: number;
  operational_setting_2: number;
  operational_setting_3: number;
  sensor_2: number;
  sensor_3: number;
  sensor_4: number;
  sensor_7: number;
  sensor_8: number;
  sensor_9: number;
  sensor_11: number;
  sensor_12: number;
  sensor_13: number;
  sensor_14: number;
  sensor_15: number;
  sensor_17: number;
  sensor_20: number;
  sensor_21: number;
}

export interface RULPredictionRequest {
  unit_id?: string | number | null;
  observations: RULObservation[];
}

export interface RULPredictionResponse {
  unit_id: string | number | null;
  predicted_rul_cycles: number;
  raw_model_prediction: number;
  rul_display: string;
  prediction_horizon_cap: number;
  history_cycle_count: number;
  history_quality: "limited_history" | "full_context";
  model_version: string;
  dataset: string;
  development_stage: true;
  warning: string;
  disclaimer: string;
}

export interface RULModelInfoResponse {
  model_name: string;
  model_version: string;
  model_family: string;
  dataset: string;
  dataset_subset: string;
  target: string;
  rul_cap: number;
  predictor_count: number;
  minimum_full_context_cycles: number;
  official_endpoint_metrics: Record<string, number>;
  near_failure_metrics: Record<string, number | string>;
  known_limitations: string[];
  output_interpretation: string;
  development_warning: string;
  disclaimer: string;
}

export interface ModelInfoResponse {
  model_version: string;
  threshold_version: string;
  model_family: string;
  calibration_method: string;
  raw_input_features: string[];
  engineered_features: string[];
  target: string;
  development_evaluation_metrics: Record<string, number>;
  output_interpretation: string;
  methodological_warnings: string[];
}

export interface FailurePredictionRequest {
  type: "L" | "M" | "H";
  air_temperature: number;
  process_temperature: number;
  rotational_speed: number;
  torque: number;
  tool_wear: number;
}

export interface FailurePredictionResponse {
  calibrated_risk_estimate: number;
  failure_risk_score: number;
  risk_category: RiskCategory;
  recommended_action: string;
  model_version: string;
  threshold_version: string;
  calibration_method: string;
  disclaimer: string;
}
