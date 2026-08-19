export type RiskCategory =
  | "Low Risk"
  | "Medium Risk"
  | "High Risk"
  | "Critical Risk";

export interface HealthResponse {
  status: "ok";
  service: string;
  model_loaded: boolean;
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
