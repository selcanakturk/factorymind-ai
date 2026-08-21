import type {
  AnomalyModelInfoResponse,
  AnomalyPredictionRequest,
  AnomalyPredictionResponse,
  FailurePredictionRequest,
  FailurePredictionResponse,
  HealthResponse,
  ModelInfoResponse,
  RULModelInfoResponse,
  RULPredictionRequest,
  RULPredictionResponse,
} from "../types/api";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ??
  "http://127.0.0.1:8000"
).replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
  ) {
    super(message);
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      headers: { "Content-Type": "application/json", ...options?.headers },
    });
  } catch {
    throw new ApiError(
      "FactoryMind API is offline. Start the backend and try again.",
    );
  }

  if (!response.ok) {
    let message = "The request could not be completed.";
    try {
      const payload = (await response.json()) as { detail?: unknown };
      if (typeof payload.detail === "string") message = payload.detail;
      if (Array.isArray(payload.detail)) {
        message = payload.detail
          .map((item) =>
            typeof item === "object" && item && "msg" in item
              ? String(item.msg)
              : "Invalid input",
          )
          .join(" · ");
      }
    } catch {
      // Preserve the safe generic message for non-JSON errors.
    }
    throw new ApiError(message, response.status);
  }

  return (await response.json()) as T;
}

export const factoryMindApi = {
  health: () => request<HealthResponse>("/health"),
  modelInfo: () => request<ModelInfoResponse>("/model/info"),
  getRulModelInfo: () => request<RULModelInfoResponse>("/model/rul/info"),
  getAnomalyModelInfo: () => request<AnomalyModelInfoResponse>("/model/anomaly/info"),
  predictFailure: (payload: FailurePredictionRequest) =>
    request<FailurePredictionResponse>("/predict/failure", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  predictRul: (payload: RULPredictionRequest) =>
    request<RULPredictionResponse>("/predict/rul", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  predictAnomaly: (payload: AnomalyPredictionRequest) =>
    request<AnomalyPredictionResponse>("/predict/anomaly", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
};
