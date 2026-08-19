import type { HealthResponse, ModelInfoResponse } from "../types/api";

interface Props {
  health: HealthResponse | null;
  modelInfo: ModelInfoResponse | null;
  loading: boolean;
  error: string | null;
  onOpenAnalysis: () => void;
  onRetry: () => void;
}

const riskDefinitions = [
  ["Low", "Routine monitoring", "low"],
  ["Medium", "Increase observation", "medium"],
  ["High", "Plan an inspection", "high"],
  ["Critical", "Prioritize review", "critical"],
] as const;

export function OverviewPage({
  health,
  modelInfo,
  loading,
  error,
  onOpenAnalysis,
  onRetry,
}: Props) {
  return (
    <div className="page-stack">
      <header className="page-heading overview-heading">
        <div>
          <p className="eyebrow">Predictive maintenance · Module 01</p>
          <h1>Machine intelligence,<br />made operational.</h1>
          <p className="page-lead">
            Evaluate machine operating conditions with a calibrated,
            development-stage failure-risk model—without inventing fleet data.
          </p>
        </div>
        <button className="primary-button" onClick={onOpenAnalysis}>
          Analyze a machine <span aria-hidden="true">→</span>
        </button>
      </header>

      {error && (
        <div className="alert alert-error" role="alert">
          <div><strong>Backend unavailable</strong><span>{error}</span></div>
          <button onClick={onRetry}>Retry connection</button>
        </div>
      )}

      <section className="metric-grid" aria-label="Platform summary">
        <article className="metric-card accent-card">
          <span className="metric-label">System status</span>
          <div className="metric-value status-value">
            <span className={`status-dot ${health?.model_loaded ? "online" : "offline"}`} />
            {loading ? "Checking" : health?.model_loaded ? "Operational" : "Offline"}
          </div>
          <p>API and model readiness</p>
        </article>
        <article className="metric-card">
          <span className="metric-label">Model version</span>
          <div className="metric-value mono">v{modelInfo?.model_version ?? "—"}</div>
          <p>{modelInfo?.model_family ?? "Loading model metadata"}</p>
        </article>
        <article className="metric-card">
          <span className="metric-label">Calibration</span>
          <div className="metric-value capitalize">{modelInfo?.calibration_method ?? "—"}</div>
          <p>Model-derived risk estimate</p>
        </article>
        <article className="metric-card">
          <span className="metric-label">Available inputs</span>
          <div className="metric-value">{modelInfo?.raw_input_features.length ?? "—"}</div>
          <p>Raw operating measurements</p>
        </article>
      </section>

      <section className="overview-grid">
        <article className="panel engine-panel">
          <div className="panel-heading">
            <div><span className="section-kicker">Active capability</span><h2>Failure Risk Engine</h2></div>
            <span className="status-pill active">Live</span>
          </div>
          <p>
            FactoryMind currently analyzes product class, temperatures,
            rotational speed, torque, and tool wear. Three physically meaningful
            interactions are created inside the model pipeline.
          </p>
          <div className="sensor-list">
            {(modelInfo?.raw_input_features ?? ["Type", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]).map((feature, index) => (
              <div key={feature}><span className="sensor-index">0{index + 1}</span><span>{feature}</span></div>
            ))}
          </div>
        </article>

        <article className="panel risk-panel">
          <div className="panel-heading"><div><span className="section-kicker">Decision support</span><h2>Risk levels</h2></div></div>
          <div className="risk-definition-list">
            {riskDefinitions.map(([name, action, tone]) => (
              <div key={name} className="risk-definition">
                <span className={`risk-marker ${tone}`} />
                <div><strong>{name} Risk</strong><span>{action}</span></div>
              </div>
            ))}
          </div>
          <p className="fine-print">Categories are returned by the backend using versioned calibrated thresholds.</p>
        </article>
      </section>

      <section className="future-strip" aria-label="Future modules">
        <div><span className="section-kicker">Platform roadmap</span><h2>Built to expand beyond failure risk.</h2></div>
        {[
          ["RUL", "Remaining Useful Life"],
          ["AD", "Anomaly Detection"],
          ["QI", "Quality Inspection"],
        ].map(([code, title]) => (
          <div className="future-item" key={code}><span>{code}</span><div><strong>{title}</strong><small>Coming Soon</small></div></div>
        ))}
      </section>
    </div>
  );
}
