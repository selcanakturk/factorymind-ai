import type { ModelInfoResponse, RULModelInfoResponse } from "../types/api";

interface Props { modelInfo: ModelInfoResponse | null; rulModelInfo: RULModelInfoResponse | null; loading: boolean; error: string | null; onRetry: () => void; }

const metricLabels: Record<string, string> = {
  roc_auc: "ROC–AUC",
  average_precision: "Average Precision",
  brier_score: "Brier Score",
  log_loss: "Log Loss",
};

export function ModelInfoPage({ modelInfo, rulModelInfo, loading, error, onRetry }: Props) {
  if (loading) return <div className="page-stack"><header className="page-heading compact-heading"><div><p className="eyebrow">Model governance</p><h1>Model Information</h1></div></header><div className="panel loading-panel"><span className="loader-ring" /><p>Loading versioned model metadata…</p></div></div>;
  if (error || !modelInfo) return <div className="page-stack"><header className="page-heading compact-heading"><div><p className="eyebrow">Model governance</p><h1>Model Information</h1></div></header><div className="alert alert-error" role="alert"><div><strong>Metadata unavailable</strong><span>{error ?? "No model metadata returned."}</span></div><button onClick={onRetry}>Retry</button></div></div>;

  return (
    <div className="page-stack">
      <header className="page-heading compact-heading"><div><p className="eyebrow">Model governance</p><h1>Model Information</h1><p className="page-lead">A transparent view of the model currently serving predictive-maintenance risk estimates.</p></div><span className="version-lock">Model v{modelInfo.model_version}</span></header>
      <section className="metric-grid model-metrics">
        {Object.entries(modelInfo.development_evaluation_metrics).map(([key, value]) => <article className="metric-card" key={key}><span className="metric-label">{metricLabels[key] ?? key}</span><div className="metric-value mono">{value.toFixed(4)}</div><p>{key === "brier_score" || key === "log_loss" ? "Lower is better" : "Development holdout"}</p></article>)}
      </section>
      <section className="model-info-grid">
        <article className="panel"><div className="panel-heading"><div><span className="section-kicker">Serving configuration</span><h2>Model specification</h2></div></div><dl className="spec-list"><div><dt>Family</dt><dd>{modelInfo.model_family}</dd></div><div><dt>Target</dt><dd>{modelInfo.target}</dd></div><div><dt>Calibration</dt><dd className="capitalize">{modelInfo.calibration_method}</dd></div><div><dt>Threshold version</dt><dd>v{modelInfo.threshold_version}</dd></div></dl></article>
        <article className="panel feature-panel"><div className="panel-heading"><div><span className="section-kicker">Feature contract</span><h2>Inputs & transformations</h2></div></div><h3>Raw client inputs</h3><div className="tag-list">{modelInfo.raw_input_features.map((feature) => <span key={feature}>{feature}</span>)}</div><h3>Internally engineered</h3><div className="tag-list engineered">{modelInfo.engineered_features.map((feature) => <span key={feature}>{feature}</span>)}</div></article>
      </section>
      <article className="panel interpretation-panel"><div><span className="section-kicker">Responsible use</span><h2>Development-stage interpretation</h2><p>{modelInfo.output_interpretation}</p></div><div className="warning-list">{modelInfo.methodological_warnings.map((warning, index) => <div key={`${index}-${warning}`}><span aria-hidden="true">!</span><p>{warning}</p></div>)}</div></article>
      {rulModelInfo && <section className="panel rul-model-panel"><div className="panel-heading"><div><span className="section-kicker">Module 02 · Remaining Useful Life</span><h2>{rulModelInfo.model_name}</h2></div><span className="version-lock">Model v{rulModelInfo.model_version}</span></div><dl className="spec-list"><div><dt>Family</dt><dd>{rulModelInfo.model_family}</dd></div><div><dt>Dataset</dt><dd>{rulModelInfo.dataset} · {rulModelInfo.dataset_subset}</dd></div><div><dt>Predictors</dt><dd>{rulModelInfo.predictor_count}</dd></div><div><dt>Horizon cap</dt><dd>{rulModelInfo.rul_cap} cycles</dd></div><div><dt>Full context</dt><dd>{rulModelInfo.minimum_full_context_cycles}+ cycles</dd></div><div><dt>Endpoint RMSE</dt><dd>{rulModelInfo.official_endpoint_metrics.rmse?.toFixed(2) ?? "—"}</dd></div><div><dt>Endpoint MAE</dt><dd>{rulModelInfo.official_endpoint_metrics.mae?.toFixed(2) ?? "—"}</dd></div><div><dt>Endpoint R²</dt><dd>{rulModelInfo.official_endpoint_metrics.r2?.toFixed(3) ?? "—"}</dd></div></dl><p className="fine-print">{rulModelInfo.output_interpretation}</p><div className="warning-list light-warnings">{rulModelInfo.known_limitations.map((item) => <div key={item}><span aria-hidden="true">!</span><p>{item}</p></div>)}</div></section>}
    </div>
  );
}
