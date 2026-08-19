"use client";

import { FormEvent, useState } from "react";
import { ApiError, factoryMindApi } from "../services/api";
import type { FailurePredictionRequest, FailurePredictionResponse, RiskCategory } from "../types/api";

type FormState = Record<Exclude<keyof FailurePredictionRequest, "type">, string> & { type: "L" | "M" | "H" };

const initialForm: FormState = {
  type: "M",
  air_temperature: "300.1",
  process_temperature: "310.4",
  rotational_speed: "1450",
  torque: "48.2",
  tool_wear: "125",
};

const fields = [
  ["air_temperature", "Air Temperature", "K", "Ambient temperature surrounding the machine"],
  ["process_temperature", "Process Temperature", "K", "Temperature measured during operation"],
  ["rotational_speed", "Rotational Speed", "rpm", "Current spindle or shaft speed"],
  ["torque", "Torque", "Nm", "Applied rotational force"],
  ["tool_wear", "Tool Wear", "min", "Accumulated tool operating time"],
] as const;

const toneClass: Record<RiskCategory, string> = {
  "Low Risk": "low",
  "Medium Risk": "medium",
  "High Risk": "high",
  "Critical Risk": "critical",
};

export function MachineAnalysisPage() {
  const [form, setForm] = useState<FormState>(initialForm);
  const [result, setResult] = useState<FailurePredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  function updateField(key: keyof FormState, value: string) {
    setForm((current) => ({ ...current, [key]: value }));
    setError(null);
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    const numericEntries = fields.map(([key]) => [key, Number(form[key])] as const);
    if (numericEntries.some(([, value]) => !Number.isFinite(value))) {
      setError("Enter a valid finite number for every sensor field.");
      return;
    }
    if (numericEntries.some(([key, value]) =>
      key === "torque" || key === "tool_wear" ? value < 0 : value <= 0,
    )) {
      setError("Temperatures and speed must be positive; torque and tool wear cannot be negative.");
      return;
    }

    const payload = {
      type: form.type,
      ...Object.fromEntries(numericEntries),
    } as FailurePredictionRequest;

    setLoading(true);
    setError(null);
    try {
      setResult(await factoryMindApi.predictFailure(payload));
    } catch (caught) {
      setResult(null);
      setError(caught instanceof ApiError ? caught.message : "Unexpected analysis error.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page-stack">
      <header className="page-heading compact-heading">
        <div><p className="eyebrow">Predictive maintenance</p><h1>Machine Analysis</h1><p className="page-lead">Enter a single operating observation to calculate a calibrated, model-derived risk score.</p></div>
      </header>

      <div className="analysis-layout">
        <form className="panel analysis-form" onSubmit={submit} noValidate>
          <div className="panel-heading"><div><span className="section-kicker">Raw inputs only</span><h2>Operating conditions</h2></div><span className="step-badge">6 inputs</span></div>
          <div className="field-group">
            <label htmlFor="type">Product Type</label>
            <div className="select-wrap">
              <select id="type" value={form.type} onChange={(event) => updateField("type", event.target.value)}>
                <option value="L">L — Low quality variant</option>
                <option value="M">M — Medium quality variant</option>
                <option value="H">H — High quality variant</option>
              </select>
            </div>
            <small>AI4I product quality category used during model training.</small>
          </div>
          <div className="form-grid">
            {fields.map(([key, label, unit, helper]) => (
              <div className="field-group" key={key}>
                <label htmlFor={key}>{label}<span className="unit">{unit}</span></label>
                <input id={key} name={key} type="number" step="any" min={key === "torque" || key === "tool_wear" ? 0 : 0.000001} required value={form[key]} onChange={(event) => updateField(key, event.target.value)} aria-describedby={`${key}-help`} />
                <small id={`${key}-help`}>{helper}</small>
              </div>
            ))}
          </div>
          {error && <div className="alert alert-error compact-alert" role="alert"><div><strong>Analysis unavailable</strong><span>{error}</span></div></div>}
          <button className="primary-button submit-button" type="submit" disabled={loading}>
            {loading ? <><span className="spinner" aria-hidden="true" /> Analyzing observation</> : <>Calculate risk score <span aria-hidden="true">→</span></>}
          </button>
          <p className="form-note">Engineered features are calculated securely inside the production model pipeline.</p>
        </form>

        <section className="result-column" aria-live="polite" aria-busy={loading}>
          {loading && <div className="panel result-card empty-result"><span className="loader-ring" /><h2>Running model inference</h2><p>Applying feature engineering, preprocessing, and sigmoid calibration.</p></div>}
          {!loading && !result && <div className="panel result-card empty-result"><div className="empty-visual"><span>FM</span></div><p className="eyebrow">Awaiting observation</p><h2>Your analysis will appear here.</h2><p>Submit operating measurements to receive a risk score, category, and deterministic maintenance guidance.</p></div>}
          {!loading && result && <RiskResult result={result} />}
        </section>
      </div>
    </div>
  );
}

function RiskResult({ result }: { result: FailurePredictionResponse }) {
  const tone = toneClass[result.risk_category];
  const score = Math.min(100, Math.max(0, result.failure_risk_score));
  return (
    <article className={`panel result-card risk-result ${tone}`}>
      <div className="result-topline"><span className="section-kicker">Analysis complete</span><span className={`risk-chip ${tone}`}>{result.risk_category}</span></div>
      <div className="score-block"><span>Failure Risk Score</span><strong>{score.toFixed(1)}<small>/100</small></strong></div>
      <div className="gauge" role="img" aria-label={`Failure risk score ${score.toFixed(1)} out of 100`}><div className="gauge-fill" style={{ width: `${score}%` }} /></div>
      <div className="recommendation"><span>Recommended action</span><p>{result.recommended_action}</p></div>
      <dl className="result-meta"><div><dt>Calibration</dt><dd className="capitalize">{result.calibration_method}</dd></div><div><dt>Model</dt><dd>v{result.model_version}</dd></div><div><dt>Thresholds</dt><dd>v{result.threshold_version}</dd></div></dl>
      <p className="disclaimer"><span aria-hidden="true">i</span>{result.disclaimer}</p>
    </article>
  );
}
