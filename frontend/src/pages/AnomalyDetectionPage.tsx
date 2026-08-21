"use client";

import { CSSProperties, FormEvent, useState } from "react";
import { ApiError, factoryMindApi } from "../services/api";
import type { AnomalyObservation, AnomalyPredictionRequest, AnomalyPredictionResponse } from "../types/api";

type ObservationForm = Record<keyof AnomalyObservation, string>;
type SensorKey = Exclude<keyof AnomalyObservation, "cycle">;

const sensorGroups: Array<{ title: string; fields: Array<[SensorKey, string]> }> = [
  { title: "Core condition sensors", fields: [["sensor_2", "Sensor 2"], ["sensor_3", "Sensor 3"], ["sensor_4", "Sensor 4"], ["sensor_7", "Sensor 7"], ["sensor_8", "Sensor 8"]] },
  { title: "Performance sensors", fields: [["sensor_9", "Sensor 9"], ["sensor_11", "Sensor 11"], ["sensor_12", "Sensor 12"], ["sensor_13", "Sensor 13"], ["sensor_14", "Sensor 14"]] },
  { title: "Late-life indicators", fields: [["sensor_15", "Sensor 15"], ["sensor_17", "Sensor 17"], ["sensor_20", "Sensor 20"], ["sensor_21", "Sensor 21"]] },
];

const normalSample: AnomalyObservation = { cycle: 1, sensor_2: 641.82, sensor_3: 1589.7, sensor_4: 1400.6, sensor_7: 554.36, sensor_8: 2388.06, sensor_9: 9046.19, sensor_11: 47.47, sensor_12: 521.66, sensor_13: 2388.02, sensor_14: 8138.62, sensor_15: 8.4195, sensor_17: 392, sensor_20: 39.06, sensor_21: 23.419 };
const anomalySample: AnomalyObservation = { cycle: 1, sensor_2: 643.54, sensor_3: 1601.41, sensor_4: 1427.2, sensor_7: 551.25, sensor_8: 2388.32, sensor_9: 9033.22, sensor_11: 48.25, sensor_12: 520.08, sensor_13: 2388.32, sensor_14: 8110.93, sensor_15: 8.5113, sensor_17: 396, sensor_20: 38.48, sensor_21: 22.9649 };

function toForm(row: AnomalyObservation): ObservationForm { return Object.fromEntries(Object.entries(row).map(([key, value]) => [key, String(value)])) as ObservationForm; }
function blankCycle(cycle: number): ObservationForm { return Object.fromEntries(["cycle", ...sensorGroups.flatMap((group) => group.fields.map(([key]) => key))].map((key) => [key, key === "cycle" ? String(cycle) : ""])) as ObservationForm; }
function sampleTrajectory(kind: "normal" | "persistent") { const base = kind === "normal" ? normalSample : anomalySample; const length = kind === "normal" ? 1 : 5; return Array.from({ length }, (_, index) => toForm({ ...base, cycle: index + 1 })); }

export function AnomalyDetectionPage({ modelAvailable }: { modelAvailable: boolean }) {
  const [unitId, setUnitId] = useState("");
  const [rows, setRows] = useState<ObservationForm[]>(sampleTrajectory("normal"));
  const [expanded, setExpanded] = useState(0);
  const [result, setResult] = useState<AnomalyPredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const persistenceReady = rows.length >= 5;

  function update(index: number, key: keyof AnomalyObservation, value: string) { setRows((current) => current.map((row, rowIndex) => rowIndex === index ? { ...row, [key]: value } : row)); setError(null); }
  function addCycle() { const last = Number(rows.at(-1)?.cycle); setRows((current) => [...current, blankCycle(Number.isInteger(last) ? last + 1 : current.length + 1)]); setExpanded(rows.length); setResult(null); }
  function removeLatest() { if (rows.length === 1) return; setRows((current) => current.slice(0, -1)); setExpanded((current) => Math.min(current, rows.length - 2)); setResult(null); }
  function loadSample(kind: "normal" | "persistent") { setRows(sampleTrajectory(kind)); setExpanded(0); setResult(null); setError(null); }

  async function submit(event: FormEvent) {
    event.preventDefault();
    const observations: AnomalyObservation[] = [];
    for (const [index, row] of rows.entries()) {
      const parsed = Object.fromEntries(Object.entries(row).map(([key, value]) => [key, value.trim() === "" ? Number.NaN : Number(value)])) as unknown as AnomalyObservation;
      if (!Number.isInteger(parsed.cycle) || parsed.cycle <= 0) { setError(`Cycle ${index + 1}: enter a positive whole-number cycle.`); return; }
      if (Object.values(parsed).some((value) => !Number.isFinite(value))) { setError(`Cycle ${parsed.cycle}: complete every sensor with a finite number.`); return; }
      observations.push(parsed);
    }
    const payload: AnomalyPredictionRequest = { observations, ...(unitId.trim() ? { unit_id: unitId.trim() } : {}) };
    setLoading(true); setError(null); setResult(null);
    try { setResult(await factoryMindApi.predictAnomaly(payload)); }
    catch (caught) { setError(caught instanceof ApiError ? readableError(caught) : "Anomaly analysis could not be completed."); }
    finally { setLoading(false); }
  }

  return <div className="page-stack">
    <header className="page-heading compact-heading"><div><p className="eyebrow">Condition monitoring · Module 03</p><h1>Anomaly Detection</h1><p className="page-lead">Compare the current engine sensor state with the development normal reference, then evaluate whether unusual behavior has persisted.</p></div></header>
    {!modelAvailable && <div className="alert alert-error" role="status"><div><strong>Anomaly module unavailable</strong><span>The backend did not report the anomaly model as ready. Entered data will be preserved if you try again later.</span></div></div>}
    <div className="history-banner" role="status"><span className={`history-icon ${persistenceReady ? "complete" : "limited"}`}>{rows.length}</span><div><strong>{persistenceReady ? "Persistent 3-of-5 monitoring available" : "Anomaly scoring is available"}</strong><p>{persistenceReady ? "The backend will evaluate exactly the latest five cycles." : "Five consecutive cycles are required for persistent-alert evaluation."}</p></div></div>
    <div className="rul-layout anomaly-layout">
      <form className="trajectory-column" onSubmit={submit} noValidate>
        <section className="panel trajectory-toolbar"><div className="field-group"><label htmlFor="anomaly-unit-id">Engine / Unit ID <span className="optional">Optional</span></label><input id="anomaly-unit-id" value={unitId} onChange={(event) => setUnitId(event.target.value)} placeholder="e.g. engine-42" /><small>Reference only; identity does not affect anomaly scoring.</small></div><div className="sample-actions"><button type="button" onClick={() => loadSample("normal")}>Load normal sample</button><button type="button" onClick={() => loadSample("persistent")}>Load persistent anomaly sample</button><small>Demo trajectories—not live factory telemetry.</small></div></section>
        <div className="trajectory-heading"><div><span className="section-kicker">Chronological input</span><h2>Sensor history</h2></div><div><button type="button" className="secondary-button" onClick={removeLatest} disabled={rows.length === 1}>Remove latest</button><button type="button" className="primary-button" onClick={addCycle}>+ Add cycle</button></div></div>
        <div className="cycle-list">{rows.map((row, index) => <article className={`cycle-card ${expanded === index ? "expanded" : ""}`} key={index}><button type="button" className="cycle-summary" onClick={() => setExpanded(expanded === index ? -1 : index)} aria-expanded={expanded === index}><span className="cycle-number">{row.cycle || "—"}</span><div><strong>Cycle {row.cycle || index + 1}</strong><small>{Object.values(row).filter(Boolean).length - 1} of 14 sensors entered</small></div><span>{expanded === index ? "Close" : "Edit"}</span></button>{expanded === index && <div className="cycle-editor"><div className="cycle-input field-group"><label htmlFor={`anomaly-cycle-${index}`}>Cycle</label><input id={`anomaly-cycle-${index}`} type="number" step="1" min="1" value={row.cycle} onChange={(event) => update(index, "cycle", event.target.value)} /></div>{sensorGroups.map((group) => <fieldset key={group.title}><legend>{group.title}</legend><div className="sensor-grid">{group.fields.map(([key, label]) => <div className="field-group" key={key}><label htmlFor={`anomaly-${key}-${index}`}>{label}<span className="unit-label">raw</span></label><input id={`anomaly-${key}-${index}`} type="number" step="any" required value={row[key]} onChange={(event) => update(index, key, event.target.value)} /></div>)}</div></fieldset>)}</div>}</article>)}</div>
        {error && <div className="alert alert-error" role="alert"><div><strong>Trajectory not analyzed</strong><span>{error}</span></div></div>}
        <button className="primary-button submit-button" type="submit" disabled={loading}>{loading ? <><span className="spinner" /> Analyzing sensor state</> : <>Analyze anomaly state <span aria-hidden="true">→</span></>}</button>
      </form>
      <aside className="rul-result-column" aria-live="polite" aria-busy={loading}>{loading ? <div className="panel rul-empty"><span className="loader-ring" /><h2>Analyzing trajectory</h2><p>Comparing sensor states with the development normal reference.</p></div> : result ? <AnomalyResult result={result} /> : <div className="panel rul-empty"><div className="horizon-symbol">AD</div><p className="eyebrow">Awaiting trajectory</p><h2>Your condition-monitoring result will appear here.</h2><p>Current unusualness and persistent alert state are evaluated separately.</p></div>}</aside>
    </div>
  </div>;
}

function AnomalyResult({ result }: { result: AnomalyPredictionResponse }) {
  const persistenceLabel = result.persistence_status === "insufficient_history" ? "Insufficient History" : result.alert_active ? "Active" : "Inactive";
  const style = { "--percentile": `${result.anomaly_percentile}%`, "--boundary": `${result.threshold_percentile}%` } as CSSProperties;
  return <article className={`panel anomaly-result ${result.alert_active ? "alert-active" : ""}`}>
    <div className="result-topline"><span className="section-kicker">Condition analysis complete</span><span className={`history-chip ${result.current_threshold_exceeded ? "limited" : "complete"}`}>{result.current_threshold_exceeded ? "Above threshold" : "Within reference range"}</span></div>
    <div className="anomaly-percentile"><span>Anomaly Percentile</span><strong>{result.anomaly_percentile.toFixed(1)}<small>/100</small></strong><p>More unusual than approximately {result.anomaly_percentile.toFixed(1)}% of the development normal-reference score distribution.</p></div>
    <div className="percentile-gauge" style={style} role="img" aria-label={`Anomaly percentile ${result.anomaly_percentile.toFixed(1)}; boundary ${result.threshold_percentile.toFixed(1)}`}><div className="percentile-fill" /><span className="threshold-marker"><i />Anomaly boundary {result.threshold_percentile.toFixed(1)}</span></div>
    <div className="anomaly-state-grid"><section><span>Current State</span><strong>{result.current_threshold_exceeded ? "Above anomaly threshold" : "Within reference range"}</strong><small>Raw score {result.current_anomaly_score.toFixed(6)}</small></section><section className={result.alert_active ? "persistent-active" : ""}><span>Persistent Anomaly Alert</span><strong>{persistenceLabel}</strong><small>{result.persistence_status === "available" ? `${result.recent_exceedance_count} of latest ${result.recent_window_size} cycles exceeded threshold` : `${result.history_cycle_count} of ${result.persistence_window_size} required cycles supplied`}</small></section></div>
    <p className="persistence-note">Persistent status reflects the recent trajectory, not only the current cycle. The policy requires {result.persistence_required_count} of the latest {result.persistence_window_size} cycles.</p>
    <section className="deviation-section"><div><span className="section-kicker">Sensor context</span><h2>Most Unusual Sensor Readings</h2><p>{result.sensor_context_label} These are not causal explanations or model feature importance.</p></div><div className="deviation-list">{result.top_sensor_deviations.map((item) => <div key={item.sensor}><div><strong>{item.sensor.replace("_", " ")}</strong><span>{item.direction === "above_normal" ? "Above normal" : "Below normal"}</span></div><div><small>Current</small><strong>{item.current_value.toFixed(3)}</strong></div><div><small>Reference</small><strong>{item.reference_mean.toFixed(3)}</strong></div><div><small>Deviation</small><strong>{item.standardized_deviation > 0 ? "+" : ""}{item.standardized_deviation.toFixed(2)}σ</strong></div></div>)}</div></section>
    <div className="rul-warning"><strong>Condition-monitoring caution</strong><p>{result.warning}</p></div><p className="disclaimer"><span aria-hidden="true">i</span>{result.disclaimer}</p>
  </article>;
}

function readableError(error: ApiError) {
  const message = error.message.toLowerCase();
  if (error.status === 503) return "The anomaly model is currently unavailable. Check the backend service and try again.";
  if (error.status === 422) {
    if (message.includes("duplicate")) return "Cycle numbers must be unique.";
    if (message.includes("consecutive") || message.includes("gap")) return "Cycles must be consecutive without gaps.";
    if (message.includes("increasing")) return "Cycles must be entered in ascending chronological order.";
    return `The backend rejected this trajectory: ${error.message}`;
  }
  if (error.status === 500) return "The anomaly model could not complete inference. Your trajectory has been preserved; try again.";
  return error.message;
}
