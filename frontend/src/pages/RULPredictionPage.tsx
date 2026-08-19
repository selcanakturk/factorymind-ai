"use client";

import { FormEvent, useState } from "react";
import { ApiError, factoryMindApi } from "../services/api";
import type { RULModelInfoResponse, RULObservation, RULPredictionRequest, RULPredictionResponse } from "../types/api";

type ObservationForm = Record<keyof RULObservation, string>;
type FieldKey = Exclude<keyof RULObservation, "cycle">;

const fieldGroups: Array<{ title: string; fields: Array<[FieldKey, string, string]> }> = [
  { title: "Operating settings", fields: [["operational_setting_1", "Setting 1", "normalized"], ["operational_setting_2", "Setting 2", "normalized"], ["operational_setting_3", "Setting 3", "%"]] },
  { title: "Core sensors", fields: [["sensor_2", "Sensor 2", "raw"], ["sensor_3", "Sensor 3", "raw"], ["sensor_4", "Sensor 4", "raw"], ["sensor_7", "Sensor 7", "raw"], ["sensor_8", "Sensor 8", "raw"], ["sensor_9", "Sensor 9", "raw"]] },
  { title: "Condition sensors", fields: [["sensor_11", "Sensor 11", "raw"], ["sensor_12", "Sensor 12", "raw"], ["sensor_13", "Sensor 13", "raw"], ["sensor_14", "Sensor 14", "raw"], ["sensor_15", "Sensor 15", "raw"], ["sensor_17", "Sensor 17", "raw"], ["sensor_20", "Sensor 20", "raw"], ["sensor_21", "Sensor 21", "raw"]] },
];

const demoBase: RULObservation = { cycle: 1, operational_setting_1: -0.0007, operational_setting_2: -0.0004, operational_setting_3: 100, sensor_2: 641.82, sensor_3: 1589.7, sensor_4: 1400.6, sensor_7: 554.36, sensor_8: 2388.06, sensor_9: 9046.19, sensor_11: 47.47, sensor_12: 521.66, sensor_13: 2388.02, sensor_14: 8138.62, sensor_15: 8.4195, sensor_17: 392, sensor_20: 39.06, sensor_21: 23.419 };

function toForm(row: RULObservation): ObservationForm { return Object.fromEntries(Object.entries(row).map(([key, value]) => [key, String(value)])) as ObservationForm; }
function blankCycle(cycle: number): ObservationForm { return Object.fromEntries(["cycle", ...fieldGroups.flatMap((group) => group.fields.map(([key]) => key))].map((key) => [key, key === "cycle" ? String(cycle) : ""])) as ObservationForm; }
function demoTrajectory(length: number): ObservationForm[] { return Array.from({ length }, (_, index) => toForm({ ...demoBase, cycle: index + 1, sensor_4: demoBase.sensor_4 + index * .4, sensor_11: demoBase.sensor_11 + index * .02, sensor_12: demoBase.sensor_12 - index * .03 })); }

export function RULPredictionPage({ modelInfo }: { modelInfo: RULModelInfoResponse | null }) {
  const [unitId, setUnitId] = useState("");
  const [rows, setRows] = useState<ObservationForm[]>(demoTrajectory(1));
  const [expanded, setExpanded] = useState(0);
  const [result, setResult] = useState<RULPredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const fullContext = rows.length >= (modelInfo?.minimum_full_context_cycles ?? 6);

  function update(index: number, key: keyof RULObservation, value: string) { setRows((current) => current.map((row, rowIndex) => rowIndex === index ? { ...row, [key]: value } : row)); setError(null); }
  function addCycle() { const last = Number(rows.at(-1)?.cycle); setRows((current) => [...current, blankCycle(Number.isInteger(last) ? last + 1 : current.length + 1)]); setExpanded(rows.length); setResult(null); }
  function removeLatest() { if (rows.length === 1) return; setRows((current) => current.slice(0, -1)); setExpanded((current) => Math.min(current, rows.length - 2)); setResult(null); }
  function loadSample(length: number) { setRows(demoTrajectory(length)); setExpanded(0); setResult(null); setError(null); }

  async function submit(event: FormEvent) {
    event.preventDefault();
    const observations: RULObservation[] = [];
    for (const [index, row] of rows.entries()) {
      const parsed = Object.fromEntries(Object.entries(row).map(([key, value]) => [key, value.trim() === "" ? Number.NaN : Number(value)])) as unknown as RULObservation;
      if (!Number.isInteger(parsed.cycle) || parsed.cycle <= 0) { setError(`Cycle ${index + 1}: enter a positive whole-number cycle.`); return; }
      if (Object.values(parsed).some((value) => !Number.isFinite(value))) { setError(`Cycle ${parsed.cycle}: complete every setting and sensor with a finite number.`); return; }
      observations.push(parsed);
    }
    const payload: RULPredictionRequest = { observations, ...(unitId.trim() ? { unit_id: unitId.trim() } : {}) };
    setLoading(true); setError(null); setResult(null);
    try { setResult(await factoryMindApi.predictRul(payload)); }
    catch (caught) { setError(caught instanceof ApiError ? readableRulError(caught) : "RUL analysis could not be completed."); }
    finally { setLoading(false); }
  }

  return <div className="page-stack">
    <header className="page-heading compact-heading"><div><p className="eyebrow">Predictive maintenance · Module 02</p><h1>Remaining Useful Life</h1><p className="page-lead">Build a chronological engine trajectory, then estimate the latest cycle&apos;s capped remaining-life horizon.</p></div></header>
    <div className="history-banner" role="status"><span className={`history-icon ${fullContext ? "complete" : "limited"}`}>{rows.length}</span><div><strong>{fullContext ? "Full temporal context available" : "Limited history"}</strong><p>{fullContext ? "6+ consecutive cycles supplied." : "The model can estimate RUL, but some temporal features are unavailable until six cycles are supplied."}</p></div></div>
    <div className="rul-layout">
      <form className="trajectory-column" onSubmit={submit} noValidate>
        <section className="panel trajectory-toolbar"><div className="field-group"><label htmlFor="unit-id">Engine / Unit ID <span className="optional">Optional</span></label><input id="unit-id" value={unitId} onChange={(event) => setUnitId(event.target.value)} placeholder="e.g. engine-42" /><small>Reference only; this identity does not affect the prediction.</small></div><div className="sample-actions" aria-label="Demo trajectories"><button type="button" onClick={() => loadSample(1)}>Load short sample</button><button type="button" onClick={() => loadSample(6)}>Load full-context sample</button><small>Documented demo observations—not live factory telemetry.</small></div></section>
        <div className="trajectory-heading"><div><span className="section-kicker">Chronological input</span><h2>Cycle history</h2></div><div><button type="button" className="secondary-button" onClick={removeLatest} disabled={rows.length === 1}>Remove latest</button><button type="button" className="primary-button" onClick={addCycle}>+ Add cycle</button></div></div>
        <div className="cycle-list">{rows.map((row, index) => <article className={`cycle-card ${expanded === index ? "expanded" : ""}`} key={index}><button type="button" className="cycle-summary" onClick={() => setExpanded(expanded === index ? -1 : index)} aria-expanded={expanded === index}><span className="cycle-number">{row.cycle || "—"}</span><div><strong>Cycle {row.cycle || index + 1}</strong><small>{Object.values(row).filter(Boolean).length - 1} of 17 measurements entered</small></div><span>{expanded === index ? "Close" : "Edit"}</span></button>{expanded === index && <div className="cycle-editor"><div className="cycle-input field-group"><label htmlFor={`cycle-${index}`}>Cycle</label><input id={`cycle-${index}`} type="number" step="1" min="1" value={row.cycle} onChange={(event) => update(index, "cycle", event.target.value)} /></div>{fieldGroups.map((group) => <fieldset key={group.title}><legend>{group.title}</legend><div className="sensor-grid">{group.fields.map(([key, label, unit]) => <div className="field-group" key={key}><label htmlFor={`${key}-${index}`}>{label}<span className="unit-label">{unit}</span></label><input id={`${key}-${index}`} type="number" step="any" value={row[key]} onChange={(event) => update(index, key, event.target.value)} required /></div>)}</div></fieldset>)}</div>}</article>)}</div>
        {error && <div className="alert alert-error" role="alert"><div><strong>Trajectory not analyzed</strong><span>{error}</span></div></div>}
        <button className="primary-button submit-button" type="submit" disabled={loading}>{loading ? <><span className="spinner" /> Analyzing trajectory</> : <>Estimate remaining useful life <span aria-hidden="true">→</span></>}</button>
      </form>
      <aside className="rul-result-column" aria-live="polite" aria-busy={loading}>{loading ? <div className="panel rul-empty"><span className="loader-ring" /><h2>Analyzing trajectory</h2><p>Engineering temporal context and scoring the latest cycle.</p></div> : result ? <RULResult result={result} /> : <div className="panel rul-empty"><div className="horizon-symbol">RUL</div><p className="eyebrow">Awaiting trajectory</p><h2>Your estimate will appear here.</h2><p>Short histories are valid. Six consecutive cycles provide full temporal context.</p></div>}</aside>
    </div>
  </div>;
}

function RULResult({ result }: { result: RULPredictionResponse }) {
  const width = Math.min(100, Math.max(0, result.predicted_rul_cycles / result.prediction_horizon_cap * 100));
  const full = result.history_quality === "full_context";
  return <article className="panel rul-result"><div className="result-topline"><span className="section-kicker">Trajectory analyzed</span><span className={`history-chip ${full ? "complete" : "limited"}`}>{full ? "Full Context" : "Limited History"}</span></div><div className="rul-value"><span>Estimated Remaining Useful Life</span><strong>{result.rul_display}</strong><small>Development-stage point estimate</small></div><div className="horizon-track" role="img" aria-label={`Estimate within ${result.prediction_horizon_cap}-cycle capped horizon`}><div style={{ width: `${width}%` }} /></div><div className="horizon-labels"><span>Near failure</span><span>{result.prediction_horizon_cap}-cycle cap</span></div><div className="cap-note">The RUL model uses a {result.prediction_horizon_cap}-cycle capped training target. Long-horizon estimates represent the healthy/early-life region rather than an exact {result.prediction_horizon_cap}-cycle forecast.</div><dl className="result-meta"><div><dt>History quality</dt><dd>{full ? "Full Context" : "Limited History"}</dd></div><div><dt>Cycles supplied</dt><dd>{result.history_cycle_count}</dd></div><div><dt>Model</dt><dd>v{result.model_version}</dd></div><div><dt>Dataset</dt><dd>{result.dataset}</dd></div><div><dt>Horizon cap</dt><dd>{result.prediction_horizon_cap} cycles</dd></div><div><dt>Status</dt><dd>{result.development_stage ? "Development stage" : "Production"}</dd></div></dl><div className="rul-warning" role="note"><strong>Engineering caution</strong><p>{result.warning}</p></div><p className="disclaimer"><span aria-hidden="true">i</span>{result.disclaimer}</p></article>;
}

function readableRulError(error: ApiError) {
  const message = error.message.toLowerCase();
  if (error.status === 503) return "The RUL model is currently unavailable. Check the backend service and try again.";
  if (error.status === 422) {
    if (message.includes("duplicate")) return "Cycle numbers must be unique. Remove the duplicate cycle and try again.";
    if (message.includes("gap") || message.includes("consecutive")) return "Cycles must be consecutive without gaps.";
    if (message.includes("sort") || message.includes("increasing")) return "Cycles must be entered in ascending chronological order.";
    return `The backend rejected this trajectory: ${error.message}`;
  }
  if (error.status === 500) return "The RUL model could not complete inference. Your trajectory has been preserved; try again.";
  return error.message;
}
