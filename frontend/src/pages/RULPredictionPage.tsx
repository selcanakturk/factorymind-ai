"use client";

import { FormEvent, useRef, useState } from "react";
import { ApiError, factoryMindApi } from "../services/api";
import type { RULModelInfoResponse, RULObservation, RULPredictionRequest, RULPredictionResponse } from "../types/api";
import { parseRulCsv, RUL_CSV_HEADERS, RUL_MEASUREMENT_COUNT, type ParsedRulCsv } from "../utils/rulCsv";

type ObservationForm = Record<keyof RULObservation, string>;
type FieldKey = Exclude<keyof RULObservation, "cycle">;
type InputMode = "csv" | "manual";
type InputSource = "Uploaded CSV" | "Manual Entry" | "Demo Data";

const fieldGroups: Array<{ title: string; fields: Array<[FieldKey, string, string]> }> = [
  { title: "Operating settings", fields: [["operational_setting_1", "Setting 1", "normalized"], ["operational_setting_2", "Setting 2", "normalized"], ["operational_setting_3", "Setting 3", "%"]] },
  { title: "Core sensors", fields: [["sensor_2", "Sensor 2", "raw"], ["sensor_3", "Sensor 3", "raw"], ["sensor_4", "Sensor 4", "raw"], ["sensor_7", "Sensor 7", "raw"], ["sensor_8", "Sensor 8", "raw"], ["sensor_9", "Sensor 9", "raw"]] },
  { title: "Condition sensors", fields: [["sensor_11", "Sensor 11", "raw"], ["sensor_12", "Sensor 12", "raw"], ["sensor_13", "Sensor 13", "raw"], ["sensor_14", "Sensor 14", "raw"], ["sensor_15", "Sensor 15", "raw"], ["sensor_17", "Sensor 17", "raw"], ["sensor_20", "Sensor 20", "raw"], ["sensor_21", "Sensor 21", "raw"]] },
];

const demoBase: RULObservation = { cycle: 1, operational_setting_1: -0.0007, operational_setting_2: -0.0004, operational_setting_3: 100, sensor_2: 641.82, sensor_3: 1589.7, sensor_4: 1400.6, sensor_7: 554.36, sensor_8: 2388.06, sensor_9: 9046.19, sensor_11: 47.47, sensor_12: 521.66, sensor_13: 2388.02, sensor_14: 8138.62, sensor_15: 8.4195, sensor_17: 392, sensor_20: 39.06, sensor_21: 23.419 };

function toForm(row: RULObservation): ObservationForm { return Object.fromEntries(Object.entries(row).map(([key, value]) => [key, String(value)])) as ObservationForm; }
function blankCycle(cycle: number): ObservationForm { return Object.fromEntries(RUL_CSV_HEADERS.map((key) => [key, key === "cycle" ? String(cycle) : ""])) as ObservationForm; }
function demoTrajectory(length: number): ObservationForm[] { return Array.from({ length }, (_, index) => toForm({ ...demoBase, cycle: index + 1, sensor_4: demoBase.sensor_4 + index * .4, sensor_11: demoBase.sensor_11 + index * .02, sensor_12: demoBase.sensor_12 - index * .03 })); }

export function RULPredictionPage({ modelInfo }: { modelInfo: RULModelInfoResponse | null }) {
  const minimumContext = modelInfo?.minimum_full_context_cycles ?? 6;
  const fileInput = useRef<HTMLInputElement>(null);
  const [mode, setMode] = useState<InputMode>("csv");
  const [unitId, setUnitId] = useState("");
  const [rows, setRows] = useState<ObservationForm[]>([blankCycle(1)]);
  const [manualSource, setManualSource] = useState<"manual" | "demo">("manual");
  const [csv, setCsv] = useState<ParsedRulCsv | null>(null);
  const [expanded, setExpanded] = useState(0);
  const [result, setResult] = useState<RULPredictionResponse | null>(null);
  const [resultSource, setResultSource] = useState<InputSource | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [csvError, setCsvError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const activeCount = mode === "csv" ? (csv?.observations.length ?? 0) : rows.length;
  const fullContext = activeCount >= minimumContext;

  function selectMode(next: InputMode) { setMode(next); setResult(null); setResultSource(null); setError(null); }
  function update(index: number, key: keyof RULObservation, value: string) { setRows((current) => current.map((row, rowIndex) => rowIndex === index ? { ...row, [key]: value } : row)); setManualSource("manual"); setError(null); }
  function addCycle() { const last = Number(rows.at(-1)?.cycle); setRows((current) => [...current, blankCycle(Number.isInteger(last) ? last + 1 : current.length + 1)]); setExpanded(rows.length); setManualSource("manual"); setResult(null); }
  function removeLatest() { if (rows.length === 1) return; setRows((current) => current.slice(0, -1)); setExpanded((current) => Math.min(current, rows.length - 2)); setManualSource("manual"); setResult(null); }
  function loadSample(length: number) { setRows(demoTrajectory(length)); setMode("manual"); setManualSource("demo"); setExpanded(0); setResult(null); setResultSource(null); setError(null); }

  async function chooseCsv(file: File | undefined) {
    setCsv(null); setCsvError(null); setError(null); setResult(null); setResultSource(null);
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".csv")) { setCsvError("Choose a file with a .csv extension."); if (fileInput.current) fileInput.current.value = ""; return; }
    try { setCsv(parseRulCsv(await file.text(), file.name)); }
    catch (caught) { setCsvError(caught instanceof Error ? caught.message : "The CSV could not be parsed. Check its format and try again."); }
    finally { if (fileInput.current) fileInput.current.value = ""; }
  }

  function manualObservations(): RULObservation[] | null {
    const observations: RULObservation[] = [];
    for (const [index, row] of rows.entries()) {
      const parsed = Object.fromEntries(Object.entries(row).map(([key, value]) => [key, value.trim() === "" ? Number.NaN : Number(value)])) as unknown as RULObservation;
      if (!Number.isInteger(parsed.cycle) || parsed.cycle <= 0) { setError(`Cycle ${index + 1}: enter a positive whole-number cycle.`); return null; }
      if (Object.values(parsed).some((value) => !Number.isFinite(value))) { setError(`Cycle ${parsed.cycle}: complete every setting and sensor with a finite number.`); return null; }
      if (index && parsed.cycle === observations[index - 1].cycle) { setError("Cycle numbers must be unique."); return null; }
      if (index && parsed.cycle < observations[index - 1].cycle) { setError("Cycles must be entered in ascending chronological order."); return null; }
      if (index && parsed.cycle !== observations[index - 1].cycle + 1) { setError("Cycles must be consecutive without gaps."); return null; }
      observations.push(parsed);
    }
    return observations;
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    const observations = mode === "csv" ? csv?.observations : manualObservations();
    if (!observations) { if (mode === "csv") setError("Select and validate a CSV trajectory before analysis."); return; }
    const source: InputSource = mode === "csv" ? "Uploaded CSV" : manualSource === "demo" ? "Demo Data" : "Manual Entry";
    const payload: RULPredictionRequest = { observations, ...(unitId.trim() ? { unit_id: unitId.trim() } : {}) };
    setLoading(true); setError(null); setResult(null); setResultSource(null);
    try { setResult(await factoryMindApi.predictRul(payload)); setResultSource(source); }
    catch (caught) { setError(caught instanceof ApiError ? readableRulError(caught) : "RUL analysis could not be completed."); }
    finally { setLoading(false); }
  }

  return <div className="page-stack">
    <header className="page-heading compact-heading"><div><p className="eyebrow">Predictive maintenance · Module 02</p><h1>Remaining Useful Life</h1><p className="page-lead">Provide chronological engine history by CSV or manual entry, then estimate the latest cycle&apos;s capped remaining-life horizon.</p></div></header>
    <section className="rul-input-intro" aria-labelledby="rul-input-title"><div><span className="section-kicker">Choose an input source</span><h2 id="rul-input-title">How would you like to provide engine history?</h2><p>Upload an existing trajectory, enter cycles for testing, or load a documented demo engine.</p></div><div className="input-mode-tabs" role="tablist" aria-label="RUL input mode"><button type="button" role="tab" aria-selected={mode === "csv"} className={mode === "csv" ? "active" : ""} onClick={() => selectMode("csv")}>Upload CSV</button><button type="button" role="tab" aria-selected={mode === "manual"} className={mode === "manual" ? "active" : ""} onClick={() => selectMode("manual")}>Manual Entry</button></div></section>
    <section className="sensor-explainer" aria-labelledby="sensor-channel-title"><span aria-hidden="true">i</span><div><strong id="sensor-channel-title">About sensor channels</strong><p>This development model uses anonymized sensor channels from the NASA C-MAPSS dataset. In a production factory deployment, these channels would be mapped to the facility&apos;s telemetry or sensor system and ingested automatically. Manual entry is provided for testing and demonstration.</p></div></section>
    <section className="panel demo-engine-panel"><div><span className="section-kicker">Explore without a file</span><h2>Try Demo Engine</h2><p>Load a sample C-MAPSS trajectory to explore the RUL model without preparing a file.</p></div><div className="sample-actions" aria-label="Demo engine trajectories"><button type="button" onClick={() => loadSample(1)}>Short-history demo</button><button type="button" onClick={() => loadSample(6)}>Full-context demo</button><small>Demo data—not live factory telemetry. No known RUL labels are sent.</small></div></section>
    {activeCount > 0 && <div className="history-banner" role="status"><span className={`history-icon ${fullContext ? "complete" : "limited"}`}>{activeCount}</span><div><strong>{fullContext ? "Full temporal context available" : "Limited history"}</strong><p>{fullContext ? `${minimumContext}+ consecutive cycles supplied.` : `The model can estimate RUL, but some temporal features are unavailable until ${minimumContext} cycles are supplied.`}</p></div></div>}
    <div className="rul-layout">
      <form className="trajectory-column" onSubmit={submit} noValidate>
        <section className="panel trajectory-toolbar rul-unit-field"><div className="field-group"><label htmlFor="unit-id">Engine / Unit ID <span className="optional">Optional</span></label><input id="unit-id" value={unitId} onChange={(event) => setUnitId(event.target.value)} placeholder="e.g. engine-42" /><small>Reference only; this identity does not affect the prediction.</small></div>{mode === "manual" && <p className="manual-purpose">Manual entry is intended for testing individual trajectories. In production, sensor history would typically be ingested automatically.</p>}</section>
        {mode === "csv" ? <CsvInput csv={csv} error={csvError} minimumContext={minimumContext} inputRef={fileInput} onFile={chooseCsv} /> : <ManualInput rows={rows} expanded={expanded} demo={manualSource === "demo"} onExpanded={setExpanded} onUpdate={update} onAdd={addCycle} onRemove={removeLatest} />}
        {error && <div className="alert alert-error" role="alert"><div><strong>Trajectory not analyzed</strong><span>{error}</span></div></div>}
        <button className="primary-button submit-button" type="submit" disabled={loading}>{loading ? <><span className="spinner" /> Analyzing trajectory</> : <>Estimate remaining useful life <span aria-hidden="true">→</span></>}</button>
      </form>
      <aside className="rul-result-column" aria-live="polite" aria-busy={loading}>{loading ? <div className="panel rul-empty"><span className="loader-ring" /><h2>Analyzing trajectory</h2><p>Engineering temporal context and scoring the latest cycle.</p></div> : result ? <RULResult result={result} source={resultSource} /> : <div className="panel rul-empty"><div className="horizon-symbol">RUL</div><p className="eyebrow">Awaiting trajectory</p><h2>Your estimate will appear here.</h2><p>CSV, manual, and demo inputs all use the same versioned RUL model.</p></div>}</aside>
    </div>
  </div>;
}

function CsvInput({ csv, error, minimumContext, inputRef, onFile }: { csv: ParsedRulCsv | null; error: string | null; minimumContext: number; inputRef: React.RefObject<HTMLInputElement | null>; onFile: (file: File | undefined) => void }) {
  const full = (csv?.observations.length ?? 0) >= minimumContext;
  return <><section className="panel csv-upload-panel"><label className="csv-dropzone" htmlFor="rul-csv"><span className="upload-mark" aria-hidden="true">CSV</span><strong>{csv ? "Choose a different CSV" : "Choose trajectory CSV"}</strong><small>The file is parsed in your browser and is not uploaded or retained.</small></label><input ref={inputRef} id="rul-csv" className="visually-hidden" type="file" accept=".csv,text/csv" onChange={(event) => void onFile(event.target.files?.[0])} />{error && <div className="csv-error" role="alert"><strong>CSV not ready</strong><span>{error}</span></div>}<details className="csv-format"><summary>View CSV format</summary><p>Exactly these model-input columns are accepted. Extra columns—including RUL, target, or failure labels—are rejected.</p><code>{RUL_CSV_HEADERS.join(",")}</code></details></section>{csv && <section className="panel csv-ready" aria-live="polite"><div className="panel-heading"><div><span className="section-kicker">Trajectory ready</span><h2>{csv.filename}</h2></div><span className={`history-chip ${full ? "complete" : "limited"}`}>{full ? "Full Context" : "Limited History"}</span></div><ul className="validation-summary"><li><span aria-hidden="true">✓</span>{csv.observations.length} cycle{csv.observations.length === 1 ? "" : "s"} detected</li><li><span aria-hidden="true">✓</span>{RUL_MEASUREMENT_COUNT} required measurements found per cycle</li><li><span aria-hidden="true">✓</span>Chronological sequence valid</li><li className={full ? "" : "warning-item"}><span aria-hidden="true">{full ? "✓" : "!"}</span>{full ? "Full temporal context available" : `Limited history — full temporal context begins at ${minimumContext} cycles`}</li></ul><div className="csv-preview"><h3>Preview · first {Math.min(3, csv.observations.length)} cycles</h3><div className="preview-table" role="table" aria-label="Parsed CSV preview"><div className="preview-row preview-header" role="row"><span>Cycle</span><span>Setting 1</span><span>Sensor 2</span><span>Sensor 11</span></div>{csv.observations.slice(0, 3).map((row) => <div className="preview-row" role="row" key={row.cycle}><span>{row.cycle}</span><span>{row.operational_setting_1}</span><span>{row.sensor_2}</span><span>{row.sensor_11}</span></div>)}</div></div></section>}</>;
}

function ManualInput({ rows, expanded, demo, onExpanded, onUpdate, onAdd, onRemove }: { rows: ObservationForm[]; expanded: number; demo: boolean; onExpanded: (index: number) => void; onUpdate: (index: number, key: keyof RULObservation, value: string) => void; onAdd: () => void; onRemove: () => void }) {
  return <><div className="trajectory-heading"><div><span className="section-kicker">Chronological input</span><h2>Cycle history</h2>{demo && <span className="demo-data-badge">Demo Data</span>}</div><div><button type="button" className="secondary-button" onClick={onRemove} disabled={rows.length === 1}>Remove latest</button><button type="button" className="primary-button" onClick={onAdd}>+ Add cycle</button></div></div><div className="cycle-list">{rows.map((row, index) => <article className={`cycle-card ${expanded === index ? "expanded" : ""}`} key={index}><button type="button" className="cycle-summary" onClick={() => onExpanded(expanded === index ? -1 : index)} aria-expanded={expanded === index}><span className="cycle-number">{row.cycle || "—"}</span><div><strong>Cycle {row.cycle || index + 1}</strong><small>{Object.values(row).filter(Boolean).length - 1} of {RUL_MEASUREMENT_COUNT} measurements entered</small></div><span>{expanded === index ? "Close" : "Edit"}</span></button>{expanded === index && <div className="cycle-editor"><div className="cycle-input field-group"><label htmlFor={`cycle-${index}`}>Cycle</label><input id={`cycle-${index}`} type="number" step="1" min="1" value={row.cycle} onChange={(event) => onUpdate(index, "cycle", event.target.value)} /></div>{fieldGroups.map((group) => <fieldset key={group.title}><legend>{group.title}</legend><div className="sensor-grid">{group.fields.map(([key, label, unit]) => <div className="field-group" key={key}><label htmlFor={`${key}-${index}`} title="An anonymized C-MAPSS sensor channel or operating setting.">{label}<span className="unit-label">{unit}</span></label><input id={`${key}-${index}`} type="number" step="any" value={row[key]} onChange={(event) => onUpdate(index, key, event.target.value)} required /></div>)}</div></fieldset>)}</div>}</article>)}</div></>;
}

function RULResult({ result, source }: { result: RULPredictionResponse; source: InputSource | null }) {
  const width = Math.min(100, Math.max(0, result.predicted_rul_cycles / result.prediction_horizon_cap * 100));
  const full = result.history_quality === "full_context";
  return <article className="panel rul-result"><div className="result-topline"><span className="section-kicker">Trajectory analyzed</span><span className={`history-chip ${full ? "complete" : "limited"}`}>{full ? "Full Context" : "Limited History"}</span></div>{source && <span className="input-source-badge">Input Source: {source}</span>}<div className="rul-value"><span>Estimated Remaining Useful Life</span><strong>{result.rul_display}</strong><small>Development-stage point estimate</small></div><div className="horizon-track" role="img" aria-label={`Estimate within ${result.prediction_horizon_cap}-cycle capped horizon`}><div style={{ width: `${width}%` }} /></div><div className="horizon-labels"><span>Near failure</span><span>{result.prediction_horizon_cap}-cycle cap</span></div><div className="cap-note">The RUL model uses a {result.prediction_horizon_cap}-cycle capped training target. Long-horizon estimates represent the healthy/early-life region rather than an exact {result.prediction_horizon_cap}-cycle forecast.</div><dl className="result-meta"><div><dt>History quality</dt><dd>{full ? "Full Context" : "Limited History"}</dd></div><div><dt>Cycles supplied</dt><dd>{result.history_cycle_count}</dd></div><div><dt>Model</dt><dd>v{result.model_version}</dd></div><div><dt>Dataset</dt><dd>{result.dataset}</dd></div><div><dt>Horizon cap</dt><dd>{result.prediction_horizon_cap} cycles</dd></div><div><dt>Status</dt><dd>{result.development_stage ? "Development stage" : "Production"}</dd></div></dl><div className="rul-warning" role="note"><strong>Engineering caution</strong><p>{result.warning}</p></div><p className="disclaimer"><span aria-hidden="true">i</span>{result.disclaimer}</p></article>;
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
