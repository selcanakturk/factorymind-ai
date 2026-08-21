"use client";

import { CSSProperties, FormEvent, useRef, useState } from "react";
import { ApiError, factoryMindApi } from "../services/api";
import type { AnomalyObservation, AnomalyPredictionRequest, AnomalyPredictionResponse } from "../types/api";
import { ANOMALY_CSV_HEADERS, ANOMALY_SENSOR_COUNT, parseAnomalyCsv, type ParsedAnomalyCsv } from "../utils/anomalyCsv";

type ObservationForm = Record<keyof AnomalyObservation, string>;
type SensorKey = Exclude<keyof AnomalyObservation, "cycle">;
type InputMode = "csv" | "manual";
type InputSource = "Uploaded CSV" | "Manual Entry" | "Demo Data";

const sensorGroups: Array<{ title: string; fields: Array<[SensorKey, string]> }> = [
  { title: "Core condition sensors", fields: [["sensor_2", "Sensor 2"], ["sensor_3", "Sensor 3"], ["sensor_4", "Sensor 4"], ["sensor_7", "Sensor 7"], ["sensor_8", "Sensor 8"]] },
  { title: "Performance sensors", fields: [["sensor_9", "Sensor 9"], ["sensor_11", "Sensor 11"], ["sensor_12", "Sensor 12"], ["sensor_13", "Sensor 13"], ["sensor_14", "Sensor 14"]] },
  { title: "Late-life indicators", fields: [["sensor_15", "Sensor 15"], ["sensor_17", "Sensor 17"], ["sensor_20", "Sensor 20"], ["sensor_21", "Sensor 21"]] },
];

const normalSample: AnomalyObservation = { cycle: 1, sensor_2: 641.82, sensor_3: 1589.7, sensor_4: 1400.6, sensor_7: 554.36, sensor_8: 2388.06, sensor_9: 9046.19, sensor_11: 47.47, sensor_12: 521.66, sensor_13: 2388.02, sensor_14: 8138.62, sensor_15: 8.4195, sensor_17: 392, sensor_20: 39.06, sensor_21: 23.419 };
const anomalySample: AnomalyObservation = { cycle: 1, sensor_2: 643.54, sensor_3: 1601.41, sensor_4: 1427.2, sensor_7: 551.25, sensor_8: 2388.32, sensor_9: 9033.22, sensor_11: 48.25, sensor_12: 520.08, sensor_13: 2388.32, sensor_14: 8110.93, sensor_15: 8.5113, sensor_17: 396, sensor_20: 38.48, sensor_21: 22.9649 };

function toForm(row: AnomalyObservation): ObservationForm { return Object.fromEntries(Object.entries(row).map(([key, value]) => [key, String(value)])) as ObservationForm; }
function blankCycle(cycle: number): ObservationForm { return Object.fromEntries(ANOMALY_CSV_HEADERS.map((key) => [key, key === "cycle" ? String(cycle) : ""])) as ObservationForm; }
function sampleTrajectory(kind: "normal" | "persistent") { const base = kind === "normal" ? normalSample : anomalySample; const length = kind === "normal" ? 1 : 5; return Array.from({ length }, (_, index) => toForm({ ...base, cycle: index + 1 })); }

export function AnomalyDetectionPage({ modelAvailable }: { modelAvailable: boolean }) {
  const fileInput = useRef<HTMLInputElement>(null);
  const [mode, setMode] = useState<InputMode>("csv");
  const [unitId, setUnitId] = useState("");
  const [rows, setRows] = useState<ObservationForm[]>([blankCycle(1)]);
  const [manualSource, setManualSource] = useState<"manual" | "demo">("manual");
  const [csv, setCsv] = useState<ParsedAnomalyCsv | null>(null);
  const [expanded, setExpanded] = useState(0);
  const [result, setResult] = useState<AnomalyPredictionResponse | null>(null);
  const [resultSource, setResultSource] = useState<InputSource | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [csvError, setCsvError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const activeCount = mode === "csv" ? (csv?.observations.length ?? 0) : rows.length;
  const persistenceReady = activeCount >= 5;

  function selectMode(next: InputMode) { setMode(next); setResult(null); setResultSource(null); setError(null); }
  function update(index: number, key: keyof AnomalyObservation, value: string) { setRows((current) => current.map((row, rowIndex) => rowIndex === index ? { ...row, [key]: value } : row)); setManualSource("manual"); setError(null); }
  function addCycle() { const last = Number(rows.at(-1)?.cycle); setRows((current) => [...current, blankCycle(Number.isInteger(last) ? last + 1 : current.length + 1)]); setExpanded(rows.length); setManualSource("manual"); setResult(null); }
  function removeLatest() { if (rows.length === 1) return; setRows((current) => current.slice(0, -1)); setExpanded((current) => Math.min(current, rows.length - 2)); setManualSource("manual"); setResult(null); }
  function loadSample(kind: "normal" | "persistent") { setRows(sampleTrajectory(kind)); setMode("manual"); setManualSource("demo"); setExpanded(0); setResult(null); setResultSource(null); setError(null); }

  async function chooseCsv(file: File | undefined) {
    setCsv(null); setCsvError(null); setError(null); setResult(null); setResultSource(null);
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".csv")) { setCsvError("Choose a file with a .csv extension."); if (fileInput.current) fileInput.current.value = ""; return; }
    try { setCsv(parseAnomalyCsv(await file.text(), file.name)); }
    catch (caught) { setCsvError(caught instanceof Error ? caught.message : "The CSV could not be parsed. Check its format and try again."); }
    finally { if (fileInput.current) fileInput.current.value = ""; }
  }

  function manualObservations(): AnomalyObservation[] | null {
    const observations: AnomalyObservation[] = [];
    for (const [index, row] of rows.entries()) {
      const parsed = Object.fromEntries(Object.entries(row).map(([key, value]) => [key, value.trim() === "" ? Number.NaN : Number(value)])) as unknown as AnomalyObservation;
      if (!Number.isInteger(parsed.cycle) || parsed.cycle <= 0) { setError(`Cycle ${index + 1}: enter a positive whole-number cycle.`); return null; }
      if (Object.values(parsed).some((value) => !Number.isFinite(value))) { setError(`Cycle ${parsed.cycle}: complete every sensor with a finite number.`); return null; }
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
    const payload: AnomalyPredictionRequest = { observations, ...(unitId.trim() ? { unit_id: unitId.trim() } : {}) };
    setLoading(true); setError(null); setResult(null); setResultSource(null);
    try { setResult(await factoryMindApi.predictAnomaly(payload)); setResultSource(source); }
    catch (caught) { setError(caught instanceof ApiError ? readableError(caught) : "Anomaly analysis could not be completed."); }
    finally { setLoading(false); }
  }

  return <div className="page-stack">
    <header className="page-heading compact-heading"><div><p className="eyebrow">Condition monitoring · Module 03</p><h1>Anomaly Detection</h1><p className="page-lead">Compare the current engine sensor state with the development normal reference, then evaluate whether unusual behavior has persisted.</p></div></header>
    {!modelAvailable && <div className="alert alert-error" role="status"><div><strong>Anomaly module unavailable</strong><span>The backend did not report the anomaly model as ready. Entered data will be preserved if you try again later.</span></div></div>}
    <section className="rul-input-intro" aria-labelledby="anomaly-input-title"><div><span className="section-kicker">Choose an input source</span><h2 id="anomaly-input-title">How would you like to provide sensor history?</h2><p>Upload an existing trajectory, enter cycles for testing, or load a documented demo engine.</p></div><div className="input-mode-tabs" role="tablist" aria-label="Anomaly input mode"><button type="button" role="tab" aria-selected={mode === "csv"} className={mode === "csv" ? "active" : ""} onClick={() => selectMode("csv")}>Upload CSV</button><button type="button" role="tab" aria-selected={mode === "manual"} className={mode === "manual" ? "active" : ""} onClick={() => selectMode("manual")}>Manual Entry</button></div></section>
    <section className="sensor-explainer" aria-labelledby="anomaly-sensor-title"><span aria-hidden="true">i</span><div><strong id="anomaly-sensor-title">About sensor channels</strong><p>This development model uses anonymized sensor channels from the NASA C-MAPSS dataset. In a production factory deployment, these channels would be mapped to the facility&apos;s telemetry or sensor system and ingested automatically. Manual entry is provided for testing and demonstration.</p></div></section>
    <section className="panel demo-engine-panel"><div><span className="section-kicker">Explore without a file</span><h2>Try Demo Engine</h2><p>Load documented C-MAPSS trajectories to explore the anomaly model without preparing a file.</p></div><div className="sample-actions" aria-label="Anomaly demo trajectories"><button type="button" onClick={() => loadSample("normal")}>Normal-state demo</button><button type="button" onClick={() => loadSample("persistent")}>Persistent-anomaly demo</button><small>Demo data—not live factory telemetry. No RUL or failure labels are sent.</small></div></section>
    {activeCount > 0 && <div className="history-banner" role="status"><span className={`history-icon ${persistenceReady ? "complete" : "limited"}`}>{activeCount}</span><div><strong>{persistenceReady ? "Persistent 3-of-5 monitoring available" : "Anomaly scoring is available"}</strong><p>{persistenceReady ? "Persistent-alert evaluation is available. The backend evaluates exactly the latest five cycles." : "Five consecutive cycles are required for persistent-alert evaluation."}</p></div></div>}
    <div className="rul-layout anomaly-layout">
      <form className="trajectory-column" onSubmit={submit} noValidate>
        <section className="panel trajectory-toolbar rul-unit-field"><div className="field-group"><label htmlFor="anomaly-unit-id">Engine / Unit ID <span className="optional">Optional</span></label><input id="anomaly-unit-id" value={unitId} onChange={(event) => setUnitId(event.target.value)} placeholder="e.g. engine-42" /><small>Reference only; identity does not affect anomaly scoring.</small></div>{mode === "manual" && <p className="manual-purpose">Manual entry is intended for testing individual trajectories. Production sensor history would typically be ingested automatically.</p>}</section>
        {mode === "csv" ? <AnomalyCsvInput csv={csv} error={csvError} inputRef={fileInput} onFile={chooseCsv} /> : <AnomalyManualInput rows={rows} expanded={expanded} demo={manualSource === "demo"} onExpanded={setExpanded} onUpdate={update} onAdd={addCycle} onRemove={removeLatest} />}
        {error && <div className="alert alert-error" role="alert"><div><strong>Trajectory not analyzed</strong><span>{error}</span></div></div>}
        <button className="primary-button submit-button" type="submit" disabled={loading}>{loading ? <><span className="spinner" /> Analyzing sensor state</> : <>Analyze anomaly state <span aria-hidden="true">→</span></>}</button>
      </form>
      <aside className="rul-result-column" aria-live="polite" aria-busy={loading}>{loading ? <div className="panel rul-empty"><span className="loader-ring" /><h2>Analyzing trajectory</h2><p>Comparing sensor states with the development normal reference.</p></div> : result ? <AnomalyResult result={result} source={resultSource} /> : <div className="panel rul-empty"><div className="horizon-symbol">AD</div><p className="eyebrow">Awaiting trajectory</p><h2>Your condition-monitoring result will appear here.</h2><p>CSV, manual, and demo inputs use the same model. Current unusualness and persistent alert state remain separate.</p></div>}</aside>
    </div>
  </div>;
}

function AnomalyCsvInput({ csv, error, inputRef, onFile }: { csv: ParsedAnomalyCsv | null; error: string | null; inputRef: React.RefObject<HTMLInputElement | null>; onFile: (file: File | undefined) => void }) {
  const ready = (csv?.observations.length ?? 0) >= 5;
  return <><section className="panel csv-upload-panel"><label className="csv-dropzone" htmlFor="anomaly-csv"><span className="upload-mark" aria-hidden="true">CSV</span><strong>{csv ? "Choose a different CSV" : "Choose anomaly trajectory CSV"}</strong><small>The file is parsed in your browser and is not uploaded or retained.</small></label><input ref={inputRef} id="anomaly-csv" className="visually-hidden" type="file" accept=".csv,text/csv" onChange={(event) => void onFile(event.target.files?.[0])} />{error && <div className="csv-error" role="alert"><strong>CSV not ready</strong><span>{error}</span></div>}<details className="csv-format"><summary>View CSV format</summary><p>Exactly these model-input columns are accepted. Extra, output, RUL, target, and failure columns are rejected.</p><code>{ANOMALY_CSV_HEADERS.join(",")}</code></details></section>{csv && <section className="panel csv-ready" aria-live="polite"><div className="panel-heading"><div><span className="section-kicker">Trajectory ready</span><h2>{csv.filename}</h2></div><span className={`history-chip ${ready ? "complete" : "limited"}`}>{ready ? "Persistence Available" : "Scoring Available"}</span></div><ul className="validation-summary"><li><span aria-hidden="true">✓</span>{csv.observations.length} cycle{csv.observations.length === 1 ? "" : "s"} detected</li><li><span aria-hidden="true">✓</span>{ANOMALY_SENSOR_COUNT} required sensor channels found per cycle</li><li><span aria-hidden="true">✓</span>Chronological sequence valid</li><li className={ready ? "" : "warning-item"}><span aria-hidden="true">{ready ? "✓" : "!"}</span>{ready ? "Persistent-alert evaluation available; exactly the latest five cycles are used" : "Anomaly scoring available; five cycles are required for persistent-alert evaluation"}</li></ul><div className="csv-preview"><h3>Preview · first {Math.min(3, csv.observations.length)} cycles</h3><p className="fine-print">Four representative fields are shown; all 14 sensor channels will be sent for inference.</p><div className="preview-table" role="table" aria-label="Parsed anomaly CSV preview"><div className="preview-row preview-header" role="row"><span>Cycle</span><span>Sensor 2</span><span>Sensor 11</span><span>Sensor 21</span></div>{csv.observations.slice(0, 3).map((row) => <div className="preview-row" role="row" key={row.cycle}><span>{row.cycle}</span><span>{row.sensor_2}</span><span>{row.sensor_11}</span><span>{row.sensor_21}</span></div>)}</div></div></section>}</>;
}

function AnomalyManualInput({ rows, expanded, demo, onExpanded, onUpdate, onAdd, onRemove }: { rows: ObservationForm[]; expanded: number; demo: boolean; onExpanded: (index: number) => void; onUpdate: (index: number, key: keyof AnomalyObservation, value: string) => void; onAdd: () => void; onRemove: () => void }) {
  return <><div className="trajectory-heading"><div><span className="section-kicker">Chronological input</span><h2>Sensor history</h2>{demo && <span className="demo-data-badge">Demo Data</span>}</div><div><button type="button" className="secondary-button" onClick={onRemove} disabled={rows.length === 1}>Remove latest</button><button type="button" className="primary-button" onClick={onAdd}>+ Add cycle</button></div></div><div className="cycle-list">{rows.map((row, index) => <article className={`cycle-card ${expanded === index ? "expanded" : ""}`} key={index}><button type="button" className="cycle-summary" onClick={() => onExpanded(expanded === index ? -1 : index)} aria-expanded={expanded === index}><span className="cycle-number">{row.cycle || "—"}</span><div><strong>Cycle {row.cycle || index + 1}</strong><small>{Object.values(row).filter(Boolean).length - 1} of {ANOMALY_SENSOR_COUNT} sensors entered</small></div><span>{expanded === index ? "Close" : "Edit"}</span></button>{expanded === index && <div className="cycle-editor"><div className="cycle-input field-group"><label htmlFor={`anomaly-cycle-${index}`}>Cycle</label><input id={`anomaly-cycle-${index}`} type="number" step="1" min="1" value={row.cycle} onChange={(event) => onUpdate(index, "cycle", event.target.value)} /></div>{sensorGroups.map((group) => <fieldset key={group.title}><legend>{group.title}</legend><div className="sensor-grid">{group.fields.map(([key, label]) => <div className="field-group" key={key}><label htmlFor={`anomaly-${key}-${index}`} title="An anonymized C-MAPSS sensor channel.">{label}<span className="unit-label">raw</span></label><input id={`anomaly-${key}-${index}`} type="number" step="any" required value={row[key]} onChange={(event) => onUpdate(index, key, event.target.value)} /></div>)}</div></fieldset>)}</div>}</article>)}</div></>;
}

function AnomalyResult({ result, source }: { result: AnomalyPredictionResponse; source: InputSource | null }) {
  const persistenceLabel = result.persistence_status === "insufficient_history" ? "Insufficient History" : result.alert_active ? "Active" : "Inactive";
  const style = { "--percentile": `${result.anomaly_percentile}%`, "--boundary": `${result.threshold_percentile}%` } as CSSProperties;
  return <article className={`panel anomaly-result ${result.alert_active ? "alert-active" : ""}`}>
    <div className="result-topline"><span className="section-kicker">Condition analysis complete</span><span className={`history-chip ${result.current_threshold_exceeded ? "limited" : "complete"}`}>{result.current_threshold_exceeded ? "Above threshold" : "Within reference range"}</span></div>
    {source && <span className="input-source-badge">Input Source: {source}</span>}
    <div className="anomaly-percentile"><span>Anomaly Percentile · Non-probabilistic</span><strong>{result.anomaly_percentile.toFixed(1)}<small>/100</small></strong><p>More unusual than approximately {result.anomaly_percentile.toFixed(1)}% of the development normal-reference score distribution. This is not a probability or confidence.</p></div>
    <div className="percentile-gauge" style={style} role="img" aria-label={`Anomaly percentile ${result.anomaly_percentile.toFixed(1)}; boundary ${result.threshold_percentile.toFixed(1)}`}><div className="percentile-fill" /><span className="threshold-marker"><i />Anomaly boundary {result.threshold_percentile.toFixed(1)}</span></div>
    <div className="anomaly-state-grid"><section><span>Current State</span><strong>{result.current_threshold_exceeded ? "Above anomaly threshold" : "Within reference range"}</strong><small>Raw score {result.current_anomaly_score.toFixed(6)} · raw boundary {result.raw_threshold.toFixed(6)}</small></section><section className={result.alert_active ? "persistent-active" : ""}><span>Persistent Anomaly Alert</span><strong>{persistenceLabel}</strong><small>{result.persistence_status === "available" ? `${result.recent_exceedance_count} of latest ${result.recent_window_size} cycles exceeded threshold` : `${result.history_cycle_count} of ${result.persistence_window_size} required cycles supplied`}</small></section></div>
    <p className="persistence-note">Persistent status reflects the recent trajectory, not only the current cycle. Recent pattern: {result.recent_exceedance_pattern.map((value) => value ? "above" : "within").join(" · ")}. The policy requires {result.persistence_required_count} of the latest {result.persistence_window_size} cycles.</p>
    <dl className="result-meta"><div><dt>Model</dt><dd>v{result.model_version}</dd></div><div><dt>Dataset</dt><dd>{result.dataset}</dd></div><div><dt>History supplied</dt><dd>{result.history_cycle_count} cycles</dd></div><div><dt>Status</dt><dd>{result.development_stage ? "Development stage" : "Production"}</dd></div></dl>
    <section className="deviation-section"><div><span className="section-kicker">Sensor context</span><h2>Most Unusual Sensor Readings</h2><p>{result.sensor_context_label} These standardized deviations are context relative to the normal reference—not causes, root causes, or model feature importance.</p></div><div className="deviation-list">{result.top_sensor_deviations.map((item) => <div key={item.sensor}><div><strong>{item.sensor.replace("_", " ")}</strong><span>{item.direction === "above_normal" ? "Above normal" : "Below normal"}</span></div><div><small>Current</small><strong>{item.current_value.toFixed(3)}</strong></div><div><small>Reference</small><strong>{item.reference_mean.toFixed(3)}</strong></div><div><small>Deviation</small><strong>{item.standardized_deviation > 0 ? "+" : ""}{item.standardized_deviation.toFixed(2)}σ</strong></div></div>)}</div></section>
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
