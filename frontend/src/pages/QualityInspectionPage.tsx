import { useEffect, useRef, useState } from "react";
import { ApiError, factoryMindApi } from "../services/api";
import type { VisualQualityPredictionResponse } from "../types/api";


const MAX_UPLOAD_BYTES = 8 * 1024 * 1024;
const ALLOWED_EXTENSIONS = new Set(["jpg", "jpeg", "png"]);
const ALLOWED_MIME_TYPES = new Set(["image/jpeg", "image/png"]);

interface SelectedImage {
  file: File;
  previewUrl: string;
  width?: number;
  height?: number;
}

function validateFile(file: File): string | null {
  const extension = file.name.split(".").pop()?.toLowerCase() ?? "";
  if (!ALLOWED_EXTENSIONS.has(extension)) return "Choose a JPEG, JPG, or PNG image.";
  if (!ALLOWED_MIME_TYPES.has(file.type)) return "The selected file must use a JPEG or PNG image type.";
  if (file.size === 0) return "The selected image is empty.";
  if (file.size > MAX_UPLOAD_BYTES) return "The selected image exceeds the 8 MB upload limit.";
  return null;
}

function readableError(error: unknown): string {
  if (!(error instanceof ApiError)) return "Inspection could not be completed. Please try again.";
  if (error.status === 422) return `The image could not be accepted. ${error.message}`;
  if (error.status === 413) return "The image exceeds the backend 8 MB upload limit.";
  if (error.status === 503) return "The Visual Quality model is temporarily unavailable.";
  if (error.status === 500) return "Visual inspection failed unexpectedly. Please try again.";
  return error.message;
}

export function QualityInspectionPage({ modelAvailable }: { modelAvailable: boolean }) {
  const [selected, setSelected] = useState<SelectedImage | null>(null);
  const [result, setResult] = useState<VisualQualityPredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const previewUrl = selected?.previewUrl;
    return () => { if (previewUrl) URL.revokeObjectURL(previewUrl); };
  }, [selected?.previewUrl]);

  function chooseFile(file?: File) {
    setError(null);
    if (!file) { setError("Select one inspection image before analysis."); return; }
    const validationError = validateFile(file);
    if (validationError) { setError(validationError); return; }
    const previewUrl = URL.createObjectURL(file);
    setSelected((previous) => {
      if (previous) URL.revokeObjectURL(previous.previewUrl);
      return { file, previewUrl };
    });
    setResult(null);
    const probe = new Image();
    probe.onload = () => setSelected((current) => current?.file === file ? { ...current, width: probe.naturalWidth, height: probe.naturalHeight } : current);
    probe.src = previewUrl;
  }

  function removeImage() {
    setSelected((previous) => {
      if (previous) URL.revokeObjectURL(previous.previewUrl);
      return null;
    });
    setResult(null); setError(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  async function analyze() {
    if (!selected) { setError("Select one inspection image before analysis."); return; }
    if (!modelAvailable) { setError("The Visual Quality model is currently unavailable."); return; }
    setLoading(true); setError(null);
    try { setResult(await factoryMindApi.predictVisualQuality(selected.file)); }
    catch (caught) { setResult(null); setError(readableError(caught)); }
    finally { setLoading(false); }
  }

  return (
    <div className="page-stack quality-page">
      <header className="page-heading compact-heading">
        <div><p className="eyebrow">Visual intelligence · Module 04</p><h1>Visual Quality Inspection</h1><p className="page-lead">Does this inspection image visually differ from the learned normal zipper reference?</p></div>
      </header>

      <div className="quality-scope" role="note">
        <span>ZIPPER v1</span><p>Normal-only anomaly detection developed for the MVTec AD zipper category. JPEG/PNG only · maximum 8 MB · score is not probability.</p>
      </div>

      <section className="quality-workspace">
        <article className="panel quality-input-panel">
          <div className="panel-heading"><div><span className="section-kicker">Inspection input</span><h2>Choose one zipper image</h2></div><span className={`status-pill ${modelAvailable ? "active" : ""}`}>{modelAvailable ? "Model ready" : "Unavailable"}</span></div>
          {!selected ? (
            <label className="image-dropzone" htmlFor="quality-image">
              <span className="upload-mark">IMG</span><strong>Select inspection image</strong><span>JPEG, JPG, or PNG · up to 8 MB</span>
              <input ref={inputRef} id="quality-image" className="visually-hidden" type="file" accept=".jpg,.jpeg,.png,image/jpeg,image/png" onChange={(event) => chooseFile(event.target.files?.[0])} />
            </label>
          ) : (
            <div className="selected-image">
              <img src={selected.previewUrl} alt={`Original inspection preview: ${selected.file.name}`} />
              <div className="file-metadata"><strong>{selected.file.name}</strong><span>{selected.file.type} · {(selected.file.size / 1024).toFixed(1)} KB{selected.width && selected.height ? ` · ${selected.width} × ${selected.height}` : ""}</span></div>
              <div className="image-actions"><button className="secondary-button" type="button" onClick={() => inputRef.current?.click()}>Replace image</button><button className="text-button" type="button" onClick={removeImage}>Remove image</button></div>
              <input ref={inputRef} id="quality-image-replace" className="visually-hidden" type="file" accept=".jpg,.jpeg,.png,image/jpeg,image/png" onChange={(event) => chooseFile(event.target.files?.[0])} />
            </div>
          )}
          {error && <div className="csv-error" role="alert" aria-live="assertive"><strong>Image not analyzed</strong><span>{error}</span></div>}
          <button className="submit-button" type="button" disabled={loading || !selected || !modelAvailable} onClick={analyze}>{loading ? <><span className="spinner" />Analyzing image…</> : "Analyze inspection image"}</button>
          <p className="form-note">The browser sends the selected file directly to FactoryMind. It is not stored by this interface.</p>
        </article>

        <article className="panel quality-result-panel" aria-live="polite" aria-busy={loading}>
          {loading ? <div className="quality-empty" role="status"><span className="loader-ring" /><h2>Inspecting visual patterns</h2><p>Comparing patch representations with the frozen normal zipper reference.</p></div> : result && selected ? <QualityResult result={result} originalUrl={selected.previewUrl} filename={selected.file.name} /> : <div className="quality-empty"><div className="quality-empty-icon">16×16</div><h2>Inspection result</h2><p>Select a zipper image to view its Visual Anomaly Score and Model anomaly map.</p></div>}
        </article>
      </section>

      <article className="panel quality-context"><span className="section-kicker">Responsible scope</span><h2>What this result means</h2><div><p><strong>Appearance difference, not diagnosis.</strong> Highlighted regions differ from the learned normal representation; they do not confirm defect cause or severity.</p><p><strong>Research-category constraint.</strong> This model was developed on MVTec AD zipper images and has no real-factory external validation.</p><p><strong>No demo imagery bundled.</strong> Dataset examples are not redistributed in the frontend because MVTec AD carries noncommercial licensing constraints.</p></div></article>
    </div>
  );
}

function QualityResult({ result, originalUrl, filename }: { result: VisualQualityPredictionResponse; originalUrl: string; filename: string }) {
  return <div className={`quality-result ${result.anomaly_detected ? "detected" : "clear"}`}>
    <div className="quality-status"><span>{result.anomaly_detected ? "Review recommended" : "Development result"}</span><h2>{result.quality_status}</h2><p>For the learned normal {result.category} reference</p></div>
    <div className="quality-score-grid"><div><span>Visual Anomaly Score</span><strong>{result.visual_anomaly_score.toFixed(4)}</strong></div><div><span>Development threshold</span><strong>{result.threshold.toFixed(4)}</strong></div></div>
    <p className="score-explainer">Higher scores indicate greater visual difference from the learned normal zipper reference. The score is a model distance, not probability or confidence.</p>
    <div className="image-comparison">
      <figure><div className="comparison-frame"><img src={originalUrl} alt={`Original uploaded inspection: ${filename}`} /></div><figcaption>Original inspection image</figcaption></figure>
      <figure><div className="comparison-frame anomaly-map-frame"><img src={`data:image/png;base64,${result.anomaly_map_image_base64}`} alt="Model anomaly map showing relative visual difference intensity" /></div><figcaption>{result.anomaly_map_label}</figcaption></figure>
    </div>
    <p className="map-explainer">Highlighted regions indicate areas whose visual representation differs most from the learned normal reference. This is not a defect mask or exact defect boundary.</p>
    <dl className="quality-details"><div><dt>Category</dt><dd>{result.category}</dd></div><div><dt>Model</dt><dd>{result.model_version}</dd></div><div><dt>Dataset</dt><dd>{result.dataset}</dd></div><div><dt>Original</dt><dd>{result.original_width} × {result.original_height}</dd></div><div><dt>Model input</dt><dd>{result.model_input_width} × {result.model_input_height}</dd></div><div><dt>Boundary</dt><dd>{(result.threshold_quantile * 100).toFixed(1)}th development-normal percentile</dd></div><div><dt>Map</dt><dd>{result.raw_anomaly_map_16x16.length}×{result.raw_anomaly_map_16x16[0]?.length ?? 0} patch anomaly map</dd></div></dl>
    <div className="quality-warning"><strong>Model warning</strong><p>{result.warning}</p></div>
    <p className="disclaimer">{result.disclaimer}</p>
  </div>;
}
