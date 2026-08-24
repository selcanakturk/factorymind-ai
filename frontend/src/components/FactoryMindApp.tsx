"use client";

import { useCallback, useEffect, useState } from "react";
import { factoryMindApi } from "../services/api";
import type { AnomalyModelInfoResponse, HealthResponse, ModelInfoResponse, RULModelInfoResponse, VisualQualityModelInfoResponse } from "../types/api";
import { AnomalyDetectionPage } from "../pages/AnomalyDetectionPage";
import { MachineAnalysisPage } from "../pages/MachineAnalysisPage";
import { ModelInfoPage } from "../pages/ModelInfoPage";
import { OverviewPage } from "../pages/OverviewPage";
import { RULPredictionPage } from "../pages/RULPredictionPage";
import { QualityInspectionPage } from "../pages/QualityInspectionPage";

type Page = "overview" | "analysis" | "rul" | "anomaly" | "quality" | "model";

const navigation = [
  ["overview", "01", "Overview"],
  ["analysis", "02", "Machine Analysis"],
  ["rul", "03", "Remaining Useful Life"],
  ["anomaly", "04", "Anomaly Detection"],
  ["quality", "05", "Quality Inspection"],
  ["model", "06", "Model Info"],
] as const;

export function FactoryMindApp() {
  const [page, setPage] = useState<Page>("overview");
  const [menuOpen, setMenuOpen] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [rulModelInfo, setRulModelInfo] = useState<RULModelInfoResponse | null>(null);
  const [anomalyModelInfo, setAnomalyModelInfo] = useState<AnomalyModelInfoResponse | null>(null);
  const [visualModelInfo, setVisualModelInfo] = useState<VisualQualityModelInfoResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadPlatformData = useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const [healthResult, modelResult, rulModelResult, anomalyModelResult, visualModelResult] = await Promise.all([
        factoryMindApi.health(), factoryMindApi.modelInfo(), factoryMindApi.getRulModelInfo(), factoryMindApi.getAnomalyModelInfo(), factoryMindApi.getVisualQualityModelInfo(),
      ]);
      setHealth(healthResult); setModelInfo(modelResult); setRulModelInfo(rulModelResult); setAnomalyModelInfo(anomalyModelResult); setVisualModelInfo(visualModelResult);
    } catch (caught) {
      setHealth(null);
      setError(caught instanceof Error ? caught.message : "Backend connection failed.");
    } finally { setLoading(false); }
  }, []);

  useEffect(() => {
    let active = true;
    Promise.all([factoryMindApi.health(), factoryMindApi.modelInfo(), factoryMindApi.getRulModelInfo(), factoryMindApi.getAnomalyModelInfo(), factoryMindApi.getVisualQualityModelInfo()])
      .then(([healthResult, modelResult, rulModelResult, anomalyModelResult, visualModelResult]) => {
        if (!active) return;
        setHealth(healthResult);
        setModelInfo(modelResult);
        setRulModelInfo(rulModelResult);
        setAnomalyModelInfo(anomalyModelResult);
        setVisualModelInfo(visualModelResult);
      })
      .catch((caught: unknown) => {
        if (!active) return;
        setHealth(null);
        setError(caught instanceof Error ? caught.message : "Backend connection failed.");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => { active = false; };
  }, []);

  function navigate(next: Page) { setPage(next); setMenuOpen(false); window.scrollTo({ top: 0, behavior: "smooth" }); }

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">Skip to content</a>
      <header className="mobile-header"><Brand /><button className="menu-button" onClick={() => setMenuOpen((open) => !open)} aria-expanded={menuOpen} aria-label="Toggle navigation"><span /><span /></button></header>
      <aside className={`sidebar ${menuOpen ? "open" : ""}`}>
        <Brand />
        <nav aria-label="Primary navigation">
          <span className="nav-label">Workspace</span>
          {navigation.map(([key, index, label]) => <button key={key} className={page === key ? "active" : ""} onClick={() => navigate(key)}><span>{index}</span>{label}</button>)}
        </nav>
        <div className="sidebar-status"><span className={`status-dot ${health?.model_loaded ? "online" : "offline"}`} /><div><strong>{health?.model_loaded ? "Model online" : "Backend offline"}</strong><span>{modelInfo ? `v${modelInfo.model_version} · ${modelInfo.calibration_method}` : "Waiting for service"}</span></div></div>
      </aside>
      {menuOpen && <button className="menu-backdrop" onClick={() => setMenuOpen(false)} aria-label="Close navigation" />}
      <main id="main-content" className="main-content">
        {page === "overview" && <OverviewPage health={health} modelInfo={modelInfo} loading={loading} error={error} onOpenAnalysis={() => navigate("analysis")} onOpenRul={() => navigate("rul")} onOpenAnomaly={() => navigate("anomaly")} onOpenQuality={() => navigate("quality")} onRetry={loadPlatformData} />}
        {page === "analysis" && <MachineAnalysisPage />}
        {page === "rul" && <RULPredictionPage modelInfo={rulModelInfo} />}
        {page === "anomaly" && <AnomalyDetectionPage modelAvailable={health?.anomaly_model_loaded ?? false} />}
        {page === "quality" && <QualityInspectionPage modelAvailable={health?.visual_quality_model_loaded ?? false} />}
        {page === "model" && <ModelInfoPage modelInfo={modelInfo} rulModelInfo={rulModelInfo} anomalyModelInfo={anomalyModelInfo} visualModelInfo={visualModelInfo} loading={loading} error={error} onRetry={loadPlatformData} />}
        <footer><span>FactoryMind AI</span><p>Development-stage machine intelligence · Built for responsible evaluation</p></footer>
      </main>
    </div>
  );
}

function Brand() {
  return <div className="brand"><span className="brand-mark"><i /><i /><i /></span><div><strong>FactoryMind</strong><small>INDUSTRIAL AI</small></div></div>;
}
