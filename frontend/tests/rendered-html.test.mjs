import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const projectFile = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

test("production build contains the FactoryMind application entry", async () => {
  const html = await readFile(new URL("../dist/index.html", import.meta.url), "utf8");
  assert.match(html, /<title>FactoryMind AI/);
  assert.match(html, /id="root"/);
  assert.match(html, /assets\/index-[^"]+\.js/);
  assert.match(html, /assets\/index-[^"]+\.css/);
});

test("RUL and anomaly navigation are active while Quality Inspection remains disabled", async () => {
  const source = await projectFile("src/components/FactoryMindApp.tsx");
  assert.match(source, /\["rul", "03", "Remaining Useful Life"\]/);
  assert.match(source, /\["anomaly", "04", "Anomaly Detection"\]/);
  assert.match(source, /disabled-nav[\s\S]*Quality Inspection/);
});

test("central API service owns both RUL endpoints", async () => {
  const source = await projectFile("src/services/api.ts");
  assert.match(source, /getRulModelInfo:[\s\S]*\/model\/rul\/info/);
  assert.match(source, /predictRul:[\s\S]*\/predict\/rul/);
});

test("central API service owns both anomaly endpoints", async () => {
  const source = await projectFile("src/services/api.ts");
  assert.match(source, /getAnomalyModelInfo:[\s\S]*\/model\/anomaly\/info/);
  assert.match(source, /predictAnomaly:[\s\S]*\/predict\/anomaly/);
});

test("anomaly page supports trajectory editing and both demo trajectories", async () => {
  const source = await projectFile("src/pages/AnomalyDetectionPage.tsx");
  assert.match(source, /function addCycle\(/);
  assert.match(source, /function removeLatest\(/);
  assert.match(source, /loadSample\("normal"\)/);
  assert.match(source, /loadSample\("persistent"\)/);
  assert.match(source, /rows\.length >= 5/);
});

test("anomaly result keeps current exceedance separate from persistent alert state", async () => {
  const source = await projectFile("src/pages/AnomalyDetectionPage.tsx");
  assert.match(source, /result\.current_threshold_exceeded/);
  assert.match(source, /result\.alert_active/);
  assert.match(source, /result\.anomaly_percentile/);
  assert.match(source, /result\.threshold_percentile/);
  assert.match(source, /result\.top_sensor_deviations/);
  assert.match(source, /result\.warning/);
  assert.match(source, /result\.disclaimer/);
});

test("anomaly validation and service failures have readable UI paths", async () => {
  const source = await projectFile("src/pages/AnomalyDetectionPage.tsx");
  assert.match(source, /error\.status === 422/);
  assert.match(source, /error\.status === 503/);
  assert.match(source, /error\.status === 500/);
});

test("anomaly persistence states and policy text are rendered from backend fields", async () => {
  const source = await projectFile("src/pages/AnomalyDetectionPage.tsx");
  assert.match(source, /persistence_status === "insufficient_history"/);
  assert.match(source, /result\.recent_exceedance_count/);
  assert.match(source, /result\.recent_window_size/);
  assert.match(source, /result\.persistence_required_count/);
  assert.match(source, /result\.persistence_window_size/);
});

test("Overview and Model Info expose the anomaly module without promoting Quality Inspection", async () => {
  const overview = await projectFile("src/pages/OverviewPage.tsx");
  const modelInfo = await projectFile("src/pages/ModelInfoPage.tsx");
  assert.match(overview, /anomaly_model_loaded/);
  assert.match(overview, /Anomaly Detection/);
  assert.match(overview, /Quality Inspection[\s\S]*Coming Soon/);
  assert.match(modelInfo, /Module 03 · Anomaly Detection/);
  assert.match(modelInfo, /anomalyModelInfo\.known_limitations/);
});

test("RUL page supports cycle add, removal, and both demo history lengths", async () => {
  const source = await projectFile("src/pages/RULPredictionPage.tsx");
  assert.match(source, /function addCycle\(/);
  assert.match(source, /function removeLatest\(/);
  assert.match(source, /loadSample\(1\)/);
  assert.match(source, /loadSample\(6\)/);
});

test("RUL result uses backend display and backend history quality", async () => {
  const source = await projectFile("src/pages/RULPredictionPage.tsx");
  assert.match(source, /result\.rul_display/);
  assert.match(source, /result\.history_quality === "full_context"/);
  assert.match(source, /result\.warning/);
  assert.match(source, /result\.disclaimer/);
});

test("RUL validation and service failures have readable UI paths", async () => {
  const source = await projectFile("src/pages/RULPredictionPage.tsx");
  assert.match(source, /error\.status === 422/);
  assert.match(source, /error\.status === 503/);
  assert.match(source, /error\.status === 500/);
});

test("failure-analysis page remains part of the application", async () => {
  const source = await projectFile("src/components/FactoryMindApp.tsx");
  assert.match(source, /page === "analysis" && <MachineAnalysisPage/);
});
