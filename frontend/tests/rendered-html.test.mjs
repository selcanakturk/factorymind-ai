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

test("RUL navigation is active while future modules remain disabled", async () => {
  const source = await projectFile("src/components/FactoryMindApp.tsx");
  assert.match(source, /\["rul", "03", "Remaining Useful Life"\]/);
  assert.match(source, /\["Anomaly Detection", "Quality Inspection"\]/);
});

test("central API service owns both RUL endpoints", async () => {
  const source = await projectFile("src/services/api.ts");
  assert.match(source, /getRulModelInfo:[\s\S]*\/model\/rul\/info/);
  assert.match(source, /predictRul:[\s\S]*\/predict\/rul/);
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
