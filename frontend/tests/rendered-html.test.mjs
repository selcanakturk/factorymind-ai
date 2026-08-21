import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { ANOMALY_CSV_HEADERS, ANOMALY_SENSOR_COUNT, parseAnomalyCsv } from "../src/utils/anomalyCsv.ts";
import { parseRulCsv, RUL_CSV_HEADERS, RUL_MEASUREMENT_COUNT } from "../src/utils/rulCsv.ts";

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
  assert.match(source, /activeCount >= 5/);
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

test("anomaly page exposes CSV, manual, and clearly labeled demo paths", async () => {
  const source = await projectFile("src/pages/AnomalyDetectionPage.tsx");
  assert.match(source, />Upload CSV</);
  assert.match(source, />Manual Entry</);
  assert.match(source, /Try Demo Engine/);
  assert.match(source, /Normal-state demo/);
  assert.match(source, /Persistent-anomaly demo/);
  assert.match(source, /About sensor channels/);
  assert.match(source, /NASA C-MAPSS/);
  assert.match(source, /mode === "csv" \? csv\?\.observations : manualObservations\(\)/);
});

const anomalyCsvRow = (cycle) => ANOMALY_CSV_HEADERS.map((header) => header === "cycle" ? cycle : 1).join(",");
const anomalyCsvText = (cycles) => [ANOMALY_CSV_HEADERS.join(","), ...cycles.map(anomalyCsvRow)].join("\n");

test("valid anomaly CSV converts to the exact frozen 14-sensor contract", () => {
  const parsed = parseAnomalyCsv(anomalyCsvText([1, 2, 3, 4, 5]), "anomaly.csv");
  assert.equal(ANOMALY_CSV_HEADERS.length, 15);
  assert.equal(ANOMALY_SENSOR_COUNT, 14);
  assert.equal(parsed.observations.length, 5);
  assert.deepEqual(Object.keys(parsed.observations[0]), [...ANOMALY_CSV_HEADERS]);
});

test("anomaly CSV rejects missing, unknown, and forbidden columns", () => {
  const missing = ANOMALY_CSV_HEADERS.filter((header) => header !== "sensor_11");
  assert.throws(() => parseAnomalyCsv([missing.join(","), missing.map(() => 1).join(",")].join("\n"), "missing.csv"), /Missing required CSV column: sensor_11/);
  assert.throws(() => parseAnomalyCsv(`${ANOMALY_CSV_HEADERS.join(",")},comment\n${anomalyCsvRow(1)},1`, "extra.csv"), /Unknown CSV column: comment/);
  for (const forbidden of ["RUL", "raw_rul", "target", "failure_label", "anomaly_score", "prediction"]) {
    assert.throws(() => parseAnomalyCsv(`${ANOMALY_CSV_HEADERS.join(",")},${forbidden}\n${anomalyCsvRow(1)},1`, "forbidden.csv"), /forbidden output, target, failure, or RUL column/);
  }
});

test("anomaly CSV rejects invalid files and measurement values", () => {
  assert.throws(() => parseAnomalyCsv("", "empty.csv"), /file is empty/);
  assert.throws(() => parseAnomalyCsv(anomalyCsvText([1]), "engine.txt"), /\.csv extension/);
  const blank = anomalyCsvRow(1).split(","); blank[1] = "";
  assert.throws(() => parseAnomalyCsv(`${ANOMALY_CSV_HEADERS.join(",")}\n${blank.join(",")}`, "blank.csv"), /blank value for sensor_2/);
  for (const value of ["not-a-number", "NaN", "Infinity", "true"]) {
    const invalid = anomalyCsvRow(1).split(","); invalid[1] = value;
    assert.throws(() => parseAnomalyCsv(`${ANOMALY_CSV_HEADERS.join(",")}\n${invalid.join(",")}`, "invalid.csv"), /invalid numeric value for sensor_2/);
  }
});

test("anomaly CSV rejects invalid chronological sequences", () => {
  assert.throws(() => parseAnomalyCsv(anomalyCsvText([1, 1]), "duplicate.csv"), /Duplicate cycle 1/);
  assert.throws(() => parseAnomalyCsv(anomalyCsvText([2, 1]), "unsorted.csv"), /increasing chronological order/);
  assert.throws(() => parseAnomalyCsv(anomalyCsvText([1, 3]), "gap.csv"), /expected cycle 2 after cycle 1/);
  assert.throws(() => parseAnomalyCsv(anomalyCsvText([0]), "zero.csv"), /positive whole-number cycle/);
  assert.throws(() => parseAnomalyCsv(anomalyCsvText([1.5]), "fraction.csv"), /positive whole-number cycle/);
});

test("anomaly CSV UI presents persistence-aware summaries and compact preview", async () => {
  const source = await projectFile("src/pages/AnomalyDetectionPage.tsx");
  assert.match(source, /Anomaly scoring available; five cycles are required/);
  assert.match(source, /Persistent-alert evaluation available; exactly the latest five cycles are used/);
  assert.match(source, /all 14 sensor channels will be sent for inference/);
  assert.match(source, /slice\(0, 3\)/);
  assert.match(source, /Input Source:/);
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

test("RUL page exposes CSV, manual, and clearly labeled demo input paths", async () => {
  const source = await projectFile("src/pages/RULPredictionPage.tsx");
  assert.match(source, />Upload CSV</);
  assert.match(source, />Manual Entry</);
  assert.match(source, /Try Demo Engine/);
  assert.match(source, /Demo Data/);
  assert.match(source, /About sensor channels/);
  assert.match(source, /NASA C-MAPSS/);
  assert.match(source, /mode === "csv" \? csv\?\.observations : manualObservations\(\)/);
});

const csvRow = (cycle) => RUL_CSV_HEADERS.map((header) => header === "cycle" ? cycle : 1).join(",");
const csvText = (cycles) => [RUL_CSV_HEADERS.join(","), ...cycles.map(csvRow)].join("\n");

test("valid RUL CSV parses into the frozen observation contract", () => {
  const parsed = parseRulCsv(csvText([1, 2, 3, 4, 5, 6]), "engine.csv");
  assert.equal(parsed.observations.length, 6);
  assert.equal(Object.keys(parsed.observations[0]).length, RUL_CSV_HEADERS.length);
  assert.equal(RUL_MEASUREMENT_COUNT, 17);
});

test("RUL CSV rejects missing, forbidden, and unknown columns", () => {
  const missing = RUL_CSV_HEADERS.filter((header) => header !== "sensor_21");
  assert.throws(() => parseRulCsv([missing.join(","), missing.map(() => 1).join(",")].join("\n"), "missing.csv"), /Missing required CSV column: sensor_21/);
  assert.throws(() => parseRulCsv(`${RUL_CSV_HEADERS.join(",")},RUL\n${csvRow(1)},20`, "target.csv"), /forbidden target or RUL column: RUL/);
  assert.throws(() => parseRulCsv(`${RUL_CSV_HEADERS.join(",")},comment\n${csvRow(1)},1`, "extra.csv"), /Unknown CSV column: comment/);
});

test("RUL CSV rejects duplicate, unsorted, and gapped cycles", () => {
  assert.throws(() => parseRulCsv(csvText([1, 1]), "duplicate.csv"), /Duplicate cycle 1/);
  assert.throws(() => parseRulCsv(csvText([2, 1]), "unsorted.csv"), /increasing chronological order/);
  assert.throws(() => parseRulCsv(csvText([1, 3]), "gap.csv"), /expected cycle 2/);
});

test("RUL CSV rejects invalid files, blank values, and non-numeric measurements", () => {
  assert.throws(() => parseRulCsv("", "empty.csv"), /file is empty/);
  assert.throws(() => parseRulCsv(csvText([1]), "engine.txt"), /\.csv extension/);
  const blank = csvRow(1).split(","); blank[4] = "";
  assert.throws(() => parseRulCsv(`${RUL_CSV_HEADERS.join(",")}\n${blank.join(",")}`, "blank.csv"), /blank value for sensor_2/);
  const invalid = csvRow(1).split(","); invalid[4] = "not-a-number";
  assert.throws(() => parseRulCsv(`${RUL_CSV_HEADERS.join(",")}\n${invalid.join(",")}`, "invalid.csv"), /invalid numeric value for sensor_2/);
});

test("RUL CSV UI includes limited/full summaries, preview, and source badge", async () => {
  const source = await projectFile("src/pages/RULPredictionPage.tsx");
  assert.match(source, /Full temporal context available/);
  assert.match(source, /Limited history/);
  assert.match(source, /Preview · first/);
  assert.match(source, /Input Source:/);
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
