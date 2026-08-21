import type { AnomalyObservation } from "../types/api";

export const ANOMALY_CSV_HEADERS = [
  "cycle",
  "sensor_2",
  "sensor_3",
  "sensor_4",
  "sensor_7",
  "sensor_8",
  "sensor_9",
  "sensor_11",
  "sensor_12",
  "sensor_13",
  "sensor_14",
  "sensor_15",
  "sensor_17",
  "sensor_20",
  "sensor_21",
] as const satisfies readonly (keyof AnomalyObservation)[];

export const ANOMALY_SENSOR_COUNT = ANOMALY_CSV_HEADERS.length - 1;

const FORBIDDEN_HEADERS = new Set([
  "rul", "raw_rul", "capped_rul", "target", "target_label", "label",
  "failure", "failure_label", "machine_failure", "anomaly_score",
  "current_anomaly_score", "anomaly_percentile", "prediction", "predicted_label",
  "current_threshold_exceeded", "alert_active", "raw_threshold", "threshold_percentile",
]);

export interface ParsedAnomalyCsv {
  observations: AnomalyObservation[];
  filename: string;
}

export function parseAnomalyCsv(text: string, filename: string): ParsedAnomalyCsv {
  if (!filename.toLowerCase().endsWith(".csv")) throw new Error("Choose a file with a .csv extension.");
  if (!text.trim()) throw new Error("The CSV file is empty.");

  const records = parseRecords(text);
  if (records.length < 2) throw new Error("The CSV must contain a header and at least one data row.");
  const headers = records[0].map((header, index) => index === 0 ? header.replace(/^\uFEFF/, "").trim() : header.trim());
  if (headers.some((header) => !header)) throw new Error("The CSV header contains a blank column name.");
  if (new Set(headers).size !== headers.length) throw new Error("The CSV header contains duplicate column names.");

  const forbidden = headers.filter((header) => FORBIDDEN_HEADERS.has(header.toLowerCase()));
  if (forbidden.length) throw new Error(`Remove forbidden output, target, failure, or RUL column${forbidden.length > 1 ? "s" : ""}: ${forbidden.join(", ")}.`);
  const missing = ANOMALY_CSV_HEADERS.filter((header) => !headers.includes(header));
  if (missing.length) throw new Error(`Missing required CSV column${missing.length > 1 ? "s" : ""}: ${missing.join(", ")}.`);
  const unknown = headers.filter((header) => !ANOMALY_CSV_HEADERS.includes(header as keyof AnomalyObservation));
  if (unknown.length) throw new Error(`Unknown CSV column${unknown.length > 1 ? "s" : ""}: ${unknown.join(", ")}. Remove columns that are not part of the anomaly input contract.`);

  const observations = records.slice(1).map((record, rowIndex) => {
    if (record.length !== headers.length) throw new Error(`CSV row ${rowIndex + 2} has ${record.length} values; expected ${headers.length}.`);
    const values = Object.fromEntries(headers.map((header, index) => {
      const raw = record[index].trim();
      if (!raw) throw new Error(`CSV row ${rowIndex + 2} has a blank value for ${header}.`);
      if (raw.toLowerCase() === "true" || raw.toLowerCase() === "false") throw new Error(`CSV row ${rowIndex + 2} has an invalid numeric value for ${header}.`);
      const value = Number(raw);
      if (!Number.isFinite(value)) throw new Error(`CSV row ${rowIndex + 2} has an invalid numeric value for ${header}.`);
      return [header, value];
    })) as unknown as AnomalyObservation;
    if (!Number.isInteger(values.cycle) || values.cycle <= 0) throw new Error(`CSV row ${rowIndex + 2} must use a positive whole-number cycle.`);
    return values;
  });

  for (let index = 1; index < observations.length; index += 1) {
    const previous = observations[index - 1].cycle;
    const current = observations[index].cycle;
    if (current === previous) throw new Error(`Duplicate cycle ${current} found in the CSV.`);
    if (current < previous) throw new Error("CSV cycles must already be in increasing chronological order.");
    if (current !== previous + 1) throw new Error(`CSV cycles must be consecutive; expected cycle ${previous + 1} after cycle ${previous}.`);
  }

  return { observations, filename };
}

function parseRecords(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (character === '"') {
      if (quoted && text[index + 1] === '"') { field += '"'; index += 1; }
      else quoted = !quoted;
    } else if (character === "," && !quoted) { row.push(field); field = ""; }
    else if ((character === "\n" || character === "\r") && !quoted) {
      if (character === "\r" && text[index + 1] === "\n") index += 1;
      row.push(field);
      if (row.some((value) => value.trim())) rows.push(row);
      row = []; field = "";
    } else field += character;
  }
  if (quoted) throw new Error("The CSV contains an unclosed quoted value.");
  row.push(field);
  if (row.some((value) => value.trim())) rows.push(row);
  return rows;
}
