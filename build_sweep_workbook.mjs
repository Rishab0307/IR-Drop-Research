import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = "/Users/rishab/Research";
const csvPath = path.join(root, "hyperparameter_sweep_runs", "raw_results.csv");
const outputDir = path.join(root, "outputs", "019fc200-9d3b-7320-b6d8-d626ff2f3540");

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const fields = line.split(",");
    return Object.fromEntries(headers.map((header, index) => [header, fields[index]]));
  });
}

const rows = parseCsv(await fs.readFile(csvPath, "utf8"));
if (rows.length !== 180) throw new Error(`Expected 180 rows, found ${rows.length}`);

const losses = ["MSE", "MAE", "Huber", "LogCosh"];
const batches = [8, 16, 32];
const learningRates = ["1e-5", "1e-4", "1e-3"];
const layers = [2, 4, 6, 8, 10];
const byKey = new Map(rows.map((r) => [
  [r.loss, r.batch_size, r.learning_rate, r.fine_tuned_layers].join("|"),
  Number(r.per_chip_range_ssim),
]));

const workbook = Workbook.create();
const summary = workbook.worksheets.add("Summary");
for (const loss of losses) workbook.worksheets.add(loss);
const raw = workbook.worksheets.add("Raw Results");

const navy = "#17324D";
const blue = "#2F75B5";
const paleBlue = "#D9EAF7";
const paleGreen = "#E2F0D9";
const light = "#F5F7FA";
const gray = "#667085";
const border = "#C7D0D9";

summary.showGridLines = false;
summary.getRange("A1:H1").merge();
summary.getRange("A1").values = [["ResNet50V2 Vortex-small Hyperparameter Sweep"]];
summary.getRange("A1:H1").format = {
  fill: navy,
  font: { bold: true, color: "#FFFFFF", size: 18 },
  verticalAlignment: "center",
};
summary.getRange("A1:H1").format.rowHeight = 34;
summary.getRange("A3:B8").values = [
  ["Sweep status", "180 / 180 verified"],
  ["Model", "ResNet50V2"],
  ["Dataset", "Vortex-small only"],
  ["Metric", "Per-chip-range SSIM"],
  ["Kernel policy", "Fresh process/kernel for every run"],
  ["Added random seeds", "None"],
];
summary.getRange("A3:A8").format = { fill: paleBlue, font: { bold: true, color: navy } };
summary.getRange("A3:B8").format.borders = { preset: "outside", style: "thin", color: border };

summary.getRange("D3:I3").values = [["Rank", "Loss", "Batch", "Learning rate", "Layers", "SSIM"]];
summary.getRange("D3:I3").format = { fill: blue, font: { bold: true, color: "#FFFFFF" }, horizontalAlignment: "center" };
const ranked = [...rows].sort((a, b) => Number(b.per_chip_range_ssim) - Number(a.per_chip_range_ssim)).slice(0, 10);
summary.getRange("D4:I13").values = ranked.map((r, i) => [
  i + 1,
  r.loss,
  Number(r.batch_size),
  r.learning_rate,
  Number(r.fine_tuned_layers),
  Number(r.per_chip_range_ssim),
]);
summary.getRange("I4:I13").format.numberFormat = "0.000000000";
summary.getRange("D4:I13").format.borders = { preset: "inside", style: "thin", color: border };
summary.getRange("D4:I4").format = { fill: paleGreen, font: { bold: true, color: "#1F4E3D" } };

summary.getRange("A11:B16").values = [
  ["Winning configuration", ""],
  ["Loss", "Huber"],
  ["Batch", 8],
  ["Learning rate", "1e-3"],
  ["Fine-tuned layers", 2],
  ["SSIM", 0.878742814],
];
summary.getRange("A11:B11").format = { fill: navy, font: { bold: true, color: "#FFFFFF" } };
summary.getRange("A12:A16").format = { fill: paleGreen, font: { bold: true, color: "#1F4E3D" } };
summary.getRange("B16").format.numberFormat = "0.000000000";
summary.getRange("A11:B16").format.borders = { preset: "outside", style: "medium", color: "#70AD47" };
summary.getRange("A18:I19").merge();
summary.getRange("A18").values = [["Each loss sheet contains all 45 combinations (3 batch sizes × 3 learning rates × 5 fine-tuned layer counts). Scores are copied directly from the verified raw-results ledger."]];
summary.getRange("A18:I19").format = { fill: light, font: { italic: true, color: gray }, wrapText: true };
summary.getRange("A:A").format.columnWidth = 21;
summary.getRange("B:B").format.columnWidth = 25;
summary.getRange("D:D").format.columnWidth = 8;
summary.getRange("E:E").format.columnWidth = 12;
summary.getRange("F:F").format.columnWidth = 9;
summary.getRange("G:G").format.columnWidth = 15;
summary.getRange("H:H").format.columnWidth = 9;
summary.getRange("I:I").format.columnWidth = 15;
summary.freezePanes.freezeRows(1);

for (const loss of losses) {
  const sheet = workbook.worksheets.getItem(loss);
  sheet.showGridLines = false;
  sheet.getRange("A1:G1").merge();
  sheet.getRange("A1").values = [[`${loss} — Per-chip-range SSIM (Vortex-small)`]];
  sheet.getRange("A1:G1").format = { fill: navy, font: { bold: true, color: "#FFFFFF", size: 16 } };
  sheet.getRange("A3:G3").values = [["Model", "Loss", "Batch", "Layers", "1e-5", "1e-4", "1e-3"]];
  sheet.getRange("A3:G3").format = { fill: blue, font: { bold: true, color: "#FFFFFF" }, horizontalAlignment: "center" };
  const data = [];
  for (const batch of batches) {
    for (const layer of layers) {
      data.push([
        "ResNet50V2",
        loss,
        batch,
        layer,
        byKey.get([loss, batch, "1e-5", layer].join("|")),
        byKey.get([loss, batch, "1e-4", layer].join("|")),
        byKey.get([loss, batch, "1e-3", layer].join("|")),
      ]);
    }
  }
  sheet.getRange("A4:G18").values = data;
  sheet.getRange("E4:G18").format.numberFormat = "0.000000000";
  sheet.getRange("C4:D18").format.horizontalAlignment = "center";
  sheet.getRange("E4:G18").format.horizontalAlignment = "right";
  sheet.getRange("A4:G18").format.borders = {
    insideHorizontal: { style: "thin", color: border },
    insideVertical: { style: "thin", color: border },
  };
  for (const startRow of [4, 9, 14]) {
    sheet.getRange(`A${startRow}:G${startRow}`).format.borders = { top: { style: "medium", color: navy } };
    sheet.getRange(`A${startRow}:D${startRow + 4}`).format.fill = light;
  }
  sheet.getRange("E4:G18").conditionalFormats.add("colorScale", {
    colors: ["#FCE8E6", "#FFF2CC", "#E2F0D9"],
    thresholds: ["min", "50%", "max"],
  });
  const lossRows = rows.filter((r) => r.loss === loss);
  const best = lossRows.sort((a, b) => Number(b.per_chip_range_ssim) - Number(a.per_chip_range_ssim))[0];
  sheet.getRange("A21:G21").merge();
  sheet.getRange("A21").values = [[`Best ${loss}: batch ${best.batch_size}, LR ${best.learning_rate}, ${best.fine_tuned_layers} layers — SSIM ${Number(best.per_chip_range_ssim).toFixed(9)}`]];
  sheet.getRange("A21:G21").format = { fill: paleGreen, font: { bold: true, color: "#1F4E3D" } };
  sheet.getRange("A:A").format.columnWidth = 18;
  sheet.getRange("B:B").format.columnWidth = 12;
  sheet.getRange("C:D").format.columnWidth = 10;
  sheet.getRange("E:G").format.columnWidth = 16;
  sheet.freezePanes.freezeRows(3);
}

raw.showGridLines = false;
raw.getRange("A1:H1").values = [["Loss", "Batch", "Learning rate", "Layers", "Per-chip-range SSIM", "Elapsed seconds", "Run ID", "Executed notebook"]];
raw.getRange("A1:H1").format = { fill: navy, font: { bold: true, color: "#FFFFFF" }, wrapText: true };
raw.getRange("A2:H181").values = rows.map((r) => [
  r.loss,
  Number(r.batch_size),
  r.learning_rate,
  Number(r.fine_tuned_layers),
  Number(r.per_chip_range_ssim),
  Number(r.elapsed_seconds),
  r.run_id,
  r.executed_notebook,
]);
raw.getRange("E2:E181").format.numberFormat = "0.000000000";
raw.getRange("F2:F181").format.numberFormat = "0.000";
raw.getRange("A1:H181").format.borders = { preset: "inside", style: "thin", color: border };
raw.getRange("A:A").format.columnWidth = 12;
raw.getRange("B:B").format.columnWidth = 9;
raw.getRange("C:C").format.columnWidth = 15;
raw.getRange("D:D").format.columnWidth = 9;
raw.getRange("E:E").format.columnWidth = 22;
raw.getRange("F:F").format.columnWidth = 16;
raw.getRange("G:G").format.columnWidth = 31;
raw.getRange("H:H").format.columnWidth = 60;
raw.freezePanes.freezeRows(1);
raw.tables.add("A1:H181", true, "RawSweepResults").style = "TableStyleMedium2";

await fs.mkdir(outputDir, { recursive: true });
const inspect = await workbook.inspect({ kind: "table", range: "Summary!A1:I19", include: "values,formulas", tableMaxRows: 20, tableMaxCols: 10 });
console.log(inspect.ndjson);
const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 300 }, summary: "final formula error scan" });
console.log(errors.ndjson);
for (const sheetName of ["Summary", ...losses, "Raw Results"]) {
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(path.join(outputDir, `${sheetName.replaceAll(" ", "_")}.png`), new Uint8Array(await preview.arrayBuffer()));
}
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(path.join(outputDir, "ResNet50V2_Vortex-small_SSIM_sweep.xlsx"));
