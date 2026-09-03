import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const repoRoot = path.resolve(process.argv[2] ?? ".");
const previewDir = path.resolve(process.argv[3] ?? path.join(repoRoot, "lectures", "source", ".week02_1_preview"));
const outputPath = path.join(repoRoot, "lectures", "source", "week02_1_probabilistic_uq.pptx");
const evidenceFigure = path.join(repoRoot, "results", "probabilistic_uq", "probabilistic_uq_validation.png");

const W = 1280;
const H = 720;
const C = {
  canvas: "#FFFFFF",
  ink: "#000000",
  muted: "#5B616B",
  panel: "#EDEDED",
  rule: "#B8BCC4",
  accent: "#6DCBF4",
  accentStrong: "#3D8DFF",
  paleBlue: "#D0EDFA",
  warn: "#D97706",
  warnPale: "#FFF3D6",
};
const FONT = "Arial";

function addText(slide, text, position, fontSize = 26, options = {}) {
  const box = slide.shapes.add({
    geometry: "textbox",
    name: options.name,
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  box.text = text;
  box.text.style = {
    fontSize,
    typeface: FONT,
    color: options.color ?? C.ink,
    bold: options.bold ?? false,
    italic: options.italic ?? false,
    alignment: options.alignment ?? "left",
    verticalAlignment: options.verticalAlignment ?? "top",
    autoFit: options.autoFit ?? "none",
    insets: options.insets ?? { top: 0, right: 0, bottom: 0, left: 0 },
  };
  return box;
}

function addRect(slide, position, fill = C.panel, options = {}) {
  return slide.shapes.add({
    geometry: options.geometry ?? "rect",
    name: options.name,
    position,
    fill,
    line: {
      style: "solid",
      fill: options.lineFill ?? fill,
      width: options.lineWidth ?? 0,
    },
    ...(options.borderRadius ? { borderRadius: options.borderRadius } : {}),
  });
}

function addLine(slide, left, top, width, color = C.rule, lineWidth = 2) {
  return slide.shapes.add({
    geometry: "straightConnector1",
    position: { left, top, width, height: 0.01 },
    fill: "none",
    line: { style: "solid", fill: color, width: lineWidth },
  });
}

function addDot(slide, left, top, fill = C.ink, size = 14) {
  return slide.shapes.add({
    geometry: "ellipse",
    position: { left, top, width: size, height: size },
    fill,
    line: { style: "solid", fill, width: 0 },
  });
}

function addHeader(slide, title, number) {
  addText(slide, title, { left: 52, top: 34, width: 1150, height: 70 }, 48, {
    bold: true,
    name: `slide-${number}-title`,
  });
  addLine(slide, 52, 118, 1176, C.rule, 1);
  addText(slide, `WEEK 2.1  •  ${number} / 14`, { left: 970, top: 674, width: 258, height: 24 }, 16, {
    color: C.muted,
    alignment: "right",
  });
}

function addKicker(slide, text, left, top, width) {
  addText(slide, text.toUpperCase(), { left, top, width, height: 28 }, 18, {
    bold: true,
    color: C.accentStrong,
  });
}

function setNotes(slide, presenterNotes, sources = []) {
  const sourceBlock = sources.length
    ? `\n\n[Sources]\n${sources.map((source) => `- ${source}`).join("\n")}\n[/Sources]`
    : "";
  slide.speakerNotes.textFrame.setText(`${presenterNotes}${sourceBlock}`);
  slide.speakerNotes.setVisible(true);
}

function newSlide(presentation) {
  const slide = presentation.slides.add();
  slide.background.fill = C.canvas;
  return slide;
}

function metricPanel(slide, left, stat, label, accent = C.accentStrong) {
  addRect(slide, { left, top: 342, width: 350, height: 252 }, C.panel);
  addRect(slide, { left, top: 342, width: 10, height: 252 }, accent);
  addText(slide, stat, { left: left + 34, top: 385, width: 288, height: 82 }, 58, { bold: true });
  addText(slide, label, { left: left + 34, top: 492, width: 288, height: 70 }, 24, { color: C.muted });
}

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.mkdir(previewDir, { recursive: true });

  const presentation = Presentation.create({ slideSize: { width: W, height: H } });
  const evidenceImageBytes = await fs.readFile(evidenceFigure);

  // 1 — Cover: selected Codex Grid slide-01 silhouette.
  {
    const slide = newSlide(presentation);
    addKicker(slide, "FlowMLLab • Week 2.1", 52, 48, 500);
    addText(slide, "Probabilistic uncertainty\nfor CFD surrogates", { left: 52, top: 205, width: 1040, height: 210 }, 72, {
      bold: true,
      verticalAlignment: "bottom",
    });
    addText(slide, "From a loss function to a calibrated predictive distribution", { left: 52, top: 505, width: 860, height: 72 }, 30, {
      color: C.muted,
    });
    addRect(slide, { left: 1128, top: 42, width: 100, height: 12 }, C.accentStrong);
    setNotes(slide, "Communication job: by the end, students should distinguish prediction error from predictive uncertainty and be able to audit a CFD surrogate distribution on untouched physical cases.", [
      "https://github.com/Ehsan-Roohi/FlowMLLab",
    ]);
  }

  // 2 — Motivation.
  {
    const slide = newSlide(presentation);
    addHeader(slide, "A point estimate cannot express confidence", 2);
    addKicker(slide, "Same mean error, different scientific risk", 54, 154, 560);
    addText(slide, "ŷ(x)", { left: 86, top: 270, width: 210, height: 90 }, 64, { bold: true, alignment: "center" });
    addText(slide, "One best guess", { left: 84, top: 370, width: 220, height: 40 }, 25, { alignment: "center", color: C.muted });
    addLine(slide, 395, 183, 0.01, C.rule, 1);
    addRect(slide, { left: 474, top: 226, width: 680, height: 94 }, C.paleBlue);
    addLine(slide, 500, 273, 628, C.accentStrong, 5);
    addDot(slide, 805, 263, C.ink, 20);
    addText(slide, "narrow distribution", { left: 474, top: 330, width: 320, height: 40 }, 24, { color: C.muted });
    addRect(slide, { left: 474, top: 430, width: 680, height: 94 }, "#F6F6F6");
    addLine(slide, 500, 477, 628, C.warn, 5);
    addDot(slide, 805, 467, C.ink, 20);
    addText(slide, "wide distribution", { left: 474, top: 534, width: 320, height: 40 }, 24, { color: C.muted });
    addText(slide, "The mean can be identical while the decision should change.", { left: 474, top: 595, width: 680, height: 48 }, 28, { bold: true });
    setNotes(slide, "Ask students which prediction they would trust near a wall or operating limit. Emphasize that RMSE alone does not communicate the width or calibration of a predictive distribution.", [
      "https://doi.org/10.1198/016214506000001437",
    ]);
  }

  // 3 — Learning bridge.
  {
    const slide = newSlide(presentation);
    addHeader(slide, "Week 2.1 extends supervised regression", 3);
    addText(slide, "Week 2 already gives us", { left: 54, top: 168, width: 410, height: 48 }, 30, { bold: true });
    addText(slide, "features and targets\nscaling\nlosses\ncase-wise splits", { left: 54, top: 240, width: 380, height: 230 }, 30, { color: C.muted });
    addRect(slide, { left: 486, top: 152, width: 8, height: 430 }, C.accentStrong);
    addText(slide, "This increment adds", { left: 548, top: 168, width: 470, height: 48 }, 30, { bold: true });
    addText(slide, "1  write an observation model\n2  compute posterior prediction\n3  score the full distribution\n4  calibrate without touching blind cases", { left: 548, top: 240, width: 624, height: 250 }, 30, { color: C.ink });
    addText(slide, "No new neural architecture is required.", { left: 548, top: 526, width: 610, height: 56 }, 28, { bold: true, color: C.accentStrong });
    setNotes(slide, "This is intentionally Week 2.1: the lesson depends on Week 2 regression and splitting concepts, then prepares students to interpret uncertainty in later surrogate models.", [
      "https://github.com/Ehsan-Roohi/FlowMLLab/blob/main/COURSE_MAP.md",
    ]);
  }

  // 4 — Taxonomy.
  {
    const slide = newSlide(presentation);
    addHeader(slide, "Uncertainty has three different causes", 4);
    const columns = [54, 456, 858];
    const colors = [C.accent, C.accentStrong, C.warn];
    const labels = [
      ["ALEATORIC", "Noise or unresolved variability", "More data may not remove it."],
      ["EPISTEMIC", "Limited knowledge of the map", "More informative data can reduce it."],
      ["DISCREPANCY", "The simulator misses reality", "A confident surrogate can still be wrong."],
    ];
    columns.forEach((left, i) => {
      addRect(slide, { left, top: 170, width: 350, height: 12 }, colors[i]);
      addText(slide, labels[i][0], { left, top: 216, width: 350, height: 40 }, 20, { bold: true, color: colors[i] });
      addText(slide, labels[i][1], { left, top: 286, width: 340, height: 100 }, 34, { bold: true });
      addText(slide, labels[i][2], { left, top: 438, width: 330, height: 100 }, 26, { color: C.muted });
    });
    addText(slide, "Do not collapse these mechanisms into one unexplained error bar.", { left: 54, top: 610, width: 1120, height: 42 }, 28, { bold: true });
    setNotes(slide, "Use measurement noise, sparse Reynolds-number sampling, and CFD model-form error as concrete examples. The distinction determines what a reported interval can legitimately mean.", [
      "https://doi.org/10.1111/1467-9868.00294",
      "https://gaussianprocess.org/gpml/",
    ]);
  }

  // 5 — Likelihood and loss.
  {
    const slide = newSlide(presentation);
    addHeader(slide, "A likelihood turns loss into a modeling choice", 5);
    addText(slide, "y = fθ(x) + ε", { left: 66, top: 188, width: 500, height: 100 }, 64, { bold: true });
    addText(slide, "ε ~ 𝒩(0, σ²)", { left: 66, top: 312, width: 500, height: 80 }, 50, { color: C.accentStrong });
    addRect(slide, { left: 610, top: 174, width: 568, height: 288 }, C.panel);
    addKicker(slide, "Gaussian observation model", 646, 212, 450);
    addText(slide, "−log p(y | θ, σ²)\n= constant + Σ(y − fθ(x))² / 2σ²", { left: 646, top: 275, width: 492, height: 130 }, 34, { bold: true });
    addText(slide, "MSE is therefore not neutral: it encodes constant-variance Gaussian residuals.", { left: 66, top: 520, width: 1080, height: 82 }, 31, { bold: true });
    setNotes(slide, "Derive the negative log likelihood up to an additive constant. Ask when constant variance is implausible in a CFD field—for example, near steep gradients or sparse operating regimes.", [
      "https://gaussianprocess.org/gpml/",
    ]);
  }

  // 6 — Bayesian update: selected Codex Grid slide-17 silhouette.
  {
    const slide = newSlide(presentation);
    addHeader(slide, "Bayesian linear regression is an exact update", 6);
    addLine(slide, 70, 340, 1110, C.ink, 2);
    const xs = [70, 492, 914];
    const labels = ["PRIOR", "LIKELIHOOD", "POSTERIOR"];
    const heads = ["Plausible before data", "Evidence from data", "Plausible after data"];
    const bodies = ["w ~ 𝒩(m₀, S₀)", "y | w ~ 𝒩(Φw, σ²I)", "w | y ~ 𝒩(mₙ, Sₙ)"];
    xs.forEach((x, i) => {
      addDot(slide, x, 331, i === 2 ? C.accentStrong : C.ink, 18);
      addText(slide, labels[i], { left: x, top: 274, width: 230, height: 30 }, 18, { bold: true, color: i === 2 ? C.accentStrong : C.muted });
      addText(slide, heads[i], { left: x, top: 390, width: 315, height: 72 }, 24, { bold: true });
      addText(slide, bodies[i], { left: x, top: 496, width: 315, height: 52 }, 26, { color: C.muted });
    });
    addText(slide, "Posterior precision = prior precision + data precision", { left: 70, top: 600, width: 1110, height: 42 }, 28, { bold: true });
    setNotes(slide, "The notebook computes this update for a noisy Poiseuille velocity profile. The analytic solution lets students inspect every assumption before moving to Gaussian processes.", [
      "https://gaussianprocess.org/gpml/",
      "https://github.com/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week02_1/Probabilistic_UQ_CFD.ipynb",
    ]);
  }

  // 7 — Predictive decomposition.
  {
    const slide = newSlide(presentation);
    addHeader(slide, "Prediction uncertainty has two terms", 7);
    addText(slide, "Var[y* | x*, D]", { left: 72, top: 208, width: 430, height: 72 }, 50, { bold: true });
    addText(slide, "=", { left: 500, top: 208, width: 58, height: 72 }, 50, { alignment: "center" });
    addRect(slide, { left: 575, top: 174, width: 265, height: 180 }, C.paleBlue);
    addText(slide, "σ²", { left: 612, top: 205, width: 190, height: 62 }, 50, { bold: true, alignment: "center" });
    addText(slide, "observation noise", { left: 602, top: 282, width: 210, height: 44 }, 22, { alignment: "center", color: C.muted });
    addText(slide, "+", { left: 848, top: 208, width: 58, height: 72 }, 50, { alignment: "center" });
    addRect(slide, { left: 924, top: 174, width: 286, height: 180 }, C.panel);
    addText(slide, "φ*ᵀ Sₙ φ*", { left: 946, top: 205, width: 242, height: 62 }, 44, { bold: true, alignment: "center" });
    addText(slide, "parameter uncertainty", { left: 944, top: 282, width: 246, height: 44 }, 22, { alignment: "center", color: C.muted });
    addText(slide, "Report which terms are included. A posterior over weights does not automatically include simulator discrepancy.", { left: 72, top: 466, width: 1110, height: 112 }, 34, { bold: true });
    setNotes(slide, "Separate uncertainty in a latent mean from observation noise. Then revisit the taxonomy: neither term here necessarily represents CFD-to-reality discrepancy.", [
      "https://gaussianprocess.org/gpml/",
      "https://doi.org/10.1111/1467-9868.00294",
    ]);
  }

  // 8 — POD-GP pipeline.
  {
    const slide = newSlide(presentation);
    addHeader(slide, "POD compresses field uncertainty", 8);
    const x = [54, 292, 530, 768, 1006];
    const heads = ["CFD cases", "POD basis", "Coefficients", "Gaussian process", "Field distribution"];
    const bodies = ["u(x; Re)", "Φ₁ … Φᵣ", "a₁(Re)…aᵣ(Re)", "p(aⱼ* | Re*)", "μu(x), σu(x)"];
    x.forEach((left, i) => {
      addRect(slide, { left, top: 238, width: 190, height: 190 }, i === 3 ? C.paleBlue : C.panel);
      addText(slide, `${i + 1}`, { left: left + 18, top: 258, width: 42, height: 42 }, 22, { bold: true, color: C.accentStrong });
      addText(slide, heads[i], { left: left + 18, top: 316, width: 154, height: 58 }, 23, { bold: true, alignment: "center" });
      addText(slide, bodies[i], { left: left + 14, top: 386, width: 162, height: 34 }, 19, { alignment: "center", color: C.muted });
      if (i < x.length - 1) {
        addLine(slide, left + 190, 333, 46, C.accentStrong, 3);
      }
    });
    addText(slide, "Low-rank structure reduces a spatial field to a few probabilistic coefficient functions.", { left: 54, top: 516, width: 1110, height: 64 }, 32, { bold: true });
    setNotes(slide, "Explain that FlowMLLab fits independent fixed-kernel Gaussian processes to a rank-4 velocity POD representation. Independence is a simplifying assumption, not a general truth about modal coefficients.", [
      "https://doi.org/10.1146/annurev.fl.25.010193.002543",
      "https://gaussianprocess.org/gpml/",
      "https://github.com/Ehsan-Roohi/FlowMLLab/blob/main/results/probabilistic_uq/protocol.json",
    ]);
  }

  // 9 — Leakage-free protocol: selected Codex Grid slide-17 silhouette.
  {
    const slide = newSlide(presentation);
    addHeader(slide, "Calibration must stop before the blind gate", 9);
    addLine(slide, 70, 340, 1110, C.ink, 2);
    const xs = [70, 492, 914];
    const labels = ["DEVELOP", "CALIBRATE", "BLIND TEST"];
    const heads = ["Fit POD + GP", "Scale interval width once", "Measure without tuning"];
    const bodies = ["Re = 100, 150, 200, 225, 250, 350, 400", "Re = 300", "Re = 175, 275, 375"];
    xs.forEach((left, i) => {
      addDot(slide, left, 331, i === 2 ? C.warn : C.accentStrong, 18);
      addText(slide, labels[i], { left, top: 274, width: 240, height: 30 }, 18, { bold: true, color: i === 2 ? C.warn : C.accentStrong });
      addText(slide, heads[i], { left, top: 390, width: 315, height: 72 }, 28, { bold: true });
      addText(slide, bodies[i], { left, top: 492, width: 315, height: 78 }, 23, { color: C.muted });
    });
    addText(slide, "The split unit is a complete Reynolds-number case—not individual grid nodes.", { left: 70, top: 608, width: 1110, height: 42 }, 28, { bold: true });
    setNotes(slide, "Make students identify every decision that must be frozen before opening Re=175, 275, and 375. Pointwise random splits would leak the same physical solution across partitions.", [
      "https://github.com/Ehsan-Roohi/FlowMLLab/blob/main/results/probabilistic_uq/protocol.json",
    ]);
  }

  // 10 — Three goals.
  {
    const slide = newSlide(presentation);
    addHeader(slide, "Judge accuracy, calibration, and sharpness", 10);
    metricPanel(slide, 54, "01", "Accuracy\nIs the predictive mass near truth?", C.accentStrong);
    metricPanel(slide, 465, "02", "Calibration\nDo nominal levels match frequency?", C.accent);
    metricPanel(slide, 876, "03", "Sharpness\nAre intervals as narrow as justified?", C.warn);
    addText(slide, "Optimizing only one can produce a useless distribution.", { left: 54, top: 175, width: 1110, height: 80 }, 36, { bold: true });
    setNotes(slide, "A very wide interval can cover everything and still be uninformative. A very sharp interval can look decisive and still be badly miscalibrated.", [
      "https://doi.org/10.1198/016214506000001437",
    ]);
  }

  // 11 — Scores.
  {
    const slide = newSlide(presentation);
    addHeader(slide, "Score the distribution, not only its mean", 11);
    addKicker(slide, "Gaussian negative log likelihood", 58, 164, 500);
    addText(slide, "NLL = ½ log(2πσ²) + ½(y−μ)²/σ²", { left: 58, top: 218, width: 1120, height: 72 }, 40, { bold: true });
    addLine(slide, 58, 326, 1120, C.rule, 1);
    addKicker(slide, "Continuous ranked probability score", 58, 364, 590);
    addText(slide, "CRPS compares the predictive CDF with the observed step CDF.", { left: 58, top: 418, width: 1080, height: 72 }, 34, { bold: true });
    addRect(slide, { left: 58, top: 536, width: 1120, height: 82 }, C.paleBlue);
    addText(slide, "Lower is better. Both scores penalize confident errors.", { left: 84, top: 557, width: 1068, height: 42 }, 29, { bold: true });
    setNotes(slide, "NLL can be dominated by a few overconfident misses. CRPS has the same units as the target and integrates distribution quality across thresholds. Use both alongside coverage and width.", [
      "https://doi.org/10.1198/016214506000001437",
    ]);
  }

  // 12 — Retained evidence.
  {
    const slide = newSlide(presentation);
    addHeader(slide, "Blind evidence beats a polished uncertainty story", 12);
    addRect(slide, { left: 44, top: 142, width: 1192, height: 500 }, "#FAFAFA", { lineFill: C.rule, lineWidth: 1 });
    slide.images.add({
      blob: evidenceImageBytes,
      contentType: "image/png",
      alt: "FlowMLLab probabilistic UQ validation figure showing blind error and calibration behavior",
      fit: "contain",
      position: { left: 60, top: 154, width: 1160, height: 476 },
    });
    setNotes(slide, "Walk from mean error to coverage. The POD–GP mean slightly improves on interpolation, but its uncertainty bands are not calibrated across all blind Reynolds cases.", [
      "https://github.com/Ehsan-Roohi/FlowMLLab/blob/main/results/probabilistic_uq/probabilistic_uq_validation.png",
      "https://github.com/Ehsan-Roohi/FlowMLLab/blob/main/results/probabilistic_uq/blind_metrics.csv",
    ]);
  }

  // 13 — Metrics and retained failure: selected Codex Grid slide-19 silhouette.
  {
    const slide = newSlide(presentation);
    addHeader(slide, "The mean improved; coverage did not", 13);
    addText(slide, "Validation-only scaling increased blind 90% coverage, but did not approach the nominal target.", { left: 54, top: 154, width: 1120, height: 90 }, 31, { bold: true });
    metricPanel(slide, 54, "0.333%", "Mean POD–GP relative L₂ error", C.accentStrong);
    metricPanel(slide, 465, "0.414%", "Mean interpolation relative L₂ error", C.accent);
    metricPanel(slide, 876, "65.2%", "Calibrated coverage at nominal 90%", C.warn);
    addRect(slide, { left: 876, top: 598, width: 350, height: 8 }, C.warn);
    setNotes(slide, "Retain the failure. The calibrated aggregate coverage is 65.2%, up from 61.5%, but far below 90%. Spatial nodes are correlated, so these values are descriptive rather than finite-sample coverage guarantees.", [
      "https://github.com/Ehsan-Roohi/FlowMLLab/blob/main/results/probabilistic_uq/summary.json",
      "https://github.com/Ehsan-Roohi/FlowMLLab/blob/main/results/probabilistic_uq/calibration.csv",
    ]);
  }

  // 14 — Close.
  {
    const slide = newSlide(presentation);
    addKicker(slide, "Week 2.1 • exit checklist", 52, 48, 520);
    addText(slide, "Make the uncertainty claim\nas auditable as the mean", { left: 52, top: 156, width: 990, height: 160 }, 62, { bold: true });
    addText(slide, "□ State what is random\n□ Keep physical cases intact\n□ Freeze calibration before blind testing\n□ Report accuracy, coverage, and width\n□ Preserve under-coverage as evidence", { left: 54, top: 360, width: 730, height: 230 }, 29);
    addRect(slide, { left: 840, top: 338, width: 370, height: 292 }, C.paleBlue);
    addText(slide, "Exit prompt", { left: 876, top: 374, width: 294, height: 38 }, 22, { bold: true, color: C.accentStrong });
    addText(slide, "What new data would separate epistemic uncertainty from CFD model discrepancy?", { left: 876, top: 438, width: 294, height: 160 }, 24, { bold: true });
    setNotes(slide, "Close by having students answer the exit prompt before opening Week 4 surrogate models. The notebook is the executable companion to this lecture.", [
      "https://github.com/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week02_1/Probabilistic_UQ_CFD.ipynb",
      "https://gaussianprocess.org/gpml/",
      "https://doi.org/10.1198/016214506000001437",
      "https://doi.org/10.1111/1467-9868.00294",
    ]);
  }

  const inspection = await presentation.inspect({
    kind: "slide,textbox,shape,image,chart,notes",
    maxChars: 12000,
  });
  await fs.writeFile(path.join(previewDir, "deck-inspect.ndjson"), inspection.ndjson, "utf8");

  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    await writeBlob(path.join(previewDir, `${stem}.png`), await presentation.export({ slide, format: "png", scale: 2 }));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(previewDir, `${stem}.layout.json`), await layout.text(), "utf8");
  }

  await writeBlob(
    path.join(previewDir, "week02_1_probabilistic_uq_montage.webp"),
    await presentation.export({ format: "webp", montage: true, scale: 1 }),
  );

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(outputPath);
  console.log(`Wrote ${outputPath}`);
  console.log(`Preview ${previewDir}`);
  process.exitCode = 0;
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
