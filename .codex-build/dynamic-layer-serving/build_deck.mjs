import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const { SKILL_DIR, TMP_DIR, FINAL_PPTX, RUNTIME_PYTHON } = process.env;
if (!path.isAbsolute(SKILL_DIR ?? "") || !path.isAbsolute(TMP_DIR ?? "") || !path.isAbsolute(FINAL_PPTX ?? "")) {
  throw new Error("Set absolute SKILL_DIR, TMP_DIR and FINAL_PPTX");
}

const workspaceDir = "/Users/stevendongyu/Desktop/Project/ai_infra/PoLar";
const {
  resolvePresentationFont,
  finalizePresentation,
} = await import(pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href);

await fs.mkdir(TMP_DIR, { recursive: true });
await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });

const family = resolvePresentationFont();
const presentation = Presentation.create({
  slideSize: { width: 1280, height: 720 },
});

const slide = presentation.slides.add();
slide.background.fill = "#F7F9FC";

function addText(slide, text, position, style = {}) {
  const box = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { fill: "none", width: 0 },
  });
  box.text = text;
  box.text.style = {
    typeface: family,
    fontSize: style.fontSize ?? 20,
    bold: style.bold ?? false,
    color: style.color ?? "#172033",
    autoFit: "shrinkText",
    ...style,
  };
  return box;
}

function addPanel(slide, position, opts = {}) {
  return slide.shapes.add({
    geometry: "roundRect",
    position,
    fill: opts.fill ?? "#FFFFFF",
    line: { style: "solid", fill: opts.line ?? "#D7DFEA", width: opts.width ?? 1 },
    borderRadius: 18,
    shadow: opts.shadow ?? "shadow-sm",
  });
}

function addCodeBox(slide, lines, position, opts = {}) {
  const box = slide.shapes.add({
    geometry: "roundRect",
    position,
    fill: opts.fill ?? "#101828",
    line: { style: "solid", fill: opts.line ?? "#26344D", width: 1 },
    borderRadius: 12,
  });
  box.text = lines.join("\n");
  box.text.style = {
    typeface: "Menlo",
    fontSize: opts.fontSize ?? 15,
    color: opts.color ?? "#D6E4FF",
    autoFit: "shrinkText",
  };
  return box;
}

function addBullet(slide, label, body, left, top, accent) {
  const dot = slide.shapes.add({
    geometry: "ellipse",
    position: { left, top: top + 5, width: 10, height: 10 },
    fill: accent,
    line: { fill: accent, width: 0 },
  });
  const title = addText(slide, label, { left: left + 20, top, width: 236, height: 24 }, {
    fontSize: 17,
    bold: true,
    color: "#172033",
  });
  const desc = addText(slide, body, { left: left + 20, top: top + 25, width: 248, height: 50 }, {
    fontSize: 13.5,
    color: "#475467",
  });
  return [dot, title, desc];
}

// Header
addText(slide, "动态 Layer Transformer 的服务化推理挑战", { left: 54, top: 34, width: 760, height: 48 }, {
  fontSize: 34,
  bold: true,
  color: "#111827",
});
addText(slide, "目标：在 skip / repeat / segment execution 下，维持固定层模型的吞吐、时延和可预测性", {
  left: 56, top: 82, width: 860, height: 28,
}, {
  fontSize: 17,
  color: "#475467",
});

const tag = slide.shapes.add({
  geometry: "roundRect",
  position: { left: 978, top: 40, width: 244, height: 40 },
  fill: "#EAF1FF",
  line: { style: "solid", fill: "#B7CCFF", width: 1 },
  borderRadius: 20,
});
tag.text = "AI Infra 难题挑战";
tag.text.style = { typeface: family, fontSize: 18, bold: true, color: "#1849A9", autoFit: "shrinkText" };

// Panels
addPanel(slide, { left: 54, top: 138, width: 338, height: 452 }, { fill: "#FFFFFF" });
addPanel(slide, { left: 429, top: 138, width: 370, height: 452 }, { fill: "#FFFFFF" });
addPanel(slide, { left: 836, top: 138, width: 390, height: 452 }, { fill: "#FFFFFF" });

addText(slide, "固定层 Serving 假设", { left: 78, top: 164, width: 286, height: 30 }, {
  fontSize: 23,
  bold: true,
  color: "#111827",
});
addText(slide, "现有 LLM serving 默认 batch 内请求按同一层序执行。", {
  left: 80, top: 198, width: 274, height: 44,
}, {
  fontSize: 15,
  color: "#475467",
});

addCodeBox(slide, [
  "Batch token step t",
  "  layer 0",
  "  layer 1",
  "  layer 2",
  "  ...",
  "  layer 35",
], { left: 82, top: 258, width: 256, height: 174 }, { fontSize: 16 });

addText(slide, "系统优化建立在这些稳定假设上：", {
  left: 82, top: 454, width: 270, height: 24,
}, {
  fontSize: 15,
  bold: true,
  color: "#344054",
});
addText(slide, "固定 layer graph\nKV cache 按 layer 管理\n调度只估计 token 与 memory", {
  left: 96, top: 484, width: 260, height: 70,
}, {
  fontSize: 15,
  color: "#475467",
});

addText(slide, "POLAR 类模型打破假设", { left: 454, top: 164, width: 304, height: 30 }, {
  fontSize: 23,
  bold: true,
  color: "#111827",
});
addText(slide, "模型执行不再是固定 forward，而是每个请求拥有自己的 layer execution program。", {
  left: 456, top: 198, width: 306, height: 52,
}, {
  fontSize: 15,
  color: "#475467",
});

addCodeBox(slide, [
  "Req A: 0 1 2 3 4 ... 35",
  "Req B: 0 1 2 3 3 4 ...",
  "Req C: 0 1 4 5 6 ... 35",
  "Req D: 0 1 2 3 4 4 5 ...",
], { left: 458, top: 270, width: 284, height: 130 }, { fontSize: 14.5, fill: "#172033" });

const middleCallout = slide.shapes.add({
  geometry: "roundRect",
  position: { left: 458, top: 426, width: 284, height: 94 },
  fill: "#FFF7E8",
  line: { style: "solid", fill: "#FEDF89", width: 1 },
  borderRadius: 14,
});
middleCallout.text = "精度收益来自动态路径，但动态路径会打散 batch、改变 KV cache 语义，并让单 token 成本变得不可预测。";
middleCallout.text.style = { typeface: family, fontSize: 15, color: "#7A2E0E", bold: true, autoFit: "shrinkText" };

addText(slide, "四个核心系统问题", { left: 862, top: 164, width: 318, height: 30 }, {
  fontSize: 23,
  bold: true,
  color: "#111827",
});

addBullet(slide, "Path-aware batching", "按 layer id、segment 或 program prefix 组 batch，降低动态路径带来的碎片化。", 864, 214, "#2E90FA");
addBullet(slide, "Execution-aware KV cache", "repeat layer 的第二次执行输入不同，cache key 需要从 layer id 扩展到 execution step。", 864, 304, "#12B76A");
addBullet(slide, "Cost-aware scheduling", "不同请求的 layer executions 不同，调度器需要估计成本并控制 P99 latency。", 864, 394, "#F79009");
addBullet(slide, "Top-k candidate serving", "多路径可提升准确率，但需要 selector、early-stop 和共享计算来限制总时延。", 864, 484, "#9E77ED");

// Bottom challenge statement
const footer = slide.shapes.add({
  geometry: "roundRect",
  position: { left: 54, top: 620, width: 1172, height: 58 },
  fill: "#111827",
  line: { fill: "#111827", width: 0 },
  borderRadius: 16,
});
footer.text = "挑战定义：让动态层执行 Transformer 像固定层 Transformer 一样高效、稳定、可预测地服务化部署。";
footer.text.style = {
  typeface: family,
  fontSize: 22,
  bold: true,
  color: "#FFFFFF",
  autoFit: "shrinkText",
};

slide.speakerNotes.textFrame.setText([
  "This slide is based on the user's POLAR serving challenge framing in the current conversation.",
  "The examples are conceptual layer execution paths, not measured production traces.",
  "Key research directions: path-aware batching, execution-aware KV cache, cost-aware scheduling, and top-k candidate serving.",
].join("\n"));

// Export preview and finalize
const preview = await presentation.export({ slide, format: "png", scale: 1 });
await fs.writeFile(path.join(TMP_DIR, "dynamic-layer-serving-slide.png"), new Uint8Array(await preview.arrayBuffer()));
const layout = await slide.export({ format: "layout" });
await fs.writeFile(path.join(TMP_DIR, "dynamic-layer-serving-slide.layout.json"), await layout.text());

const candidatePath = path.join(TMP_DIR, "candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const stagingDir = path.join(workspaceDir, ".codex-finalizer");
await fs.mkdir(stagingDir, { recursive: true });

await finalizePresentation({
  explicitTotalSlideCount: 1,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
  workspaceDir,
  candidatePath,
  finalPath: FINAL_PPTX,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000",
    "--validate-heading-fit",
  ],
  fontPolicy: { basis: "design", families: [family, "Menlo"] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, `${path.basename(FINAL_PPTX)}.validation.json`),
});

console.log(JSON.stringify({
  final: FINAL_PPTX,
  preview: path.join(TMP_DIR, "dynamic-layer-serving-slide.png"),
  layout: path.join(TMP_DIR, "dynamic-layer-serving-slide.layout.json"),
}, null, 2));
