#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
import re
import uuid
from pathlib import Path


ROOT = Path("/Users/stevendongyu/Desktop/Project/ai_infra/PoLar")
SKILL = Path("/Users/stevendongyu/.agents/skills/aico-ppt")
TEMPLATE = SKILL / "assets/tech-share-deck.html"
OUT = ROOT / "outputs/presentations/layer_transformer_huawei_aico.html"

spec = importlib.util.spec_from_file_location("eb", SKILL / "scripts/edit-bundle.py")
eb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eb)


def page_id() -> str:
    return "page-" + uuid.uuid4().hex


def editor_id() -> str:
    return "element-" + uuid.uuid4().hex


def slide(label: str, idx: int, body: str, active: bool = False) -> str:
    active_attr = ' data-active=""' if active else ""
    return f'''<div class="slide-fit" data-idx="{idx}"{active_attr}><div class="slide-canvas">
      <section data-label="{label}" data-page-id="{page_id()}" style="width:100%;height:100%;position:relative;overflow:hidden;font-family:'Noto Sans SC','Microsoft YaHei',sans-serif;background:#f7f8fb;color:#171717;">
{body}
      </section>
    </div></div>'''


def common_bg() -> str:
    return '''
        <div style="position:absolute;inset:0;background:radial-gradient(circle at 86% 10%,rgba(196,0,22,.18),transparent 34%),linear-gradient(135deg,#0d1726 0%,#172033 48%,#111827 100%);"></div>
        <div style="position:absolute;right:-150px;top:-110px;width:520px;height:520px;border:1px solid rgba(255,255,255,.16);border-radius:50%;"></div>
        <div style="position:absolute;right:96px;bottom:88px;width:360px;height:360px;border:1px solid rgba(255,255,255,.11);border-radius:50%;"></div>
        <div style="position:absolute;left:0;top:0;width:14px;height:100%;background:#c7000b;"></div>
    '''


cover = slide(
    "封面",
    0,
    f'''
        {common_bg()}
        <div data-editor-id="{editor_id()}" style="position:absolute;left:132px;top:116px;font-size:24px;letter-spacing:.14em;color:rgba(255,255,255,.72);font-weight:700;">AI INFRA & SYSTEMS ARCHITECTURE</div>
        <div data-editor-id="{editor_id()}" style="position:absolute;left:132px;top:190px;width:1260px;font-size:70px;line-height:1.18;font-weight:800;color:#fff;letter-spacing:-.02em;">动态 Layer Transformer 的<br/>高性能服务化推理挑战</div>
        <div data-editor-id="{editor_id()}" style="position:absolute;left:132px;top:386px;width:1120px;font-size:31px;line-height:1.45;color:rgba(255,255,255,.82);">POLAR、Loop Transformer、Skip-layer 与 Mixture-of-Depth 正在打破固定层序 Serving 的基础假设。</div>
        <div data-editor-id="{editor_id()}" style="position:absolute;left:132px;bottom:138px;display:flex;gap:18px;align-items:center;">
          <span style="display:inline-block;background:#c7000b;color:#fff;font-size:24px;font-weight:800;padding:13px 22px;border-radius:6px;">EXEC_STATUS</span>
          <span style="display:inline-block;border:1px solid rgba(255,255,255,.28);color:rgba(255,255,255,.9);font-size:23px;padding:12px 22px;border-radius:6px;">BREAKING_CHANGES</span>
          <span style="display:inline-block;border:1px solid rgba(255,255,255,.28);color:rgba(255,255,255,.9);font-size:23px;padding:12px 22px;border-radius:6px;">BATCH / KV / SLA</span>
        </div>
        <div data-editor-id="{editor_id()}" style="position:absolute;right:118px;bottom:92px;width:430px;padding:26px 30px;border-left:5px solid #c7000b;background:rgba(255,255,255,.08);backdrop-filter:blur(6px);color:#fff;font-size:26px;line-height:1.45;font-weight:700;">从 fixed-depth serving 走向 layer-programmable serving</div>
    ''',
    True,
)

content = slide(
    "核心挑战",
    1,
    f'''
        <div style="position:absolute;left:0;top:0;width:100%;height:100%;background:#f5f7fb;"></div>
        <div style="position:absolute;left:0;top:0;width:100%;height:18px;background:#c7000b;"></div>
        <div data-editor-id="{editor_id()}" style="position:absolute;left:88px;top:54px;font-size:34px;font-weight:800;color:#171717;">动态路径带来精度收益，也破坏固定层 Serving 的系统假设</div>
        <div data-editor-id="{editor_id()}" style="position:absolute;right:96px;top:62px;font-size:22px;color:#667085;font-family:'JetBrains Mono',monospace;">Layer-Programmable Transformer Serving</div>

        <div style="position:absolute;left:88px;top:132px;width:492px;height:772px;background:#fff;border:1px solid #d8dee9;border-radius:18px;box-shadow:0 10px 30px rgba(16,24,40,.08);"></div>
        <div data-editor-id="{editor_id()}" style="position:absolute;left:124px;top:172px;font-size:28px;font-weight:800;color:#171717;">01 · 固定层 Serving 假设</div>
        <div data-editor-id="{editor_id()}" style="position:absolute;left:124px;top:224px;width:410px;font-size:22px;line-height:1.55;color:#475467;">现有高性能 LLM 推理默认 batch 内请求按同一层序锁步推进。</div>
        <div data-editor-id="{editor_id()}" style="position:absolute;left:124px;top:328px;width:402px;height:238px;border-radius:14px;background:#101828;color:#e6edf7;font-family:'JetBrains Mono',monospace;font-size:24px;line-height:1.38;padding:28px 30px;box-shadow:inset 0 0 0 1px rgba(255,255,255,.08);">Batch token step t<br/>  layer 0<br/>  layer 1<br/>  layer 2<br/>  ...<br/>  layer 35</div>
        <div data-editor-id="{editor_id()}" style="position:absolute;left:124px;top:604px;width:410px;font-size:22px;line-height:1.7;color:#344054;">
          <b style="color:#171717;">系统优化基石</b><br/>
          固定算子图<br/>
          KV Cache 按 Layer ID 管理<br/>
          Continuous Batching / Paged Attention / Graph
        </div>

        <div style="position:absolute;left:638px;top:132px;width:542px;height:772px;background:#fff;border:1px solid #d8dee9;border-radius:18px;box-shadow:0 10px 30px rgba(16,24,40,.08);"></div>
        <div data-editor-id="{editor_id()}" style="position:absolute;left:674px;top:172px;font-size:28px;font-weight:800;color:#171717;">02 · 动态 Execution 打破假设</div>
        <div data-editor-id="{editor_id()}" style="position:absolute;left:674px;top:224px;width:462px;font-size:22px;line-height:1.55;color:#475467;">不同请求、不同候选 path 拥有不同的 layer execution program。</div>
        <div data-editor-id="{editor_id()}" style="position:absolute;left:674px;top:326px;width:466px;height:206px;border-radius:14px;background:#172033;color:#e6edf7;font-family:'JetBrains Mono',monospace;font-size:23px;line-height:1.45;padding:28px 30px;">Req A: 0 1 2 3 4 ... 35<br/>Req B: 0 1 2 3 3 4 ...<br/>Req C: 0 1 4 5 6 ... 35<br/>Req D: 0 1 2 3 4 4 5 ...</div>
        <div data-editor-id="{editor_id()}" style="position:absolute;left:674px;top:570px;width:466px;padding:28px 30px;border-left:6px solid #c7000b;border-radius:12px;background:#fff5f5;color:#7a271a;font-size:24px;line-height:1.52;font-weight:800;">核心冲突：算法收益来自动态路径选择，但工程侧会打散 batch 同步、破坏 KV cache 语义，并让单 token 成本不可预测。</div>

        <div style="position:absolute;left:1238px;top:132px;width:594px;height:772px;background:#fff;border:1px solid #d8dee9;border-radius:18px;box-shadow:0 10px 30px rgba(16,24,40,.08);"></div>
        <div data-editor-id="{editor_id()}" style="position:absolute;left:1274px;top:172px;font-size:28px;font-weight:800;color:#171717;">03 · 四个核心系统挑战</div>
        <div data-editor-id="{editor_id()}" style="position:absolute;left:1274px;top:234px;width:500px;display:grid;grid-template-columns:1fr;gap:22px;">
          <div style="border-left:6px solid #c7000b;padding:0 0 0 20px;"><div style="font-size:23px;font-weight:800;color:#171717;">Path-Aware Batching</div><div style="margin-top:8px;font-size:20px;line-height:1.45;color:#475467;">按 Layer、Segment 或 Prefix 聚合请求，降低动态路径导致的 batch 碎片化。</div></div>
          <div style="border-left:6px solid #2e90fa;padding:0 0 0 20px;"><div style="font-size:23px;font-weight:800;color:#171717;">Execution-Aware KV Cache</div><div style="margin-top:8px;font-size:20px;line-height:1.45;color:#475467;">Repeat layer 的 hidden state 已改变，cache key 需从 layer id 扩展到 execution step。</div></div>
          <div style="border-left:6px solid #f79009;padding:0 0 0 20px;"><div style="font-size:23px;font-weight:800;color:#171717;">Cost-Aware Scheduling</div><div style="margin-top:8px;font-size:20px;line-height:1.45;color:#475467;">不同 path 的 total layer executions 不同，调度器需要控制吞吐与 P99 latency。</div></div>
          <div style="border-left:6px solid #12b76a;padding:0 0 0 20px;"><div style="font-size:23px;font-weight:800;color:#171717;">Top-k Candidate Serving</div><div style="margin-top:8px;font-size:20px;line-height:1.45;color:#475467;">多路径提升准确率，但需要 path 共享、early-stop 与轻量 verifier 限制总时延。</div></div>
        </div>

        <div data-editor-id="{editor_id()}" style="position:absolute;left:88px;bottom:50px;width:1744px;height:74px;background:#111827;border-radius:12px;color:#fff;font-size:30px;font-weight:800;display:flex;align-items:center;padding:0 36px;">挑战定义：让动态层执行 Transformer 像固定层 Transformer 一样高效、稳定、可预测地服务化部署。</div>
    ''',
)

thanks = slide(
    "结语页",
    2,
    f'''
        {common_bg()}
        <div data-editor-id="{editor_id()}" style="position:absolute;left:132px;top:124px;font-size:28px;letter-spacing:.12em;color:rgba(255,255,255,.74);font-weight:800;">MISSION</div>
        <div data-editor-id="{editor_id()}" style="position:absolute;left:132px;top:208px;width:1280px;font-size:58px;line-height:1.25;font-weight:850;color:#fff;">构建软硬协同运行时，<br/>让动态层模型具备生产级服务化能力</div>
        <div data-editor-id="{editor_id()}" style="position:absolute;left:132px;top:402px;width:1180px;font-size:28px;line-height:1.62;color:rgba(255,255,255,.82);">面向 POLAR 类 layer execution program，系统需要同时解决路径感知组 batch、执行步感知 KV cache、成本感知调度和 Top-k 候选服务化。</div>
        <div data-editor-id="{editor_id()}" style="position:absolute;left:132px;bottom:132px;display:grid;grid-template-columns:repeat(4,260px);gap:24px;">
          <div style="height:112px;border-top:5px solid #c7000b;background:rgba(255,255,255,.08);padding:22px 24px;color:#fff;font-size:23px;font-weight:800;">Batch<br/><span style="font-size:19px;font-weight:500;color:rgba(255,255,255,.68);">按路径聚合</span></div>
          <div style="height:112px;border-top:5px solid #c7000b;background:rgba(255,255,255,.08);padding:22px 24px;color:#fff;font-size:23px;font-weight:800;">Cache<br/><span style="font-size:19px;font-weight:500;color:rgba(255,255,255,.68);">按执行步索引</span></div>
          <div style="height:112px;border-top:5px solid #c7000b;background:rgba(255,255,255,.08);padding:22px 24px;color:#fff;font-size:23px;font-weight:800;">Scheduler<br/><span style="font-size:19px;font-weight:500;color:rgba(255,255,255,.68);">按真实成本决策</span></div>
          <div style="height:112px;border-top:5px solid #c7000b;background:rgba(255,255,255,.08);padding:22px 24px;color:#fff;font-size:23px;font-weight:800;">Selector<br/><span style="font-size:19px;font-weight:500;color:rgba(255,255,255,.68);">按收益选择候选</span></div>
        </div>
    ''',
)


def replace_nav(s: str) -> str:
    ns = s.find("const nav = [")
    ne = s.find("];", ns)
    nav = """const nav = [
      { i:0, code:'封面', label:'封面' },
      { i:1, code:'挑战', label:'核心挑战' },
      { i:2, code:'使命', label:'结语页' },
    ];"""
    return s[:ns] + nav + s[ne + 2 :]


def replace_chapters(s: str) -> str:
    cs = s.find("const chapters = [")
    ce = s.find("];", cs)
    chapters = """const chapters = [
      { name:'01 · 动态层服务化挑战', start:1 },
    ];"""
    return s[:cs] + chapters + s[ce + 2 :]


def main() -> None:
    lines = eb.load(TEMPLATE)
    s = eb.get_template(lines)

    stage_start = s.find('<div class="stage" id="stage">')
    if stage_start < 0:
        raise RuntimeError("stage not found")
    first_slide = s.find('<div class="slide-fit"', stage_start)
    labels = re.findall(r'<section data-label="([^"]+)"', s)
    last_label = labels[-1]
    last_section = s.find(f'<section data-label="{last_label}"')
    stage_content_end = s.find("</div></div>", s.find("</section>", last_section)) + len("</div></div>")

    custom = "\n\n    " + "\n\n    ".join([cover, content, thanks]) + "\n\n  "
    s = s[:first_slide] + custom + s[stage_content_end:]
    s = replace_nav(s)
    s = replace_chapters(s)

    eb.set_template(lines, s)
    eb.save(OUT, lines)
    eb.verify(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
