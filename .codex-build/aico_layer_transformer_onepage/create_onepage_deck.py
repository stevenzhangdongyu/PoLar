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
OUT = ROOT / "outputs/presentations/layer_transformer_challenge_huawei_onepage.html"

spec = importlib.util.spec_from_file_location("eb", SKILL / "scripts/edit-bundle.py")
eb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eb)


def uid(prefix: str) -> str:
    return prefix + "-" + uuid.uuid4().hex


def editor_id() -> str:
    return uid("element")


def panel(x: int, y: int, w: int, h: int, num: str, title: str, subtitle: str, body_html: str, accent: str = "#c7000b") -> str:
    return f'''
      <div data-editor-id="{editor_id()}" style="position:absolute;left:{x}px;top:{y}px;width:{w}px;height:{h}px;background:#fff;border:1px solid #d8dee9;border-radius:18px;box-shadow:0 12px 30px rgba(16,24,40,.07);padding:28px 30px;">
        <div style="display:flex;align-items:center;gap:16px;margin-bottom:18px;">
          <div style="width:48px;height:48px;border-radius:50%;background:{accent};color:#fff;display:flex;align-items:center;justify-content:center;font-family:'JetBrains Mono',monospace;font-size:22px;font-weight:850;">{num}</div>
          <div>
            <div style="font-size:28px;font-weight:850;color:#171717;line-height:1.15;">{title}</div>
            <div style="margin-top:5px;font-size:17px;color:#667085;font-family:'JetBrains Mono',monospace;">{subtitle}</div>
          </div>
        </div>
        {body_html}
      </div>
    '''


def main() -> None:
    lines = eb.load(TEMPLATE)
    s = eb.get_template(lines)

    body_bg = '''
      <div style="position:absolute;inset:0;background:#f5f7fb;"></div>
      <div style="position:absolute;left:0;top:0;width:100%;height:18px;background:#c7000b;"></div>
      <div style="position:absolute;right:-160px;top:-160px;width:520px;height:520px;border:1px solid rgba(199,0,11,.16);border-radius:50%;"></div>
      <div style="position:absolute;left:-180px;bottom:-220px;width:620px;height:620px;border:1px solid rgba(23,32,51,.10);border-radius:50%;"></div>
    '''

    background_body = '''
        <div style="font-size:22px;line-height:1.52;color:#475467;">Transformer 推理系统长期假设每个请求按照固定层序执行，围绕固定 graph、固定 KV cache layout 和 continuous batching 做优化。</div>
        <div style="margin-top:22px;border-radius:12px;background:#111827;color:#e6edf7;font-family:'JetBrains Mono',monospace;font-size:18px;line-height:1.42;padding:22px 24px;">Base forward<br/>  layer 0 → layer 1 → ... → layer N</div>
        <div style="margin-top:22px;font-size:22px;line-height:1.52;color:#475467;">POLAR / Loop / Skip / MoD 把 layer path 变成模型能力的一部分，推理系统需要理解动态执行程序。</div>
    '''

    challenge_body = '''
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
          <div style="border-left:5px solid #c7000b;background:#f8fafc;border-radius:10px;padding:15px 16px;"><div style="font-size:20px;font-weight:850;color:#171717;">Batch 变碎</div><div style="margin-top:6px;font-size:18px;line-height:1.38;color:#475467;">不同请求路径不一致，同层同步执行被打散。</div></div>
          <div style="border-left:5px solid #2e90fa;background:#f8fafc;border-radius:10px;padding:15px 16px;"><div style="font-size:20px;font-weight:850;color:#171717;">KV 语义变化</div><div style="margin-top:6px;font-size:18px;line-height:1.38;color:#475467;">repeat layer 的 hidden state 已变化，不能只按 layer id 复用。</div></div>
          <div style="border-left:5px solid #f79009;background:#f8fafc;border-radius:10px;padding:15px 16px;"><div style="font-size:20px;font-weight:850;color:#171717;">成本不可见</div><div style="margin-top:6px;font-size:18px;line-height:1.38;color:#475467;">token 成本还取决于 total layer executions 与 top-k 数量。</div></div>
          <div style="border-left:5px solid #12b76a;background:#f8fafc;border-radius:10px;padding:15px 16px;"><div style="font-size:20px;font-weight:850;color:#171717;">收益难落地</div><div style="margin-top:6px;font-size:18px;line-height:1.38;color:#475467;">oracle@k 需要 selector / verifier 才能转成线上收益。</div></div>
        </div>
        <div style="margin-top:22px;border-radius:12px;background:#fff5f5;border-left:6px solid #c7000b;padding:18px 20px;font-size:21px;line-height:1.42;color:#7a271a;font-weight:800;">困难点不是“能否跑通”，而是能否保持吞吐、P99 latency、KV 正确性和 SLA 可控。</div>
    '''

    goal_body = '''
        <div style="font-size:22px;line-height:1.52;color:#475467;">目标是把动态路径收益转化为生产可用能力，而不是停留在离线 oracle 指标。</div>
        <div style="margin-top:22px;display:grid;gap:14px;">
          <div style="display:flex;gap:14px;align-items:flex-start;"><span style="width:10px;height:10px;border-radius:50%;background:#c7000b;margin-top:11px;flex:0 0 auto;"></span><div style="font-size:21px;line-height:1.45;color:#344054;"><b>Path-aware runtime</b>：将 layer execution program 作为运行时输入。</div></div>
          <div style="display:flex;gap:14px;align-items:flex-start;"><span style="width:10px;height:10px;border-radius:50%;background:#c7000b;margin-top:11px;flex:0 0 auto;"></span><div style="font-size:21px;line-height:1.45;color:#344054;"><b>Execution-step cache</b>：按执行步和 path 管理 KV / activation。</div></div>
          <div style="display:flex;gap:14px;align-items:flex-start;"><span style="width:10px;height:10px;border-radius:50%;background:#c7000b;margin-top:11px;flex:0 0 auto;"></span><div style="font-size:21px;line-height:1.45;color:#344054;"><b>Budget controller</b>：按负载和 SLA 调整 top-k、repeat 与 fallback。</div></div>
        </div>
        <div style="margin-top:22px;border-radius:12px;background:#111827;color:#fff;padding:18px 20px;font-size:21px;line-height:1.45;font-weight:850;">价值：在准确率收益、吞吐、长尾时延和部署成本之间建立可配置策略。</div>
    '''

    page = f'''<div class="slide-fit" data-idx="0" data-active=""><div class="slide-canvas">
      <section data-label="挑战总览" data-page-id="{uid("page")}" style="width:100%;height:100%;position:relative;overflow:hidden;font-family:'Noto Sans SC','Microsoft YaHei',sans-serif;background:#f7f8fb;color:#171717;">
        {body_bg}
        <div data-editor-id="{editor_id()}" style="position:absolute;left:86px;top:54px;font-size:38px;font-weight:850;color:#171717;">动态 Layer Transformer 的服务化推理挑战</div>
        <div data-editor-id="{editor_id()}" style="position:absolute;left:88px;top:108px;width:1130px;font-size:22px;line-height:1.45;color:#475467;">按“技术背景 → 技术挑战和困难点 → 技术诉求与价值收益”组织，定义 layer-programmable Transformer 在生产推理中的核心难题。</div>
        <div data-editor-id="{editor_id()}" style="position:absolute;right:88px;top:66px;font-size:22px;color:#667085;font-family:'JetBrains Mono',monospace;">ONE-PAGE CHALLENGE BRIEF</div>

        {panel(88, 192, 548, 660, "01", "技术背景", "BACKGROUND", background_body, "#c7000b")}
        {panel(686, 192, 548, 660, "02", "技术挑战和困难点", "CHALLENGES", challenge_body, "#c7000b")}
        {panel(1284, 192, 548, 660, "03", "技术诉求 / 目标 / 价值", "GOALS & VALUE", goal_body, "#c7000b")}

        <div data-editor-id="{editor_id()}" style="position:absolute;left:88px;bottom:64px;width:1744px;height:72px;background:#111827;border-radius:12px;color:#fff;font-size:27px;font-weight:850;display:flex;align-items:center;padding:0 32px;">核心挑战：让动态层执行 Transformer 像固定层 Transformer 一样高效、稳定、可预测地服务化部署。</div>
      </section>
    </div></div>'''

    stage_start = s.find('<div class="stage" id="stage">')
    first_slide = s.find('<div class="slide-fit"', stage_start)
    labels = re.findall(r'<section data-label="([^"]+)"', s)
    last_label = labels[-1]
    last_section = s.find(f'<section data-label="{last_label}"')
    stage_content_end = s.find("</div></div>", s.find("</section>", last_section)) + len("</div></div>")

    s = s[:first_slide] + "\n\n    " + page + "\n\n  " + s[stage_content_end:]

    ns = s.find("const nav = [")
    ne = s.find("];", ns)
    nav = """const nav = [
      { i:0, code:'总览', label:'挑战总览' },
    ];"""
    s = s[:ns] + nav + s[ne + 2 :]

    cs = s.find("const chapters = [")
    ce = s.find("];", cs)
    chapters = """const chapters = [
      { name:'01 · 动态层服务化挑战', start:0 },
    ];"""
    s = s[:cs] + chapters + s[ce + 2 :]

    eb.set_template(lines, s)
    eb.save(OUT, lines)
    eb.verify(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
