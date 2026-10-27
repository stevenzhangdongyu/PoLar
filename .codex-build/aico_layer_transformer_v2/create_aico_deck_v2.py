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
OUT = ROOT / "outputs/presentations/layer_transformer_challenge_huawei_v2.html"

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


def dark_bg() -> str:
    return '''
        <div style="position:absolute;inset:0;background:radial-gradient(circle at 84% 12%,rgba(199,0,11,.20),transparent 34%),linear-gradient(135deg,#0b1320 0%,#172033 50%,#101828 100%);"></div>
        <div style="position:absolute;left:0;top:0;width:14px;height:100%;background:#c7000b;"></div>
        <div style="position:absolute;right:-160px;top:-140px;width:560px;height:560px;border:1px solid rgba(255,255,255,.14);border-radius:50%;"></div>
        <div style="position:absolute;right:180px;bottom:-140px;width:420px;height:420px;border:1px solid rgba(255,255,255,.10);border-radius:50%;"></div>
    '''


def top_bar(title: str, kicker: str) -> str:
    return f'''
        <div style="position:absolute;left:0;top:0;width:100%;height:18px;background:#c7000b;"></div>
        <div data-editor-id="{editor_id()}" style="position:absolute;left:88px;top:54px;font-size:34px;font-weight:850;color:#171717;">{title}</div>
        <div data-editor-id="{editor_id()}" style="position:absolute;right:92px;top:64px;font-size:21px;color:#667085;font-family:'JetBrains Mono',monospace;">{kicker}</div>
    '''


def card(x: int, y: int, w: int, h: int, title: str, body: str, accent: str = "#c7000b") -> str:
    return f'''
        <div data-editor-id="{editor_id()}" style="position:absolute;left:{x}px;top:{y}px;width:{w}px;height:{h}px;background:#fff;border:1px solid #d8dee9;border-radius:16px;box-shadow:0 10px 28px rgba(16,24,40,.07);padding:28px 30px;">
          <div style="height:5px;width:68px;background:{accent};border-radius:999px;margin-bottom:22px;"></div>
          <div style="font-size:28px;font-weight:850;color:#171717;margin-bottom:14px;">{title}</div>
          <div style="font-size:22px;line-height:1.55;color:#475467;">{body}</div>
        </div>
    '''


cover = slide(
    "封面",
    0,
    f'''
        {dark_bg()}
        <div data-editor-id="{editor_id()}" style="position:absolute;left:132px;top:120px;font-size:24px;letter-spacing:.14em;color:rgba(255,255,255,.72);font-weight:800;">AI INFRA CHALLENGE</div>
        <div data-editor-id="{editor_id()}" style="position:absolute;left:132px;top:196px;width:1320px;font-size:70px;line-height:1.18;font-weight:850;color:#fff;letter-spacing:-.02em;">动态 Layer Transformer<br/>服务化推理挑战</div>
        <div data-editor-id="{editor_id()}" style="position:absolute;left:132px;top:400px;width:1180px;font-size:31px;line-height:1.48;color:rgba(255,255,255,.82);">围绕技术背景、系统困难点与工程诉求，定义 layer-programmable Transformer 在生产推理中的关键攻坚方向。</div>
        <div data-editor-id="{editor_id()}" style="position:absolute;left:132px;bottom:134px;display:flex;gap:18px;">
          <span style="display:inline-block;background:#c7000b;color:#fff;font-size:24px;font-weight:850;padding:13px 22px;border-radius:6px;">POLAR</span>
          <span style="display:inline-block;border:1px solid rgba(255,255,255,.28);color:rgba(255,255,255,.9);font-size:23px;padding:12px 22px;border-radius:6px;">Loop / Skip / MoD</span>
          <span style="display:inline-block;border:1px solid rgba(255,255,255,.28);color:rgba(255,255,255,.9);font-size:23px;padding:12px 22px;border-radius:6px;">Serving Runtime</span>
        </div>
    ''',
    True,
)

toc = slide(
    "目录",
    1,
    f'''
        <div style="position:absolute;inset:0;background:#f5f7fb;"></div>
        {top_bar("目录：从算法变化推导系统诉求", "AGENDA")}
        <div data-editor-id="{editor_id()}" style="position:absolute;left:154px;top:186px;width:1610px;height:700px;display:grid;grid-template-columns:repeat(3,1fr);gap:34px;">
          <div style="background:#fff;border:1px solid #d8dee9;border-radius:18px;padding:46px 42px;box-shadow:0 12px 30px rgba(16,24,40,.07);">
            <div style="font-family:'JetBrains Mono',monospace;font-size:30px;color:#c7000b;font-weight:850;">01</div>
            <div style="margin-top:34px;font-size:36px;font-weight:850;color:#171717;">技术背景</div>
            <div style="margin-top:24px;font-size:24px;line-height:1.55;color:#475467;">模型执行从固定层序走向动态 layer program，算法收益开始进入推理系统核心路径。</div>
          </div>
          <div style="background:#fff;border:1px solid #d8dee9;border-radius:18px;padding:46px 42px;box-shadow:0 12px 30px rgba(16,24,40,.07);">
            <div style="font-family:'JetBrains Mono',monospace;font-size:30px;color:#c7000b;font-weight:850;">02</div>
            <div style="margin-top:34px;font-size:36px;font-weight:850;color:#171717;">挑战和困难点</div>
            <div style="margin-top:24px;font-size:24px;line-height:1.55;color:#475467;">现有 serving 假设被打破，batch、KV cache、调度和 top-k 候选都出现新的复杂度。</div>
          </div>
          <div style="background:#fff;border:1px solid #d8dee9;border-radius:18px;padding:46px 42px;box-shadow:0 12px 30px rgba(16,24,40,.07);">
            <div style="font-family:'JetBrains Mono',monospace;font-size:30px;color:#c7000b;font-weight:850;">03</div>
            <div style="margin-top:34px;font-size:36px;font-weight:850;color:#171717;">技术诉求与价值</div>
            <div style="margin-top:24px;font-size:24px;line-height:1.55;color:#475467;">目标是把动态路径的精度潜力转化为可部署、可控、可规模化的在线推理能力。</div>
          </div>
        </div>
    ''',
)

background = slide(
    "技术背景",
    2,
    f'''
        <div style="position:absolute;inset:0;background:#f5f7fb;"></div>
        {top_bar("技术背景：Transformer 执行范式正在从固定深度转向动态路径", "01 / BACKGROUND")}
        <div data-editor-id="{editor_id()}" style="position:absolute;left:96px;top:146px;width:820px;height:760px;background:#111827;border-radius:18px;padding:42px 48px;color:#fff;">
          <div style="font-size:28px;font-weight:850;color:#fff;margin-bottom:28px;">传统固定层模型</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:28px;line-height:1.55;color:#e6edf7;background:#172033;border-radius:14px;padding:34px 38px;">Token step t<br/>  layer 0<br/>  layer 1<br/>  layer 2<br/>  ...<br/>  layer N</div>
          <div style="margin-top:36px;font-size:24px;line-height:1.6;color:rgba(255,255,255,.78);">推理系统围绕固定 graph 做优化：continuous batching、paged attention、CUDA/NPU graph、按层组织 KV cache。</div>
        </div>
        <div data-editor-id="{editor_id()}" style="position:absolute;left:1004px;top:146px;width:820px;height:760px;background:#fff;border:1px solid #d8dee9;border-radius:18px;padding:42px 48px;box-shadow:0 12px 30px rgba(16,24,40,.07);">
          <div style="font-size:28px;font-weight:850;color:#171717;margin-bottom:28px;">动态层执行模型</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:27px;line-height:1.58;color:#172033;background:#f0f3f8;border-radius:14px;padding:34px 38px;">Req A: 0 1 2 3 4 ... N<br/>Req B: 0 1 2 3 3 4 ...<br/>Req C: 0 1 4 5 6 ... N<br/>Req D: 0 1 2 3 4 4 5 ...</div>
          <div style="margin-top:36px;font-size:24px;line-height:1.6;color:#475467;">POLAR、Loop Transformer、Skip-layer 和 MoD 让 layer 执行路径成为模型能力的一部分，推理系统必须理解 path。</div>
        </div>
        <div data-editor-id="{editor_id()}" style="position:absolute;left:96px;bottom:50px;width:1728px;height:72px;background:#fff5f5;border-left:8px solid #c7000b;border-radius:12px;color:#7a271a;font-size:27px;font-weight:850;display:flex;align-items:center;padding:0 34px;">背景判断：算法侧的动态执行收益，正在把“层序选择”从模型内部实现问题变成在线推理系统的一等调度对象。</div>
    ''',
)

challenge = slide(
    "挑战困难",
    3,
    f'''
        <div style="position:absolute;inset:0;background:#f5f7fb;"></div>
        {top_bar("技术挑战和困难点：现有 Serving 栈默认固定层序，动态路径会打散关键优化", "02 / CHALLENGES")}
        {card(96, 152, 405, 278, "Batch 变碎", "不同请求执行不同 layer path，传统 token-level batching 难以保持同层同步，GPU/NPU 利用率下降。", "#c7000b")}
        {card(536, 152, 405, 278, "KV 语义变化", "repeat layer 的第二次输入 hidden state 已不同，KV cache 不能只按 layer id 静态寻址。", "#2e90fa")}
        {card(976, 152, 405, 278, "调度成本不可见", "请求成本由 token 数和 path 共同决定，长 path、repeat 和 top-k 会显著拉高 P99 latency。", "#f79009")}
        {card(1416, 152, 405, 278, "Top-k 难落地", "多候选 path 带来 oracle 收益，但真实服务需要执行、验证和选择，成本容易成倍放大。", "#12b76a")}
        <div data-editor-id="{editor_id()}" style="position:absolute;left:96px;top:492px;width:1725px;height:330px;background:#fff;border:1px solid #d8dee9;border-radius:18px;box-shadow:0 12px 30px rgba(16,24,40,.07);padding:36px 44px;">
          <div style="font-size:30px;font-weight:850;color:#171717;margin-bottom:26px;">困难点不是“能否跑通”，而是“能否稳定高效地服务化”</div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:28px;">
            <div style="background:#f6f8fb;border-radius:14px;padding:28px 30px;height:200px;"><div style="font-size:24px;font-weight:850;color:#171717;">正确性边界</div><div style="margin-top:14px;font-size:22px;line-height:1.5;color:#475467;">skip / repeat 后 residual、norm、KV cache 和 activation cache 的语义需要重新定义。</div></div>
            <div style="background:#f6f8fb;border-radius:14px;padding:28px 30px;height:200px;"><div style="font-size:24px;font-weight:850;color:#171717;">性能边界</div><div style="margin-top:14px;font-size:22px;line-height:1.5;color:#475467;">动态 path 导致 graph cache、batch packing、kernel fusion 和显存规划都更难预测。</div></div>
            <div style="background:#f6f8fb;border-radius:14px;padding:28px 30px;height:200px;"><div style="font-size:24px;font-weight:850;color:#171717;">收益边界</div><div style="margin-top:14px;font-size:22px;line-height:1.5;color:#475467;">top-k 有候选价值，但没有 selector / verifier 时，无法把 oracle 收益稳定转化为线上收益。</div></div>
          </div>
        </div>
        <div data-editor-id="{editor_id()}" style="position:absolute;left:96px;bottom:50px;width:1725px;height:58px;background:#111827;border-radius:12px;color:#fff;font-size:25px;font-weight:850;display:flex;align-items:center;padding:0 32px;">挑战本质：动态 layer execution 破坏了固定层模型的同步执行、静态缓存和可预测成本模型。</div>
    ''',
)

goals = slide(
    "技术诉求",
    4,
    f'''
        <div style="position:absolute;inset:0;background:#f5f7fb;"></div>
        {top_bar("技术诉求和目标：把动态路径收益转化为可部署的系统能力", "03 / GOALS & VALUE")}
        <div data-editor-id="{editor_id()}" style="position:absolute;left:96px;top:150px;width:530px;height:700px;background:#fff;border:1px solid #d8dee9;border-radius:18px;padding:38px 40px;box-shadow:0 12px 30px rgba(16,24,40,.07);">
          <div style="font-size:30px;font-weight:850;color:#171717;">系统诉求</div>
          <div style="margin-top:34px;display:grid;gap:26px;">
            <div><div style="font-size:24px;font-weight:850;color:#c7000b;">Path-aware runtime</div><div style="font-size:22px;line-height:1.5;color:#475467;margin-top:8px;">推理引擎接收 layer execution program，而不是假设固定 forward。</div></div>
            <div><div style="font-size:24px;font-weight:850;color:#c7000b;">Execution-step KV</div><div style="font-size:22px;line-height:1.5;color:#475467;margin-top:8px;">KV cache、activation cache 和中间状态按执行步与 path 管理。</div></div>
            <div><div style="font-size:24px;font-weight:850;color:#c7000b;">Budget controller</div><div style="font-size:22px;line-height:1.5;color:#475467;margin-top:8px;">根据 SLA、负载和请求类型动态决定 top-k、repeat 和 fallback 策略。</div></div>
          </div>
        </div>
        <div data-editor-id="{editor_id()}" style="position:absolute;left:696px;top:150px;width:530px;height:700px;background:#fff;border:1px solid #d8dee9;border-radius:18px;padding:38px 40px;box-shadow:0 12px 30px rgba(16,24,40,.07);">
          <div style="font-size:30px;font-weight:850;color:#171717;">阶段目标</div>
          <div style="margin-top:36px;display:grid;gap:22px;">
            <div style="border-left:6px solid #c7000b;padding-left:20px;"><div style="font-size:24px;font-weight:850;color:#171717;">可正确执行</div><div style="font-size:21px;line-height:1.45;color:#475467;margin-top:8px;">定义 skip / repeat 的状态和 cache 语义。</div></div>
            <div style="border-left:6px solid #c7000b;padding-left:20px;"><div style="font-size:24px;font-weight:850;color:#171717;">可高效组批</div><div style="font-size:21px;line-height:1.45;color:#475467;margin-top:8px;">按 layer、segment、prefix 或成本分桶调度。</div></div>
            <div style="border-left:6px solid #c7000b;padding-left:20px;"><div style="font-size:24px;font-weight:850;color:#171717;">可控时延</div><div style="font-size:21px;line-height:1.45;color:#475467;margin-top:8px;">约束 total layer executions，控制 P99 latency。</div></div>
            <div style="border-left:6px solid #c7000b;padding-left:20px;"><div style="font-size:24px;font-weight:850;color:#171717;">可证明收益</div><div style="font-size:21px;line-height:1.45;color:#475467;margin-top:8px;">用 selector / verifier 把 top-k 候选转为线上准确率提升。</div></div>
          </div>
        </div>
        <div data-editor-id="{editor_id()}" style="position:absolute;left:1296px;top:150px;width:526px;height:700px;background:#111827;border-radius:18px;padding:42px 42px;color:#fff;">
          <div style="font-size:30px;font-weight:850;">价值收益</div>
          <div style="margin-top:38px;display:grid;gap:28px;">
            <div style="background:rgba(255,255,255,.08);border-top:5px solid #c7000b;border-radius:10px;padding:24px 26px;"><div style="font-size:25px;font-weight:850;">模型侧</div><div style="margin-top:10px;font-size:22px;line-height:1.45;color:rgba(255,255,255,.72);">释放动态深度带来的准确率和泛化收益。</div></div>
            <div style="background:rgba(255,255,255,.08);border-top:5px solid #c7000b;border-radius:10px;padding:24px 26px;"><div style="font-size:25px;font-weight:850;">系统侧</div><div style="margin-top:10px;font-size:22px;line-height:1.45;color:rgba(255,255,255,.72);">形成支持 layer-programmable model 的推理运行时能力。</div></div>
            <div style="background:rgba(255,255,255,.08);border-top:5px solid #c7000b;border-radius:10px;padding:24px 26px;"><div style="font-size:25px;font-weight:850;">工程侧</div><div style="margin-top:10px;font-size:22px;line-height:1.45;color:rgba(255,255,255,.72);">在精度、吞吐、P99 和成本之间建立可配置策略。</div></div>
          </div>
        </div>
        <div data-editor-id="{editor_id()}" style="position:absolute;left:96px;bottom:52px;width:1726px;height:58px;background:#fff5f5;border-left:8px solid #c7000b;border-radius:12px;color:#7a271a;font-size:25px;font-weight:850;display:flex;align-items:center;padding:0 32px;">目标状态：动态层模型不是离线实验特性，而是可纳入生产推理 SLA 的模型执行形态。</div>
    ''',
)

thanks = slide(
    "结语页",
    5,
    f'''
        {dark_bg()}
        <div data-editor-id="{editor_id()}" style="position:absolute;left:132px;top:130px;font-size:28px;letter-spacing:.12em;color:rgba(255,255,255,.72);font-weight:850;">TAKEAWAY</div>
        <div data-editor-id="{editor_id()}" style="position:absolute;left:132px;top:214px;width:1320px;font-size:60px;line-height:1.25;font-weight:850;color:#fff;">动态 Layer Transformer 的服务化难题，<br/>本质是运行时从固定图走向动态程序</div>
        <div data-editor-id="{editor_id()}" style="position:absolute;left:132px;top:420px;width:1250px;font-size:29px;line-height:1.6;color:rgba(255,255,255,.80);">后续技术路线应围绕 path-aware batching、execution-aware cache、cost-aware scheduling 和 deployable top-k selection 展开。</div>
        <div data-editor-id="{editor_id()}" style="position:absolute;left:132px;bottom:130px;width:1280px;height:92px;background:#c7000b;border-radius:10px;color:#fff;font-size:32px;font-weight:850;display:flex;align-items:center;padding:0 34px;">核心挑战：让动态层执行 Transformer 像固定层 Transformer 一样高效、稳定、可预测地服务化部署。</div>
    ''',
)


def replace_nav(s: str) -> str:
    ns = s.find("const nav = [")
    ne = s.find("];", ns)
    nav = """const nav = [
      { i:0, code:'封面', label:'封面' },
      { i:1, code:'目录', label:'目录' },
      { i:2, code:'背景', label:'技术背景' },
      { i:3, code:'挑战', label:'挑战困难' },
      { i:4, code:'诉求', label:'技术诉求' },
      { i:5, code:'结语', label:'结语页' },
    ];"""
    return s[:ns] + nav + s[ne + 2 :]


def replace_chapters(s: str) -> str:
    cs = s.find("const chapters = [")
    ce = s.find("];", cs)
    chapters = """const chapters = [
      { name:'01 · 技术背景', start:2 },
      { name:'02 · 技术挑战和困难点', start:3 },
      { name:'03 · 技术诉求与价值收益', start:4 },
    ];"""
    return s[:cs] + chapters + s[ce + 2 :]


def main() -> None:
    lines = eb.load(TEMPLATE)
    s = eb.get_template(lines)

    stage_start = s.find('<div class="stage" id="stage">')
    first_slide = s.find('<div class="slide-fit"', stage_start)
    labels = re.findall(r'<section data-label="([^"]+)"', s)
    last_label = labels[-1]
    last_section = s.find(f'<section data-label="{last_label}"')
    stage_content_end = s.find("</div></div>", s.find("</section>", last_section)) + len("</div></div>")

    pages = [cover, toc, background, challenge, goals, thanks]
    custom = "\n\n    " + "\n\n    ".join(pages) + "\n\n  "
    s = s[:first_slide] + custom + s[stage_content_end:]
    s = replace_nav(s)
    s = replace_chapters(s)

    eb.set_template(lines, s)
    eb.save(OUT, lines)
    eb.verify(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
