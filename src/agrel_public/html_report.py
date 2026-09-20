# ComputeProof 算证 Community — GPU scientific-computing reliability check
# Copyright (C) 2026 The ComputeProof 算证 authors
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE file
# (GNU GPL v3) shipped with this package or <https://www.gnu.org/licenses/>.
"""Local static report: no scripts, remote assets, cookies, or telemetry.
Two layers per card: ordinary scenario outcomes (descriptive) and exact probes
(arithmetic verdicts). They are shown side by side, never merged."""
from html import escape
from pathlib import Path
from .common import atomic_bytes

STATUS_TEXT = {
    "NO_ANOMALY_OBSERVED": "本次探针未观察到异常",
    "NUMERICAL_ANOMALY_DETECTED": "观察到数值异常",
    "INCOMPLETE": "检测未完成（不是通过）",
}


def _metric(m):
    if not m:
        return "—"
    if not m.get("finite_output", True):
        return "非有限输出"
    parts = []
    for k in ("accuracy", "rmse", "recall_at_5", "ndcg_at_10", "cluster_agreement"):
        if m.get(k) is not None:
            parts.append(f"{k}={m[k]:.6g}")
    if "evaluated_rows" in m:
        parts.append(f"rows={m['evaluated_rows']}")
    if "queries" in m:
        parts.append(f"queries={m['queries']}")
    return escape(", ".join(parts) or "—")


def render_report(report: dict, path: Path) -> None:
    pieces = []
    for i, card in enumerate(report["cards"]):
        d = card["diagnosis"]
        status = card.get("overall_status", d["status"])
        a = d.get("assessment") or d.get("software_assessment") or {}
        software_note = ("<p class='INCOMPLETE'>CPU 软件运行：下列探针只验证软件流程与参考一致性，不是 GPU 判定。</p>"
                         if report.get("cpu_demo") else "")
        rows = "".join(
            "<tr><td>" + escape(str(r.get("title", r.get("scenario")))) + "</td><td>" +
            escape(f"b{r.get('config', {}).get('batch_size', '?')}/{r.get('config', {}).get('context', '?')}") +
            "</td><td>" + escape(str(r.get("status"))) + "</td><td>" + _metric(r.get("metric")) +
            "</td><td>" + escape(str(r.get("arithmetic_verdict", "NOT_ASSESSED"))) + "</td></tr>"
            for r in card.get("scenarios", []))
        probes = "".join(
            "<tr><td>" + escape(str(p.get("probe_id"))) + "</td><td>" + escape("×".join(str(x) for x in p.get("shape", []))) +
            "</td><td>" + escape(str(p.get("layout", ""))) + "</td><td>" + escape(str(p.get("iterations_completed", 0))) +
            "</td><td>" + escape(f"{p.get('checked_values', 0):,}") + "</td><td>" + escape(f"{p.get('bad_values', 0):,}") +
            "</td><td class='" + ("bad" if p.get("arithmetic_verdict") == "FAIL_NUMERICAL" else "ok") + "'>" +
            escape(str(p.get("arithmetic_verdict", "NOT_ASSESSED"))) + "</td></tr>"
            for p in d.get("probes", []))
        native = card.get("native") or {}
        native_rows = "".join(
            "<tr><td>" + escape(str(k.get("kernel"))) + "</td><td>" + escape(str(k.get("iterations_completed", 0))) + " × " +
            escape(str(k.get("blocks_per_iteration", 0))) + " 块</td><td>" + escape(f"{len(k.get('observed_logical_sms', []))}/{k.get('sm_count', '?')}") +
            "</td><td>" + escape(str(k.get("migrated_or_invalid_blocks", 0))) + "</td><td>" + escape(f"{k.get('checked_values', 0):,}") +
            "</td><td>" + escape(f"{k.get('bad_values', 0):,}") + "</td><td>" + escape(", ".join(str(x) for x in k.get("bad_sms", [])) or "—") +
            "</td><td class='" + ("bad" if k.get("arithmetic_verdict") == "FAIL_NUMERICAL" else "ok") + "'>" +
            escape(str(k.get("arithmetic_verdict", "NOT_ASSESSED"))) + " / " + escape(str(k.get("coverage_verdict", ""))) + "</td></tr>"
            for k in native.get("kernels", []))
        per_sm_rows = ""
        for k in native.get("kernels", []):
            if k.get("bad_sms"):
                per_sm_rows += "<p>" + escape(str(k["kernel"])) + " 逐逻辑 SM 不一致值下界：" + escape(", ".join(
                    f"SM{sm}: {v['bad_values']:,}/{v['blocks']} 块" for sm, v in k.get("per_sm", {}).items() if v.get("bad_values"))) + "</p>"
        native_html = ("<h3>原生 SM 覆盖探针（native-signed-unit-smid-v1，驱动 API 加载预编译核函数）</h3><p>状态：<strong>" +
                       escape(str(native.get("status", "未运行"))) + "</strong> " + escape(str(native.get("reason", ""))) +
                       "</p><div class='wrap'><table><thead><tr><th>核函数</th><th>迭代 × 块</th><th>观察到的逻辑 SM</th><th>迁移/无效块</th><th>核对值</th><th>不一致值下界</th><th>坏值 SM</th><th>判定 / 覆盖</th></tr></thead><tbody>" +
                       (native_rows or "<tr><td colspan='8'>未运行</td></tr>") + "</tbody></table></div>" + per_sm_rows +
                       "<p><small>SMID 是逻辑标识；覆盖只统计起止 SMID 相同的块；不映射物理核心或张量核单元。</small></p>") if native else ""
        pack = card.get("pack_diagnosis")
        pack_html = ("<p>维护者签名包：<strong>" + escape(str(pack.get("status"))) + "</strong> · " +
                     escape(str(pack.get("provenance", ""))) + " · " + escape(str(pack.get("pack_id", ""))) + "</p>") if pack else ""
        pieces.append(
            "<section><h2>卡 " + str(i + 1) + " · " + escape(str(card["model"])) + "</h2><p>" + str(int(card["memory_mib"])) +
            " MiB · 启动前频率快照 " + escape(str(card.get("core_clock_mhz") if card.get("core_clock_mhz") is not None else "未知")) +
            " MHz · 检测：<strong class='" + escape(status) + "'>" + escape(status) + "</strong> — " + escape(STATUS_TEXT.get(status, "")) +
            "</p><p>" + escape(str(d.get("reason", ""))) + "</p>" + software_note +
            "<p>探针 " + str(a.get("probes_completed", 0)) + "/" + str(a.get("probes_total", 0)) + " 完成，核对 " +
            f"{a.get('checked_values', 0):,}" + " 个值，不一致至少 " + f"{a.get('bad_values', 0):,}" + " 个。</p>" + pack_html +
            "<h3>场景应用（自然工作负载，指标为描述性结果）</h3><div class='wrap'><table><thead><tr><th>场景</th><th>配置</th><th>执行状态</th><th>任务指标</th><th>算术判定</th></tr></thead><tbody>" +
            (rows or "<tr><td colspan='5'>本卡未运行场景应用</td></tr>") + "</tbody></table></div>" +
            "<h3>精确探针（torch-signed-unit-exact-v1）</h3><div class='wrap'><table><thead><tr><th>探针</th><th>形状 b×m×k×n</th><th>布局</th><th>次数</th><th>核对值</th><th>不一致值下界</th><th>判定</th></tr></thead><tbody>" +
            (probes or "<tr><td colspan='7'>本卡未运行精确探针</td></tr>") + "</tbody></table></div>" + native_html + "</section>")
    sel = report.get("selection", {})
    excluded = "".join("<li>" + escape(str(e.get("scenario"))) + "：" + escape(str(e.get("reason"))) + "</li>" for e in sel.get("excluded", []))
    ref = report.get("reference", {})
    page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src 'none'">
<title>ComputeProof 算证 · 本地场景与检测报告</title><style>
body{margin:0;background:#f5f7fa;color:#172330;font:15px/1.6 system-ui,sans-serif}
main{max-width:1180px;margin:auto;padding:36px 24px}h1{font-size:30px;margin:6px 0}h3{margin:22px 0 8px}
section{margin:32px 0;background:white;border:1px solid #dce3eb;border-radius:12px;padding:20px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #e5eaf0;padding:6px 8px;text-align:left;vertical-align:top}
th{background:#f0f4f8}.wrap{overflow-x:auto}.bad{color:#a11;font-weight:600}.ok{color:#176}
.notice{background:#e8eef5;padding:14px 18px;border-radius:10px}small{color:#526474}
.NO_ANOMALY_OBSERVED{color:#176}.NUMERICAL_ANOMALY_DETECTED{color:#a11}.INCOMPLETE{color:#8a5a00}
</style></head><body><main><small>COMPUTEPROOF / LOCAL ONLY</small><h1>你的科研计算场景与数值检测记录</h1>
<p>先运行真实科研工作负载，再在新进程中以代表性 GEMM 形状运行精确探针。两层证据分别记录。</p>
<div class="notice">场景指标是描述性结果，不能单独给硬件定罪；精确探针不一致是计算/传输链路的反例，不自动指向某个物理单元。
未观察到异常只覆盖本次探针形状、种子与预算；检测未完成不是通过。本文件完全离线，没有外部资源和上传行为。
姓名、邮箱、数值分享必须在命令行另行同意。</div>'''
    import json
    page += "<p><small>开始：" + escape(str(report.get('started_at',''))) + "；结束：" + escape(str(report.get('finished_at',''))) + "；环境：" + escape(json.dumps(report.get('environment',{}),ensure_ascii=False)) + "；源码构建摘要：" + escape(str(report.get('build_id',''))) + "</small></p>"
    page += "".join(pieces)
    page += ("<section><h3>选择与参考</h3><p>可选大场景处理：" + escape(str(sel.get("large_mode", ""))) + "；已选：" +
             escape(", ".join(sel.get("selected", []))) + "</p>" + ("<ul>" + excluded + "</ul>" if excluded else "") +
             "<p>冻结参考 " + escape(str(ref.get("version", ""))) + " · SHA-256 " + escape(str(ref.get("sha256", ""))) +
             " · 契约 " + escape(str(ref.get("contract", ""))) + " · 种子数 " + escape(str(ref.get("iterations", ""))) + "</p>" +
             "<p>运行标识：" + escape(report["run_id"]) + " · 客户端 " + escape(str(report.get("client_version", ""))) + "</p></section></main></body></html>")
    atomic_bytes(path, page.encode("utf-8"))
