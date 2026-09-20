# Aritheia Community — GPU scientific-computing reliability check
# Copyright (C) 2026 The Aritheia authors
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE file
# (GNU GPL v3) shipped with this package or <https://www.gnu.org/licenses/>.
from __future__ import annotations
import argparse
import json
import os
import sys
import uuid
import threading
import copy
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from . import __version__
from .common import canonical, load_json, save_json
from .config import Config
from .console_input import TimedConsoleInput
from .devices import inventory, select_devices, bootstrap_payload
from .network import PackJob, Transport
from .packs import cached_pack, load_pack
from .processes import run_worker
from .scenarios import assets
from .scenarios.registry import BUILTIN, OPTIONAL, ORDER, SCENARIOS, configs
from .selection import choose, parse_scenarios
from .sharing import prepare, upload_preview, withdraw

NARRATIVE = {
    "balanced": "先在你的显卡上运行真实科研计算场景，再用精确探针独立检查数值。场景完成不等于整卡健康。",
    "ai": "先看看你的显卡能完成哪些科研计算：每个场景都有真实数据与可见结果，不要求先懂硬件。",
    "health": "先运行真实工作负载，再做精确检查。出现差异是复测线索，不等于显卡已经损坏。",
    "research": "场景与探针分层记录；测试后可自愿贡献数值，并单独选择联系与致谢署名。",
}


def say(text):
    print(text, flush=True)


def yes(prompt, default=False):
    if not sys.stdin.isatty():
        return default if default else False
    ans = input(prompt + (" [Y/n] " if default else " [y/N] ")).strip().lower()
    return (ans in ("y", "yes", "是")) or (not ans and default)


def add_config(parser):
    parser.add_argument("--config", type=Path)


def parser():
    p = argparse.ArgumentParser(prog="computeproof",
                                description="真实科研场景 + 精确探针的单卡数值完整性检查；详细研究 JSON 另行同意")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("check", "guided", "doctor", "fetch"):
        s = sub.add_parser(name)
        add_config(s)
        s.add_argument("--devices", "--device", dest="devices", default="cuda:0")
        s.add_argument("--all-devices", action="store_true")
        if name != "doctor":
            s.add_argument("--offline", action="store_true", help=argparse.SUPPRESS)
            s.add_argument("--allow-hardware-metadata", action="store_true",
                           help="同意只发送所选显卡的型号、显存、当前频率以获取维护者签名包")
        if name in ("check", "guided"):
            s.add_argument("--telemetry", choices=["auto", "off"], default=None, help=argparse.SUPPRESS)
            s.add_argument("--parallel-gpus", type=int, default=None, help="同时运行的显卡数，默认 2；每卡独立进程")
            s.add_argument("--out", type=Path, default=None)
            s.add_argument("--pack", type=Path, help="已签名维护者实验包（附加证据），不要求联网")
            s.add_argument("--cpu-demo", "--cpu-software-run", dest="cpu_demo", action="store_true",
                           help="仅在 CPU 验收软件流程；不是 GPU 检测")
            s.add_argument("--intent", choices=list(NARRATIVE), default="balanced")
            s.add_argument("--no-share-prompt", action="store_true")
            s.add_argument("--scenarios", default=None,
                           help="显式场景列表：逗号分隔 ID、all、builtin 或 none")
            s.add_argument("--large-scenarios", choices=["skip", "include", "ask"], default=None,
                           help="四个可选大场景：skip=不加入，include=全部加入，ask=逐项询问（交互默认）")
            s.add_argument("--accept-download", action="store_true",
                           help="授权下载所选可选场景缺失且已钉住 SHA-256 的资产包；不授权上传")
            s.add_argument("--asset-release-url", default=None, help="资产包正式发布地址（HTTPS）")
            s.add_argument("--iterations", type=int, default=None, help="每个探针的冻结种子数 1..8；正式检查用默认 8")
            s.add_argument("--steps", type=int, default=8, help="微调场景的 AdamW 更新步数")
            s.add_argument("--probes-only", action="store_true", help="跳过场景应用，只运行精确探针")
            s.add_argument("--no-native", action="store_true", help="跳过原生 SM 覆盖探针（驱动 API 加载预编译核函数）")
            if name == "guided":
                s.set_defaults(scenarios=None, large_scenarios="skip")
    s = sub.add_parser("assets")
    a = s.add_subparsers(dest="assets_command", required=True)
    a.add_parser("status")
    x = a.add_parser("import", help="维护者：从 v0.3.x 冻结资产树导入 frozen_asset 场景")
    x.add_argument("--from", dest="source", type=Path, required=True)
    x.add_argument("--scenarios", default="genomics_splice,medical_ultrasound,materials_screen")
    x.add_argument("--dest", type=Path, default=None)
    a.add_parser("freeze", help="维护者：记录随包资产 SHA-256 清单")
    a.add_parser("verify", help="核对随包资产与清单")
    d = a.add_parser("download", help="显式下载一个可选资产包")
    d.add_argument("pack", choices=["singlecell", "literature", "language"])
    d.add_argument("--asset-release-url", default=None)
    d.add_argument("--accept-download", action="store_true")
    add_config(d)
    inst=a.add_parser('install',help='离线安装并校验一个已下载的正式资产 ZIP')
    inst.add_argument('pack',choices=['singlecell','literature','language'])
    inst.add_argument('archive',type=Path)
    s = sub.add_parser("reference")
    r = s.add_subparsers(dest="reference_command", required=True)
    v = r.add_parser("verify", help="校验固定参考完整性和探针定义；不做 CPU/GPU 数值回算")
    v.add_argument("--limit", type=int, default=None)
    v.add_argument("--device", default=None)
    r.add_parser("list")
    s = sub.add_parser("research", help="研究数据：重试摘要、预览/上传详细 JSON 或撤回")
    s.add_argument("action", choices=["retry", "preview-detail", "upload-detail", "withdraw"])
    s.add_argument("run_dir", type=Path); add_config(s)
    s.add_argument("--confirm", help="详细预览 manifest 的 SHA-256")
    s = sub.add_parser("deep", help="异常报告的深度测试；编译包下载、执行、结果上传分别确认")
    s.add_argument("action", choices=["fetch","preview-download","download","preview","run","preview-result","upload-result"])
    s.add_argument("run_dir", type=Path); add_config(s)
    s.add_argument("--case", dest="case_file", type=Path)
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--out", type=Path)
    s.add_argument("--confirm", help="对应预览的 SHA-256：download 仅授权下载；run 仅授权执行；upload-result 仅授权上传")
    s = sub.add_parser("report"); s.add_argument("run_dir", type=Path)
    s = sub.add_parser("replay", help="无需 GPU 或 CPU 矩阵回算：核对固定摘要与已保存的反例")
    s.add_argument("probe_dir", type=Path)
    s = sub.add_parser("export"); s.add_argument("run_dir", type=Path)
    s.add_argument("--share-values", action="store_true")
    s.add_argument("--name", default=""); s.add_argument("--email", default="")
    s.add_argument("--allow-contact", action="store_true"); s.add_argument("--acknowledge", action="store_true")
    s = sub.add_parser("upload"); s.add_argument("run_dir", type=Path); s.add_argument("--confirm", required=True); add_config(s)
    s = sub.add_parser("withdraw"); s.add_argument("run_dir", type=Path); s.add_argument("--yes", action="store_true"); add_config(s)
    return p


def fmt_metric(m):
    if not m:
        return ""
    if not m.get("finite_output", True):
        return "非有限输出"
    for key in ("accuracy", "rmse", "recall_at_5", "cluster_agreement", "queries"):
        if key in m and m[key] is not None:
            v = m[key]
            return f"{key}={v:.6g}" if isinstance(v, float) else f"{key}={v}"
    return ""


NOT_APPLICABLE = ("SKIPPED_BY_USER",)


def overall_status(card: dict) -> str:
    """Combine the exact-probe channel and the native SM channel. Anomaly in either wins;
    requested channels which fail to run remain INCOMPLETE, even if other channels pass."""
    exact = card["diagnosis"]["status"]
    native = card.get("native") or {}
    if exact == "NUMERICAL_ANOMALY_DETECTED" or native.get("status") == "NUMERICAL_ANOMALY_DETECTED" or (card.get("pack_diagnosis") or {}).get("status") == "NUMERICAL_ANOMALY_DETECTED":
        return "NUMERICAL_ANOMALY_DETECTED"
    if exact != "NO_ANOMALY_OBSERVED":
        return exact
    if card.get("workflow_complete") is False:
        return "INCOMPLETE"
    if native.get("status") == "NO_ANOMALY_OBSERVED":
        return "NO_ANOMALY_OBSERVED"
    if native.get("reason") in NOT_APPLICABLE:
        return "NO_ANOMALY_OBSERVED"
    return "INCOMPLETE"


def display_report(report):
    say("\n── 本地结果：场景应用与精确探针分层 ──")
    for i, c in enumerate(report["cards"]):
        rows = c.get("scenarios", [])
        done = {r["scenario"] for r in rows if r.get("status") in ("COMPLETED", "MONITOR_ALERT")}
        d = c["diagnosis"]
        a = d.get("assessment") or d.get("software_assessment") or {}
        say(f"卡 {i + 1} · {c['model']} | 场景完成 {len(done)}/{len({r['scenario'] for r in rows}) or 0} | 探针 "
            f"{a.get('probes_completed', 0)}/{a.get('probes_total', 0)} | 已核对 {a.get('checked_values', 0):,} 值 | {c.get('overall_status', d['status'])}")
        if d.get("reason") and d["status"] != "NO_ANOMALY_OBSERVED":
            say("  精确探针: " + str(d["reason"]))
        n = c.get("native")
        if n:
            say(f"  原生 SM 探针: {n['status']}" + (f"（{n.get('reason')}）" if n["status"] != "NO_ANOMALY_OBSERVED" else ""))
            for k in n.get("kernels", []):
                say(f"    {k['kernel']}: {k['arithmetic_verdict']}，{len(k.get('observed_logical_sms', []))}/{k.get('sm_count')} 逻辑 SM，"
                    f"核对 {k.get('checked_values', 0):,} 值，不一致至少 {k.get('bad_values', 0):,}" + (f"，坏值所在逻辑 SM {k['bad_sms']}" if k.get("bad_sms") else ""))
        for r in rows:
            cfg = r.get("config", {})
            say(f"  · {r.get('title', r['scenario'])} b{cfg.get('batch_size', '?')}/{cfg.get('context', '?')}: {r['status']} {fmt_metric(r.get('metric'))}")
        for f in a.get("failed_probes", []):
            say("  ✗ 精确探针不一致: " + f)
        if c.get("pack_diagnosis"):
            say(f"  维护者签名包: {c['pack_diagnosis'].get('status')} ({c['pack_diagnosis'].get('provenance')})")
    for e in report.get("selection", {}).get("excluded", []):
        say(f"  未运行 {e['scenario']}: {e['reason']}")
    say("未观察到异常仅覆盖本次探针形状、种子与预算；检测未完成不是通过；场景完成也不是健康认证。")


def selected_cards(args):
    cards = inventory()
    if not cards:
        raise RuntimeError("没有可用 CUDA GPU。CPU 软件验收不能当作 GPU 检测。")
    return select_devices(cards, "all" if args.all_devices else args.devices)


def consent_bootstrap(args, config, cards):
    if args.offline or not config.api_url:
        return False
    preview = bootstrap_payload(cards)
    say("维护者签名包请求仅发送：" + canonical(preview).decode())
    say("不会发送 UUID、用户名、运算结果、姓名或邮箱。服务器能看到连接源 IP；应用不保存 IP。")
    return args.allow_hardware_metadata or yes("允许这一次硬件信息交换并在场景运行期间下载签名实验包？")


def make_downloader(config, release_url, offline):
    if offline:
        return None
    url = release_url or config.asset_release_url or os.environ.get("AGREL_ASSET_RELEASE_URL", "")
    transport = Transport(max(config.network_timeout, 30.0), config.allow_local_http)

    def download(pack, manifest, cancel=None):
        last = [0]
        def progress(done, total):
            pct = done * 100 // max(total, 1)
            if pct >= last[0] + 10:
                last[0] = pct
                say(f"    下载 {pack}: {pct}% ({done:,}/{total:,} bytes)")
        assets.download_pack(pack, transport, release_url=url or None, manifest=manifest, progress=progress, cancel=cancel)
        say(f"    资产包 {pack} 已校验并安装到 {assets.user_root()}")
    download.release_url = url
    return download


class OptionalAssetJob:
    """Download each pinned optional pack once while the eight built-in scenes run."""
    def __init__(self, downloader):
        self.downloader = downloader
        self.cancel = threading.Event()
        self.thread = None
        self.completed = []
        self.errors = []

    def start(self):
        def work():
            try:
                manifest = assets.optional_manifest()
                packs = []
                for name in OPTIONAL:
                    pack = SCENARIOS[name]['pack']
                    if not assets.status(name)['ready'] and pack not in packs:
                        packs.append(pack)
                for pack in packs:
                    if self.cancel.is_set():
                        break
                    say(f"  后台准备可选资产：{pack}")
                    try:
                        self.downloader(pack, manifest, self.cancel)
                        self.completed.append(pack)
                    except Exception as exc:
                        self.errors.append({'pack': pack, 'error': type(exc).__name__})
                        say(f"  可选资产 {pack} 未完成（{type(exc).__name__}）；不影响当前八场景检测。")
            finally:
                say("  可选资产后台任务结束。")
        self.thread = threading.Thread(target=work, name='computeproof-optional-assets', daemon=True)
        self.thread.start()

    def stop(self):
        self.cancel.set()
        if self.thread:
            self.thread.join(timeout=5)


def guided_asset_countdown(args, config):
    """TTY default is download-after-countdown; unattended runs require an explicit flag."""
    if args.offline:
        args.optional_skip_reason = "DOWNLOAD_DECLINED"
        say("本次未启动可选场景资产下载。")
        return None
    downloader = make_downloader(config, args.asset_release_url, False)
    if not downloader or not downloader.release_url:
        args.optional_skip_reason = "ASSET_RELEASE_URL_MISSING"
        say("未配置 GitHub Release 资产地址；本次运行八个内置场景，不启动可选资产下载。")
        return None
    if not sys.stdin.isatty() and not args.accept_download:
        args.optional_skip_reason = "DOWNLOAD_DECLINED"
        say("无人值守运行未提供 --accept-download；跳过可选资产下载。")
        return None
    if sys.stdin.isatty() and not args.accept_download:
        say("10 秒后将在后台下载四个可选场景的三个校验包；输入 c 后回车可取消。")
        console = TimedConsoleInput()
        for remaining in range(10, 0, -1):
            say(f"  下载倒计时 {remaining:02d}s")
            line = console.poll(1.0)
            if line is not None and line.strip().lower() in ('c', 'cancel', '取消'):
                args.optional_skip_reason = "OPTIONAL_DOWNLOAD_CANCELLED"
                say("已取消可选资产下载；八个内置场景照常运行。")
                return None
    job = OptionalAssetJob(downloader)
    job.start()
    return job


def guided_optional_scope(job, missing_reason="DOWNLOAD_UNAVAILABLE"):
    ready, excluded = [], []
    for name in OPTIONAL:
        state = assets.status(name)
        if state["ready"]:
            ready.append(name)
        else:
            reason = "ASSET_INTEGRITY_ERROR" if state.get("integrity_error") else (
                "DOWNLOAD_FAILED" if job and job.errors else
                missing_reason if not job else "ASSET_STILL_MISSING")
            excluded.append({"scenario": name, "reason": reason})
    return ready, excluded


def check(args):
    config = Config.read(args.config)
    cards = [] if args.cpu_demo else selected_cards(args)
    out = args.out or Path("runs") / ("ag-" + uuid.uuid4().hex[:10])
    if out.exists():
        raise FileExistsError("输出目录已存在，拒绝覆盖：" + str(out))
    iterations = config.iterations if args.iterations is None else args.iterations
    if not 1 <= iterations <= 8:
        raise ValueError("--iterations 必须在 1..8")
    if not 1 <= args.steps <= 10000:
        raise ValueError("--steps 必须在 1..10000")
    parallel = args.parallel_gpus or config.max_parallel_gpus
    if not 1 <= parallel <= 32:
        raise ValueError("--parallel-gpus must be 1..32")
    research_enabled = not args.offline and not args.cpu_demo and (args.telemetry or config.telemetry) == "auto"
    if research_enabled:
        from .telemetry import NOTICE
        say(NOTICE)
    out.mkdir(parents=True)
    try:
        out.chmod(0o700)
    except OSError:
        pass
    say("ComputeProof 算证 " + __version__ + " · " + NARRATIVE[args.intent])
    for d in cards:
        say(f"选中 {d.name}: {d.model} / {d.memory_mib} MiB / 频率 {d.core_clock_mhz if d.core_clock_mhz is not None else '未知（不猜测）'} MHz")
    interactive = sys.stdin.isatty() and not args.cpu_demo
    guided_full = args.command == 'guided' and args.scenarios in (None, "all") and not args.cpu_demo and not args.probes_only
    optional_job = guided_asset_countdown(args, config) if guided_full else None
    large = args.large_scenarios or ("ask" if interactive and args.scenarios is None else "skip")
    selection = choose(explicit=list(BUILTIN) if guided_full else parse_scenarios(args.scenarios), large=large, interactive=interactive, ask=yes, say=say,
                       accept_download=args.accept_download,
                       downloader=make_downloader(config, args.asset_release_url, args.offline), probes_only=args.probes_only)
    selection["large_mode"] = large
    if guided_full:
        selection["requested"] = list(ORDER)
        say("本次目标为十二场景：先跑八个内置场景，后台资产就绪后继续四个可选场景；缺失/取消会明确保留。")
    if not selection["selected"]:
        say("没有可运行的场景（资产缺失或已拒绝下载）。")
        for e in selection["excluded"]:
            say(f"  {e['scenario']}: {e['reason']}")
    say("本次场景：" + ", ".join(SCENARIOS[n]["title"] for n in selection["selected"]))
    for e in selection["excluded"]:
        say(f"  不运行 {SCENARIOS[e['scenario']]['title']}：{e['reason']}" + (f"（{e['detail']}）" if e.get("detail") else ""))
    allowed = False if args.cpu_demo else consent_bootstrap(args, config, cards)
    download = PackJob(config, bootstrap_payload(cards)) if allowed else None
    active_card = None
    first_scene = False
    qualified_failure = {"probes": False, "native": False, "pack": False}
    active_phase = "scenarios"

    def event(e):
        nonlocal first_scene
        kind = e.get("event")
        if e.get("qualified") is True and e.get("matched") is False:
            qualified_failure[active_phase] = True
        if kind == "scenario_start":
            say(f"\n[{e['index']:02d}/{e['total']:02d}] {e['title']} · {e['config']} · 正在计算")
            if not first_scene:
                first_scene = True
                if download:
                    download.start()  # First visible computation starts before any network wait.
        elif kind == "scenario_done":
            say(f"  → {e['status']} {fmt_metric(e.get('metric'))} ({e['wall_s']:.1f}s)")
        elif kind == "scenario_error":
            say(f"  → {e['status']} ({e['error_type']})")
        elif kind == "probe_start":
            say(f"精确探针 [{e['index']}/{e['total']}] {e['probe']}")
        elif kind == "probe_done":
            mark = "一致" if e["verdict"] == "PASS_OBSERVED" else e["verdict"]
            say(f"  → {mark}，核对 {e['checked_values']:,} 值，不一致至少 {e['bad_values']:,}")
            if e["bad_values"]:
                say("  数值不一致；首个反例已保留本地，稍后再决定是否分享。")
        elif kind == "native_iteration":
            if e["iteration"] == 0:
                say(f"原生 SM 探针 {e['kernel']}：观察到 {e['observed_sms']}/{e['sm_count']} 个逻辑 SM" + ("" if e["matched"] else "，本轮有不一致"))
        elif kind == "native_done":
            say(f"  原生探针：{e['status']}" + (f"（{e['reason']}）" if e["status"] != "NO_ANOMALY_OBSERVED" else ""))
            for k in e["kernels"]:
                mark = "一致" if k["arithmetic_verdict"] == "PASS_OBSERVED" else k["arithmetic_verdict"]
                say(f"    {k['kernel']}: {mark}，核对 {k['checked_values']:,} 值，不一致至少 {k['bad_values']:,}，SM 覆盖 {k['coverage_verdict']}"
                    + (f"，坏值所在逻辑 SM {k['bad_sms']}" if k["bad_sms"] else ""))
        elif kind in {"fatal", "timeout"}:
            say("  本卡此阶段未完成；其余已选卡仍可继续。")
        if download and kind == "scenario_done" and e["index"] % 4 == 0:
            d = download.state
            say(f"  签名包：{d.status}" + (f" {d.bytes_done}/{d.bytes_total} bytes" if d.bytes_total else ""))

    names = [d.name for d in cards] if cards else ["cpu"]
    run_configs = [c for n in selection["selected"] for c in configs(n)]
    from . import reference
    from .capabilities import execution_stack, timestamp, build_id
    report = {"schema": "aritheia.local.v2", "run_id": uuid.uuid4().hex, "client_version": __version__,
              "cpu_demo": args.cpu_demo, "selection": selection, "started_at": timestamp(),
              "environment": execution_stack(), "build_id": build_id(),
              "reference": {"version": reference.REFERENCE_VERSION, "sha256": reference.REFERENCE_SHA256,
                            "contract": "torch-signed-unit-exact-v1", "iterations": iterations},
              "protocol": "scenarios-then-fresh-probe-process; no physical GPU reset; no clock/power changes", "cards": []}
    report["budget"] = {"steps": args.steps, "native_blocks": config.native_blocks,
                        "native_iterations": config.native_iterations, "probes_only": args.probes_only,
                        "no_native": args.no_native, "parallel_gpus": parallel}
    checkpoint_lock = threading.RLock()
    cancelled = threading.Event()
    download_started = [False]
    report["cards"] = [{**(cards[i].public_card() if cards else {"model":"CPU software run", "memory_mib":0, "core_clock_mhz":None}), "local_device":d, "scenarios":[], "diagnosis":{"status":"INCOMPLETE", "reason":"NOT_STARTED"}, "overall_status":"INCOMPLETE", "workflow_complete":False} for i,d in enumerate(names)]

    def run_card(i, device):
        qualified_failure = {"probes": False, "native": False, "pack": False}
        active_phase = "scenarios"
        first_scene = False
        card = cards[i].public_card() if cards else {"model": "CPU software run", "memory_mib": 0, "core_clock_mhz": None}
        card.update({"local_device": device, "scenarios": [], "diagnosis": {"status": "INCOMPLETE", "reason": "NOT_STARTED"},
                     "native": None, "pack_diagnosis": None, "overall_status": "INCOMPLETE",
                     "clock_scope": "preflight_snapshot_not_load_frequency", "workflow_complete": False})
        card['compute_capability'] = cards[i].compute_capability if cards else None
        card['selection'] = copy.deepcopy(selection)
        card_configs = list(run_configs)
        def checkpoint():
            with checkpoint_lock:
                report["cards"][i] = copy.deepcopy(card)
                save_json(out / "report.json", report)
        def event(e):
            nonlocal first_scene
            if cancelled.is_set():
                raise InterruptedError("GPU run cancelled")
            kind = e.get("event")
            if kind in ("scenario_start", "probe_start", "native_done"):
                say(f"[{device}]")
            if e.get("qualified") is True and e.get("matched") is False:
                qualified_failure[active_phase] = True
            if kind == "scenario_start":
                width = 24
                filled = int((e['index'] - 1) * width / max(e['total'], 1))
                say(f"\n[{'█' * filled}{'░' * (width-filled)}] {e['index']:02d}/{e['total']:02d} {e['title']} · {e['config']} · 正在计算")
                if not first_scene:
                    first_scene = True
                    if download:
                        with checkpoint_lock:
                            if not download_started[0]:
                                download.start()
                                download_started[0] = True
            elif kind == "scenario_done":
                filled = int(e['index'] * 24 / max(e['total'], 1))
                say(f"  [{'█' * filled}{'░' * (24-filled)}] → {e['status']} {fmt_metric(e.get('metric'))} ({e['wall_s']:.1f}s)，剩余 {e['total']-e['index']} 项")
            elif kind == "scenario_error":
                say(f"  → {e['status']} ({e['error_type']})")
            elif kind == "probe_start":
                say(f"精确探针 [{e['index']}/{e['total']}] {e['probe']}")
            elif kind == "probe_done":
                mark = "一致" if e["verdict"] == "PASS_OBSERVED" else e["verdict"]
                say(f"  → {mark}，核对 {e['checked_values']:,} 值，不一致至少 {e['bad_values']:,}")
                if e["bad_values"]:
                    say("  数值不一致；首个反例已保留本地，稍后再决定是否分享。")
            elif kind == "native_iteration":
                if e["iteration"] == 0:
                    say(f"原生 SM 探针 {e['kernel']}：观察到 {e['observed_sms']}/{e['sm_count']} 个逻辑 SM" + ("" if e["matched"] else "，本轮有不一致"))
            elif kind == "native_done":
                say(f"  原生探针：{e['status']}" + (f"（{e['reason']}）" if e["status"] != "NO_ANOMALY_OBSERVED" else ""))
                for k in e["kernels"]:
                    mark = "一致" if k["arithmetic_verdict"] == "PASS_OBSERVED" else k["arithmetic_verdict"]
                    say(f"    {k['kernel']}: {mark}，核对 {k['checked_values']:,} 值，不一致至少 {k['bad_values']:,}，SM 覆盖 {k['coverage_verdict']}"
                        + (f"，坏值所在逻辑 SM {k['bad_sms']}" if k["bad_sms"] else ""))
            elif kind in {"fatal", "timeout"}:
                say("  本卡此阶段未完成；其余已选卡仍可继续。")
            if download and kind == "scenario_done" and e["index"] % 4 == 0:
                d = download.state
                say(f"  签名包：{d.status}" + (f" {d.bytes_done}/{d.bytes_total} bytes" if d.bytes_total else ""))
        event.cancel_event = cancelled
        try:
            checkpoint()
            if run_configs and not args.probes_only:
                p = out / f"card-{i + 1}" / "scenarios"
                run_worker({"mode": "scenarios", "device": device, "out": str(p.resolve()), "configs": run_configs,
                            "precision": "bf16", "steps": args.steps}, event, timeout=config.scenario_timeout)
                card["scenarios"] = load_json(p / "scenarios.json") if (p / "scenarios.json").exists() else []
                checkpoint()
            if guided_full:
                # Wait on the same download, never launch one download per GPU.
                last_notice = time.monotonic()
                while optional_job and optional_job.thread and optional_job.thread.is_alive():
                    if cancelled.is_set():
                        raise InterruptedError("Optional phase cancelled")
                    optional_job.thread.join(timeout=0.5)
                    if time.monotonic() - last_notice >= 10:
                        say(f"[{device}] 八场景已结束，等待本次后台资产下载；可用 Ctrl-C 停止并保留结果。")
                        last_notice = time.monotonic()
                extras, excluded = guided_optional_scope(optional_job, getattr(args,'optional_skip_reason','DOWNLOAD_UNAVAILABLE'))
                card['selection']['selected'] += extras
                card['selection']['excluded'] += excluded
                extra_configs = [c for name in extras for c in configs(name)]
                card_configs += extra_configs
                checkpoint()
                if extra_configs:
                    say(f"[{device}] 继续本次会话的 {len(extras)} 个可选场景，不重新运行已完成八场景。")
                    p = out / f"card-{i + 1}" / "optional-scenarios"
                    run_worker({"mode": "scenarios", "device": device, "out": str(p.resolve()),
                                "configs": extra_configs, "precision": "bf16", "steps": args.steps},
                               event, timeout=config.scenario_timeout)
                    card["scenarios"] += load_json(p / "scenarios.json") if (p / "scenarios.json").exists() else []
                    checkpoint()
            say("\n场景阶段结束；现在在新进程中运行精确探针，避免把场景调用混入固定探针序列。")
            active_phase = "probes"
            d = out / f"card-{i + 1}" / "probes"
            run_worker({"mode": "probes", "device": device, "out": str(d.resolve()), "scenarios": selection["requested"],
                        "iterations": iterations, "memory_roundtrip_mib": config.memory_roundtrip_mib,
                        "cpu_software_run": args.cpu_demo}, event, timeout=config.worker_timeout * max(1, len(run_configs)))
            card["diagnosis"] = load_json(d / "diagnosis.json") if (d / "diagnosis.json").exists() else \
                {"status": "INCOMPLETE", "reason": "PROBE_WORKER_FAILED", "probes": []}
            if not args.cpu_demo:
                from .exact import assess
                recovered = {r['probe_id']:r for r in card['diagnosis'].get('probes', [])}
                for rp in sorted(d.glob('*/result.json')):
                    row=load_json(rp); recovered[row['probe_id']]=row
                expected = len(reference.select(reference.load(), selection['requested'], iterations))+1
                assessment=assess(list(recovered.values()), expected_total=expected)
                prior_failure = card['diagnosis'].get('status') == 'NUMERICAL_ANOMALY_DETECTED'
                card['diagnosis'].update(status=assessment['status'], assessment=assessment, probes=list(recovered.values()),
                                         reference_sha256=reference.REFERENCE_SHA256, provenance='exact_integer_contract')
                if qualified_failure['probes'] or prior_failure:
                    card['diagnosis']['status']='NUMERICAL_ANOMALY_DETECTED'
                    if not assessment['failed_probes']:
                        card['diagnosis'].update(reason='QUALIFIED_FAILURE_EVENT_EVIDENCE_INCOMPLETE', evidence_complete=False)
            scene_ok = args.probes_only or (len(card['scenarios']) == len(card_configs) and all(r.get('status') in ('COMPLETED','MONITOR_ALERT') for r in card['scenarios']))
            card['workflow_complete'] = bool(selection['requested']) and not card['selection']['excluded'] and scene_ok
            checkpoint()
            if not args.no_native and not args.cpu_demo:
                say("\n原生 SM 覆盖探针：通过驱动 API 加载预编译核函数，在新进程中运行。")
                active_phase = "native"
                nd = out / f"card-{i + 1}" / "native"
                run_worker({"mode": "native", "device": device, "out": str(nd.resolve()), "blocks": config.native_blocks,
                            "iterations": config.native_iterations}, event, timeout=config.worker_timeout)
                card["native"] = load_json(nd / "native.json") if (nd / "native.json").exists() else \
                    {"status": "INCOMPLETE", "reason": "NATIVE_WORKER_FAILED", "kernels": []}
                if qualified_failure['native'] or any(k.get('arithmetic_verdict')=='FAIL_NUMERICAL' for k in card['native'].get('kernels', [])):
                    card['native']['status']='NUMERICAL_ANOMALY_DETECTED'
            elif args.no_native:
                card["native"] = {"status": "INCOMPLETE", "reason": "SKIPPED_BY_USER", "kernels": []}
            card["overall_status"] = overall_status(card)
            checkpoint()
        except Exception as exc:
            card["workflow_complete"] = False
            card["worker_error_type"] = type(exc).__name__
        finally:
            for phase, key in (("probes", "diagnosis"), ("native", "native")):
                if qualified_failure[phase]:
                    card[key] = {**(card.get(key) or {}), "status": "NUMERICAL_ANOMALY_DETECTED"}
            card["overall_status"] = overall_status(card)
            checkpoint()

    try:
        pool = ThreadPoolExecutor(max_workers=min(parallel, len(names)))
        futures = [pool.submit(run_card, i, device) for i, device in enumerate(names)]
        try:
            for future in futures:
                future.result()
        except BaseException:
            cancelled.set()
            for future in futures:
                future.cancel()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        chosen = args.pack
        if chosen:
            load_pack(chosen, config.public_keys, config.max_pack_bytes)
        elif download and download.state.path:
            chosen = download.state.path
        else:
            chosen = cached_pack(config.pack_cache, config.public_keys, config.max_pack_bytes)
        if not args.cpu_demo and chosen:
            say("\n附加证据：维护者签名包在独立进程运行。")
            for i, device in enumerate(names):
                active_card=report['cards'][i]
                active_phase='pack'
                qualified_failure={'probes':False,'native':False,'pack':False}
                d = out / f"card-{i + 1}" / "pack"
                job = {"mode": "check", "device": device, "out": str(d.resolve()), "pack": str(Path(chosen).resolve()),
                       "public_keys": config.public_keys, "max_pack_bytes": config.max_pack_bytes,
                       "max_probe_memory_mib": config.max_probe_memory_mib, "max_probe_repeats": config.max_probe_repeats}
                run_worker(job, event, timeout=config.worker_timeout)
                result = load_json(d / "diagnosis.json") if (d / "diagnosis.json").exists() else {"status": "INCOMPLETE", "reason": "DETECTOR_WORKER_FAILED"}
                if qualified_failure['pack']:
                    result['status']='NUMERICAL_ANOMALY_DETECTED'
                report["cards"][i]["pack_diagnosis"] = result
                if result.get('status')=='INCOMPLETE':
                    report['cards'][i]['workflow_complete']=False
                if result.get("status") == "NUMERICAL_ANOMALY_DETECTED":
                    report["cards"][i]["diagnosis"]["status"] = "NUMERICAL_ANOMALY_DETECTED"
                    report["cards"][i]["diagnosis"]["reason"] = "SIGNED_PACK_MISMATCH"
                report["cards"][i]["overall_status"] = overall_status(report["cards"][i])
                save_json(out / "report.json", report)
        if download and download.state.session:
            save_json(out / "session.private.json", {**download.state.session, "api_url": config.api_url.rstrip("/")})
        if optional_job and optional_job.thread:
            say("等待后台可选资产任务收尾；当前 GPU 检测结果已经完成。")
            optional_job.thread.join()
            report['optional_asset_download'] = {'completed': optional_job.completed, 'errors': optional_job.errors}
    finally:
        if active_card is not None:
            last=active_card
            for channel,key in (('probes','diagnosis'),('native','native'),('pack','pack_diagnosis')):
                if qualified_failure.get(channel):
                    current=last.get(key) or {}
                    current.update(status='NUMERICAL_ANOMALY_DETECTED')
                    last[key]=current
            last['overall_status']=overall_status(last)
        if download:
            download.stop()
        if optional_job:
            optional_job.stop()
        report["finished_at"] = timestamp()
        save_json(out / "report.json", report)
    from .html_report import render_report
    render_report(report, out / "report.html")
    display_report(report)
    say("本地结果：" + str(out) + "；可直接打开 report.html")
    if research_enabled:
        from . import telemetry
        try:
            state = telemetry.prepare_auto(out, report, cards)
            state['endpoint'] = config.telemetry_url.rstrip('/')
            save_json(out / 'research.private.json', state)
            telemetry.transmit(out, config)
            say("最小研究摘要已接收；回执与撤回凭据保存在本地 research.private.json。")
        except Exception as exc:
            say(f"研究摘要未上传（{type(exc).__name__}）；检测结果不受影响，可显式 research retry。")
        if not args.no_share_prompt and sys.stdin.isatty():
            research_dialog(out, config)
    elif not args.offline and not args.no_share_prompt and not args.cpu_demo and sys.stdin.isatty():
        share_dialog(out, config)
    codes = [c.get("overall_status", c["diagnosis"]["status"]) for c in report["cards"]]
    if args.cpu_demo:
        ok = all(c.get("workflow_complete") for c in report["cards"]) and all(all(r.get("status") in ("COMPLETED", "MONITOR_ALERT") for r in c["scenarios"]) for c in report["cards"])
        ok = ok and all((c["diagnosis"].get("software_assessment") or {}).get("status") == "NO_ANOMALY_OBSERVED" for c in report["cards"])
        return 0 if ok else 2
    return 1 if "NUMERICAL_ANOMALY_DETECTED" in codes else (2 if any(c != "NO_ANOMALY_OBSERVED" for c in codes) else 0)


def detail_manifest(out):
    from .telemetry import detailed
    report = load_json(out / 'report.json')
    contribution=out/'contribution'
    contribution.mkdir(mode=0o700,exist_ok=True)
    rows=[]
    import hashlib
    for i,card in enumerate(report['cards']):
        payload=detailed(card)
        path=contribution/f'research-detail-{i+1}.json'
        save_json(path,payload)
        rows.append({'file':path.name,'sha256':hashlib.sha256(canonical(payload)).hexdigest(),'bytes':len(canonical(payload))})
    manifest={'files':rows}
    save_json(contribution/'manifest.json',manifest)
    (contribution/'README.txt').write_text(
        'ComputeProof contribution preview. These files stay local until you explicitly confirm upload.\n'
        'They contain whitelisted probe/scenario/SM summaries, not arbitrary logs or raw GPU UUIDs.\n',
        encoding='utf-8')
    return hashlib.sha256(canonical(manifest)).hexdigest()


def research_dialog(out, config):
    from .telemetry import transmit
    try:
        sha=detail_manifest(out)
        say("Contribution 目录已生成："+str(out/'contribution'))
        say("其中含详细 JSON、manifest 和说明文件；包含探针摘要、少量数值、场景指标与 SM 计数，不含任意原始日志/张量。")
        contact=None
        if yes("是否可选提供研究联系邮箱或贡献致谢名字？"):
            email=input("研究联系邮箱（可空）：").strip()
            name=input("贡献致谢名字（可空）：").strip()
            contact={'name':name or None,'email':email or None,
                     'allow_contact':bool(email) and yes("同意就本次研究数据通过此邮箱联系？"),
                     'acknowledge':bool(name) and yes("同意审核后用此名字公开致谢？")}
            if not contact['allow_contact']: contact['email']=None
            if not contact['acknowledge']: contact['name']=None
            if not (contact['email'] or contact['name']): contact=None
        say("Contribution manifest SHA-256："+sha)
        if yes("已查看 contribution 目录，确认上传完整研究 JSON？拒绝不影响使用"):
            transmit(out,config,detail=True,confirmed_detail_sha=sha,contact=contact)
    except Exception as exc:
        say(f"自愿贡献未完成（{type(exc).__name__}）；本地检测结果不受影响。")


def share_dialog(out, config):
    if not yes("愿意预览并分享本次摘要（场景状态/指标与探针判定）及少量实际运算数值，帮助复测研究吗？拒绝不影响使用"):
        return
    email = input("联系邮箱（可空；填写表示同意研究团队就本次数据联系）：").strip()
    name = input("希望出现在贡献致谢名单的名字（可空；不是作者资格承诺）：").strip()
    acknowledge = bool(name) and yes("同意研究团队审核后以该名字公开致谢？")
    if not acknowledge:
        name = ""
    path, sha = prepare(out, share_values=True, name=name, email=email, allow_contact=bool(email), acknowledge=acknowledge)
    say(path.read_text(encoding="utf-8"))
    say("预览 SHA-256：" + sha)
    if yes("确认发送上述内容？姓名/邮箱可以与数值分别留空"):
        try:
            reply = upload_preview(out, config, sha)
            say("接收回执（不是硬件认证）：" + str(reply["receipt_id"]))
        except Exception as e:
            say(f"上传未完成（{type(e).__name__}）；预览保留本地，可稍后显式 upload。")


def doctor(args):
    Config.read(args.config)
    cards = inventory()
    for d in cards:
        say(f"{d.name} · {d.model} · {d.memory_mib} MiB · {d.core_clock_mhz or '未知'} MHz")
    if not cards:
        say("没有可用 CUDA GPU；CPU 软件验收不能当作 GPU 检测。")
    say("索引是进程可见索引；不自动改频率、不上传、不安装驱动。")
    from . import reference
    healthy = True
    try:
        doc = reference.load()
        from .native_smid import frozen_reference
        frozen_reference()
        from .capabilities import native_bf16_supported
        import torch
        healthy = native_bf16_supported(torch)
        say(f"冻结精确参考 {doc['version']}：{len(doc['probes'])} 个探针 × {doc['iterations']} 种子，SHA-256 已核对。")
    except Exception as e:
        healthy = False
        say(f"冻结精确参考不可用：{type(e).__name__}: {e}")
    problems = assets.verify_manifest()
    say("随包资产清单：" + ("全部一致" if not problems else "; ".join(problems)))
    say("场景资产状态：")
    for st in assets.all_status():
        flag = "就绪" if st["ready"] else ("缺失: " + ",".join(st["missing"]) if st["missing"] else "完整性错误: " + str(st["integrity_error"]))
        say(f"  {st['scenario']:<22} {st['tier']:<12} {flag}")
    return 0 if cards and healthy and not problems and all(assets.status(n)["ready"] for n in BUILTIN) else 2


def assets_command(args):
    if args.assets_command == 'install':
        say(str(assets.install_pack(args.pack,args.archive)))
        return 0
    if args.assets_command == "status":
        for st in assets.all_status():
            say(json.dumps(st, ensure_ascii=False))
        try:
            m = assets.optional_manifest()
            for pack in m["packs"]:
                say(json.dumps(assets.pack_status(pack, m), ensure_ascii=False))
        except Exception as e:
            say(f"optional manifest: {type(e).__name__}: {e}")
        return 0
    if args.assets_command == "import":
        names = parse_scenarios(args.scenarios)
        copied = assets.import_tree(args.source, names, args.dest)
        for k, v in copied.items():
            say(f"{v}  {k}")
        say("已导入并核对冻结链；发布前运行 `assets freeze` 更新清单。")
        return 0
    if args.assets_command == "freeze":
        m = assets.freeze_manifest()
        say(f"已记录 {len(m['files'])} 个随包文件的 SHA-256 → {assets.MANIFEST}")
        return 0
    if args.assets_command == "verify":
        problems = assets.verify_manifest()
        say("随包资产清单：" + ("全部一致" if not problems else "\n".join(problems)))
        return 0 if not problems else 2
    if args.assets_command == "download":
        config = Config.read(args.config)
        m = assets.optional_manifest()
        ps = assets.pack_status(args.pack, m)
        say(f"资产包 {args.pack}（{ps['label']}）约 {assets.human_bytes(ps['bytes'])}；钉住: {ps['pinned']}；许可：{ps['license']}")
        if ps["installed"]:
            say("已安装，无需下载。")
            return 0
        if not (args.accept_download or yes("下载并校验该资产包？")):
            say("未下载。")
            return 2
        downloader = make_downloader(config, args.asset_release_url, False)
        downloader(args.pack, m)
        return 0
    return 2


def reference_command(args):
    from . import reference
    if args.reference_command == "list":
        doc = reference.load()
        for s in doc["probes"]:
            say(f"{s['id']:<70} {s['layout']:<11} e={s['exponent']:+d} {s['precision']}")
        say(f"{len(doc['probes'])} probes × {doc['iterations']} seeds · {doc['version']} · sha256 {reference.REFERENCE_SHA256}")
        return 0
    if args.reference_command == "verify":
        def progress(i, n, pid):
            if i % 10 == 0:
                say(f"  [{i + 1}/{n}] {pid}")
        r = reference.verify(limit=args.limit, progress=progress, device=args.device)
        say(json.dumps(r, ensure_ascii=False, indent=2))
        return 0 if r["ok"] and r["pinned_match"] else 2
    return 2


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "deep":
            from . import deep_cases
            config=Config.read(args.config)
            if args.action == "fetch":
                files=deep_cases.fetch(args.run_dir,config)
                say(f"取得 {len(files)} 份签名测试说明，未下载编译包，未执行测试。")
                for path in files: say(str(path))
                return 0
            if not args.case_file:
                raise ValueError("需要 --case 指定已签名 case 文件")
            if args.action in ("preview-download", "download"):
                if args.action == "preview-download":
                    review, sha = deep_cases.download_preview(args.run_dir, args.case_file, config)
                    say(json.dumps(review, ensure_ascii=False, indent=2))
                    say("仅下载确认 SHA-256：" + sha)
                else:
                    path = deep_cases.download(args.run_dir, args.case_file, config, args.confirm)
                    say("编译包已校验并保存于 " + str(path) + "；尚未执行，也未上传结果。")
                return 0
            if args.action in ("preview-result","upload-result"):
                if not args.out: raise ValueError("需要 --out 指定深测结果目录")
                if args.action=="preview-result":
                    path,sha,_=deep_cases.preview_result(args.run_dir,args.case_file,args.out,config)
                    say("请查看 "+str(path)); say("确认 SHA-256："+sha)
                else:
                    deep_cases.upload_result(args.run_dir,args.case_file,args.out,config,args.confirm)
                    say("已接收本次深测白名单结果；不含原始张量和任意日志。")
                return 0
            review,sha,_,_=deep_cases.preview(args.run_dir,args.case_file,config)
            say(json.dumps(review,ensure_ascii=False,indent=2)); say("确认 SHA-256："+sha)
            if args.action == "preview": return 0
            if not args.out: raise ValueError("需要 --out 指定新目录")
            selected=select_devices(inventory(),args.device)
            if len(selected)!=1: raise ValueError("深测一次只能绑定一张原始报告显卡")
            device=selected[0]
            result=deep_cases.run(args.run_dir,args.case_file,config,device,args.out,args.confirm)
            say("深测结果保留本地；执行许可不包含上传。")
            return 1 if result.get('status')=='NUMERICAL_ANOMALY_DETECTED' else 0 if result.get('status')=='NO_ANOMALY_OBSERVED' else 2
        if args.command == "research":
            from . import telemetry
            config=Config.read(args.config)
            if args.action=='preview-detail':
                say(detail_manifest(args.run_dir)); return 0
            if args.action=='upload-detail':
                if not args.confirm or args.confirm != detail_manifest(args.run_dir):
                    raise ValueError('请先 research preview-detail，查看预览并使用 --confirm SHA256')
                telemetry.transmit(args.run_dir,config,detail=True,confirmed_detail_sha=args.confirm)
            elif args.action=='withdraw':
                telemetry.erase(args.run_dir,config)
            else:
                telemetry.transmit(args.run_dir,config)
            return 0
        if args.command in ("check", "guided"):
            return check(args)
        if args.command == "doctor":
            return doctor(args)
        if args.command == "assets":
            return assets_command(args)
        if args.command == "reference":
            return reference_command(args)
        if args.command == "fetch":
            config = Config.read(args.config); cards = selected_cards(args)
            if not consent_bootstrap(args, config, cards):
                say("未联网；现有缓存不受影响。"); return 2
            job = PackJob(config, bootstrap_payload(cards)); job.start()
            try:
                job.thread.join(timeout=185)
                if job.state.path:
                    say(str(job.state.path)); return 0
                say(job.state.message or "未完成，部分文件保留，可稍后再次 fetch。"); return 2
            finally:
                job.stop()
        if args.command == "report":
            display_report(load_json(args.run_dir / "report.json")); return 0
        if args.command == "replay":
            from .exact import replay_failure
            r = replay_failure(args.probe_dir)
            say(json.dumps(r, ensure_ascii=False, indent=2))
            return 1 if r["arithmetic_verdict"] == "FAIL_NUMERICAL" else 0
        if args.command == "export":
            path, sha = prepare(args.run_dir, share_values=args.share_values, name=args.name, email=args.email,
                                allow_contact=args.allow_contact, acknowledge=args.acknowledge)
            say(path.read_text(encoding="utf-8")); say("确认摘要：" + sha); return 0
        if args.command == "upload":
            say(json.dumps(upload_preview(args.run_dir, Config.read(args.config), args.confirm), ensure_ascii=False)); return 0
        if args.command == "withdraw":
            if args.yes or yes("删除本次服务端提交与联系信息？"):
                say(json.dumps(withdraw(args.run_dir, Config.read(args.config)), ensure_ascii=False)); return 0
            return 2
    except KeyboardInterrupt:
        say("已停止；已生成结果保留本地，未自动上传。"); return 130
    except Exception as e:
        say(f"未完成：{type(e).__name__}: {e}"); return 2
    return 2
