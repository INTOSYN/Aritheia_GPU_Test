# Aritheia Community — GPU scientific-computing reliability check
# Copyright (C) 2026 The Aritheia authors
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE file
# (GNU GPL v3) shipped with this package or <https://www.gnu.org/licenses/>.
"""Opt-in sharing. Everything leaves the host through an explicit whitelist built
from report.json, previewed as JSON, and bound to a SHA-256 the user confirms.
No directory walk, no UUID, hostname, path, token, tensor or prompt text."""
from __future__ import annotations
import re
import time
from pathlib import Path
from .common import canonical, digest, load_json, save_json
from .network import Transport

CONSENT_VERSION = "2026-09-v3"
METRIC_KEYS = ("accuracy", "rmse", "recall_at_5", "ndcg_at_10", "cluster_agreement")


REASONS = {'NATIVE_EXACT_CONTRACT_OBSERVED', 'SM_COVERAGE_INCOMPLETE', 'NATIVE_RUNTIME_ERROR',
    'NATIVE_INITIALIZATION_ERROR', 'NATIVE_CONTEXT_CLEANUP_ERROR', 'NATIVE_WORKER_FAILED',
    'DRIVER_TORCH_DEVICE_MISMATCH', 'UNSUPPORTED_COMPUTE_CAPABILITY_BELOW_80', 'UNSUPPORTED_BINARY',
    'SKIPPED_BY_USER', 'CPU_SOFTWARE_RUN_NOT_GPU_VALIDATION'}

def public_reason(value):
    return value if value in REASONS else ('UNSPECIFIED_ERROR' if value else '')


def short_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", ".", text)[:80]


def scenario_rows(card: dict) -> list[dict]:
    rows = []
    for r in card.get("scenarios", [])[:32]:
        m = r.get("metric") or {}
        metric = next(((k, m[k]) for k in METRIC_KEYS if m.get(k) is not None), (None, None))
        cfg = r.get("config", {})
        rows.append({"scenario": r["scenario"], "batch_size": int(cfg.get("batch_size", 0)), "context": str(cfg.get("context", "short")),
                     "status": r["status"], "finite_output": bool(m.get("finite_output", False)),
                     "metric_name": metric[0], "metric_value": float(metric[1]) if metric[1] is not None else None,
                     "task_wall_s": float(r.get("task_wall_s", 0.0))})
    return rows


def make_submission(run_dir: Path, *, share_values=False, name="", email="",
                    allow_contact=False, acknowledge=False) -> dict:
    report = load_json(run_dir / "report.json")
    name, email = name.strip(), email.strip()
    if len(name) > 80 or any(ord(c) < 32 for c in name):
        raise ValueError("Name is too long or contains control characters")
    if email and (len(email) > 254 or not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email)):
        raise ValueError("Email format invalid")
    if email and not allow_contact:
        raise ValueError("Email requires separate contact consent")
    if name and not acknowledge:
        raise ValueError("Name requires separate acknowledgement consent")
    if acknowledge and not name:
        raise ValueError("Acknowledgement consent requires a chosen display name")
    if allow_contact and not email:
        raise ValueError("Contact consent requires an email")
    cards, samples = [], []
    ref = report.get("reference", {})
    for slot, c in enumerate(report["cards"]):
        d = c["diagnosis"]
        pack = c.get("pack_diagnosis") or {}
        card = {k: c[k] for k in ("model", "memory_mib", "core_clock_mhz")}
        completed = {r["scenario"] for r in c.get("scenarios", []) if r.get("status") in ("COMPLETED", "MONITOR_ALERT")}
        native = c.get("native") or {}
        card.update({"diagnosis": c.get("overall_status", d["status"]),
                     "exact_probe_status": d["status"],
                     "native": {"status": native.get("status", "NOT_RUN"), "reason": public_reason(native.get("reason")),
                                "kernels": [{"kernel": k["kernel"], "arithmetic_verdict": k["arithmetic_verdict"], "coverage_verdict": k["coverage_verdict"],
                                             "sm_count": int(k["sm_count"]), "observed_sms": len(k.get("observed_logical_sms", [])),
                                             "migrated_blocks": int(k.get("migrated_or_invalid_blocks", 0)),
                                             "checked_values": int(k["checked_values"]), "bad_values": int(k["bad_values"]), "bad_value_count_kind": k.get("bad_value_count_kind", "exact"),
                                             "unattributed_bad_values": int(k.get("unattributed_bad_values",0)),
                                             "bad_sms": [int(x) for x in k.get("bad_sms", [])][:256],
                                             "bad_values_by_sm": {str(sm): int(v["bad_values"]) for sm, v in k.get("per_sm", {}).items() if v.get("bad_values")}}
                                            for k in native.get("kernels", []) if "kernel" in k]} if native else None,
                     "pack_id": pack.get("pack_id") or (ref.get("version") if d.get("probes") else None),
                     "pack_sha256": pack.get("pack_sha256") or (ref.get("sha256") if d.get("probes") else None),
                     "provenance": pack.get("provenance") or d.get("provenance", "NOT_TESTED"),
                     "experience_completed": min(len(completed), 12),
                     "scenarios": scenario_rows(c),
                     "probes_completed": int((d.get("assessment") or {}).get("probes_completed", 0)),
                     "checked_values": int((d.get("assessment") or {}).get("checked_values", 0)),
                     "bad_values": int((d.get("assessment") or {}).get("bad_values", 0)),
                     "bad_value_count_kind": (d.get("assessment") or {}).get("bad_value_count_kind", "exact"),
                     "workflow_complete": bool(c.get("workflow_complete", False)),
                     "clock_scope": "preflight_snapshot_not_load_frequency"})
        cards.append(card)
        if share_values:
            for r in c.get("scenarios", []):
                if "values" in r:
                    samples.append({"card_slot": slot, "kind": "experience", "source_id": short_id(r["scenario"]),
                                    "shape": r["output_shape"], "values": r["values"][:16]})
            for r in d.get("probes", []) + pack.get("probes", []):
                if "values" in r and r.get("values"):
                    row = {"card_slot": slot, "kind": "probe", "source_id": short_id(r["probe_id"]),
                           "iteration": max(int(r.get("iterations_completed", 1)) - 1, 0) if "iterations_completed" in r else r.get("iteration"),
                           "shape": r.get("output_shape") or [max(1, int(x)) for x in r.get("shape", [1])],
                           "values": r["values"][:32], "failed": bool(r.get("failed"))}
                    if isinstance(r.get("output_sha256"), list) and r["output_sha256"]:
                        row["actual_sha256"] = r["output_sha256"][-1]
                        row["expected_sha256"] = r["expected_sha256"][-1]
                    elif isinstance(r.get("actual_sha256"), str):
                        row["actual_sha256"] = r["actual_sha256"]
                        row["expected_sha256"] = r["expected_sha256"]
                    samples.append(row)
    payload = {"schema": "aritheia.submission.v1", "run_id": report["run_id"],
               "client_version": report.get("client_version"), "build_id": report.get("build_id"),
               "started_at": report.get("started_at"), "finished_at": report.get("finished_at"),
               "environment": {k:report.get("environment",{}).get(k) for k in ("python","os","torch","cuda","driver","numpy","transformers","cryptography")}, "cards": cards, "numeric_samples": samples,
               "consent": {"summary": True, "numeric_values": bool(share_values), "version": CONSENT_VERSION},
               "contact": {"name": name or None, "email": email or None, "allow_contact": bool(allow_contact), "acknowledge": bool(acknowledge)}}
    if len(canonical(payload)) > 2 * 1024 * 1024:
        raise ValueError("Upload preview exceeds 2 MiB; export fewer cards or samples")
    return payload


def prepare(run_dir: Path, **kwargs) -> tuple[Path, str]:
    payload = make_submission(run_dir, **kwargs)
    p = run_dir / "upload_preview.json"
    save_json(p, payload)
    return p, digest(canonical(payload))


def _vault(run_dir):
    p=run_dir/'sessions.private.json'
    vault=load_json(p) if p.exists() else {'schema':'computeproof.sessions.v1','sessions':[]}
    legacy=run_dir/'session.private.json'
    if legacy.exists():
        session=load_json(legacy)
        if session.get('api_url') and not any(v['id']==session['id'] and v['api_url']==session['api_url'] for v in vault['sessions']):
            vault['sessions'].append(session)
            save_json(p,vault)
    return p,vault


def _validate_session(session):
    if not re.fullmatch(r'[a-f0-9]{32}',session['id']) or not isinstance(session['token'],str) or not session['token']:
        raise ValueError('Invalid collector session')


def upload_preview(run_dir: Path, config, confirmed_sha: str) -> dict:
    if load_json(run_dir/'report.json').get('cpu_demo'):
        raise ValueError('CPU 软件演示结果不进入 GPU 征集数据库')
    payload=load_json(run_dir/'upload_preview.json',2*1024*1024)
    if digest(canonical(payload)) != confirmed_sha:
        raise ValueError('预览已改变；请重新确认摘要，未上传。')
    if not config.api_url:
        raise ValueError('未配置公共接收地址；预览保存在本地。')
    origin=config.api_url.rstrip('/')
    t=Transport(config.network_timeout,config.allow_local_http)
    path,vault=_vault(run_dir)
    # An immutable prior receipt is sufficient for an identical explicit retry, even after TTL.
    for v in vault['sessions']:
        if v['api_url']==origin and not v.get('withdrawn') and v.get('payload_sha256')==confirmed_sha and v.get('receipt'):
            return {**v['receipt'],'idempotent_replay':True}
    session=next((v for v in reversed(vault['sessions']) if v['api_url']==origin and not v.get('withdrawn')
                  and v.get('expires_unix',0)>=int(time.time()) and v.get('payload_sha256') in (None,confirmed_sha)),None)
    if session is None:
        cards={'cards':[{k:c[k] for k in ('model','memory_mib','core_clock_mhz')} for c in payload['cards']]}
        session={**t.json(origin+'/v1/bootstrap',payload=cards)['session'],'api_url':origin}
        _validate_session(session)
        vault['sessions'].append(session)
    _validate_session(session)
    # Persist credentials BEFORE submission; uncertain network outcomes remain withdrawable/retryable.
    session['payload_sha256']=confirmed_sha
    save_json(path,vault)
    reply=t.json(origin+f"/v1/sessions/{session['id']}/submissions",payload=payload,token=session['token'])
    session['receipt']=reply
    save_json(path,vault)
    save_json(run_dir/'upload_receipt.json',reply)
    return reply


def withdraw(run_dir: Path, config) -> dict:
    origin=config.api_url.rstrip('/')
    path,vault=_vault(run_dir)
    sessions=[s for s in vault['sessions'] if s['api_url']==origin and not s.get('withdrawn')]
    if not sessions:
        if any(s['api_url']==origin and s.get('withdrawn') for s in vault['sessions']):
            return {'withdrawn':True,'already_withdrawn':True}
        raise ValueError('此接收地址没有保存的撤回凭据；请使用原提交配置。旧版未绑定地址的凭据须先核实来源。')
    t=Transport(config.network_timeout,config.allow_local_http)
    receipts=[]
    for session in sessions:
        _validate_session(session)
        reply=t.json(origin+f"/v1/sessions/{session['id']}",method='DELETE',token=session['token'])
        session['withdrawn']=True; session['withdrawal_receipt']=reply
        save_json(path,vault); receipts.append(reply)
    result={'withdrawn':True,'deleted':True,'sessions_withdrawn':len(receipts),'receipts':receipts}
    save_json(run_dir/'withdrawal_receipt.json',result)
    return result
