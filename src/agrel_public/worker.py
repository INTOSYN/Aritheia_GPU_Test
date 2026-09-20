# Aritheia Community — GPU scientific-computing reliability check
# Copyright (C) 2026 The Aritheia authors
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE file
# (GNU GPL v3) shipped with this package or <https://www.gnu.org/licenses/>.
"""One-shot worker process. Three modes, each in its own fresh process:
  scenarios  ordinary scenario workloads (natural lane, NOT_ASSESSED arithmetic)
  probes     exact signed-unit probes + byte round trip (arithmetic verdicts)
  check      optional maintainer-signed pack (additional evidence line)
  deep       report-bound, consented compiled follow-up (never a natural run)
A CUDA failure in one mode never leaks into another mode's process."""
from __future__ import annotations
import json
import sys
from pathlib import Path
from .common import load_json, save_json


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def scenarios_mode(job, device, out):
    from .scenarios.engine import run_config
    from .scenarios.registry import config_id, scenario
    rows = []
    total = len(job["configs"])
    for i, config in enumerate(job["configs"]):
        cid = config_id(config)
        title = scenario(config["scenario"])["title"]
        emit({"event": "scenario_start", "index": i + 1, "total": total, "config": cid, "title": title})
        try:
            row = run_config(config, device, job.get("precision", "bf16"), out / cid.replace("/", "_"),
                             steps=job.get("steps", 8), seed=job.get("seed", 20260910))
            rows.append(row)
            emit({"event": "scenario_done", "index": i + 1, "total": total, "config": cid, "title": title,
                  "status": row["status"], "metric": row["metric"], "wall_s": row["task_wall_s"]})
        except Exception as e:
            reason = "DEPENDENCY_MISSING" if isinstance(e, ImportError) or str(e).startswith("DEPENDENCY_MISSING") else type(e).__name__
            status = "DEPENDENCY_MISSING" if reason.startswith("DEPENDENCY_MISSING") else "ASSET_MISSING" if isinstance(e, FileNotFoundError) else "RUNTIME_ERROR"
            rows.append({"scenario": config["scenario"], "title": title, "config": config, "status": status,
                         "arithmetic_verdict": "NOT_ASSESSED", "error_type": type(e).__name__, "reason": reason})
            emit({"event": "scenario_error", "index": i + 1, "total": total, "config": cid, "title": title,
                  "status": status, "error_type": type(e).__name__})
            if status == "RUNTIME_ERROR" and device.startswith("cuda"):
                # A CUDA error can poison this process: stop this card's scenario phase.
                for remaining in job["configs"][i+1:]:
                    rows.append(dict(scenario=remaining['scenario'], title=scenario(remaining['scenario'])['title'], config=remaining,
                                     status='NOT_RUN', arithmetic_verdict='NOT_ASSESSED', reason='PREVIOUS_SCENARIO_RUNTIME_ERROR'))
                break
        save_json(out / "scenarios.json", rows)
    save_json(out / "scenarios.json", rows)
    emit({"event": "worker_done", "completed": sum(r["status"] in ("COMPLETED", "MONITOR_ALERT") for r in rows)})


def probes_mode(job, device, out):
    import torch
    from . import exact, reference
    if not device.startswith("cuda"):
        result = {"status": "INCOMPLETE", "reason": "CPU_SOFTWARE_RUN_NOT_GPU_VALIDATION", "probes": [], "contract": exact.CONTRACT}
        if job.get("cpu_software_run"):
            # Software validation only: the same code path, explicitly labelled.
            doc = reference.load()
            specs = reference.select(doc, job["scenarios"], job["iterations"])
            results = []
            subset = specs[: job.get("cpu_probe_limit", 3)]
            for i, spec in enumerate(subset):
                emit({"event": "probe_start", "index": i + 1, "total": len(subset), "probe": spec["id"]})
                r = exact.run_probe(spec, device, out / f"probe-{i + 1:03d}", emit, frozen=spec)
                results.append(r)
                emit({"event": "probe_done", "index": i + 1, "total": len(subset), "probe": spec["id"], "verdict": r["arithmetic_verdict"],
                      "bad_values": r["bad_values"], "checked_values": r["checked_values"]})
            result["probes"] = results
            result["software_assessment"] = exact.assess(results, expected_total=len(subset))
        save_json(out / "diagnosis.json", result)
        emit({"event": "diagnosis", "status": result["status"]})
        return
    if job.get("precision", "bf16") == "bf16" and not __import__("agrel_public.capabilities", fromlist=["native_bf16_supported"]).native_bf16_supported(torch):
        result = {"status": "INCOMPLETE", "reason": "NATIVE_BF16_UNSUPPORTED", "probes": [], "contract": exact.CONTRACT}
        save_json(out / "diagnosis.json", result)
        emit({"event": "diagnosis", "status": result["status"]})
        return
    doc = reference.load()
    specs = reference.select(doc, job["scenarios"], job["iterations"])
    results = []
    total = len(specs) + 1
    for i, spec in enumerate(specs):
        emit({"event": "probe_start", "index": i + 1, "total": total, "probe": spec["id"]})
        r = exact.run_probe(spec, device, out / f"probe-{i + 1:03d}", emit, frozen=spec)
        results.append(r)
        emit({"event": "probe_done", "index": i + 1, "total": total, "probe": spec["id"], "verdict": r["arithmetic_verdict"],
              "bad_values": r["bad_values"], "checked_values": r["checked_values"]})
        a = exact.assess(results, expected_total=total)
        save_json(out / "diagnosis.json", {"status": a['status'], "assessment": a, "execution_complete": False, "probes": results})
        if r.get('status') in ('RUNTIME_ERROR', 'REFERENCE_MISMATCH'):
            emit({'event':'diagnosis', 'status':a['status']})
            return
    emit({"event": "probe_start", "index": total, "total": total, "probe": "memory_roundtrip"})
    m = exact.run_memory(job["memory_roundtrip_mib"], min(job["iterations"], 4), job.get("seed", 20260910), device,
                         out / "memory_roundtrip", emit)
    results.append(m)
    emit({"event": "probe_done", "index": total, "total": total, "probe": "memory_roundtrip", "verdict": m["arithmetic_verdict"],
          "bad_values": m["bad_values"], "checked_values": m["checked_values"]})
    a = exact.assess(results, expected_total=total)
    result = {"status": a["status"], "reason": "EXACT_CONTRACT_OBSERVED", "assessment": a, "contract": exact.CONTRACT,
              "reference_version": doc["version"], "reference_sha256": reference.REFERENCE_SHA256,
              "provenance": "exact_integer_contract", "iterations": job["iterations"], "probes": results,
              "math_flags": {k: getattr(torch.backends.cuda.matmul, k, None)
                             for k in ("allow_tf32", "allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction")},
              "coverage": "Only the selected scenarios' GEMM shapes, this card, the current stack and the iteration budget"}
    save_json(out / "diagnosis.json", result)
    emit({"event": "diagnosis", "status": result["status"]})


def native_mode(job, device, out):
    from . import native_smid
    try:
        record = native_smid.run_native(device, out, emit, blocks=job['blocks'], chains=8, depth=8,
                                        iterations=job['iterations'], seed=job.get('seed',20260912))
    except Exception:
        record = load_json(out/'native.json') if (out/'native.json').exists() else {'kernels':[]}
        failed = any(k.get('arithmetic_verdict') == 'FAIL_NUMERICAL' for k in record['kernels'])
        record.update(status='NUMERICAL_ANOMALY_DETECTED' if failed else 'INCOMPLETE', reason='NATIVE_INITIALIZATION_ERROR', execution_complete=False)
    save_json(out/'native.json', record)
    emit(dict(event='native_done', status=record['status'], reason=record.get('reason'),
              kernels=[{k:r.get(k) for k in ('kernel','arithmetic_verdict','coverage_verdict','bad_values','checked_values','sm_count','bad_sms')} for r in record['kernels']],
              observed=[len(r.get('observed_logical_sms',[])) for r in record['kernels']]))


def check_mode(job, device, out):
    from .packs import load_pack
    from .probes import run_pack
    payload, sha = load_pack(job["pack"], job["public_keys"], job["max_pack_bytes"])
    result = run_pack(payload, device, out, emit, memory_mib=job["max_probe_memory_mib"], max_repeats=job["max_probe_repeats"])
    result["pack_sha256"] = sha
    save_json(out / "diagnosis.json", result)
    emit({"event": "diagnosis", "status": result["status"]})


def main():
    job = load_json(sys.argv[1])
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=True)
    device = job["device"]
    try:
        import torch
        if device.startswith("cuda"):
            torch.cuda.set_device(device)  # Process-visible ordinal; never SMI physical ordinal.
        elif device != "cpu":
            raise ValueError("Invalid worker device")
        torch.set_num_threads(job.get("threads", 4))
        mode = job["mode"]
        if mode == "scenarios":
            scenarios_mode(job, device, out)
        elif mode == "probes":
            probes_mode(job, device, out)
        elif mode == "native":
            native_mode(job, device, out)
        elif mode == "check":
            check_mode(job, device, out)
        elif mode == 'deep':
            from .deep_cases import run_compiled_worker
            result = run_compiled_worker(job, device, out, emit)
            save_json(out / 'diagnosis.json', result)
            emit({'event':'diagnosis', 'status':result['status']})
        else:
            raise ValueError("Unknown worker mode")
        return 0
    except Exception as e:
        save_json(out / "worker_error.json", {"error_type": type(e).__name__, "status": "INCOMPLETE", "message": str(e)[:300]})
        emit({"event": "fatal", "error_type": type(e).__name__})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
