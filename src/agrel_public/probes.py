# Aritheia Community — GPU scientific-computing reliability check
# Copyright (C) 2026 The Aritheia authors
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE file
# (GNU GPL v3) shipped with this package or <https://www.gnu.org/licenses/>.
"""Optional signed-pack executor. No runtime reference GPU or host GEMM is required.
Only signed, explicitly qualified packs can yield an observed detector result.
Compiled follow-ups cannot be invoked by ordinary signed-pack execution.
"""
from __future__ import annotations
from pathlib import Path
import math
from .common import digest, save_json, strict_keys

NO_ANOMALY = "NO_ANOMALY_OBSERVED"
ANOMALY = "NUMERICAL_ANOMALY_DETECTED"
INCOMPLETE = "INCOMPLETE"


def validate_spec(spec: dict, memory_mib: int, max_repeats: int):
    import re
    strict_keys(spec, {"id", "op", "m", "n", "k", "batch", "seeds", "expected_sha256"}, {"layout"})
    if not re.fullmatch(r"[a-zA-Z0-9_.-]{1,80}", spec["id"]):
        raise ValueError("Invalid probe ID")
    if spec["op"] not in {"mm", "bmm"}:
        raise ValueError("Only mm/bmm are supported")
    if any(type(spec[k]) is not int or spec[k] <= 0 for k in ("m","n","k","batch")):
        raise ValueError("Dimensions must be positive integers")
    if spec["k"] > 128 or (spec["op"] == "mm" and spec["batch"] != 1):
        raise ValueError("Out of exact arithmetic contract")
    if spec.get("layout", "contiguous") not in {"contiguous", "transpose_b"}:
        raise ValueError("Unsupported layout")
    if not 1 <= len(spec["seeds"]) <= max_repeats:
        raise ValueError("Repeat budget exceeded")
    if len(spec["seeds"]) != len(spec["expected_sha256"]):
        raise ValueError("Each seed needs a frozen reference")
    if any(type(s) is not int or not 0 <= s < 2**32 for s in spec["seeds"]):
        raise ValueError("Bad seed")
    if any(not re.fullmatch(r"[0-9a-f]{64}", h) for h in spec["expected_sha256"]):
        raise ValueError("Bad output digest")
    b,m,n,k = (spec[x] for x in ("batch","m","n","k"))
    # Conservative host+device working-set bound, not a performance claim.
    if 32 * b * (m*k + k*n + m*n) > memory_mib * 1024**2:
        raise ValueError("Probe exceeds this run's explicit memory budget")


def sign_array(shape, seed: int):
    import numpy as np
    # Version-independent integer sequence; fixtures and client use same defined rule.
    total = math.prod(shape)
    x = np.arange(total, dtype=np.uint64) + np.uint64(seed)
    with np.errstate(over="ignore"):
        x = (x ^ (x >> np.uint64(16))) * np.uint64(0x45d9f3b)
        x = (x ^ (x >> np.uint64(16))) * np.uint64(0x45d9f3b)
        x = x ^ (x >> np.uint64(16))
    return ((x & 1).astype(np.float32) * 2 - 1).reshape(shape)


def operands(spec, seed):
    b,m,n,k = (spec[x] for x in ("batch","m","n","k"))
    prefix = (b,) if spec["op"] == "bmm" else ()
    a = sign_array(prefix + (m,k), seed)
    z = sign_array(prefix + (k,n), seed ^ 0x5a5a5a5a)
    return a,z


def output_bytes(array):
    import numpy as np
    # Explicitly canonicalize signed zero; all other float32 decoded bits retained.
    v = np.array(array, dtype="<f4", order="C", copy=True)
    v[v == 0] = 0.0
    return v.tobytes(order="C")


def run_pack(payload: dict, device: str, out: Path, emit, *, memory_mib=512, max_repeats=8):
    if payload['backend'] == 'private_core_v1':
        return {'status': INCOMPLETE, 'reason': 'EXPLICIT_DEEP_CONSENT_REQUIRED', 'probes': []}
    import torch
    import numpy as np
    if not device.startswith("cuda"):
        return {"status": INCOMPLETE, "reason": "CPU_DEMO_NOT_GPU_VALIDATION", "probes": []}
    if not __import__("agrel_public.capabilities", fromlist=["native_bf16_supported"]).native_bf16_supported(torch):
        return {"status": INCOMPLETE, "reason": "NATIVE_BF16_UNSUPPORTED", "probes": []}
    specs = payload["data"].get("probes", [])
    if not isinstance(specs, list) or not 1 <= len(specs) <= 128:
        raise ValueError("No probes or excessive probe count")
    for s in specs:
        validate_spec(s, memory_mib, max_repeats)
    if len({s["id"] for s in specs}) != len(specs):
        raise ValueError("Duplicate probe IDs")
    results = []
    first_saved = False
    expected_runs=sum(len(s['seeds']) for s in specs)
    def checkpoint():
        qualified=payload['provenance']=='publisher_validated'
        failed=any(r['failed'] for r in results)
        status=ANOMALY if qualified and failed else INCOMPLETE
        return dict(status=status, reason='PARTIAL_SIGNED_PACK', pack_id=payload['pack_id'],
                    provenance=payload['provenance'], probes=results, execution_complete=False,
                    iterations_completed=len(results), iterations_requested=expected_runs)
    for i,s in enumerate(specs):
        emit({"event":"probe_start", "index":i+1, "total":len(specs), "probe":s["id"]})
        for iteration,seed in enumerate(s["seeds"]):
            a0,b0 = operands(s, seed)
            a = torch.from_numpy(a0).to(device=device, dtype=torch.bfloat16)
            b = torch.from_numpy(b0).to(device=device, dtype=torch.bfloat16)
            if s.get("layout") == "transpose_b":
                b = b.transpose(-1,-2).contiguous().transpose(-1,-2)
            with torch.inference_mode():
                y = torch.mm(a,b) if s["op"] == "mm" else torch.bmm(a,b)
            torch.cuda.synchronize(device)
            actual = y.float().cpu().numpy()
            h = digest(output_bytes(actual))
            expected = s["expected_sha256"][iteration]
            failed = h != expected
            flat = actual.reshape(-1)
            samples = [float(x) if np.isfinite(x) else None for x in flat[:32]]
            row = {"probe_id":s["id"], "iteration":iteration, "failed":failed,
                   "actual_sha256":h, "expected_sha256":expected,
                   "output_shape":list(actual.shape), "values":samples}
            results.append(row)
            emit({'event':'pack_iteration','qualified':payload['provenance']=='publisher_validated','matched':not failed})
            save_json(out/'diagnosis.json',checkpoint())
            if failed and not first_saved:
                out.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out / "first_failure.npz", a=a0, b=b0, output=actual)
                save_json(out / "first_failure.json", row)
                first_saved = True
            emit({"event":"probe_result", "probe":s["id"], "iteration":iteration,
                  "matched":not failed})
            del a,b,y,actual,a0,b0
    status = ANOMALY if any(r["failed"] for r in results) else NO_ANOMALY
    if payload["provenance"] != "publisher_validated":
        status = INCOMPLETE
    return {"status":status, "reason":"OBSERVED_CONTRACT_ONLY" if status != INCOMPLETE else "SOFTWARE_FIXTURE_NOT_CERTIFICATION",
            "pack_id":payload["pack_id"], "provenance":payload["provenance"],
            "probes":results, "coverage":"Only this pack, selected card, current stack and iteration budget"}
