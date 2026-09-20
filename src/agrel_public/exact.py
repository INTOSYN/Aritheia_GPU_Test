# Aritheia Community — GPU scientific-computing reliability check
# Copyright (C) 2026 The Aritheia authors
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE file
# (GNU GPL v3) shipped with this package or <https://www.gnu.org/licenses/>.
"""Exact signed-unit probes at representative GEMM shapes of each scenario configuration.

Contract `torch-signed-unit-exact-v1` (unchanged from Aritheia 0.3.1):
  * A has at most 128 nonzero entries per row, each ±2^e; B entries are ±2^-e,
    e ∈ {-8, 0, 8}. Every nonzero product is exactly ±1 and every partial sum of
    any ordinary summation order lies in [-128, 128].
  * Inputs, products, partial sums and outputs are therefore exactly representable
    in BF16, FP16, FP32 and FP64: no rounding, subnormal, overflow or final
    quantisation ambiguity exists on any correct implementation, whatever
    accumulation order, split-K strategy or intermediate precision a kernel uses.
  * The GPU executes dense mm/bmm. A800/3090 Ti qualified publisher data supply
    fixed whole-output and block hashes; the client never rebuilds a host GEMM.
    This contract applies to ordinary multiply/add implementations, not arbitrary
    approximate algorithms. It does not certify full neural network outputs.

A mismatch is a counterexample against the computation/transfer stack of the
selected card in its current software environment. It is not by itself a claim
about a particular physical tensor core, SM or memory bank.
"""
from __future__ import annotations
import hashlib
import time
from pathlib import Path
import numpy as np
from .common import digest, save_json

CONTRACT = "torch-signed-unit-exact-v1"
MEMORY_CONTRACT = "torch-byte-roundtrip-v1"
ACTIVE = 128
EXPONENTS = (0, 8, -8)
PASS = "PASS_OBSERVED"
FAIL = "FAIL_NUMERICAL"
MAX_DIM = 65536
MAX_ELEMENTS = 64_000_000  # per operand/output, hard budget independent of config


def validate_spec(spec: dict) -> None:
    import re
    from .common import strict_keys
    strict_keys(spec, {"id", "scenario", "config", "op", "batch", "m", "k", "n", "layout", "exponent", "precision", "seeds"},
                {"expected_sha256", "operand_sha256", "expected_blocks_sha256"})
    if not re.fullmatch(r"[A-Za-z0-9_./:-]{1,120}", spec["id"]):
        raise ValueError("Invalid probe ID")
    if spec["op"] not in ("mm", "bmm"):
        raise ValueError("Only mm/bmm are covered by the exact contract")
    for key in ("batch", "m", "k", "n"):
        if type(spec[key]) is not int or not 1 <= spec[key] <= MAX_DIM:
            raise ValueError("Probe dimensions must be integers in 1..65536")
    if spec["op"] == "mm" and spec["batch"] != 1:
        raise ValueError("mm probes have batch 1")
    b, m, k, n = (spec[x] for x in ("batch", "m", "k", "n"))
    if max(b * m * k, b * k * n, b * m * n) > MAX_ELEMENTS:
        raise ValueError("Probe exceeds the finite element budget")
    if spec["layout"] not in ("contiguous", "transposed"):
        raise ValueError("Invalid layout")
    if spec["exponent"] not in EXPONENTS:
        raise ValueError("Invalid exact scaling exponent")
    if spec["precision"] not in ("bf16", "fp16", "fp32"):
        raise ValueError("Invalid probe precision")
    seeds = spec["seeds"]
    if not isinstance(seeds, list) or not 1 <= len(seeds) <= 32 or any(type(s) is not int or not 0 <= s < 2 ** 32 for s in seeds):
        raise ValueError("Invalid seed list")
    for key in ("expected_sha256", "operand_sha256"):
        if key in spec and (len(spec[key]) != len(seeds) or any(not re.fullmatch(r"[0-9a-f]{64}", h) for h in spec[key])):
            raise ValueError("Reference digest list does not match seeds")

    if "expected_blocks_sha256" in spec:
        nblocks = (b*m*n+4095)//4096
        rows = spec["expected_blocks_sha256"]
        if len(rows) != len(seeds) or any(len(row) != nblocks or any(not isinstance(h,str) or not re.fullmatch(r"[0-9a-f]{64}",h) for h in row) for row in rows):
            raise ValueError("Block reference shape/digests do not match probe")


def operands(spec: dict, seed: int):
    """Declared signed-unit operands (int8) and the indices of the ≤128 active columns."""
    b, m, k, n = (spec[x] for x in ("batch", "m", "k", "n"))
    rng = np.random.RandomState(seed)  # legacy generator: frozen across NumPy versions
    active = np.sort(rng.choice(k, min(k, ACTIVE), replace=False))
    prefix = (b,) if spec["op"] == "bmm" else ()
    a = np.zeros(prefix + (m, k), np.int8)
    a[..., active] = rng.choice([-1, 1], prefix + (m, len(active))).astype(np.int8)
    bm = rng.choice([-1, 1], prefix + (k, n)).astype(np.int8)
    return a, bm, active


def canonical_bytes(array) -> bytes:
    v = np.array(array, dtype="<f4", order="C", copy=True)
    v[v == 0] = 0.0  # signed zero is a legal outcome of summing zero products
    return v.tobytes(order="C")


def output_digest(array) -> str:
    return digest(canonical_bytes(array))


def operand_digest(a, b) -> str:
    h = hashlib.sha256(str((a.shape, b.shape)).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    h.update(np.ascontiguousarray(b).tobytes())
    return h.hexdigest()


def first_bad(bad: np.ndarray, limit: int = 32):
    flat = np.flatnonzero(bad.reshape(-1))[:limit]
    return [np.unravel_index(int(i), bad.shape) for i in flat]


def compare_exact(actual: np.ndarray, expected: np.ndarray):
    if actual.shape != expected.shape:
        raise ValueError("Output shape does not match the exact contract")
    bad = (actual != expected) | ~np.isfinite(actual)
    witnesses = []
    for ix in first_bad(bad):
        witnesses.append({"index": [int(v) for v in ix], "expected": float(expected[ix]), "actual": float(actual[ix]) if np.isfinite(actual[ix]) else None,
                          "actual_f32_bits": f"{int(np.float32(actual[ix]).view(np.uint32)):08x}"})
    return int(bad.sum()), witnesses


def estimate_bytes(spec: dict) -> int:
    b, m, k, n = (spec[x] for x in ("batch", "m", "k", "n"))
    return b * ((m * k + k * n) * (2 + 8) + m * n * (2 + 4))


def run_probe(spec: dict, device: str, out: Path, emit, *, frozen: dict | None = None) -> dict:
    """Compare GPU outputs with publisher-frozen block hashes; no CPU GEMM oracle."""
    import torch
    from .frozen import compare_blocks, CHUNK_ELEMENTS, json_number, ReferenceIntegrityError
    validate_spec(spec)
    frozen = frozen or spec
    if not all(k in frozen for k in ("expected_sha256", "operand_sha256", "expected_blocks_sha256")):
        raise ValueError("Publisher-frozen output and operand hashes are required")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[spec["precision"]]
    records, out_digests, values = [], [], []
    checked = bad_total = mismatch_total = 0
    first = None
    started = time.monotonic()
    status, reason = "COMPLETED", None
    reference_ok = True
    out.mkdir(parents=True, exist_ok=True)
    def result():
        return dict(probe_id=spec["id"], scenario=spec["scenario"], config=spec["config"], op=spec["op"],
            shape=[spec["batch"], spec["m"], spec["k"], spec["n"]], layout=spec["layout"], exponent=spec["exponent"],
            precision=spec["precision"], contract=CONTRACT, status=("RUNNING" if status == "COMPLETED" and len(records) < len(spec["seeds"]) else status), reason=reason,
            arithmetic_verdict=FAIL if bad_total else (PASS if status == "COMPLETED" and len(records) == len(spec["seeds"]) else "NOT_ASSESSED"),
            iterations_completed=len(records), iterations_requested=len(spec["seeds"]), checked_values=int(checked),
            bad_values=int(bad_total), bad_value_count_kind="lower_bound", mismatch_blocks=mismatch_total,
            first_anomaly_s=first, output_sha256=out_digests,
            expected_sha256=frozen["expected_sha256"][:len(records)], frozen_reference_match=reference_ok,
            values=values, failed=bool(bad_total), runtime_reference="frozen_output_hashes_no_host_gemm",
            coverage={"sm_coverage_measured": False, "physical_unit_mapping": "NOT_OBSERVABLE"})
    try:
        if device.startswith("cuda"):
            free, _ = torch.cuda.mem_get_info(device)
            if estimate_bytes(spec) + 256 * 1024 ** 2 > free:
                status, reason = "RESOURCE_LIMIT", "INSUFFICIENT_DEVICE_MEMORY"
                return result()
        for iteration, seed in enumerate(spec["seeds"]):
            a_i, b_i, _ = operands(spec, seed)
            if operand_digest(a_i, b_i) != frozen["operand_sha256"][iteration]:
                reference_ok = False
                status, reason = "REFERENCE_MISMATCH", "HOST_INPUT_DIGEST_MISMATCH"
                break
            e = spec["exponent"]
            a_np = np.ldexp(a_i.astype(np.float32), e)
            b_np = np.ldexp(b_i.astype(np.float32), -e)
            a = torch.from_numpy(a_np).to(device=device, dtype=dtype)
            b = torch.from_numpy(b_np).to(device=device, dtype=dtype)
            if spec["layout"] == "transposed":
                a = a.transpose(-1, -2).contiguous().transpose(-1, -2)
                b = b.transpose(-1, -2).contiguous().transpose(-1, -2)
            with torch.inference_mode():
                c = torch.mm(a, b) if spec["op"] == "mm" else torch.bmm(a, b)
                a_back = a.float().cpu().numpy()
                b_back = b.float().cpu().numpy()
                actual = c.float().cpu().numpy()
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            checks, row_bad = [], 0
            for name, got, want in (("input_a", a_back, a_np), ("input_b", b_back, b_np)):
                n_bad, witnesses = compare_exact(got, want)
                checks.append(dict(name=name, checked_values=int(got.size), bad_values=n_bad,
                                   bad_value_count_kind="exact", witnesses=witnesses))
                checked += got.size
                row_bad += n_bad
            diff = compare_blocks(actual, frozen["expected_blocks_sha256"][iteration], expected_hash=frozen["expected_sha256"][iteration])
            checks.append(dict(name=spec["op"], checked_values=int(actual.size), **diff))
            checked += actual.size
            row_bad += diff["bad_values"]
            bad_total += row_bad
            mismatch_total += diff["mismatch_blocks"]
            out_digests.append(diff["actual_sha256"])
            records.append(dict(iteration=iteration, seed=seed, checks=checks,
                                output_sha256=diff["actual_sha256"], reference_sha256=frozen["expected_sha256"][iteration]))
            values = [json_number(x) for x in actual.reshape(-1)[:32]]
            if row_bad and first is None:
                first = round(time.monotonic() - started, 3)
            emit({"event": "probe_iteration", "probe": spec["id"], "iteration": iteration,
                  "matched": row_bad == 0, "qualified": True, "bad_values": row_bad})
            save_json(out / "result.json", result())
            save_json(out / "checks.json", records)
            if row_bad and not (out / "first_failure.json").exists():
                meta = dict(schema="computeproof.failure.exact.v1", probe=spec, iteration=iteration,
                            seed=seed, checks=checks, expected_sha256=frozen["expected_sha256"][iteration])
                save_json(out / "first_failure.json", meta)
                np.savez_compressed(out / "first_failure.npz", a=a_i, b=b_i,
                                    input_a_actual=a_back, input_b_actual=b_back, actual=actual)
            del a, b, c, a_np, b_np, a_back, b_back, actual
    except ReferenceIntegrityError:
        status, reason, reference_ok = "REFERENCE_MISMATCH", "FROZEN_OUTPUT_HASH_CONFLICT", False
    except Exception as exc:
        status, reason = "RUNTIME_ERROR", type(exc).__name__
        # No exception message is exported; errors never erase prior qualified FAILs.
    r = result()
    try:
        save_json(out / "result.json", r)
    except OSError:
        r["evidence_complete"] = False
    return r


def run_memory(memory_mib: int, iterations: int, seed: int, device: str, out: Path, emit) -> dict:
    """Byte round trip with exact XOR; not a host matrix reference or a whole-VRAM test."""
    import torch
    if not 1 <= memory_mib <= 1024 or not 1 <= iterations <= 32:
        raise ValueError("Memory probe budget out of range")
    out.mkdir(parents=True, exist_ok=True)
    checked = bad_total = 0
    records = []
    status, reason = "COMPLETED", None
    def result():
        return dict(probe_id="memory_roundtrip", scenario=None, config=None, op="memory", shape=[memory_mib],
                    contract=MEMORY_CONTRACT, status=("RUNNING" if status == "COMPLETED" and len(records) < iterations else status), reason=reason,
                    arithmetic_verdict=FAIL if bad_total else (PASS if status == "COMPLETED" and len(records) == iterations else "NOT_ASSESSED"),
                    iterations_completed=len(records), iterations_requested=iterations, checked_values=int(checked),
                    bad_values=int(bad_total), bad_value_count_kind="exact", failed=bool(bad_total), values=[],
                    frozen_reference_match=True, coverage={"whole_vram_tested": False, "allocated_memory_mib": memory_mib})
    try:
        if device.startswith("cuda"):
            free, _ = torch.cuda.mem_get_info(device)
            if memory_mib * 1024 ** 2 * 3 + 256 * 1024 ** 2 > free:
                status, reason = "RESOURCE_LIMIT", "INSUFFICIENT_DEVICE_MEMORY"
                return result()
        for iteration in range(iterations):
            s = (seed + iteration * 104729) % 2 ** 32
            rng = np.random.RandomState(s)
            source = rng.randint(0, 256, size=memory_mib * 1024 ** 2, dtype=np.uint8)
            if iteration % 4 != 3:
                source[:] = (0x00, 0xff, 0xaa)[iteration % 4]
                source[1::2] ^= np.uint8(0xff)
            gpu = torch.from_numpy(source).to(device)
            copied = gpu.clone()
            copied.bitwise_xor_(0xff)
            actual = copied.cpu().numpy()
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            n_bad, witnesses = compare_exact(actual, source ^ np.uint8(0xff))
            checked += actual.size
            bad_total += n_bad
            records.append(dict(iteration=iteration, seed=s, bad_values=n_bad, witnesses=witnesses))
            emit({"event": "probe_iteration", "probe": "memory_roundtrip", "iteration": iteration,
                  "matched": n_bad == 0, "qualified": True, "bad_values": n_bad})
            save_json(out / "result.json", result())
            save_json(out / "checks.json", records)
            if n_bad and not (out / "first_failure.json").exists():
                save_json(out / "first_failure.json", dict(schema="computeproof.failure.memory.v1", seed=s, iteration=iteration,
                                                           source_sha256=digest(source.tobytes()), bytes=int(source.size)))
                np.savez_compressed(out / "first_failure.npz", source=source, actual=actual)
            del gpu, copied
    except Exception as exc:
        status, reason = "RUNTIME_ERROR", type(exc).__name__
    return result()


def assess(results: list[dict], *, expected_total: int | None = None) -> dict:
    """Qualified failures are monotonic; incomplete execution is a separate axis."""
    failed = [r["probe_id"] for r in results if r.get("arithmetic_verdict") == FAIL]
    gaps = [r["probe_id"] for r in results if r.get("status") != "COMPLETED" or r.get("iterations_completed", 1) != r.get("iterations_requested", 1) or r.get("arithmetic_verdict") not in (PASS, FAIL)]
    missing = expected_total is not None and len(results) < expected_total
    status = "NUMERICAL_ANOMALY_DETECTED" if failed else ("INCOMPLETE" if not results or gaps or missing else "NO_ANOMALY_OBSERVED")
    return dict(status=status, failed_probes=failed, gaps=[g for g in gaps if g not in failed],
                execution_complete=bool(results) and not gaps and not missing,
                probes_completed=sum(r.get("status") == "COMPLETED" for r in results), probes_total=expected_total or len(results),
                checked_values=sum(r.get("checked_values", 0) for r in results),
                bad_values=sum(r.get("bad_values", 0) for r in results),
                bad_value_count_kind="lower_bound" if any(r.get("bad_value_count_kind") == "lower_bound" for r in results) else "exact")


def replay_failure(probe_dir: Path) -> dict:
    """Verify stored evidence against pinned output hashes, without a GPU or CPU GEMM."""
    from .common import load_json
    from .frozen import compare_blocks
    from . import reference as published
    probe_dir = Path(probe_dir)
    meta = load_json(probe_dir / "first_failure.json")
    with np.load(probe_dir / "first_failure.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    schema = meta.get("schema")
    if schema == "computeproof.failure.native.v1":
        from .native_smid import replay_native
        return replay_native(meta, arrays)
    if schema == "computeproof.failure.memory.v1":
        source, actual = arrays["source"], arrays["actual"]
        if source.dtype != np.uint8 or source.size != meta["bytes"] or digest(source.tobytes()) != meta["source_sha256"]:
            raise ValueError("Stored memory input digest mismatch")
        bad, witnesses = compare_exact(actual, source ^ np.uint8(0xff))
        return dict(schema="computeproof.replay.v1", channel="memory", arithmetic_verdict=FAIL if bad else PASS,
                    bad_values=bad, bad_value_count_kind="exact", witnesses=witnesses)
    if schema != "computeproof.failure.exact.v1":
        raise ValueError("Unsupported evidence schema; use the original version's publisher replay for historical evidence")
    matches = [s for s in published.load()["probes"] if s["id"] == meta["probe"]["id"]]
    if len(matches) != 1:
        raise ValueError("Probe is not in this pinned reference")
    spec = matches[0]
    i = meta["iteration"]
    if not 0 <= i < len(spec["seeds"]) or spec["seeds"][i] != meta["seed"]:
        raise ValueError("Evidence seed is not in the pinned reference")
    a, b = arrays["a"], arrays["b"]
    if operand_digest(a, b) != spec["operand_sha256"][i]:
        raise ValueError("Stored operands differ from the pinned reference")
    expected_shape = ((spec["batch"],) if spec["op"] == "bmm" else ()) + (spec["m"], spec["n"])
    if arrays["actual"].shape != expected_shape:
        raise ValueError("Stored output shape differs from reference")
    bad = 0
    channels = []
    for name, want in (("input_a_actual", np.ldexp(a.astype(np.float32), spec["exponent"])),
                       ("input_b_actual", np.ldexp(b.astype(np.float32), -spec["exponent"]))):
        n, _ = compare_exact(arrays[name], want)
        bad += n
        channels.append({"channel": name, "bad_values": n, "bad_value_count_kind": "exact"})
    diff = compare_blocks(arrays["actual"], spec["expected_blocks_sha256"][i], expected_hash=spec["expected_sha256"][i])
    bad += diff["bad_values"]
    return dict(schema="computeproof.replay.v1", probe_id=spec["id"], arithmetic_verdict=FAIL if bad else PASS,
                bad_values=bad, bad_value_count_kind="lower_bound", channels=channels, output=diff,
                verification="pinned_reference_hashes; no host GEMM; no GPU", operands_match_declared_seed=True)
