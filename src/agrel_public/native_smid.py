# Aritheia Community — GPU scientific-computing reliability check
# Copyright (C) 2026 The Aritheia authors
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE file
# (GNU GPL v3) shipped with this package or <https://www.gnu.org/licenses/>.
"""Native SM-coverage probes (contract native-signed-unit-smid-v1) with nothing to install.

A maintainer-precompiled fatbinary (assets/native/smid_probe.fatbin, see
NATIVE_MANIFEST.json) is loaded through the CUDA driver API from the driver's own
libcuda — no nvcc, no toolkit, no extra Python package. Two kernels run ±1
signed-unit arithmetic whose exact result is order-independent (partial sums
within [-128, 128]) and record the logical SM id (%smid) of every block:

  smid_mma_bf16   BF16 -> FP32 tensor-core mma.sync m16n8k16 chains
  smid_ffma_fp32  FP32 fmaf chains

The host compares fixed block hashes qualified on A800 GPU7 and a healthy
3090 Ti. Mismatching-value counts are lower bounds, grouped by logical SM. Logical SM ids may be non-contiguous; a block whose
start and end SMID differ is counted as migrated/invalid rather than as coverage.
Nothing here maps to a physical core or tensor-core unit.
"""
from __future__ import annotations
import ctypes
import ctypes.util
import sys
import time
from pathlib import Path
import numpy as np
from .common import load_json, save_json, sha256_file

CONTRACT = "native-signed-unit-smid-v1"
NATIVE_DIR = Path(__file__).resolve().parent / "assets" / "native"
FATBIN = NATIVE_DIR / "smid_probe.fatbin"
MANIFEST = NATIVE_DIR / "NATIVE_MANIFEST.json"
PASS, FAIL = "PASS_OBSERVED", "FAIL_NUMERICAL"
NATIVE_REFERENCE = NATIVE_DIR / "native_reference_v1.json"
NATIVE_REFERENCE_SHA256 = "89b2370f18d73d08a3152ef5d5e912f352b8fa447d9ecca58a5b826e054c4b8c"


class DriverError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


CUDA_ERRORS = {100: "CUDA_ERROR_NO_DEVICE", 101: "CUDA_ERROR_INVALID_DEVICE", 200: "CUDA_ERROR_INVALID_IMAGE",
               209: "CUDA_ERROR_NO_BINARY_FOR_GPU", 218: "CUDA_ERROR_INVALID_PTX", 222: "CUDA_ERROR_UNSUPPORTED_PTX_VERSION",
               300: "CUDA_ERROR_INVALID_SOURCE", 700: "CUDA_ERROR_ILLEGAL_ADDRESS", 719: "CUDA_ERROR_LAUNCH_FAILED"}


# ------------------------------------------------------------------ driver API (ctypes)
class Driver:
    def __init__(self):
        name = "nvcuda.dll" if sys.platform.startswith("win") else (ctypes.util.find_library("cuda") or "libcuda.so.1")
        try:
            self.lib = ctypes.CDLL(name)
        except OSError as e:
            raise RuntimeError(f"NVIDIA driver library not found ({name}); the native probe needs only the driver") from e
        self.ctx = None
        self.check(self.lib.cuInit(0), "cuInit")

    def check(self, rc, what):
        if rc != 0:
            raise DriverError(rc, f"{what} failed: CUresult {rc} {CUDA_ERRORS.get(rc, '')}".strip())

    def open(self, ordinal: int):
        dev = ctypes.c_int()
        self.check(self.lib.cuDeviceGet(ctypes.byref(dev), ordinal), "cuDeviceGet")
        ctx = ctypes.c_void_p()
        self.check(self.lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "cuDevicePrimaryCtxRetain")  # shared with Torch
        self.check(self.lib.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
        self.dev, self.ctx = dev, ctx
        uuid = (ctypes.c_ubyte * 16)()
        self.check(self.lib.cuDeviceGetUuid(uuid, dev), "cuDeviceGetUuid")
        return bytes(uuid)  # transient identity check only; never serialised

    def load(self, image: bytes):
        mod = ctypes.c_void_p()
        buf = ctypes.create_string_buffer(image, len(image))
        self.check(self.lib.cuModuleLoadData(ctypes.byref(mod), buf), "cuModuleLoadData")
        self.mod = mod
        self._keep = buf

    def function(self, name: str):
        fn = ctypes.c_void_p()
        self.check(self.lib.cuModuleGetFunction(ctypes.byref(fn), self.mod, name.encode()), "cuModuleGetFunction " + name)
        return fn

    def launch(self, fn, grid: int, block: int, args: list):
        params = (ctypes.c_void_p * len(args))(*[ctypes.addressof(a) for a in args])
        self.check(self.lib.cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, None, params, None), "cuLaunchKernel")
        self.check(self.lib.cuCtxSynchronize(), "cuCtxSynchronize")

    def close(self):
        if self.ctx is not None:
            self.lib.cuDevicePrimaryCtxRelease(self.dev)
            self.ctx = None


# ------------------------------------------------------------------ probe runner
def _uuid_bytes(value):
    """Transient 16-byte device identity for the Torch/driver cross-check; never serialised."""
    import uuid
    try:
        if value is None:
            return None
        if hasattr(value, "bytes"):
            return bytes(value.bytes)
        if isinstance(value, (bytes, bytearray)) and len(value) == 16:
            return bytes(value)
        return uuid.UUID(str(value).lower().replace("gpu-", "")).bytes
    except Exception:
        return None


def manifest() -> dict:
    if not FATBIN.exists() or not MANIFEST.exists():
        raise FileNotFoundError("Native probe fatbinary/manifest not shipped in this installation")
    m = load_json(MANIFEST)
    if m.get("contract") != CONTRACT or sha256_file(FATBIN) != m["fatbin_sha256"]:
        raise ValueError("Native fatbinary does not match its manifest; refusing to load")
    return m


def frozen_reference():
    if sha256_file(NATIVE_REFERENCE) != NATIVE_REFERENCE_SHA256:
        raise ValueError("Native reference digest does not match the pinned reference")
    doc = load_json(NATIVE_REFERENCE)
    if doc["contract"] != CONTRACT or doc["fatbin_sha256"] != manifest()["fatbin_sha256"]:
        raise ValueError("Native reference/kernel contract mismatch")
    return doc


def _per_sm(sm_start, sm_end, bad_per_block):
    valid = (sm_start >= 0) & (sm_end >= 0) & (sm_start == sm_end)
    observed = sorted(int(s) for s in np.unique(sm_start[valid]))
    per = {}
    for s in observed:
        mask = valid & (sm_start == s)
        per[str(s)] = {"blocks": int(mask.sum()), "bad_values": int(bad_per_block[mask].sum()),
                       "bad_blocks": int((bad_per_block[mask] > 0).sum())}
    return observed, int((~valid).sum()), per


def run_native(device: str, out: Path, emit, *, blocks: int = 4096, chains: int = 8, depth: int = 8,
               iterations: int = 4, seed: int = 20260912) -> dict:
    """Hash-check frozen native outputs; preserve findings before any subsequent error."""
    import torch
    from .frozen import compare_blocks
    if not device.startswith("cuda"):
        return {"status": "INCOMPLETE", "reason": "CPU_SOFTWARE_RUN_NOT_GPU_VALIDATION", "contract": CONTRACT, "kernels": []}
    doc = frozen_reference()
    if not (1 <= blocks <= doc["blocks"] and 1 <= iterations <= len(doc["seeds"]) and
            chains == doc["chains"] and depth == doc["depth"] and seed == doc["base_seed"]):
        raise ValueError("Native budget is outside the frozen reference")
    out.mkdir(parents=True, exist_ok=True)
    m = manifest()
    index = torch.device(device).index or 0
    props = torch.cuda.get_device_properties(index)
    sm_count = int(props.multi_processor_count)
    results = []
    error_reason = None
    def snapshot():
        failed = any(r["arithmetic_verdict"] == FAIL for r in results)
        complete = len(results) == 2 and not error_reason and all(r["iterations_completed"] == iterations and r["coverage_verdict"] == "COMPLETE_OBSERVED" for r in results)
        return {"status": "NUMERICAL_ANOMALY_DETECTED" if failed else ("NO_ANOMALY_OBSERVED" if complete else "INCOMPLETE"),
                "contract": CONTRACT, "fatbin_sha256": m["fatbin_sha256"], "reference_sha256": NATIVE_REFERENCE_SHA256,
                "kernels": results, "reason": error_reason or ("NATIVE_EXACT_CONTRACT_OBSERVED" if complete else "SM_COVERAGE_INCOMPLETE"),
                "execution_complete": complete, "bad_value_count_kind": "lower_bound", "physical_unit_claim": False,
                "runtime_reference": "frozen_block_hashes_no_host_gemm"}
    if props.major < 8:
        return {**snapshot(), "reason": "UNSUPPORTED_COMPUTE_CAPABILITY_BELOW_80"}
    drv = None
    try:
        torch.cuda.set_device(index)
        torch.zeros(1, device=device)
        drv = Driver()
        drv_uuid = drv.open(index)
        torch_uuid = _uuid_bytes(getattr(props, "uuid", None))
        if torch_uuid is not None and torch_uuid != drv_uuid:
            error_reason = "DRIVER_TORCH_DEVICE_MISMATCH"
            return snapshot()
        drv.load(FATBIN.read_bytes())
        for kernel, block_threads in (("smid_mma_bf16", 32), ("smid_ffma_fp32", 128)):
            fn = drv.function(kernel)
            elems_per_block = chains * 128
            r = {"kernel": kernel, "contract": CONTRACT, "status": "RUNNING", "arithmetic_verdict": "NOT_ASSESSED",
                 "coverage_verdict": "INCOMPLETE_OBSERVED", "iterations_completed": 0, "iterations_requested": iterations,
                 "blocks_per_iteration": blocks, "chains": chains, "depth": depth, "checked_values": 0, "bad_values": 0,
                 "bad_value_count_kind": "lower_bound", "first_anomaly_s": None, "sm_count": sm_count,
                 "observed_logical_sms": [], "migrated_or_invalid_blocks": 0, "unattributed_bad_values": 0,
                 "bad_sms": [], "per_sm": {}, "sass_gate_declared": m["sass_gate"].get(f"sm_{props.major}{props.minor}", {}).get(kernel, {}).get("gate_passed"), "wall_s": 0.0}
            results.append(r)
            started = time.monotonic()
            observed_all = set()
            for it in range(iterations):
                s = doc["seeds"][it]
                out_t = torch.full((blocks * elems_per_block,), float("nan"), dtype=torch.float32, device=device)
                sm0 = torch.full((blocks,), -1, dtype=torch.int32, device=device)
                sm1 = torch.full((blocks,), -2, dtype=torch.int32, device=device)
                args = [ctypes.c_void_p(out_t.data_ptr()), ctypes.c_void_p(sm0.data_ptr()), ctypes.c_void_p(sm1.data_ptr()),
                        ctypes.c_uint(s), ctypes.c_uint(chains), ctypes.c_uint(depth)]
                drv.launch(fn, blocks, block_threads, args)
                actual = out_t.cpu().numpy()
                sm_start, sm_end = sm0.cpu().numpy().astype(np.int64), sm1.cpu().numpy().astype(np.int64)
                diff = compare_blocks(actual, doc["kernels"][kernel.removeprefix("smid_")][it]["expected_blocks_sha256"][:blocks], elements=elems_per_block)
                bad_per_block = np.zeros(blocks, dtype=np.int64)
                for j in diff["failed_block_indices"]:
                    values = actual[j * elems_per_block:(j + 1) * elems_per_block]
                    bad_per_block[j] = max(1, int(((~np.isfinite(values)) | (np.abs(values) > 128) | (values != np.trunc(values))).sum()))
                observed, invalid, per = _per_sm(sm_start, sm_end, bad_per_block)
                observed_all.update(observed)
                r["migrated_or_invalid_blocks"] += invalid
                r["unattributed_bad_values"] += int(bad_per_block.sum()) - sum(v["bad_values"] for v in per.values())
                for key, values in per.items():
                    acc = r["per_sm"].setdefault(key, {"blocks": 0, "bad_values": 0, "bad_blocks": 0})
                    for f in acc: acc[f] += values[f]
                r["checked_values"] += int(actual.size)
                r["bad_values"] += diff["bad_values"]
                r["iterations_completed"] += 1
                r["observed_logical_sms"] = sorted(observed_all)
                r["bad_sms"] = sorted(int(k) for k, v in r["per_sm"].items() if v["bad_values"])
                r["arithmetic_verdict"] = FAIL if r["bad_values"] else PASS
                r["coverage_verdict"] = "COMPLETE_OBSERVED" if len(observed_all) == sm_count and r["migrated_or_invalid_blocks"] == 0 else "INCOMPLETE_OBSERVED"
                r["wall_s"] = round(time.monotonic() - started, 3)
                if r["bad_values"] and r["first_anomaly_s"] is None: r["first_anomaly_s"] = r["wall_s"]
                emit({"event": "native_iteration", "kernel": kernel, "iteration": it, "matched": diff["matched"],
                      "qualified": True, "bad_values": diff["bad_values"], "observed_sms": len(observed), "sm_count": sm_count})
                save_json(out / "native.json", snapshot())
                if not diff["matched"] and not (out / "first_failure.json").exists():
                    save_json(out / "first_failure.json", {"schema": "computeproof.failure.native.v1", "kernel": kernel,
                        "seed": s, "iteration": it, "blocks": blocks, "chains": chains, "depth": depth,
                        "reference_sha256": NATIVE_REFERENCE_SHA256, "bad_value_count_kind": "lower_bound"})
                    np.savez_compressed(out / "first_failure.npz", actual=actual, sm_start=sm_start, sm_end=sm_end)
                del out_t, sm0, sm1, actual
            r["status"] = "COMPLETED"
    except DriverError as exc:
        error_reason = "UNSUPPORTED_BINARY" if exc.code in (209, 222) and not results else "NATIVE_RUNTIME_ERROR"
    except Exception as exc:
        error_reason = "NATIVE_RUNTIME_ERROR"
    finally:
        if drv is not None:
            try: drv.close()
            except Exception: error_reason = error_reason or "NATIVE_CONTEXT_CLEANUP_ERROR"
    record = snapshot()
    try: save_json(out / "native.json", record)
    except OSError: record["evidence_complete"] = False
    return record


def replay_native(meta, arrays):
    from .frozen import compare_blocks
    doc = frozen_reference()
    kernel, i, blocks = meta["kernel"], meta["iteration"], meta["blocks"]
    if meta["reference_sha256"] != NATIVE_REFERENCE_SHA256 or not 0 <= i < len(doc["seeds"]) or meta["seed"] != doc["seeds"][i]:
        raise ValueError("Native evidence does not match pinned reference")
    if not 1 <= blocks <= doc["blocks"] or meta["chains"] != doc["chains"] or meta["depth"] != doc["depth"]:
        raise ValueError("Native evidence dimensions differ from reference")
    if arrays["actual"].size != blocks * doc["chains"] * 128:
        raise ValueError("Native output size differs from reference")
    diff = compare_blocks(arrays["actual"], doc["kernels"][kernel.removeprefix("smid_")][i]["expected_blocks_sha256"][:blocks], elements=doc["chains"] * 128)
    return {"schema": "computeproof.replay.v1", "channel": "native", "kernel": kernel,
            "arithmetic_verdict": PASS if diff["matched"] else FAIL, **diff,
            "verification": "pinned_reference_hashes; no host GEMM; no GPU"}
