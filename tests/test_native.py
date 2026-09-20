"""Public interfaces for native SM coverage and per-SM grouping.
The driver-API launch itself requires a supported CUDA GPU."""
import json
import numpy as np
import pytest
from agrel_public import native_smid as n
from agrel_public.common import sha256_file


def test_frozen_native_reference_bound_to_binary():
    d=n.frozen_reference()
    assert d['blocks']==4096 and d['chains']==8 and d['depth']==8 and len(d['seeds'])==4
    for rows in d['kernels'].values():
        assert len(rows)==4 and all(len(r['expected_blocks_sha256'])==4096 for r in rows)
    assert not hasattr(n,'expected_ffma') and not hasattr(n,'expected_mma')

def test_shipped_fatbin_matches_manifest_and_gate():
    m = n.manifest()
    assert m["contract"] == "native-signed-unit-smid-v1"
    assert sha256_file(n.FATBIN) == m["fatbin_sha256"]
    for arch in ("sm_80", "sm_86", "sm_89", "sm_90", "sm_100", "sm_120"):
        gate = m["sass_gate"][arch]
        assert gate["smid_mma_bf16"]["gate_passed"] and gate["smid_mma_bf16"]["instruction_counts"]["HMMA.16816.F32.BF16"] >= 1
        assert gate["smid_ffma_fp32"]["gate_passed"] and gate["smid_ffma_fp32"]["instruction_counts"]["FFMA"] >= 1
    assert n.FATBIN.stat().st_size < 1_000_000


def test_tampered_fatbin_refused(tmp_path, monkeypatch):
    bad = tmp_path / "smid_probe.fatbin"; bad.write_bytes(n.FATBIN.read_bytes() + b"x")
    monkeypatch.setattr(n, "FATBIN", bad)
    with pytest.raises(ValueError, match="manifest"):
        n.manifest()


def test_per_sm_grouping_counts_migration():
    sm_start = np.array([3, 3, 7, 7, 9]); sm_end = np.array([3, 3, 7, 8, 9]); bad = np.array([0, 2, 0, 5, 0])
    observed, migrated, per = n._per_sm(sm_start, sm_end, bad)
    assert observed == [3, 7, 9] and migrated == 1
    assert per["3"] == {"blocks": 2, "bad_values": 2, "bad_blocks": 1} and per["7"]["bad_values"] == 0


def test_cpu_native_is_not_gpu_validation(tmp_path):
    r = n.run_native("cpu", tmp_path, lambda e: None)
    assert r["status"] == "INCOMPLETE" and r["reason"] == "CPU_SOFTWARE_RUN_NOT_GPU_VALIDATION"


def test_overall_status_combination():
    from agrel_public.cli import overall_status
    ok = {"diagnosis": {"status": "NO_ANOMALY_OBSERVED"}}
    assert overall_status({**ok, "native": {"status": "NO_ANOMALY_OBSERVED"}}) == "NO_ANOMALY_OBSERVED"
    assert overall_status({**ok, "native": {"status": "NUMERICAL_ANOMALY_DETECTED"}}) == "NUMERICAL_ANOMALY_DETECTED"
    assert overall_status({**ok, "native": {"status": "INCOMPLETE", "reason": "UNSUPPORTED_COMPUTE_CAPABILITY_BELOW_80"}}) == "INCOMPLETE"
    assert overall_status({**ok, "native": {"status": "INCOMPLETE", "reason": "SM_COVERAGE_INCOMPLETE:smid_mma_bf16"}}) == "INCOMPLETE"
    assert overall_status({"diagnosis": {"status": "INCOMPLETE"}, "native": {"status": "NO_ANOMALY_OBSERVED"}}) == "INCOMPLETE"
    assert overall_status({"diagnosis": {"status": "NUMERICAL_ANOMALY_DETECTED"}, "native": None}) == "NUMERICAL_ANOMALY_DETECTED"


def test_share_whitelist_includes_native_summary_without_identifiers(tmp_path):
    from agrel_public.common import save_json, load_json
    from agrel_public.sharing import prepare
    native = {"status": "NUMERICAL_ANOMALY_DETECTED", "reason": "NATIVE_EXACT_CONTRACT_OBSERVED", "fatbin_sha256": "f" * 64,
              "kernels": [{"kernel": "smid_mma_bf16", "arithmetic_verdict": "FAIL_NUMERICAL", "coverage_verdict": "COMPLETE_OBSERVED",
                           "sm_count": 68, "observed_logical_sms": list(range(68)), "migrated_or_invalid_blocks": 0,
                           "checked_values": 100, "bad_values": 7, "bad_sms": [45], "per_sm": {"45": {"blocks": 60, "bad_values": 7, "bad_blocks": 3}},
                           "secret_path": "/home/NEVER"}]}
    save_json(tmp_path / "report.json", {"run_id": "a" * 32, "cpu_demo": False, "reference": {}, "selection": {}, "cards": [{
        "model": "gpu", "memory_mib": 1, "core_clock_mhz": None, "scenarios": [], "native": native, "overall_status": "NUMERICAL_ANOMALY_DETECTED",
        "diagnosis": {"status": "NO_ANOMALY_OBSERVED", "probes": [], "assessment": {}}}]})
    p, _ = prepare(tmp_path)
    body = load_json(p); text = p.read_text()
    assert body["cards"][0]["diagnosis"] == "NUMERICAL_ANOMALY_DETECTED" and body["cards"][0]["exact_probe_status"] == "NO_ANOMALY_OBSERVED"
    k = body["cards"][0]["native"]["kernels"][0]
    assert k["bad_sms"] == [45] and k["bad_values_by_sm"] == {"45": 7} and k["observed_sms"] == 68
    assert "NEVER" not in text and "observed_logical_sms" not in text
