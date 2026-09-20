"""Exact signed-unit probes: contract, CPU reference, frozen digests, checker sensitivity."""
import numpy as np
import pytest
from agrel_public import exact, reference

def spec(**kw):
    s = {"id":"t/x","scenario":"cytology","config":"cytology/b8/short","op":"mm","batch":1,"m":8,"k":512,"n":16,
         "layout":"contiguous","exponent":0,"precision":"bf16","seeds":[1,2]}
    s.update(kw); return s

def test_operands_obey_contract():
    a, b, active = exact.operands(spec(k=4096, m=64, n=32), 7)
    assert a.shape == (64,4096) and b.shape == (4096,32)
    assert len(active) == 128 and np.all(np.count_nonzero(a, axis=1) <= 128)
    assert set(np.unique(a)) <= {-1,0,1} and set(np.unique(b)) <= {-1,1}
    e = (a[..., active].astype(np.int64) @ b[..., active, :].astype(np.int64)).astype(np.float32)
    assert e.dtype == np.float32 and np.max(np.abs(e)) <= 128 and np.array_equal(e, np.round(e))
    # exact against int64 over the full dense product (zeros contribute nothing)
    assert np.array_equal(e, (a.astype(np.int64) @ b.astype(np.int64)).astype(np.float32))

def test_bmm_operands():
    s = spec(op="bmm", batch=3, m=5, k=64, n=7)
    a, b, active = exact.operands(s, 9)
    assert a.shape == (3,5,64) and b.shape == (3,64,7) and len(active) == 64
    e = (a[..., active].astype(np.int64) @ b[..., active, :].astype(np.int64)).astype(np.float32)
    assert np.array_equal(e, (a.astype(np.int64) @ b.astype(np.int64)).astype(np.float32))

def test_determinism_and_signed_zero():
    a1,b1,_ = exact.operands(spec(), 123); a2,b2,_ = exact.operands(spec(), 123)
    assert np.array_equal(a1,a2) and np.array_equal(b1,b2)
    assert exact.output_digest(np.array([-0.0],dtype="f4")) == exact.output_digest(np.array([0.0],dtype="f4"))

@pytest.mark.parametrize("field,value",[("op","conv"),("m",0),("k",70000),("layout","x"),("exponent",3),("precision","fp8"),("seeds",[]),("batch",2)])
def test_spec_limits(field,value):
    with pytest.raises(ValueError):
        exact.validate_spec(spec(**{field:value}))

def test_element_budget():
    with pytest.raises(ValueError):
        exact.validate_spec(spec(m=10000,n=10000))

def test_probe_passes_on_cpu_and_matches_frozen(tmp_path):
    import torch; torch.set_num_threads(2)
    doc = reference.load()
    s = reference.select(doc, ["cytology"], 2)[0]
    r = exact.run_probe(s, "cpu", tmp_path/"p", lambda e: None, frozen=s)
    assert r["arithmetic_verdict"] == "PASS_OBSERVED" and r["bad_values"] == 0
    assert r["frozen_reference_match"] and r["iterations_completed"] == 2
    assert r["expected_sha256"] == s["expected_sha256"]
    assert r["checked_values"] > 0 and not (tmp_path/"p"/"first_failure.npz").exists()

def test_corrupted_kernel_fixture_is_caught(tmp_path, monkeypatch):
    """A test of the checker with a synthetic corruption; not a GPU fault observation."""
    import torch; torch.set_num_threads(2)
    orig = torch.mm
    def bad(a, b):
        c = orig(a, b); c[0, 0] = c[0, 0] + 1; return c
    monkeypatch.setattr(torch, "mm", bad)
    s = reference.select(reference.load(), ["cytology"], 2)[0]
    r = exact.run_probe(s, "cpu", tmp_path/"p", lambda e: None, frozen=s)
    assert r["arithmetic_verdict"] == "FAIL_NUMERICAL" and r["bad_values"] == 2
    assert (tmp_path/"p"/"first_failure.npz").exists() and r["first_anomaly_s"] is not None
    z = np.load(tmp_path/"p"/"first_failure.npz")
    assert set(z.files) == {"a","b","actual","input_a_actual","input_b_actual"}
    assert exact.assess([r])["status"] == "NUMERICAL_ANOMALY_DETECTED"

def test_reference_mismatch_is_host_problem_not_pass(tmp_path):
    import torch; torch.set_num_threads(2)
    s = dict(reference.select(reference.load(), ["cytology"], 1)[0])
    s["expected_sha256"] = ["0"*64]
    r = exact.run_probe(s, "cpu", tmp_path/"p", lambda e: None, frozen=s)
    assert r["status"] == "REFERENCE_MISMATCH" and not r["frozen_reference_match"]
    assert exact.assess([r])["status"] == "INCOMPLETE"

def test_memory_roundtrip(tmp_path):
    r = exact.run_memory(2, 2, 5, "cpu", tmp_path/"m", lambda e: None)
    assert r["arithmetic_verdict"] == "PASS_OBSERVED" and r["checked_values"] == 2*2*1024**2

def test_assess_levels():
    ok = {"probe_id":"a","status":"COMPLETED","arithmetic_verdict":"PASS_OBSERVED","checked_values":1,"bad_values":0}
    gap = {"probe_id":"b","status":"RESOURCE_LIMIT","arithmetic_verdict":"NOT_ASSESSED","checked_values":0,"bad_values":0}
    assert exact.assess([ok])["status"] == "NO_ANOMALY_OBSERVED"
    assert exact.assess([ok,gap])["status"] == "INCOMPLETE"
    assert exact.assess([])["status"] == "INCOMPLETE"

def test_frozen_reference_pinned_and_reproducible():
    doc = reference.load()
    assert len(doc["probes"]) == 68 and doc["iterations"] == 8
    r = reference.verify(limit=6)
    assert r["ok"] and r["pinned_match"] and r["probes_checked"] == 68
    ids = [s["id"] for s in reference.probe_specs()]
    assert ids == [s["id"] for s in doc["probes"]]

def test_modified_reference_refused(tmp_path):
    p = tmp_path/"ref.json"
    p.write_bytes(reference.REFERENCE_PATH.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="pinned"):
        reference.load(p)

def test_select_truncates_iterations():
    doc = reference.load()
    sel = reference.select(doc, ["drug_screen"], 3)
    assert len(sel) == 6 and all(len(s["seeds"]) == 3 and len(s["expected_sha256"]) == 3 for s in sel)
    with pytest.raises(ValueError):
        reference.select(doc, ["drug_screen"], 9)

def test_replay_confirms_saved_counterexample_and_rejects_tampering(tmp_path, monkeypatch):
    import torch; torch.set_num_threads(2)
    orig = torch.mm
    def bad(a, b):
        c = orig(a, b); c[0, 0] = c[0, 0] + 1; return c
    monkeypatch.setattr(torch, "mm", bad)
    s = reference.select(reference.load(), ["cytology"], 1)[0]
    exact.run_probe(s, "cpu", tmp_path/"p", lambda e: None, frozen=s)
    r = exact.replay_failure(tmp_path/"p")
    assert r["arithmetic_verdict"] == "FAIL_NUMERICAL" and r["bad_values"] == 1 and r["operands_match_declared_seed"]
    z = dict(np.load(tmp_path/"p"/"first_failure.npz"))
    z["a"][0, 0] += 1
    np.savez_compressed(tmp_path/"p"/"first_failure.npz", **z)
    with pytest.raises(ValueError, match="operands"):
        exact.replay_failure(tmp_path/"p")
