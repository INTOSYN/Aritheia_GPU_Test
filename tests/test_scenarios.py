"""Real scenario layer on CPU: frozen assets, ordinary workloads, descriptive metrics."""
import pytest
from agrel_public.scenarios import assets
from agrel_public.scenarios.registry import BUILTIN, OPTIONAL, ORDER, SCENARIOS, configs, gemms, config_id

def test_registry_tiers():
    assert len(ORDER) == 12 and len(BUILTIN) == 8 and len(OPTIONAL) == 4
    assert [n for n in ORDER if SCENARIOS[n]["tier"] == "bundled"] == ["drug_screen","drug_finetune","molecule_neighbors","cytology","ocr_cache"]
    assert all(len(configs(n)) == 2 for n in ORDER)

def test_gemm_shapes_follow_real_workloads():
    g = gemms({"scenario":"molecule_neighbors","batch_size":1536,"context":"short"})
    assert g == [{"op":"mm","m":1536,"k":2048,"n":4991}]
    g = gemms({"scenario":"cytology","batch_size":512,"context":"short"})
    assert [x["k"] for x in g] == [30,512,256] and g[-1]["n"] == 2
    g = gemms({"scenario":"biomed_rag","batch_size":1536,"context":"short"})
    assert g[0]["m"] == 300  # never more queries than the frozen dataset has
    assert any(x["op"] == "bmm" for x in gemms({"scenario":"agent_tools","batch_size":1,"context":"long"}))

def test_builtin_assets_ready_and_manifest_matches():
    for name in BUILTIN:
        st = assets.status(name)
        assert st["ready"], st
    assert assets.verify_manifest() == []
    for name in OPTIONAL:
        st = assets.status(name)
        assert st["tier"] == "optional" and (st["ready"] or st["missing"])

def test_public_assets_carry_no_raw_gpu_uuid():
    import re
    pattern = re.compile(r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F-]{27,}")
    for p in assets.PACKAGE_ASSETS.rglob("*.json"):
        assert not pattern.search(p.read_text(encoding="utf-8")), p
    m = assets.optional_manifest()
    assert not pattern.search(str(m))

def test_frozen_chain_detects_tampering(tmp_path, monkeypatch):
    import shutil
    root = tmp_path / "assets"
    for rel in assets.required_files("cytology"):
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(assets.PACKAGE_ASSETS / rel, dst)
    monkeypatch.setattr(assets, "PACKAGE_ASSETS", root)
    monkeypatch.setattr(assets, "roots", lambda: [root])
    assert assets.status("cytology")["ready"]
    p = root / "models/breast_cancer/weights.npz"
    p.write_bytes(p.read_bytes() + b"x")
    st = assets.status("cytology")
    assert not st["ready"] and "integrity" in st["integrity_error"].lower() or "mismatch" in st["integrity_error"].lower()

@pytest.mark.parametrize("name,batch", [("drug_screen",8),("drug_finetune",8),("molecule_neighbors",512),("cytology",512),("ocr_cache",8),
                                        ("genomics_splice",512),("medical_ultrasound",512),("materials_screen",512)])
def test_builtin_scenarios_run_and_stay_not_assessed(tmp_path, name, batch):
    """Exercise the software path with a tiny row budget."""
    import torch
    torch.set_num_threads(2)
    from agrel_public.scenarios.engine import run_config
    row = run_config({"scenario":name,"batch_size":batch,"context":"short"}, "cpu", "bf16", tmp_path / name, steps=2, max_items=64)
    assert row["status"] == "COMPLETED"
    assert row["arithmetic_verdict"] == "NOT_ASSESSED"
    assert row["metric"]["finite_output"] and row["values"]
    assert row["dataset_sha256"] and (row["model_sha256"] or SCENARIOS[name]["kind"] == "retrieval")
    assert (tmp_path / name / "outputs.npz").exists()

def test_metrics_match_frozen_baseline_reports(tmp_path):
    """Descriptive metrics on the complete bundled sets reproduce the frozen training reports."""
    import torch
    torch.set_num_threads(2)
    from agrel_public.common import load_json
    from agrel_public.scenarios.engine import run_config
    row = run_config({"scenario":"cytology","batch_size":512,"context":"short"}, "cpu", "fp32", tmp_path / "c")
    report = load_json(assets.locate("models/breast_cancer/training_report.json"))
    assert abs(row["metric"]["accuracy"] - report["test_accuracy"]) < 1e-6
    assert row["metric"]["evaluated_rows"] == report["test_rows"]

def test_nonfinite_scores_are_alerts_not_verdicts():
    import numpy as np
    from agrel_public.scenarios.engine import metrics
    m = metrics({"scores":np.array([[1.0,float("inf")]],dtype="f4"),"y":np.array([0]),"test_mask":np.array([True])},{"task":"classification"})
    assert m["finite_output"] is False and m["nonfinite_rows"] == 1
