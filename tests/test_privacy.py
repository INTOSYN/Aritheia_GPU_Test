import copy
import pytest
from agrel_public.common import canonical,digest,save_json,load_json
from agrel_public.sharing import prepare,make_submission,upload_preview
from agrel_public.config import Config

@pytest.fixture
def run(tmp_path):
    save_json(tmp_path/"report.json",{"run_id":"a"*32,"cpu_demo":False,"schema":"aritheia.local.v2",
        "reference":{"version":"ref-v","sha256":"c"*64},
        "selection":{"selected":["drug_screen"],"excluded":[{"scenario":"x","reason":"y","detail":"/home/NEVER_UPLOAD"}]},
        "cards":[{
        "model":"NVIDIA Test","memory_mib":24000,"core_clock_mhz":1710,
        "uuid":"NEVER_UPLOAD","hostname":"NEVER_UPLOAD","local_device":"cuda:7",
        "scenarios":[{"scenario":"drug_screen","title":"t","config":{"batch_size":8,"context":"short"},"status":"COMPLETED",
                      "metric":{"finite_output":True,"rmse":0.47,"evaluated_rows":158},"output_shape":[2],"values":[0.1,0.2],
                      "task_wall_s":0.5,"detail":{"actions":"NEVER_UPLOAD"},"dataset_sha256":"d"*64}],
        "diagnosis":{"status":"NO_ANOMALY_OBSERVED","reason":"EXACT_CONTRACT_OBSERVED","provenance":"exact_integer_contract",
                     "assessment":{"probes_completed":1,"probes_total":1,"checked_values":100,"bad_values":0},
                     "probes":[{"probe_id":"drug_screen/b8/short/g0-mm-1x8x2048x512","iterations_completed":2,"shape":[1,8,2048,512],
                                "values":[3.0,-1.0],"failed":False,"output_sha256":["a"*64,"b"*64],"expected_sha256":["a"*64,"b"*64]}]},
        "pack_diagnosis":None}]})
    return tmp_path

def test_default_export_omits_values_and_pii(run):
    p,sha=prepare(run)
    text=p.read_text();body=load_json(p)
    assert body["numeric_samples"]==[]
    assert body["contact"]["name"] is None
    assert "NEVER_UPLOAD" not in text and "cuda:7" not in text and "/home" not in text
    assert body["cards"][0]["scenarios"][0]["metric_value"]==0.47 and body["cards"][0]["provenance"]=="exact_integer_contract"
    assert body["cards"][0]["pack_sha256"]=="c"*64 and body["consent"]["version"]=="2026-09-v3"
    assert sha==digest(canonical(body))

def test_optin_values(run):
    p,_=prepare(run,share_values=True)
    samples=load_json(p)["numeric_samples"]
    assert samples[0]["values"]==[0.1,0.2] and samples[0]["kind"]=="experience"
    assert samples[1]["kind"]=="probe" and samples[1]["iteration"]==1 and samples[1]["actual_sha256"]=="b"*64
    import re
    assert re.fullmatch(r"[A-Za-z0-9_.-]{1,80}",samples[1]["source_id"])

@pytest.mark.parametrize("kwargs",[
    {"name":"name"},{"email":"a@example.org"},{"acknowledge":True},{"allow_contact":True},
    {"email":"invalid","allow_contact":True},{"name":"x\ny","acknowledge":True}])
def test_separate_contact_consent(run,kwargs):
    with pytest.raises(ValueError):make_submission(run,**kwargs)

def test_contact_independent_of_values(run):
    p=make_submission(run,name="Research contributor",acknowledge=True)
    assert not p["consent"]["numeric_values"]
    assert p["contact"]["email"] is None

def test_confirm_binds_bytes(run):
    p,sha=prepare(run,share_values=True)
    data=load_json(p);data["numeric_samples"][0]["values"][0]=0.3;save_json(p,data)
    with pytest.raises(ValueError,match="预览已改变"):
        upload_preview(run,Config(),sha)

def test_unconfigured_server_no_network(run):
    _,sha=prepare(run)
    with pytest.raises(ValueError,match="未配置"):
        upload_preview(run,Config(),sha)
