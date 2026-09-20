import copy
import json
from pathlib import Path
import pytest
from agrel_public.common import canonical,digest,save_json,load_json
from agrel_public.devices import Device,bootstrap_payload,select_devices,normalize_uuid
from agrel_public.config import Config
from agrel_public.network import validate_url
from agrel_public.packs import verify_pack,load_pack,cached_pack,PackError
from agrel_public.probes import validate_spec,operands,output_bytes,run_pack

@pytest.mark.parametrize("selection,expected",[("0",[0]),("cuda:2",[2]),("cuda:2,cuda:0",[2,0]),("all",[0,2])])
def test_multi_gpu_selection(selection,expected):
    cards=[Device(0,"same-model",24000,1500),Device(2,"same-model",24000,1600)]
    assert [x.logical_index for x in select_devices(cards,selection)]==expected

@pytest.mark.parametrize("selection",["cuda:1","0,0","-1","cuda:all","cuda:0;rm","0,","cpu"])
def test_bad_selection(selection):
    with pytest.raises(ValueError):
        select_devices([Device(0,"gpu",24000)],selection)

def test_only_three_bootstrap_fields():
    body=bootstrap_payload([Device(7,"gpu",24000,1710),Device(2,"gpu",12000,None)])
    assert set(body)=={"cards"}
    assert all(set(c)=={"model","memory_mib","core_clock_mhz"} for c in body["cards"])
    assert "logical_index" not in canonical(body).decode()

def test_uuid_local_matching():
    assert normalize_uuid("GPU-01234567-1234-1234-1234-123456789abc")=="gpu-01234567-1234-1234-1234-123456789abc"
    assert normalize_uuid("unknown") is None
    assert normalize_uuid(None) is None

@pytest.mark.parametrize("url",["http://example.com", "file:///tmp/a", "https://user:pass@example.com", "https://example.com/#x", "ftp://host", "https://"])
def test_bad_url(url):
    with pytest.raises(ValueError):
        validate_url(url)

def test_dev_http_only_loopback():
    assert validate_url("http://127.0.0.1:80",True)
    with pytest.raises(ValueError):
        validate_url("http://192.168.1.1",True)

def test_signature_and_tamper(signed_pack,tmp_path):
    keys,payload,sign=signed_pack
    raw=sign();assert verify_pack(raw,keys)==payload
    envelope=json.loads(raw);envelope["payload"]["pack_id"]="tampered"
    with pytest.raises(PackError): verify_pack(canonical(envelope),keys)
    with pytest.raises(PackError): verify_pack(raw,{})
    with pytest.raises(PackError): verify_pack(raw,keys,len(raw)-1)
    p=tmp_path/"ok.agpack";p.write_bytes(raw)
    assert load_pack(p,keys,99999)[1]==digest(raw)
    assert cached_pack(tmp_path,keys,99999)==p

@pytest.mark.parametrize("backend",["exec","python","shell","downloaded_module", "../evil"])
def test_no_pack_code(backend,signed_pack):
    keys,payload,sign=signed_pack;p=copy.deepcopy(payload);p["backend"]=backend
    with pytest.raises(PackError):verify_pack(sign(p),keys)

def test_no_fake_qualification(signed_pack):
    keys,p,sign=signed_pack;p=copy.deepcopy(p);p["provenance"]="publisher_validated"
    with pytest.raises(PackError):verify_pack(sign(p),keys)

def spec():
    return {"id":"p1","op":"mm","m":8,"n":8,"k":128,"batch":1,"seeds":[1],"expected_sha256":["a"*64]}

def test_exact_operands_and_determinism():
    import numpy as np
    s=spec();a,b=operands(s,123);c,d=operands(s,123)
    assert np.array_equal(a,c) and np.array_equal(b,d)
    out=a@b
    assert np.max(np.abs(out))<=128 and np.array_equal(out,np.round(out))
    assert output_bytes(np.array([-0.0],dtype="f4"))==output_bytes(np.array([0.0],dtype="f4"))

@pytest.mark.parametrize("field,value",[("k",129),("m",0),("batch",2),("op","conv"),("layout","other"),("seeds",[]),("expected_sha256",["x"]),("m",10**9)])
def test_probe_contract_limits(field,value):
    s=spec();s[field]=value
    with pytest.raises(ValueError):validate_spec(s,512,8)

def test_cpu_cannot_pass(tmp_path):
    result=run_pack({"backend":"fixed_mm_v1"},"cpu",tmp_path,lambda _:None)
    assert result["status"]=="INCOMPLETE"

def test_config_defaults_and_limits(tmp_path):
    p=tmp_path/"config.json";save_json(p,{"network_timeout":0})
    with pytest.raises(ValueError):Config.read(p)
    save_json(p,{"upload_results_automatically":True})
    with pytest.raises(ValueError):Config.read(p)
    assert Config().api_url==""

def test_json_no_nonfinite(tmp_path):
    with pytest.raises(ValueError):canonical({"x":float("nan")})
    p=tmp_path/"bad.json";p.write_text('{"x": NaN}')
    with pytest.raises(ValueError):load_json(p)
