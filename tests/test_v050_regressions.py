"""Software fault injection exercises failure handling; it is not hardware evidence."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from agrel_public import exact, reference, native_smid, frozen
from agrel_public.common import save_json,load_json,digest,canonical

@pytest.mark.parametrize('value',[float('nan'),float('inf'),float('-inf')])
def test_nonfinite_is_durable_and_replayable(value,tmp_path,monkeypatch):
    import torch
    mm=torch.mm
    def corrupt(a,b):
        c=mm(a,b);c[0,0]=value;return c
    monkeypatch.setattr(torch,'mm',corrupt)
    s=reference.select(reference.load(),['cytology'],1)[0]
    r=exact.run_probe(s,'cpu',tmp_path,lambda e:None)
    assert r['arithmetic_verdict']==exact.FAIL and r['bad_values']>=1
    assert exact.replay_failure(tmp_path)['arithmetic_verdict']==exact.FAIL
    assert load_json(tmp_path/'first_failure.json')['schema']=='computeproof.failure.exact.v1'
    assert np.isnan(np.load(tmp_path/'first_failure.npz')['actual'][0,0]) if np.isnan(value) else True


def test_no_runtime_host_oracle_and_reference_verify_nonempty(monkeypatch,tmp_path):
    assert not hasattr(exact,'reference') and not hasattr(reference,'compute_digests')
    def forbidden(*a,**k): raise AssertionError('host matmul called')
    monkeypatch.setattr(np,'matmul',forbidden)
    assert reference.verify()['ok']
    with pytest.raises(ValueError): reference.verify(limit=0)
    s=reference.select(reference.load(),['cytology'],1)[0]
    assert exact.run_probe(s,'cpu',tmp_path,lambda e:None)['arithmetic_verdict']==exact.PASS


def test_input_reference_tamper_prevents_gpu_work(monkeypatch,tmp_path):
    import torch
    monkeypatch.setattr(torch,'mm',lambda *a:pytest.fail('must validate inputs first'))
    s=reference.select(reference.load(),['cytology'],1)[0];s['operand_sha256']=['0'*64]
    r=exact.run_probe(s,'cpu',tmp_path,lambda e:None)
    assert r['status']=='REFERENCE_MISMATCH' and r['bad_values']==0


def test_fail_then_runtime_error_stays_failed(tmp_path,monkeypatch):
    import torch
    mm=torch.mm;calls=[];events=[]
    def corrupt_then_fail(a,b):
        calls.append(1)
        if len(calls)>1:raise RuntimeError('simulated CUDA error')
        c=mm(a,b);c[0,0]+=1;return c
    monkeypatch.setattr(torch,'mm',corrupt_then_fail)
    s=reference.select(reference.load(),['cytology'],2)[0]
    r=exact.run_probe(s,'cpu',tmp_path,events.append)
    assert r['status']=='RUNTIME_ERROR' and r['arithmetic_verdict']==exact.FAIL
    assert r['iterations_completed']==1 and not exact.assess([r])['execution_complete']
    assert exact.assess([r],expected_total=69)['status']=='NUMERICAL_ANOMALY_DETECTED'
    assert events[0]['qualified'] and events[0]['matched'] is False


def test_partial_checkpoint_cannot_look_complete(tmp_path,monkeypatch):
    import torch
    mm=torch.mm;calls=[]
    def stop(a,b):
        calls.append(1)
        if len(calls)==2:raise KeyboardInterrupt()
        return mm(a,b)
    monkeypatch.setattr(torch,'mm',stop)
    s=reference.select(reference.load(),['cytology'],2)[0]
    with pytest.raises(KeyboardInterrupt):exact.run_probe(s,'cpu',tmp_path,lambda e:None)
    r=load_json(tmp_path/'result.json')
    assert r['status']=='RUNNING' and exact.assess([r])['status']=='INCOMPLETE'


def test_input_only_failure_replays(tmp_path,monkeypatch):
    import torch
    orig=exact.compare_exact
    def corrupt_readback(got,want):
        got.flat[0]+=1
        return orig(got,want)
    monkeypatch.setattr(exact,'compare_exact',corrupt_readback)
    s=reference.select(reference.load(),['cytology'],1)[0]
    r=exact.run_probe(s,'cpu',tmp_path,lambda e:None)
    monkeypatch.setattr(exact,'compare_exact',orig)
    replay=exact.replay_failure(tmp_path)
    assert r['arithmetic_verdict']==exact.FAIL and replay['arithmetic_verdict']==exact.FAIL
    assert replay['output']['matched'] and sum(c['bad_values'] for c in replay['channels'])==2


def test_memory_schema_replays_and_checks_source(tmp_path):
    src=np.array([0,1,2],dtype=np.uint8);actual=src^255;actual[1]=0
    save_json(tmp_path/'first_failure.json',dict(schema='computeproof.failure.memory.v1',source_sha256=digest(src.tobytes()),bytes=3))
    np.savez(tmp_path/'first_failure.npz',source=src,actual=actual)
    assert exact.replay_failure(tmp_path)['bad_values']==1
    src[0]=3;np.savez(tmp_path/'first_failure.npz',source=src,actual=actual)
    with pytest.raises(ValueError):exact.replay_failure(tmp_path)


def test_invalid_smid_not_attributed():
    observed,invalid,per=native_smid._per_sm(np.array([-1,2,4,9]),np.array([-1,3,4,9]),np.array([8,5,2,0]))
    assert observed==[4,9] and invalid==2 and sum(v['bad_values'] for v in per.values())==2

@pytest.mark.parametrize('major,cuda,expected',[(8,'11.8',True),(7,'12.4',False),(9,None,False)])
def test_old_torch_bf16_signature(major,cuda,expected):
    from agrel_public.capabilities import native_bf16_supported
    def old():return True
    fake=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,is_bf16_supported=old,current_device=lambda:0,
            get_device_properties=lambda i:SimpleNamespace(major=major)),version=SimpleNamespace(cuda=cuda))
    assert native_bf16_supported(fake)==expected

@pytest.mark.parametrize('reason',['NATIVE_RUNTIME_ERROR','DRIVER_TORCH_DEVICE_MISMATCH','UNSUPPORTED_BINARY','NATIVE_INITIALIZATION_ERROR'])
def test_native_gap_cannot_green(reason):
    from agrel_public.cli import overall_status
    c={'diagnosis':{'status':'NO_ANOMALY_OBSERVED'},'native':{'status':'INCOMPLETE','reason':reason}}
    assert overall_status(c)=='INCOMPLETE'
    c['diagnosis']['status']='NUMERICAL_ANOMALY_DETECTED'
    assert overall_status(c)=='NUMERICAL_ANOMALY_DETECTED'


def test_probes_only_never_demands_optional_assets(monkeypatch):
    from agrel_public.selection import choose
    from agrel_public.scenarios import assets
    monkeypatch.setattr(assets,'status',lambda n:pytest.fail('no app assets needed'))
    r=choose(explicit=['llm_biomed_tables'],large='skip',interactive=False,ask=lambda *a:False,say=lambda *a:None,accept_download=False,probes_only=True)
    assert r['selected']==r['requested']==['llm_biomed_tables']


def test_missing_requested_assets_retained(monkeypatch):
    from agrel_public.selection import choose
    from agrel_public.scenarios import assets
    monkeypatch.setattr(assets,'status',lambda n:dict(ready=False,missing=['x'],integrity_error=None))
    r=choose(explicit=['cytology'],large='skip',interactive=False,ask=lambda *a:False,say=lambda *a:None,accept_download=False)
    assert r['selected']==[] and r['requested']==['cytology'] and len(r['excluded'])==1


def test_hash_blocks_are_lower_bounds():
    expected=np.zeros(8192,dtype=np.float32);actual=expected.copy();actual[0:20]=2
    r=frozen.compare_blocks(actual,frozen.block_hashes(expected),expected_hash=frozen.output_hash(expected))
    assert r['bad_values']==1 and r['bad_value_count_kind']=='lower_bound'
    actual[4096:4100]=np.nan
    assert frozen.compare_blocks(actual,frozen.block_hashes(expected))['bad_values']==5


def test_native_failure_survives_second_kernel_error(tmp_path,monkeypatch):
    import ctypes,torch
    props=SimpleNamespace(major=8,minor=0,multi_processor_count=1,uuid=None)
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda i:props)
    monkeypatch.setattr(torch.cuda,'set_device',lambda i:None)
    orig_full,orig_zeros=torch.full,torch.zeros
    monkeypatch.setattr(torch,'full',lambda *a,**k:orig_full(*a,**{**k,'device':'cpu'}))
    monkeypatch.setattr(torch,'zeros',lambda *a,**k:orig_zeros(*a,**{**k,'device':'cpu'}))
    doc={'blocks':1,'chains':8,'depth':8,'base_seed':20260912,'seeds':[20260912],
         'kernels':{'mma_bf16':[{'expected_blocks_sha256':frozen.block_hashes(np.zeros(1024,dtype='f4'),1024)}]}}
    monkeypatch.setattr(native_smid,'frozen_reference',lambda:doc)
    class Driver:
        def open(self,i):return b'a'*16
        def load(self,image):pass
        def function(self,name):
            if 'ffma' in name:raise RuntimeError('second launch fails')
            return name
        def launch(self,fn,grid,block,args):
            out=np.ctypeslib.as_array((ctypes.c_float*1024).from_address(args[0].value));out[:]=0;out[0]=1
            for a in args[1:3]:ctypes.c_int.from_address(a.value).value=0
        def close(self):pass
    monkeypatch.setattr(native_smid,'Driver',Driver)
    r=native_smid.run_native('cuda:0',tmp_path,lambda e:None,blocks=1,iterations=1)
    assert r['status']=='NUMERICAL_ANOMALY_DETECTED' and r['reason']=='NATIVE_RUNTIME_ERROR'
    assert r['kernels'][0]['bad_values']==1 and r['execution_complete'] is False
    assert load_json(tmp_path/'native.json')['status']=='NUMERICAL_ANOMALY_DETECTED'


def test_sessions_retain_old_credentials_and_bind_origin(tmp_path,monkeypatch):
    import agrel_public.sharing as sharing
    from agrel_public.config import Config
    save_json(tmp_path/'report.json',{'cpu_demo':False})
    body={'cards':[{'model':'gpu','memory_mib':8000,'core_clock_mhz':None}]}
    save_json(tmp_path/'upload_preview.json',body);sha=digest(canonical(body))
    old={'id':'a'*32,'token':'old-token','expires_unix':1,'api_url':'https://collector.example'}
    save_json(tmp_path/'session.private.json',old)
    calls=[]
    class Transport:
        def __init__(self,*a):pass
        def json(self,url,**kwargs):
            calls.append((url,kwargs))
            if url.endswith('/bootstrap'):
                return {'session':{'id':'b'*32,'token':'new-token','expires_unix':9999999999}}
            if kwargs.get('method')=='DELETE':return {'deleted':True}
            return {'receipt_id':'r','payload_sha256':sha}
    monkeypatch.setattr(sharing,'Transport',Transport)
    config=Config(api_url='https://collector.example')
    sharing.upload_preview(tmp_path,config,sha)
    vault=load_json(tmp_path/'sessions.private.json')
    assert len(vault['sessions'])==2
    count=len(calls)
    assert sharing.upload_preview(tmp_path,config,sha)['idempotent_replay']
    assert len(calls)==count
    with pytest.raises(ValueError):sharing.withdraw(tmp_path,Config(api_url='https://another.example'))
    assert len(calls)==count
    assert sharing.withdraw(tmp_path,config)['sessions_withdrawn']==2
    assert {c[1].get('token') for c in calls if c[1].get('method')=='DELETE'}=={'old-token','new-token'}
    assert (tmp_path/'sessions.private.json').stat().st_mode & 0o077 == 0


def test_native_reason_export_never_contains_arbitrary_text():
    from agrel_public.sharing import public_reason
    assert public_reason('failed /home/private secret-token')=='UNSPECIFIED_ERROR'
    assert public_reason('NATIVE_RUNTIME_ERROR')=='NATIVE_RUNTIME_ERROR'


def test_native_replay_uses_fixed_blocks_without_reference_arithmetic(tmp_path,monkeypatch):
    expected=np.zeros(1024,dtype='f4');actual=expected.copy();actual[3]=1
    doc={'blocks':1,'chains':8,'depth':8,'seeds':[11], 'kernels':{'mma_bf16':[{'expected_blocks_sha256':frozen.block_hashes(expected,1024)}]}}
    monkeypatch.setattr(native_smid,'frozen_reference',lambda:doc)
    save_json(tmp_path/'first_failure.json',dict(schema='computeproof.failure.native.v1',kernel='smid_mma_bf16',seed=11,iteration=0,blocks=1,chains=8,depth=8,reference_sha256=native_smid.NATIVE_REFERENCE_SHA256))
    np.savez(tmp_path/'first_failure.npz',actual=actual)
    r=exact.replay_failure(tmp_path)
    assert r['arithmetic_verdict']==exact.FAIL and r['bad_values']==1


def test_doctor_integrity_error_is_nonzero(monkeypatch):
    from agrel_public import cli
    from agrel_public.devices import Device
    monkeypatch.setattr(cli,'inventory',lambda:[Device(0,'gpu',8000)])
    monkeypatch.setattr(reference,'load',lambda:(_ for _ in ()).throw(ValueError('tampered')))
    assert cli.main(['doctor'])==2


def test_parent_keeps_failure_event_when_worker_checkpoint_is_missing(monkeypatch,tmp_path):
    from agrel_public import cli
    from agrel_public.devices import Device
    monkeypatch.setattr(cli,'inventory',lambda:[Device(0,'gpu',8000)])
    def worker(job,event,timeout):
        event({'event':'probe_iteration','qualified':True,'matched':False,'bad_values':1})
        return 124
    monkeypatch.setattr(cli,'run_worker',worker)
    assert cli.main(['check','--offline','--probes-only','--no-native','--scenarios','cytology','--no-share-prompt','--out',str(tmp_path/'run')])==1
    r=load_json(tmp_path/'run/report.json')
    assert r['cards'][0]['diagnosis']['evidence_complete'] is False


def test_requested_application_failure_prevents_zero_exit(monkeypatch,tmp_path):
    from agrel_public import cli
    from agrel_public.devices import Device
    from test_workflow import healthy_probes
    monkeypatch.setattr(cli,'inventory',lambda:[Device(0,'gpu',8000)])
    def worker(job,event,timeout):
        p=Path(job['out'])
        if job['mode']=='scenarios':save_json(p/'scenarios.json',[])
        else:save_json(p/'diagnosis.json',{'status':'NO_ANOMALY_OBSERVED','probes':healthy_probes(job)})
        return 0
    monkeypatch.setattr(cli,'run_worker',worker)
    assert cli.main(['check','--offline','--no-native','--scenarios','cytology','--no-share-prompt','--out',str(tmp_path/'run')])==2
    r=load_json(tmp_path/'run/report.json')
    assert r['cards'][0]['diagnosis']['status']=='NO_ANOMALY_OBSERVED'
    assert r['cards'][0]['workflow_complete'] is False


def test_signed_pack_partial_failure_is_durable(tmp_path,monkeypatch):
    import torch
    from agrel_public import probes,capabilities
    spec={'id':'fixture','op':'mm','m':4,'n':4,'k':4,'batch':1,'seeds':[1,2],'expected_sha256':[]}
    for seed in spec['seeds']:
        a,b=probes.operands(spec,seed);spec['expected_sha256'].append(digest(probes.output_bytes(a@b)))
    to=torch.Tensor.to;mm=torch.mm;count=[]
    def cpu_to(self,*a,**kw):
        if str(kw.get('device','')).startswith('cuda'):kw['device']='cpu'
        return to(self,*a,**kw)
    monkeypatch.setattr(torch.Tensor,'to',cpu_to)
    monkeypatch.setattr(capabilities,'native_bf16_supported',lambda t:True)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda *a:None)
    def bad(a,b):
        count.append(1)
        if len(count)>1:raise RuntimeError('synthetic second iteration failure')
        c=mm(a,b);c[0,0]+=1;return c
    monkeypatch.setattr(torch,'mm',bad)
    payload={'backend':'fixed_mm_v1','pack_id':'unit-fixture','provenance':'publisher_validated','data':{'probes':[spec]}}
    with pytest.raises(RuntimeError):probes.run_pack(payload,'cuda:0',tmp_path,lambda e:None)
    r=load_json(tmp_path/'diagnosis.json')
    assert r['status']=='NUMERICAL_ANOMALY_DETECTED' and r['execution_complete'] is False
