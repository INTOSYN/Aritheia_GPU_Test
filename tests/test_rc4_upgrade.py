"""Protocol/software fixtures only. No GPU defect observations or production I/O."""
import copy
import hashlib
from importlib import import_module
from pathlib import Path
import pytest
from agrel_public import cli, telemetry, deep_cases
from agrel_public.common import canonical, save_json, load_json
from agrel_public.config import Config
from agrel_public.devices import Device
from agrel_public.scenarios.registry import ORDER, BUILTIN, OPTIONAL, SCENARIOS, configs
from test_research import fixture_report
from test_workflow import healthy_probes


def test_healthy_summary_keeps_twenty_four_config_denominators():
    r=fixture_report(); c=r['cards'][0]
    c.update(overall_status='NO_ANOMALY_OBSERVED',workflow_complete=True,compute_capability='8.6')
    s=telemetry.summary(r,c)
    assert s['schema']=='computeproof.summary.v3' and len(s['coverage'])==24
    assert sum(x['selected'] for x in s['coverage'])==2
    assert s['compute_capability']==[8,6]
    assert 'diagnostics' not in s and 'stack' in s and 'reference_sha256' in s
    assert 'metrics' not in str(s['coverage'])


def test_monitor_alert_not_hidden_by_clean_independent_probes():
    r=fixture_report(); c=r['cards'][0]
    c.update(overall_status='NO_ANOMALY_OBSERVED',workflow_complete=True)
    c['scenarios']=[dict(scenario='cytology',config=configs('cytology')[0],status='MONITOR_ALERT')]
    s=telemetry.summary(r,c)
    assert s['result_class']=='ATTENTION_REQUIRED'
    assert s['channels']['task_monitor']=='ALERT'
    assert s['diagnostics']['overall']=='NO_ANOMALY_OBSERVED'  # Not a fabricated hardware fault.


def test_detail_preserves_multiple_metrics_flags_and_decision_hashes():
    r=fixture_report(); c=r['cards'][0]
    c['scenarios']=[dict(scenario='biomed_rag',config=configs('biomed_rag')[0],status='COMPLETED',
        precision='bf16',metric={'finite_output':True,'recall_at_5':.5,'ndcg_at_10':.7,'private':'secret'},
        scores_sha256='1'*64,dataset_sha256='2'*64,model_sha256=None,
        math_flags={'allow_tf32':False,'private':'secret'},
        decision_audit={'order_sha256':'3'*64,'membership_sha256':'4'*64,
                        'rule':'stable_descending_index','nonfinite_count':0})]
    d=telemetry.detailed(c); row=d['scenario_evidence'][0]
    assert row['metrics']=={'recall_at_5':.5,'ndcg_at_10':.7}
    assert row['scores_sha256']=='1'*64 and row['decision_membership_sha256']=='4'*64
    assert row['arithmetic']=='NOT_ASSESSED' and b'secret' not in canonical(d)


@pytest.mark.parametrize('name',ORDER)
def test_each_scenario_has_actual_public_entry(name):
    module=import_module('agrel_public.scenarios.'+name)
    assert module.SPEC is SCENARIOS[name] and callable(module.execute)
    root=Path(module.__file__).parent
    assert (root/'__main__.py').is_file() and (root/'README.md').is_file()


@pytest.mark.parametrize('optional_ready',[True,False])
def test_guided_runs_ready_optionals_in_same_session(tmp_path,monkeypatch,optional_ready):
    monkeypatch.setattr(cli,'inventory',lambda:[Device(0,'GPU',8000)])
    monkeypatch.setattr(cli.assets,'status',lambda n:dict(ready=n in BUILTIN or optional_ready,missing=[],integrity_error=[]))
    monkeypatch.setattr(cli,'guided_asset_countdown',lambda *a:None)
    phases=[]
    def worker(job,event,timeout):
        phases.append(job['mode'])
        if job['mode']=='scenarios':
            rows=[dict(scenario=c['scenario'],config=c,status='COMPLETED',metric={'finite_output':True})
                  for c in job['configs']]
            save_json(Path(job['out'])/'scenarios.json',rows)
        else:
            save_json(Path(job['out'])/'diagnosis.json',dict(status='NO_ANOMALY_OBSERVED',probes=healthy_probes(job)))
    monkeypatch.setattr(cli,'run_worker',worker)
    out=tmp_path/'guided'
    rc=cli.main(['guided','--offline','--no-native','--iterations','1','--out',str(out)])
    report=load_json(out/'report.json'); c=report['cards'][0]
    assert len(c['scenarios'])==(24 if optional_ready else 16)
    assert phases==(['scenarios','scenarios','probes'] if optional_ready else ['scenarios','probes'])
    assert c['workflow_complete']==optional_ready
    assert rc==(0 if optional_ready else 2)


@pytest.fixture
def signed_case(signed_pack):
    keys,pack,sign=signed_pack
    pack=copy.deepcopy(pack);pack.update(provenance='publisher_validated',reference_receipt='software-test-only')
    record={'id':'a'*32,'summary':{'result_class':'ATTENTION_REQUIRED','gpu_model':'Fixture GPU','device_hash':'b'*64}}
    p={'schema':'computeproof.case.v1','case_id':'fixture-case','report_id':record['id'],
       'source_summary_sha256':hashlib.sha256(canonical(record['summary'])).hexdigest(),
       'issued_unix':100,'expires_unix':200,'purpose':'confirm_exact_mismatch',
       'budget':{'timeout_s':240,'memory_mib':512,'max_repeats':8},'pack':__import__('json').loads(sign(pack))}
    return keys,record,p,lambda p:__import__('json').loads(sign(p))


def test_case_signature_source_expiry_budget_and_no_remote_code(signed_case):
    keys,r,p,sign=signed_case
    assert deep_cases.validate_case(sign(p),keys,r,now=150)[0]['case_id']=='fixture-case'
    for key,value in [('report_id','c'*32),('source_summary_sha256','d'*64),('expires_unix',120),
                      ('budget',{'timeout_s':1801,'memory_mib':512,'max_repeats':8})]:
        changed=copy.deepcopy(p);changed[key]=value
        with pytest.raises(ValueError): deep_cases.validate_case(sign(changed),keys,r,now=150)
    changed=sign(p);changed['payload']['budget']['timeout_s']=30
    with pytest.raises(Exception):deep_cases.validate_case(changed,keys,r,now=150)
    changed=copy.deepcopy(p);changed['script']='print(1)'
    with pytest.raises(ValueError):deep_cases.validate_case(sign(changed),keys,r,now=150)


def test_deep_run_requires_review_and_same_gpu_before_worker(tmp_path,monkeypatch,signed_case):
    keys,r,p,sign=signed_case
    monkeypatch.setattr(deep_cases.time,'time',lambda:150)
    cfg=Config(public_keys=keys)
    save_json(tmp_path/'research.private.json',{'endpoint':cfg.telemetry_url,'records':[r]})
    path=tmp_path/'case.json';save_json(path,sign(p))
    review,sha,_,_=deep_cases.preview(tmp_path,path,cfg)
    with pytest.raises(ValueError):
        deep_cases.run(tmp_path,path,cfg,Device(0,'Fixture GPU',8000),tmp_path/'run',None)
    with pytest.raises(ValueError):
        deep_cases.run(tmp_path,path,cfg,Device(0,'Fixture GPU',8000),tmp_path/'run',sha)
    assert not (tmp_path/'run').exists() and review['upload_results'] is False

def test_deep_followup_download_execute_and_upload_are_separate(tmp_path,monkeypatch,signed_case):
    from agrel_public import processes
    keys,r,p,sign=signed_case
    monkeypatch.setattr(deep_cases.time,'time',lambda:150)
    cfg=Config(public_keys=keys)
    r.update(token='x'*43,receipt={'accepted':True})
    save_json(tmp_path/'research.private.json',{'endpoint':cfg.telemetry_url,'records':[r]})
    calls=[]
    def network(self,url,**kwargs):
        calls.append((url,kwargs))
        if kwargs['method']=='GET':return {'cases':[sign(p)]}
        return {'accepted':True,'review_state':'UNVERIFIED_CLIENT_REPORT'}
    monkeypatch.setattr(deep_cases.Transport,'json',network)
    files=deep_cases.fetch(tmp_path,cfg)
    assert len(files)==1 and len(calls)==1
    path=files[0]; _,review,_,_=deep_cases.preview(tmp_path,path,cfg)
    monkeypatch.setattr(telemetry,'device_hash',lambda identity:'b'*64)
    def worker(job,event,timeout):
        assert timeout==240 and job['max_probe_memory_mib']==512 and job['max_probe_repeats']==8
        assert job['mode']=='check' and job['device']=='cuda:0'
        event({'event':'pack_iteration','qualified':True,'matched':False,'private':'must-not-share'})
        save_json(Path(job['out'])/'diagnosis.json',{'status':'NUMERICAL_ANOMALY_DETECTED','probes':[
            {'probe_id':'software-fixture','iteration':0,'actual_sha256':'1'*64,'expected_sha256':'2'*64,
             'secret':'must-not-share'}]})
    monkeypatch.setattr(processes,'run_worker',worker)
    out=tmp_path/'followup'
    result=deep_cases.run(tmp_path,path,cfg,Device(0,'Fixture GPU',8000),out,review)
    assert result['status']=='NUMERICAL_ANOMALY_DETECTED' and len(calls)==1
    assert load_json(out/'case-result.json')['qualified_failure_event']
    with pytest.raises(FileExistsError):deep_cases.run(tmp_path,path,cfg,Device(0,'Fixture GPU',8000),out,review)
    shared,sha,_=deep_cases.preview_result(tmp_path,path,out,cfg)
    assert b'must-not-share' not in shared.read_bytes()
    with pytest.raises(ValueError):deep_cases.upload_result(tmp_path,path,out,cfg,'0'*64)
    assert len(calls)==1
    deep_cases.upload_result(tmp_path,path,out,cfg,sha)
    assert len(calls)==2 and calls[-1][1]['payload']['consent'] is True
    assert calls[-1][1]['payload']['execution_lane']=='separate_followup_not_original_natural_run'
    # Expiry after execution is not retroactively represented as a new run.
    monkeypatch.setattr(deep_cases.time,'time',lambda:250)
    assert deep_cases.preview_result(tmp_path,path,out,cfg)[1]==sha


def test_deep_interruption_preserves_failure_event_without_claiming_completion(tmp_path,monkeypatch,signed_case):
    from agrel_public import processes
    keys,r,p,sign=signed_case
    cfg=Config(public_keys=keys)
    monkeypatch.setattr(deep_cases.time,'time',lambda:150)
    save_json(tmp_path/'research.private.json',{'endpoint':cfg.telemetry_url,'records':[r]})
    path=tmp_path/'case.json';save_json(path,sign(p))
    _,sha,_,_=deep_cases.preview(tmp_path,path,cfg)
    monkeypatch.setattr(telemetry,'device_hash',lambda identity:'b'*64)
    def interrupted(job,event,timeout):
        event({'event':'pack_iteration','qualified':True,'matched':False})
        raise InterruptedError('software fixture')
    monkeypatch.setattr(processes,'run_worker',interrupted)
    out=tmp_path/'attempt'
    with pytest.raises(InterruptedError):deep_cases.run(tmp_path,path,cfg,Device(0,'Fixture GPU',8000),out,sha)
    result=load_json(out/'case-result.json')
    assert result['status']=='INCOMPLETE' and result['qualified_failure_event'] and result['uploaded'] is False


def test_offline_and_telemetry_help_policy_is_unchanged():
    help_text=cli.parser()._subparsers._group_actions[0].choices['guided'].format_help()
    assert '--offline' not in help_text and '--telemetry' not in help_text
