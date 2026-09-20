import copy
import hashlib
import threading
from pathlib import Path
import pytest
from agrel_public import telemetry, cli, reference
from agrel_public.common import save_json,load_json,canonical
from agrel_public.devices import Device
from agrel_public.config import Config
from test_workflow import healthy_probes


def fixture_report():
    return {'cpu_demo':False,'client_version':'0.5.0rc3','build_id':'a'*64,
            'reference':{'sha256':'b'*64,'iterations':1},'selection':{'requested':['cytology']},'environment':{},
            'cards':[{'model':'NVIDIA Test GPU','memory_mib':24000,'core_clock_mhz':None,'diagnosis':{
                'status':'INCOMPLETE','probes':[],'assessment':{}},'overall_status':'INCOMPLETE','scenarios':[]}]}


def test_privacy_pseudonyms_and_minimum_schema(tmp_path,monkeypatch):
    monkeypatch.setattr(telemetry,'config_home',lambda:tmp_path/'a')
    first=telemetry.device_hash('gpu-secret')
    assert first==telemetry.device_hash('gpu-secret') and len(first)==64
    assert first!=telemetry.device_hash('gpu-second')
    monkeypatch.setattr(telemetry,'config_home',lambda:tmp_path/'b')
    assert first!=telemetry.device_hash('gpu-secret')
    r=fixture_report();r.update(hostname='secret-host',ip='private-ip',token='secret-token')
    c=r['cards'][0];c.update(uuid='secret-uuid',path='/private/users')
    payload=telemetry.summary(r,c,first)
    raw=canonical(payload)
    assert len(payload['diagnostics']['scenarios'])==12 and len(raw)<16384
    assert all(value not in raw for value in (b'secret',b'/private',b'private-ip'))
    assert payload['diagnostics']['scenarios'][0]['numerical']=='NOT_ASSESSED'
    r['cpu_demo']=True
    with pytest.raises(ValueError): telemetry.summary(r,c)


def test_all_pass_upload_omits_experiment_results():
    r=fixture_report();c=r['cards'][0]
    c.update(overall_status='NO_ANOMALY_OBSERVED',workflow_complete=True)
    payload=telemetry.summary(r,c,'a'*64)
    assert payload['result_class']=='ALL_PASS'
    assert 'diagnostics' not in payload
    assert not any(k in payload for k in ('scenarios','checked_values','native'))


def test_parallel_workers_overlap_and_failures_do_not_cross_cards(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'inventory',lambda:[Device(0,'GPU A',8000),Device(1,'GPU B',8000)])
    barrier=threading.Barrier(2); arrivals=[]
    def worker(job,event,timeout):
        assert job['mode']=='probes'
        arrivals.append(job['device']);barrier.wait(timeout=5)
        if job['device']=='cuda:0':
            event({'event':'iteration_done','qualified':True,'matched':False})
            raise RuntimeError('one GPU worker crashed')
        save_json(Path(job['out'])/'diagnosis.json',{'status':'NO_ANOMALY_OBSERVED','probes':healthy_probes(job)})
    monkeypatch.setattr(cli,'run_worker',worker)
    out=tmp_path/'parallel'
    code=cli.main(['check','--all-devices','--offline','--probes-only','--no-native','--scenarios','cytology','--out',str(out)])
    r=load_json(out/'report.json')
    assert len(arrivals)==2 and code==1
    assert [c['overall_status'] for c in r['cards']]==['NUMERICAL_ANOMALY_DETECTED','NO_ANOMALY_OBSERVED']
    assert not (out/'research.private.json').exists()


def test_auto_retry_credentials_saved_before_request_and_delete(tmp_path,monkeypatch):
    r=fixture_report();save_json(tmp_path/'report.json',r)
    state=telemetry.prepare_auto(tmp_path,r,[Device(0,'GPU',8000)])
    seen=[]
    class T:
        def __init__(self,*a): pass
        def json(self,url,**kwargs):
            assert (tmp_path/'research.private.json').exists()
            seen.append((url,kwargs)); return {'receipt_id':state['records'][0]['id']}
    monkeypatch.setattr(telemetry,'Transport',T)
    telemetry.transmit(tmp_path,Config());telemetry.transmit(tmp_path,Config())
    assert len(seen)==1 and seen[0][1]['method']=='PUT'
    telemetry.erase(tmp_path,Config());telemetry.transmit(tmp_path,Config())
    assert len(seen)==2 and seen[-1][1]['method']=='DELETE'
    with pytest.raises(ValueError):telemetry.transmit(tmp_path,Config(telemetry_url='https://example.org'))


def test_partial_scene_never_passes():
    r=fixture_report();c=r['cards'][0]
    specs=reference.select(reference.load(),['cytology'],1)
    c['diagnosis']['probes']=[{'scenario':'cytology','arithmetic_verdict':'PASS_OBSERVED','status':'COMPLETED'}]
    outcomes=telemetry.outcomes(r,c)
    assert next(x for x in outcomes if x['scenario']=='cytology')['numerical']=='NOT_ASSESSED'
    c['diagnosis']['probes'][0]['arithmetic_verdict']='FAIL_NUMERICAL'
    assert next(x for x in telemetry.outcomes(r,c) if x['scenario']=='cytology')['numerical']=='FAIL_NUMERICAL'


def test_offline_suppresses_even_interactive_share(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'inventory',lambda:[Device(0,'GPU',8000)])
    monkeypatch.setattr(cli.sys.stdin,'isatty',lambda:True)
    monkeypatch.setattr(cli,'share_dialog',lambda *a:pytest.fail('offline must not prompt for a network upload'))
    monkeypatch.setattr(telemetry,'transmit',lambda *a:pytest.fail('offline upload'))
    def worker(job,event,timeout):
        save_json(Path(job['out'])/'diagnosis.json',{'status':'NO_ANOMALY_OBSERVED','probes':healthy_probes(job)})
    monkeypatch.setattr(cli,'run_worker',worker)
    assert cli.main(['check','--offline','--probes-only','--no-native','--scenarios','cytology','--out',str(tmp_path/'offline')])==0


def test_detail_preview_tamper_is_rejected_before_network(tmp_path,monkeypatch):
    r=fixture_report();save_json(tmp_path/'report.json',r)
    telemetry.prepare_auto(tmp_path,r,[Device(0,'GPU',8000)])
    sha=cli.detail_manifest(tmp_path)
    d=load_json(tmp_path/'contribution'/'research-detail-1.json');d['consent']=False
    save_json(tmp_path/'contribution'/'research-detail-1.json',d)
    class Forbidden:
        def __init__(self,*a):pass
        def json(self,*a,**k):pytest.fail('changed preview must not be sent')
    monkeypatch.setattr(telemetry,'Transport',Forbidden)
    with pytest.raises(ValueError):telemetry.transmit(tmp_path,Config(),detail=True,confirmed_detail_sha=sha)


def test_telemetry_off_disables_default_collection(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'inventory',lambda:[Device(0,'GPU',8000)])
    monkeypatch.setattr(telemetry,'prepare_auto',lambda *a:pytest.fail('disabled collection'))
    def worker(job,event,timeout):
        save_json(Path(job['out'])/'diagnosis.json',{'status':'NO_ANOMALY_OBSERVED','probes':healthy_probes(job)})
    monkeypatch.setattr(cli,'run_worker',worker)
    assert cli.main(['check','--telemetry','off','--no-share-prompt','--probes-only','--no-native','--scenarios','cytology','--out',str(tmp_path/'off')])==0


def test_guided_unattended_download_requires_explicit_flag(monkeypatch):
    class Downloader:
        release_url='https://example.test/release'
    monkeypatch.setattr(cli,'make_downloader',lambda *a:Downloader())
    monkeypatch.setattr(cli.sys.stdin,'isatty',lambda:False)
    args=cli.parser().parse_args(['guided'])
    assert cli.guided_asset_countdown(args,Config()) is None


def test_guided_parser_uses_two_phase_default():
    args=cli.parser().parse_args(['guided','--offline'])
    assert args.scenarios is None and args.large_scenarios=='skip'


def test_network_controls_are_hidden_from_help():
    help_text=cli.parser()._subparsers._group_actions[0].choices['check'].format_help()
    assert '--offline' not in help_text and '--telemetry' not in help_text


def test_contribution_directory_precedes_optional_contact_and_final_upload(tmp_path,monkeypatch):
    r=fixture_report();save_json(tmp_path/'report.json',r)
    answers=iter([True,True,True,True])
    monkeypatch.setattr(cli,'yes',lambda *a,**k:next(answers))
    entries=iter(['research@example.org','Contributor Name'])
    monkeypatch.setattr('builtins.input',lambda *a:next(entries))
    sent=[]
    monkeypatch.setattr(telemetry,'transmit',lambda *a,**k:sent.append(k))
    cli.research_dialog(tmp_path,Config())
    assert (tmp_path/'contribution'/'README.txt').is_file()
    assert (tmp_path/'contribution'/'manifest.json').is_file()
    assert sent and sent[0]['detail'] is True
    assert sent[0]['contact']['email']=='research@example.org'
    assert sent[0]['contact']['name']=='Contributor Name'
