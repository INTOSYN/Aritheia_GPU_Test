import json
from pathlib import Path
from agrel_public.cli import main
from agrel_public.common import save_json, load_json
from agrel_public.devices import Device
from agrel_public.network import DownloadState
from agrel_public.html_report import render_report


def healthy_probes(job):
    from agrel_public import reference
    specs=reference.select(reference.load(),job['scenarios'],job['iterations'])
    return [{'probe_id':x['id'],'status':'COMPLETED','arithmetic_verdict':'PASS_OBSERVED',
             'iterations_completed':job['iterations'],'iterations_requested':job['iterations'],
             'checked_values':1,'bad_values':0} for x in specs]+[{'probe_id':'memory_roundtrip','status':'COMPLETED',
             'arithmetic_verdict':'PASS_OBSERVED','checked_values':1,'bad_values':0}]


def test_scenarios_precede_bootstrap_probes_run_fresh_and_nothing_uploads(tmp_path, monkeypatch):
    import agrel_public.cli as cli
    timeline = []
    cards = [Device(1, 'same gpu', 24000, 1600), Device(0, 'same gpu', 24000, None)]
    monkeypatch.setattr(cli, 'inventory', lambda: cards)
    seen = []
    class FakeDownload:
        def __init__(self, config, body):
            seen.append(body); self.state = DownloadState(status='UNAVAILABLE'); self.state.session = None
        def start(self): timeline.append('download_start')
        def stop(self): timeline.append('download_stop')
    def worker(job, callback, timeout):
        timeline.append(job['mode'] + ':' + job['device'])
        out = Path(job['out'])
        if job['mode'] == 'scenarios':
            rows = []
            for i, cfg in enumerate(job['configs']):
                timeline.append('scenario_start')
                callback({'event': 'scenario_start', 'index': i + 1, 'total': len(job['configs']), 'config': 'c', 'title': 't'})
                rows.append({'scenario': cfg['scenario'], 'title': 't', 'config': cfg, 'status': 'COMPLETED', 'arithmetic_verdict': 'NOT_ASSESSED',
                             'metric': {'finite_output': True, 'accuracy': 0.9}, 'output_shape': [2], 'values': [1.0, 2.0], 'task_wall_s': 0.1})
                callback({'event': 'scenario_done', 'index': i + 1, 'total': len(job['configs']), 'config': 'c', 'title': 't', 'status': 'COMPLETED', 'metric': rows[-1]['metric'], 'wall_s': 0.1})
            save_json(out / 'scenarios.json', rows)
        elif job['mode'] == 'native':
            save_json(out / 'native.json', {'status': 'NO_ANOMALY_OBSERVED', 'reason': 'NATIVE_EXACT_CONTRACT_OBSERVED', 'kernels': []})
        elif job['mode'] == 'probes':
            assert job['iterations'] == 8 and job['scenarios']
            save_json(out/'diagnosis.json', {'status':'NO_ANOMALY_OBSERVED','probes':healthy_probes(job)})
        return 0
    monkeypatch.setattr(cli, 'PackJob', FakeDownload)
    monkeypatch.setattr(cli, 'run_worker', worker)
    cfg = tmp_path / 'cfg.json'; save_json(cfg, {'telemetry':'off', 'api_url': 'https://example.org', 'pack_cache': str(tmp_path / 'packs')})
    out = tmp_path / 'run'
    code = main(['check', '--config', str(cfg), '--devices', '1,0', '--allow-hardware-metadata', '--no-share-prompt',
                 '--large-scenarios', 'skip', '--out', str(out)])
    assert code == 0
    assert timeline.index('scenario_start') < timeline.index('download_start')
    assert timeline.count('download_start') == 1
    for device in ('cuda:1','cuda:0'):
        assert timeline.index('scenarios:'+device) < timeline.index('probes:'+device) < timeline.index('native:'+device)
    assert set(seen[0]) == {'cards'} and len(seen[0]['cards']) == 2
    assert all(set(x) == {'model', 'memory_mib', 'core_clock_mhz'} for x in seen[0]['cards'])
    r = load_json(out / 'report.json')
    assert r['schema'] == 'aritheia.local.v2'
    assert [c['local_device'] for c in r['cards']] == ['cuda:1', 'cuda:0']
    assert all(c['diagnosis']['status'] == 'NO_ANOMALY_OBSERVED' and c['overall_status'] == 'NO_ANOMALY_OBSERVED' for c in r['cards'])
    assert r['selection']['selected'] == ['drug_screen', 'drug_finetune', 'molecule_neighbors', 'cytology', 'ocr_cache',
                                          'genomics_splice', 'medical_ultrasound', 'materials_screen']
    assert r['selection']['excluded'] == []
    assert len(r['cards'][0]['scenarios']) == 16
    assert not (out / 'upload_preview.json').exists()
    assert (out / 'report.html').exists()
    assert main(['report', str(out)]) == 0


def test_anomaly_and_incomplete_exit_codes(tmp_path, monkeypatch):
    import agrel_public.cli as cli
    monkeypatch.setattr(cli, 'inventory', lambda: [Device(0, 'gpu', 8000, 1500)])
    status = {'value': 'NUMERICAL_ANOMALY_DETECTED'}
    def worker(job, callback, timeout):
        out = Path(job['out'])
        if job['mode'] == 'scenarios':
            save_json(out / 'scenarios.json', [])
        elif job['mode'] == 'native':
            save_json(out / 'native.json', {'status': 'NO_ANOMALY_OBSERVED', 'kernels': []})
        else:
            save_json(out / 'diagnosis.json', {'status': status['value'], 'probes': [], 'assessment': {}})
        return 0
    monkeypatch.setattr(cli, 'run_worker', worker)
    assert main(['check', '--offline', '--no-share-prompt', '--scenarios', 'cytology', '--out', str(tmp_path / 'a')]) == 1
    status['value'] = 'INCOMPLETE'
    assert main(['check', '--offline', '--no-share-prompt', '--scenarios', 'cytology', '--out', str(tmp_path / 'b')]) == 2
    assert main(['check', '--offline', '--no-share-prompt', '--scenarios', 'cytology', '--out', str(tmp_path / 'b')]) == 2  # refuses overwrite


def test_probes_only_skips_scenarios(tmp_path, monkeypatch):
    import agrel_public.cli as cli
    monkeypatch.setattr(cli, 'inventory', lambda: [Device(0, 'gpu', 8000, 1500)])
    modes = []
    def worker(job, callback, timeout):
        modes.append(job['mode'])
        save_json(Path(job['out']) / 'diagnosis.json', {'status': 'NO_ANOMALY_OBSERVED', 'probes': healthy_probes(job), 'assessment': {}})
        return 0
    monkeypatch.setattr(cli, 'run_worker', worker)
    assert main(['check', '--offline', '--no-share-prompt', '--scenarios', 'builtin', '--probes-only', '--no-native', '--out', str(tmp_path / 'a')]) == 0
    assert modes == ['probes']
    r = load_json(tmp_path / 'a' / 'report.json')
    assert r['cards'][0]['native']['reason'] == 'SKIPPED_BY_USER' and r['cards'][0]['overall_status'] == 'NO_ANOMALY_OBSERVED'


def test_native_anomaly_drives_overall_status_and_exit_code(tmp_path, monkeypatch):
    import agrel_public.cli as cli
    monkeypatch.setattr(cli, 'inventory', lambda: [Device(0, 'gpu', 8000, 1500)])
    def worker(job, callback, timeout):
        out = Path(job['out'])
        if job['mode'] == 'probes':
            save_json(out / 'diagnosis.json', {'status': 'NO_ANOMALY_OBSERVED', 'probes': [], 'assessment': {}})
        elif job['mode'] == 'native':
            save_json(out / 'native.json', {'status': 'NUMERICAL_ANOMALY_DETECTED', 'reason': 'NATIVE_EXACT_CONTRACT_OBSERVED', 'kernels': [
                {'kernel': 'smid_mma_bf16', 'arithmetic_verdict': 'FAIL_NUMERICAL', 'coverage_verdict': 'COMPLETE_OBSERVED', 'sm_count': 68,
                 'observed_logical_sms': list(range(68)), 'migrated_or_invalid_blocks': 0, 'checked_values': 10, 'bad_values': 3, 'bad_sms': [45], 'per_sm': {}}]})
        return 0
    monkeypatch.setattr(cli, 'run_worker', worker)
    assert main(['check', '--offline', '--no-share-prompt', '--scenarios', 'none', '--probes-only', '--out', str(tmp_path / 'a')]) == 1
    r = load_json(tmp_path / 'a' / 'report.json')
    assert r['cards'][0]['overall_status'] == 'NUMERICAL_ANOMALY_DETECTED' and r['cards'][0]['diagnosis']['status'] == 'INCOMPLETE'
    assert (tmp_path / 'a' / 'report.html').read_text().count('45') >= 1


def test_static_report_escapes_text_and_has_no_remote_assets(tmp_path):
    report = {'run_id': 'a' * 32, 'cards': [{'model': '<script>alert(1)</script>', 'memory_mib': 10, 'core_clock_mhz': None,
              'diagnosis': {'status': 'INCOMPLETE', 'reason': '<b>', 'probes': [{'probe_id': '<x>', 'shape': [1], 'arithmetic_verdict': 'FAIL_NUMERICAL'}]},
              'scenarios': [{'scenario': 's', 'title': '<i>', 'config': {'batch_size': 8, 'context': 'short'}, 'status': 'COMPLETED', 'metric': {'finite_output': True, 'rmse': 0.5}}]}],
              'selection': {'selected': ['s'], 'excluded': [{'scenario': '<e>', 'reason': 'X'}]}, 'reference': {'version': 'v', 'sha256': 'f' * 64}}
    p = tmp_path / 'report.html'; render_report(report, p); text = p.read_text()
    assert '<script>' not in text and '<b>' not in text and '<i>' not in text and '<x>' not in text and '<e>' not in text
    assert '&lt;script&gt;' in text
    assert 'http://' not in text and 'https://' not in text


def test_cpu_software_run_is_not_gpu_verdict(tmp_path):
    out = tmp_path / 'run'
    assert main(['check', '--cpu-demo', '--offline', '--no-share-prompt', '--scenarios', 'cytology', '--iterations', '1', '--out', str(out)]) == 0
    r = load_json(out / 'report.json')
    assert r['cpu_demo'] and len(r['cards'][0]['scenarios']) == 2
    assert r['cards'][0]['diagnosis']['status'] == 'INCOMPLETE'
    assert r['cards'][0]['diagnosis']['reason'] == 'CPU_SOFTWARE_RUN_NOT_GPU_VALIDATION'
    assert r['cards'][0]['diagnosis']['software_assessment']['status'] == 'NO_ANOMALY_OBSERVED'
    assert not (out / 'session.private.json').exists()
    assert not (out / 'upload_receipt.json').exists()
    assert main(['check', '--cpu-demo', '--offline', '--out', str(out)]) == 2
