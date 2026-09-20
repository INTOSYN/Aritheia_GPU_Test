"""Small, disclosed research summaries; no raw hardware identifiers or directory uploads."""
from __future__ import annotations
import hashlib
import hmac
import json
import math
import os
import secrets
import re
from pathlib import Path
from .common import canonical, config_home, load_json, save_json
from .network import Transport
from .scenarios.registry import ORDER, configs, config_id
from .sharing import scenario_rows

POLICY = 'computeproof-research-2026-09-v3'
ENDPOINT = 'https://data.intc.ca:8443/sci-test'
NOTICE = ('联网模式在测试结束后自动贡献最小研究记录：安装范围设备假名、GPU 型号/显存/启动频率、软件与协议版本。'
          '正常、异常及未完成均保留 24 配置完成范围、预算和检查覆盖；只有异常或未完成时发送数值异常计数及异常逻辑 SM。'
          '基础记录不上传场景分数、候选名单或张量；任务告警和算术异常分别记录。'
          '不发送原始 UUID、主机名、路径、IP 字段、姓名、邮箱或原始日志。'
          '哈希可关联本安装的重复测试，属于假名标识；服务端在连接期间能看到来源 IP。'
          '详细 JSON 与联系方式只在展示 contribution 目录后另行同意。')


def device_hash(identity):
    if not identity:
        return None
    root = config_home(); root.mkdir(parents=True, exist_ok=True)
    keyfile = root / 'device-hash-key.private'
    try:
        fd = os.open(keyfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, 'wb') as f:
            f.write(secrets.token_bytes(32))
    key = keyfile.read_bytes()
    if len(key) != 32:
        raise ValueError('Invalid local pseudonym key')
    return hmac.new(key, b'computeproof-device-v1\0' + identity.encode(), hashlib.sha256).hexdigest()


def count(value):
    return max(0, min(int(value or 0), 10**13))


def verdict(value):
    return value if value in ('PASS_OBSERVED', 'FAIL_NUMERICAL') else 'NOT_ASSESSED'


def outcomes(report, card):
    result = []
    for name in ORDER:
        probes = [p for p in card['diagnosis'].get('probes', []) if p.get('scenario') == name]
        rows = [r for r in card.get('scenarios', []) if r.get('scenario') == name]
        pv = [verdict(p.get('arithmetic_verdict')) for p in probes]
        v = 'FAIL_NUMERICAL' if 'FAIL_NUMERICAL' in pv else ('PASS_OBSERVED' if pv and all(x == 'PASS_OBSERVED' for x in pv) else 'NOT_ASSESSED')
        # A completed subset is not all requested work: require the reference's expected probe count.
        if v == 'PASS_OBSERVED':
            from . import reference
            expected = len(reference.select(reference.load(), [name], 1))
            if len(probes) != expected or any(p.get('status') != 'COMPLETED' for p in probes):
                v = 'NOT_ASSESSED'
        statuses = {r.get('status') for r in rows}
        application = ('MONITOR_ALERT' if 'MONITOR_ALERT' in statuses else
                       'COMPLETED' if rows and len(rows) == 2 and statuses == {'COMPLETED'} else
                       'INCOMPLETE' if rows else 'NOT_RUN')
        result.append({'scenario': name, 'selected': name in card.get('selection', report.get('selection', {})).get('requested', []),
                       'application': application, 'numerical': v,
                       'checked_values': sum(count(p.get('checked_values')) for p in probes),
                       'bad_values_lower_bound': sum(count(p.get('bad_values')) for p in probes)})
    return result


def summary(report, card, pseudonym=None):
    if report.get('cpu_demo'):
        raise ValueError('CPU software runs must not enter the GPU research database')
    assessment = card['diagnosis'].get('assessment') or {}
    native = card.get('native') or {}
    kernels = []
    for k in native.get('kernels', [])[:2]:
        kernels.append({'kernel': k['kernel'], 'numerical': verdict(k.get('arithmetic_verdict')),
                        'sm_count': count(k.get('sm_count')), 'observed_sms': len(k.get('observed_logical_sms', [])),
                        'checked_values': count(k.get('checked_values')), 'bad_values_lower_bound': count(k.get('bad_values')),
                        'unattributed_bad_values': count(k.get('unattributed_bad_values')),
                        'bad_logical_sms': sorted(set(int(x) for x in k.get('bad_sms', []) if int(x) >= 0))[:1024]})
    task_alert = any(r.get('status') == 'MONITOR_ALERT' for r in card.get('scenarios', []))
    all_pass = card.get('overall_status') == 'NO_ANOMALY_OBSERVED' and bool(card.get('workflow_complete')) and not task_alert
    result = {'schema': 'computeproof.summary.v3', 'policy': POLICY,
              'device_hash': pseudonym, 'identity_scope': 'installation_hmac_sha256' if pseudonym else 'unavailable',
              'gpu_model': card['model'], 'memory_mib': card['memory_mib'], 'core_clock_mhz': card.get('core_clock_mhz'),
              'clock_scope': 'preflight_snapshot_not_load_frequency',
              'client_version': report.get('client_version'), 'build_id': report.get('build_id'),
              'reference_sha256': report.get('reference', {}).get('sha256'),
              'stack': {k: report.get('environment', {}).get(k) for k in ('torch', 'cuda', 'driver')},
              'iterations': report.get('reference', {}).get('iterations', 0),
              'result_class': 'ALL_PASS' if all_pass else 'ATTENTION_REQUIRED',
              'run_id': report.get('run_id'),
              'compute_capability': [int(v) for v in card['compute_capability'].split('.')]
                                    if isinstance(card.get('compute_capability'), str) and re.fullmatch(r'[0-9]{1,2}\.[0-9]{1,2}', card['compute_capability']) else None,
              'coverage': configuration_coverage(report, card),
              'budget': {k: report.get('budget', {}).get(k) for k in
                         ('steps', 'native_blocks', 'native_iterations', 'probes_only', 'no_native', 'parallel_gpus')},
              'channels': {'exact': card['diagnosis'].get('status', 'INCOMPLETE'),
                           'native': native.get('status', 'NOT_RUN'),
                           'task_monitor': 'ALERT' if task_alert else 'NO_ALERT_OBSERVED',
                           'workflow_complete': bool(card.get('workflow_complete'))},
              'probe_coverage': {'completed': count(assessment.get('probes_completed')),
                                 'requested': count(assessment.get('probes_total')),
                                 'checked_values': count(assessment.get('checked_values')),
                                 'native': [{k: n[k] for k in ('kernel', 'sm_count', 'observed_sms', 'checked_values')}
                                            for n in kernels]},
              'evidence_relation': 'natural_tasks_and_fresh_process_probes_are_separate'}
    if not all_pass:
        result['diagnostics'] = {
            'overall': card.get('overall_status', card['diagnosis']['status']),
            'workflow_complete': bool(card.get('workflow_complete')),
            'probes_completed': count(assessment.get('probes_completed')),
            'probes_total': count(assessment.get('probes_total')),
            'checked_values': count(assessment.get('checked_values')),
            'bad_values_lower_bound': count(assessment.get('bad_values')),
            'scenarios': outcomes(report, card),
            'native_status': native.get('status', 'NOT_RUN'),
            'native': kernels,
        }
    if len(canonical(result)) > 16384:
        raise ValueError('Summary exceeds 16 KiB per GPU')
    return result


def configuration_coverage(report, card):
    """Fixed public IDs only; no scores or arbitrary exception text."""
    by_id = {config_id(r['config']): r for r in card.get('scenarios', []) if r.get('config')}
    selection = card.get('selection', report.get('selection', {}))
    requested = set(selection.get('requested', []))
    excluded = {r['scenario']: r.get('reason') for r in selection.get('excluded', [])}
    statuses = {'COMPLETED', 'MONITOR_ALERT', 'ASSET_MISSING', 'DEPENDENCY_MISSING', 'RUNTIME_ERROR', 'NOT_RUN'}
    reasons = {'DOWNLOAD_DECLINED', 'DOWNLOAD_FAILED', 'DOWNLOAD_UNAVAILABLE', 'ASSET_STILL_MISSING',
               'ASSET_UNPINNED', 'ASSET_INTEGRITY_ERROR', 'ASSET_MANIFEST_INVALID', 'ASSET_RELEASE_URL_MISSING',
               'FROZEN_ASSET_MISSING', 'OPTIONAL_DOWNLOAD_CANCELLED'}
    result = []
    for name in ORDER:
        for cfg in configs(name):
            cid = config_id(cfg); row = by_id.get(cid, {})
            reason = excluded.get(name)
            result.append({'config_id': cid, 'selected': name in requested,
                           'application': row.get('status') if row.get('status') in statuses else 'NOT_RUN',
                           'reason': reason if reason in reasons else
                           ('APPLICATIONS_SKIPPED' if report.get('budget', {}).get('probes_only') else
                            'NOT_SELECTED' if name not in requested else
                            'NONE' if row.get('status') in ('COMPLETED', 'MONITOR_ALERT') else 'UNAVAILABLE')})
    return result


def detailed(card):
    """All regular probe rows/hashes, scenario metrics, native per-SM counts. No arbitrary logs."""
    probes = []
    for p in card['diagnosis'].get('probes', [])[:1024]:
        row = {'probe_id': p['probe_id'], 'numerical': verdict(p.get('arithmetic_verdict')),
               'checked_values': count(p.get('checked_values')), 'bad_values_lower_bound': count(p.get('bad_values')),
               'iterations_completed': count(p.get('iterations_completed')), 'iterations_requested': count(p.get('iterations_requested')),
               'actual_sha256': p.get('output_sha256', []) if isinstance(p.get('output_sha256', []), list) else [],
               'expected_sha256': p.get('expected_sha256', []) if isinstance(p.get('expected_sha256', []), list) else [],
               'values': [v if isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) else None for v in p.get('values', [])[:32]]}
        probes.append(row)
    native = [{'kernel': k['kernel'], 'per_sm': {str(sm): {key: count(v.get(key)) for key in ('blocks', 'bad_blocks', 'bad_values')}
               for sm, v in k.get('per_sm', {}).items()}} for k in (card.get('native') or {}).get('kernels', [])[:2]]
    from .research_evidence import scenario_evidence
    result = {'schema': 'computeproof.detail.v2', 'consent': True, 'probes': probes,
              'scenarios': scenario_rows(card), 'native': native,
              'scenario_evidence': scenario_evidence(card),
              'evidence_relation': 'natural_tasks_and_fresh_process_probes_are_separate'}
    if len(canonical(result)) > 512 * 1024:
        raise ValueError('Detailed JSON exceeds 512 KiB; no silent truncation or upload')
    return result


def prepare_auto(out, report, cards):
    path = out / 'research.private.json'
    state = {'endpoint': ENDPOINT, 'policy': POLICY, 'records': []}
    for i, card in enumerate(report['cards']):
        payload = summary(report, card, device_hash(getattr(cards[i], '_identity', None)))
        rid = secrets.token_hex(16)
        save_json(out / f'research-summary-{i+1}.json', payload)
        state['records'].append({'id': rid, 'token': secrets.token_urlsafe(32), 'slot': i, 'summary': payload})
    save_json(path, state)
    return state


def transmit(out, config, *, detail=False, confirmed_detail_sha=None, contact=None):
    state = load_json(out / 'research.private.json')
    endpoint = config.telemetry_url.rstrip('/')
    if state['endpoint'] != endpoint:
        raise ValueError('Research credentials are bound to their original endpoint')
    transport = Transport(config.network_timeout, config.allow_local_http)
    details = []
    if detail:
        contribution=out/'contribution'
        manifest=load_json(contribution/'manifest.json')
        if not confirmed_detail_sha or hashlib.sha256(canonical(manifest)).hexdigest()!=confirmed_detail_sha:
            raise ValueError('Detailed preview requires an unchanged confirmed digest')
        if len(manifest['files'])!=len(state['records']):
            raise ValueError('Detail card count changed')
        for i, entry in enumerate(manifest['files']):
            if entry['file']!=f'research-detail-{i+1}.json':
                raise ValueError('Unexpected detail preview path')
            payload=load_json(contribution/entry['file'],512*1024)
            if hashlib.sha256(canonical(payload)).hexdigest()!=entry['sha256']:
                raise ValueError('Detailed JSON changed after preview')
            details.append(payload)
    for row in state['records']:
        if row.get('deleted'):
            continue
        url = endpoint + '/v1/reports/' + row['id']
        if not row.get('receipt'):
            row['receipt'] = transport.json(url, method='PUT', payload=row['summary'], token=row['token'])
            save_json(out / 'research.private.json', state)
        if detail:
            payload = details[row['slot']]
            row['detail_receipt'] = transport.json(url+'/detail', method='PUT', payload=payload, token=row['token'])
        if contact:
            row['contact_receipt'] = transport.json(url+'/contact', method='PUT', payload=contact, token=row['token'])
        save_json(out / 'research.private.json', state)
    return state


def erase(out, config):
    state = load_json(out / 'research.private.json')
    if state['endpoint'] != config.telemetry_url.rstrip('/'):
        raise ValueError('Research endpoint differs from saved credentials')
    t = Transport(config.network_timeout, config.allow_local_http)
    for row in state['records']:
        if not row.get('deleted'):
            t.json(state['endpoint']+'/v1/reports/'+row['id'], method='DELETE', token=row['token'])
            row['deleted'] = True
            save_json(out/'research.private.json', state)
    return {'deleted': True}
