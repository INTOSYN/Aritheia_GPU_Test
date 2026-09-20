"""Report-bound follow-ups: data-only v1 or separately consented native v2."""
import base64
import hashlib
import json
import re
import time
import secrets
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from .common import canonical, load_json, save_json, strict_keys, atomic_bytes
from .packs import verify_pack
from .network import Transport

MAX_BYTES = 16 * 1024 * 1024
NATIVE_CODE_NOTICE = (
    'Downloads and runs native code from the trusted publisher with your user-process permissions. '
    'Signature and hashes identify the publisher and files, not harmlessness. '
    'A separate worker and timeout are not a security sandbox; the memory budget is an '
    'implementation requirement, not an OS-enforced memory limit. No result upload is authorized.')


def validate_case(envelope, keys, record, *, now=None, allow_local_http=False):
    strict_keys(envelope, {'payload', 'key_id', 'signature'})
    p = envelope['payload']
    Ed25519PublicKey.from_public_bytes(base64.b64decode(keys[envelope['key_id']], validate=True)).verify(
        base64.b64decode(envelope['signature'], validate=True), canonical(p))
    strict_keys(p, {'schema','case_id','report_id','source_summary_sha256','issued_unix','expires_unix',
                    'purpose','budget','pack'}, {'binary'})
    if p['schema'] not in {'computeproof.case.v1', 'computeproof.case.v2'} or not re.fullmatch(r'[a-zA-Z0-9_.-]{1,80}', p['case_id']):
        raise ValueError('Unsupported case identity')
    if p['purpose'] not in ('confirm_exact_mismatch','characterize_path','repeat_fixed_budget'):
        raise ValueError('Unsupported case purpose')
    if record['summary'].get('result_class') != 'ATTENTION_REQUIRED':
        raise ValueError('Deep cases require an attention report, not a presumed damaged device')
    if p['report_id'] != record['id'] or p['source_summary_sha256'] != hashlib.sha256(canonical(record['summary'])).hexdigest():
        raise ValueError('Case belongs to a different source report')
    now = int(time.time()) if now is None else now
    if type(p['issued_unix']) is not int or type(p['expires_unix']) is not int or not (
        p['issued_unix'] <= now < p['expires_unix'] <= p['issued_unix'] + 30*86400):
        raise ValueError('Case expired, not yet valid, or exceeds 30-day validity')
    budget = p['budget']
    strict_keys(budget, {'timeout_s','memory_mib','max_repeats'})
    for key, lo, hi in (('timeout_s',30,1800),('memory_mib',16,2048),('max_repeats',1,32)):
        if type(budget[key]) is not int or not lo <= budget[key] <= hi:
            raise ValueError('Case exceeds client budget')
    pack = verify_pack(canonical(p['pack']), keys, MAX_BYTES)
    if pack['provenance'] != 'publisher_validated':
        raise ValueError('Deep cases need a publisher-qualified pack; fixtures are not field evidence')
    if p['schema'] == 'computeproof.case.v2':
        from .deep_binary import validate_descriptor
        if pack['backend'] != 'private_core_v1' or 'binary' not in p:
            raise ValueError('Compiled cases require one explicit binary descriptor')
        validate_descriptor(p['binary'], match_runtime=False, allow_local_http=allow_local_http)
    elif 'binary' in p or pack['backend'] != 'fixed_mm_v1':
        raise ValueError('A compiled follow-up needs a v2 case and separate download consent')
    return p, pack


def records(run_dir, config):
    state = load_json(run_dir/'research.private.json')
    if state['endpoint'] != config.telemetry_url.rstrip('/'):
        raise ValueError('Source report credentials belong to another endpoint')
    return state


def fetch(run_dir, config):
    state = records(run_dir, config)
    root = run_dir/'deep-cases'; root.mkdir(mode=0o700, exist_ok=True)
    t = Transport(config.network_timeout, config.allow_local_http)
    saved = []
    for record in state['records']:
        if record.get('deleted') or not record.get('receipt') or record['summary']['result_class'] != 'ATTENTION_REQUIRED':
            continue
        reply = t.json(state['endpoint']+'/v1/reports/'+record['id']+'/cases',
                       method='GET', token=record['token'], limit=MAX_BYTES)
        strict_keys(reply, {'cases'})
        if not isinstance(reply['cases'], list) or len(reply['cases']) > 4:
            raise ValueError('Too many assigned cases')
        for envelope in reply['cases']:
            p, _ = validate_case(envelope, config.public_keys, record, allow_local_http=config.allow_local_http)
            sha = hashlib.sha256(canonical(envelope)).hexdigest()
            path = root/(sha+'.case.json')
            if path.exists() and load_json(path,MAX_BYTES) != envelope:
                raise ValueError('Immutable case differs')
            save_json(path, envelope); saved.append(path)
    return saved


def preview(run_dir, case_file, config, *, validation_time=None):
    state = records(run_dir, config)
    envelope = load_json(case_file, MAX_BYTES)
    rid = envelope.get('payload', {}).get('report_id')
    record = next((r for r in state['records'] if r['id']==rid and not r.get('deleted')), None)
    if record is None:
        raise ValueError('No active source record')
    p, pack = validate_case(envelope, config.public_keys, record, now=validation_time,
                            allow_local_http=config.allow_local_http)
    review = {'schema':'computeproof.case-review.v1','case_id':p['case_id'],
              'case_sha256':hashlib.sha256(canonical(envelope)).hexdigest(),
              'source_summary_sha256':p['source_summary_sha256'], 'purpose':p['purpose'],
              'gpu_model':record['summary']['gpu_model'], 'budget':p['budget'],
              'expires_unix':p['expires_unix'], 'backend':pack['backend'],
              'provenance':pack['provenance'], 'upload_results':False,
              'hardware_changes':False, 'execution_lane':'separate_followup_not_original_natural_run'}
    if p['schema'] == 'computeproof.case.v2':
        review['binary'] = p['binary']
        review['native_code_notice'] = NATIVE_CODE_NOTICE
    return review, hashlib.sha256(canonical(review)).hexdigest(), record, envelope


def _download_review(review):
    if 'binary' not in review:
        raise ValueError('This data-only case has no compiled binary to download')
    result = {'schema':'computeproof.case-download-review.v1',
              'case_id':review['case_id'], 'case_sha256':review['case_sha256'],
              'source_summary_sha256':review['source_summary_sha256'],
              'binary':review['binary'], 'execute':False, 'upload_results':False,
              'native_code_notice':review['native_code_notice']}
    return result, hashlib.sha256(canonical(result)).hexdigest()


def download_preview(run_dir, case_file, config):
    review, _, _, _ = preview(run_dir, case_file, config)
    from .deep_binary import validate_descriptor
    result, sha = _download_review(review)
    validate_descriptor(result['binary'], allow_local_http=config.allow_local_http)
    return result, sha


def _download_receipt_path(run_dir, case_sha):
    return run_dir / 'deep-cases' / (case_sha + '.download-consent.json')


def download(run_dir, case_file, config, confirmed_sha):
    review, sha = download_preview(run_dir, case_file, config)
    if not confirmed_sha or confirmed_sha != sha:
        raise ValueError('Review and explicitly confirm this unchanged binary download SHA-256')
    from .deep_binary import download_bundle
    bundle = download_bundle(review['binary'], run_dir / 'deep-binaries',
                             Transport(config.network_timeout, config.allow_local_http))
    save_json(_download_receipt_path(run_dir, review['case_sha256']),
              {'scope':'download_this_binary_no_execution_no_upload', 'review':review,
               'review_sha256':sha, 'accepted_unix':int(time.time())})
    return bundle


def _require_download_consent(receipt, review):
    expected, sha = _download_review(review)
    if (receipt.get('scope') != 'download_this_binary_no_execution_no_upload' or
            receipt.get('review_sha256') != sha or receipt.get('review') != expected or
            type(receipt.get('accepted_unix')) is not int):
        raise ValueError('Matching explicit deep-binary download consent is required')


def run(run_dir, case_file, config, device, out, confirmed_sha):
    review, sha, record, envelope = preview(run_dir, case_file, config)
    if not confirmed_sha or sha != confirmed_sha:
        raise ValueError('Review the case and confirm its unchanged SHA-256 before running')
    from .telemetry import device_hash
    identity = device_hash(getattr(device, '_identity', None))
    if identity is None or identity != record['summary'].get('device_hash'):
        raise ValueError('Selected GPU cannot be bound to the original installation-scoped device')
    binary_job = {}
    if 'binary' in review:
        from .deep_binary import verify_cache
        bundle = run_dir / 'deep-binaries' / review['binary']['sha256']
        receipt_path = _download_receipt_path(run_dir, review['case_sha256'])
        receipt = load_json(receipt_path)
        _require_download_consent(receipt, review)
        verify_cache(bundle, review['binary'], allow_local_http=config.allow_local_http)
        binary_job = {'mode':'deep', 'binary_cache':str(bundle.resolve()),
                      'download_consent':receipt,
                      'case_file':str((out / 'case.json').resolve()),
                      'execution_consent':str((out / 'consent.json').resolve()),
                      'source_record':{'id':record['id'], 'summary':record['summary']},
                      'allow_local_http':config.allow_local_http}
    if out.exists():
        raise FileExistsError('New deep-run directory required; existing attempts never overwritten')
    out.mkdir(parents=True, mode=0o700)
    p = envelope['payload']; pack_file = out/'case.agpack'
    atomic_bytes(pack_file, canonical(p['pack']))
    save_json(out/'case.json', envelope)
    save_json(out/'consent.json', {'scope':'execute_this_case_no_upload',
                                  'review_sha256':sha,'review':review,'accepted_unix':int(time.time())})
    from .processes import run_worker
    events = []; result = {'status':'INCOMPLETE','reason':'NOT_STARTED'}
    def event(e):
        # Persist only enumerated event fields, not messages or arbitrary worker logs.
        events.append({k:e[k] for k in ('event','iteration','qualified','matched','index','total') if k in e})
        save_json(out/'events.json', events)
    budget = p['budget']
    try:
        run_worker({'mode':'check','device':device.name,'out':str((out/'probe').resolve()),
                    'pack':str(pack_file.resolve()),'public_keys':config.public_keys,
                    'max_pack_bytes':MAX_BYTES,'max_probe_memory_mib':budget['memory_mib'],
                    'max_probe_repeats':budget['max_repeats'], **binary_job}, event, timeout=budget['timeout_s'])
        if (out/'probe/diagnosis.json').exists():
            result=load_json(out/'probe/diagnosis.json')
    finally:
        # Even interruption remains a separate immutable follow-up record.
        save_json(out/'case-result.json', {'schema':'computeproof.case-result.v1',
                   'attempt_id':secrets.token_hex(16),
                   'case_sha256':review['case_sha256'],'source_summary_sha256':p['source_summary_sha256'],
                   'execution_lane':review['execution_lane'],'status':result.get('status','INCOMPLETE'),
                   'qualified_failure_event':any(e.get('qualified') is True and e.get('matched') is False for e in events),
                   'uploaded':False,'finished_unix':int(time.time())})
    return result


def run_compiled_worker(job, device, out, emit):
    """Recheck signed identity and both local consents before any native import."""
    envelope = load_json(job['case_file'], MAX_BYTES)
    p, pack = validate_case(envelope, job['public_keys'], job['source_record'],
                            allow_local_http=job.get('allow_local_http', False))
    if p['schema'] != 'computeproof.case.v2':
        raise ValueError('Only a v2 case can enter the compiled worker')
    consent = load_json(job['execution_consent'])
    review = consent.get('review', {})
    # Reconstruct the preview independently, with no credentials or network calls.
    expected = {'schema':'computeproof.case-review.v1', 'case_id':p['case_id'],
                'case_sha256':hashlib.sha256(canonical(envelope)).hexdigest(),
                'source_summary_sha256':p['source_summary_sha256'], 'purpose':p['purpose'],
                'gpu_model':job['source_record']['summary']['gpu_model'], 'budget':p['budget'],
                'expires_unix':p['expires_unix'], 'backend':pack['backend'],
                'provenance':pack['provenance'], 'upload_results':False,
                'hardware_changes':False, 'execution_lane':'separate_followup_not_original_natural_run',
                'binary':p['binary'],
                'native_code_notice':NATIVE_CODE_NOTICE}
    if (review != expected or consent.get('scope') != 'execute_this_case_no_upload' or
            consent.get('review_sha256') != hashlib.sha256(canonical(expected)).hexdigest() or
            type(consent.get('accepted_unix')) is not int or
            not p['issued_unix'] <= consent['accepted_unix'] <= int(time.time()) < p['expires_unix']):
        raise ValueError('Matching explicit deep-binary execution consent is required')
    _require_download_consent(job['download_consent'], expected)
    if not p['issued_unix'] <= job['download_consent']['accepted_unix'] <= consent['accepted_unix']:
        raise ValueError('Download consent must precede execution consent for this case')
    if job['max_probe_memory_mib'] != p['budget']['memory_mib'] or job['max_probe_repeats'] != p['budget']['max_repeats']:
        raise ValueError('Worker budget differs from the reviewed case')
    if not device.startswith('cuda'):
        return {'status':'INCOMPLETE', 'reason':'CPU_DEMO_NOT_GPU_VALIDATION', 'probes':[]}
    from .deep_binary import load_reviewed_extension
    module = load_reviewed_extension(job['binary_cache'], p['binary'],
                                     allow_local_http=job.get('allow_local_http', False))
    result = module.run_pack(pack, device, str(out), emit,
                             memory_mib=p['budget']['memory_mib'], max_repeats=p['budget']['max_repeats'])
    if not isinstance(result, dict) or result.get('status') not in {'NO_ANOMALY_OBSERVED', 'NUMERICAL_ANOMALY_DETECTED', 'INCOMPLETE'}:
        raise ValueError('Compiled deep probe returned an invalid result contract')
    result['pack_id'] = pack['pack_id']
    result['provenance'] = pack['provenance']
    result['binary_sha256'] = p['binary']['sha256']
    result['execution_lane'] = expected['execution_lane']
    return result


def preview_result(run_dir, case_file, result_dir, config):
    consent=load_json(result_dir/'consent.json')
    review, sha, record, envelope = preview(run_dir,case_file,config,validation_time=consent['accepted_unix'])
    result=load_json(result_dir/'case-result.json')
    if result['case_sha256']!=review['case_sha256'] or consent['review_sha256']!=sha:
        raise ValueError('Deep result provenance differs from reviewed case')
    diagnosis=load_json(result_dir/'probe/diagnosis.json') if (result_dir/'probe/diagnosis.json').exists() else {}
    rows=[]
    if len(diagnosis.get('probes',[]))>4096:
        raise ValueError('Too many deep probe rows; no silent truncation')
    for r in diagnosis.get('probes',[]):
        actual=r.get('actual_sha256'); expected=r.get('expected_sha256')
        if not (isinstance(actual,str) and isinstance(expected,str) and
                re.fullmatch(r'[a-f0-9]{64}',actual) and re.fullmatch(r'[a-f0-9]{64}',expected)):
            raise ValueError('Deep probe hashes are missing or invalid; no silent omission')
        rows.append({'probe_id':r['probe_id'],'iteration':r.get('iteration',0),
                     'actual_sha256':actual,'expected_sha256':expected,
                     'matched':actual==expected})
    payload={'schema':'computeproof.deep-result.v1','consent':True,
             'attempt_id':result['attempt_id'],'case_sha256':review['case_sha256'],
             'source_summary_sha256':review['source_summary_sha256'],
             'status':result['status'],'qualified_failure_event':result['qualified_failure_event'],
             'execution_lane':review['execution_lane'],'probes':rows}
    if len(canonical(payload))>512*1024:
        raise ValueError('Deep contribution exceeds 512 KiB; no silent truncation')
    path=result_dir/'contribution/deep-result.json'
    save_json(path,payload)
    return path,hashlib.sha256(canonical(payload)).hexdigest(),record


def upload_result(run_dir,case_file,result_dir,config,confirmed_sha):
    path,sha,record=preview_result(run_dir,case_file,result_dir,config)
    if not confirmed_sha or confirmed_sha!=sha:
        raise ValueError('Explicit unchanged deep-result preview SHA required')
    payload=load_json(path,512*1024)
    state=records(run_dir,config)
    url=state['endpoint']+'/v1/reports/'+record['id']+'/cases/'+payload['case_sha256']+'/result'
    reply=Transport(config.network_timeout,config.allow_local_http).json(
        url,method='PUT',payload=payload,token=record['token'])
    save_json(result_dir/'contribution/receipt.json',reply)
    return reply
