"""Signed synthetic protocol fixtures only; no native code, GPU or network runs."""
import copy
import hashlib
import io
import json
import stat
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from agrel_public import deep_binary, deep_cases, probes, telemetry
from agrel_public.common import canonical, load_json, save_json
from agrel_public.config import Config
from agrel_public.devices import Device


def bundle_fixture(extra=None, symlink=False):
    target = deep_binary.runtime_target()
    name = deep_binary.MODULE + target['extension_suffix']
    raw = b'synthetic fixture only, never a loadable native extension'
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as bundle:
        info = zipfile.ZipInfo(name)
        if symlink:
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
        bundle.writestr(info, raw)
        if extra:
            bundle.writestr(extra, b'not allowed')
    data = stream.getvalue()
    descriptor = {'schema':'computeproof.deep-binary.v1', 'target':target,
                  'url':'https://example.invalid/deep-probe.zip',
                  'sha256':hashlib.sha256(data).hexdigest(), 'bytes':len(data),
                  'unpacked_bytes':len(raw), 'module':deep_binary.MODULE, 'entrypoint':name,
                  'files':[{'path':name, 'bytes':len(raw), 'sha256':hashlib.sha256(raw).hexdigest(), 'kind':'extension'}]}
    return descriptor, data


class FixtureTransport:
    local = False
    def __init__(self, data):
        self.data = data
        self.calls = []
    def open(self, url, **kwargs):
        self.calls.append((url, kwargs))
        stream = io.BytesIO(self.data)
        stream.status = 200
        return stream


@pytest.fixture
def native_case(tmp_path, monkeypatch, signed_pack):
    root = tmp_path.resolve()
    keys, pack, sign = signed_pack
    descriptor, data = bundle_fixture()
    pack = copy.deepcopy(pack)
    pack.update(backend='private_core_v1', provenance='publisher_validated', reference_receipt='software-fixture')
    record = {'id':'a' * 32, 'summary':{'result_class':'ATTENTION_REQUIRED',
              'gpu_model':'Fixture GPU', 'device_hash':'b' * 64}}
    payload = {'schema':'computeproof.case.v2', 'case_id':'native-fixture', 'report_id':record['id'],
               'source_summary_sha256':hashlib.sha256(canonical(record['summary'])).hexdigest(),
               'issued_unix':100, 'expires_unix':200, 'purpose':'characterize_path',
               'budget':{'timeout_s':240, 'memory_mib':512, 'max_repeats':8},
               'pack':json.loads(sign(pack)), 'binary':descriptor}
    envelope = json.loads(sign(payload))
    cfg = Config(public_keys=keys)
    save_json(root / 'research.private.json', {'endpoint':cfg.telemetry_url, 'records':[record]})
    path = root / 'case.json'
    save_json(path, envelope)
    transport = FixtureTransport(data)
    monkeypatch.setattr(deep_cases, 'Transport', lambda *args:transport)
    monkeypatch.setattr(deep_cases.time, 'time', lambda:150)
    monkeypatch.setattr(telemetry, 'device_hash', lambda identity:'b' * 64)
    return root, path, cfg, record, payload, sign, transport


def test_data_only_pack_cannot_import_preinstalled_private_core(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail('An ordinary pack imported a protected extension')
    monkeypatch.setitem(sys.modules, 'agrel_detector_core', SimpleNamespace(run_pack=forbidden))
    result = probes.run_pack({'backend':'private_core_v1'}, 'cuda:0', tmp_path, forbidden)
    assert result == {'status':'INCOMPLETE', 'reason':'EXPLICIT_DEEP_CONSENT_REQUIRED', 'probes':[]}


def test_download_needs_specific_preview_and_never_executes(native_case, monkeypatch):
    root, path, cfg, record, payload, sign, transport = native_case
    monkeypatch.setattr(deep_binary, 'load_reviewed_extension', lambda *a, **k:pytest.fail('download executed code'))
    review, sha = deep_cases.download_preview(root, path, cfg)
    assert review['execute'] is False and review['upload_results'] is False
    assert 'not a security sandbox' in review['native_code_notice']
    for consent in (None, '0' * 64):
        with pytest.raises(ValueError):
            deep_cases.download(root, path, cfg, consent)
    assert not transport.calls
    cached = deep_cases.download(root, path, cfg, sha)
    assert len(transport.calls) == 1
    assert deep_binary.verify_cache(cached, payload['binary']).is_file()
    assert not (root / 'result').exists()
    # Consent is required even when these exact bytes are already cached.
    with pytest.raises(ValueError):
        deep_cases.download(root, path, cfg, None)
    assert deep_cases.download(root, path, cfg, sha) == cached and len(transport.calls) == 1


def test_case_report_signature_and_platform_are_checked_before_download(native_case):
    root, path, cfg, record, payload, sign, transport = native_case
    for mutation in ('report', 'signature', 'platform', 'legacy'):
        changed = copy.deepcopy(payload)
        if mutation == 'report':
            changed['source_summary_sha256'] = '0' * 64
        elif mutation == 'platform':
            changed['binary']['target']['machine'] = ('aarch64' if deep_binary.runtime_target()['machine'] == 'x86_64' else 'x86_64')
        elif mutation == 'legacy':
            changed['schema'] = 'computeproof.case.v1'
            changed.pop('binary')
        envelope = json.loads(sign(changed))
        if mutation == 'signature':
            envelope['payload']['purpose'] = 'repeat_fixed_budget'
        save_json(path, envelope)
        with pytest.raises(Exception):
            deep_cases.download_preview(root, path, cfg)
    assert not transport.calls


@pytest.mark.parametrize('name', ['../escape.bin', 'sub/file.bin', 'setup.py', 'evil.dll',
                                  'CON.json', 'AUX.bin', 'nul.npz', 'COM1.bin', 'LPT1.json', 'ending.'])
def test_descriptor_rejects_source_paths_extra_libraries_and_windows_aliases(name):
    descriptor, _ = bundle_fixture()
    descriptor['files'].append({'path':name, 'bytes':1, 'sha256':'a' * 64, 'kind':'asset'})
    descriptor['unpacked_bytes'] += 1
    with pytest.raises(ValueError):
        deep_binary.validate_descriptor(descriptor)


@pytest.mark.parametrize('extra,symlink', [('setup.py', False), ('../escape.bin', False), (None, True)])
def test_archive_content_is_rejected_without_extraction(tmp_path, extra, symlink):
    descriptor, data = bundle_fixture(extra=extra, symlink=symlink)
    with pytest.raises(ValueError):
        deep_binary.download_bundle(descriptor, tmp_path.resolve() / 'cache', FixtureTransport(data))
    assert not (tmp_path / 'cache' / descriptor['sha256']).exists()
    assert not (tmp_path / 'escape.bin').exists()


@pytest.mark.parametrize('change', ['hash', 'truncated', 'oversize'])
def test_archive_size_and_hash_rejected(tmp_path, change):
    descriptor, data = bundle_fixture()
    if change == 'hash':
        descriptor['sha256'] = '0' * 64
    elif change == 'truncated':
        data = data[:-1]
    else:
        data += b'x'
    with pytest.raises(ValueError):
        deep_binary.download_bundle(descriptor, tmp_path.resolve() / 'cache', FixtureTransport(data))


def test_cache_modification_is_not_silently_replaced(tmp_path):
    descriptor, data = bundle_fixture()
    transport = FixtureTransport(data)
    cached = deep_binary.download_bundle(descriptor, tmp_path.resolve() / 'cache', transport)
    (cached / descriptor['entrypoint']).write_bytes(b'changed')
    with pytest.raises(ValueError):
        deep_binary.download_bundle(descriptor, tmp_path.resolve() / 'cache', transport)
    assert len(transport.calls) == 1
    assert (cached / descriptor['entrypoint']).read_bytes() == b'changed'


def test_native_execution_requires_both_consents_and_worker_rechecks(native_case, monkeypatch):
    from agrel_public import processes
    root, path, cfg, record, payload, sign, transport = native_case
    _, execution_sha, _, _ = deep_cases.preview(root, path, cfg)
    device = Device(0, 'Fixture GPU', 8000)
    with pytest.raises(FileNotFoundError):
        deep_cases.run(root, path, cfg, device, root / 'result', execution_sha)
    assert not (root / 'result').exists()
    _, download_sha = deep_cases.download_preview(root, path, cfg)
    deep_cases.download(root, path, cfg, download_sha)
    loads = []
    def loader(cache, descriptor, **kwargs):
        loads.append(cache)
        return SimpleNamespace(run_pack=lambda *a, **k:{'status':'NO_ANOMALY_OBSERVED', 'probes':[]})
    monkeypatch.setattr(deep_binary, 'load_reviewed_extension', loader)
    jobs = []
    def worker(job, emit, timeout):
        jobs.append(copy.deepcopy(job))
        assert timeout == payload['budget']['timeout_s'] and job['mode'] == 'deep'
        result = deep_cases.run_compiled_worker(job, job['device'], Path(job['out']), emit)
        save_json(Path(job['out']) / 'diagnosis.json', result)
    monkeypatch.setattr(processes, 'run_worker', worker)
    result = deep_cases.run(root, path, cfg, device, root / 'result', execution_sha)
    assert result['binary_sha256'] == payload['binary']['sha256'] and len(loads) == 1
    assert load_json(root / 'result/case-result.json')['uploaded'] is False
    for field, value in [('download_consent', {}), ('max_probe_memory_mib', 2048)]:
        job = copy.deepcopy(jobs[0])
        job[field] = value
        with pytest.raises(ValueError):
            deep_cases.run_compiled_worker(job, job['device'], Path(job['out']), lambda e:None)
    assert len(loads) == 1
    consent = load_json(root / 'result/consent.json')
    consent['review_sha256'] = '0' * 64
    save_json(root / 'result/consent.json', consent)
    with pytest.raises(ValueError):
        deep_cases.run_compiled_worker(jobs[0], 'cuda:0', root / 'other', lambda e:None)
    assert len(loads) == 1


def test_expiry_blocks_new_execution_not_preview_of_existing_result(native_case, monkeypatch):
    from agrel_public import processes
    root, path, cfg, record, payload, sign, transport = native_case
    _, download_sha = deep_cases.download_preview(root, path, cfg)
    deep_cases.download(root, path, cfg, download_sha)
    _, execution_sha, _, _ = deep_cases.preview(root, path, cfg)
    monkeypatch.setattr(processes, 'run_worker', lambda job, emit, timeout:
                        save_json(Path(job['out']) / 'diagnosis.json', {'status':'INCOMPLETE', 'probes':[]}))
    deep_cases.run(root, path, cfg, Device(0, 'Fixture GPU', 8000), root / 'result', execution_sha)
    original = deep_cases.preview_result(root, path, root / 'result', cfg)[1]
    monkeypatch.setattr(deep_cases.time, 'time', lambda:250)
    assert deep_cases.preview_result(root, path, root / 'result', cfg)[1] == original
    with pytest.raises(ValueError):
        deep_cases.run(root, path, cfg, Device(0, 'Fixture GPU', 8000), root / 'another', execution_sha)
