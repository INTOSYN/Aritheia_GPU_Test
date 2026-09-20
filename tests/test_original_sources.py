"""Software fixtures only: consent, identity, download integrity, no GPU tests."""
import hashlib
import io
import threading
from types import SimpleNamespace
import pytest
from agrel_public import cli, selection
from agrel_public.config import Config
from agrel_public.scenarios import assets, llm, upstream


def test_model_and_prompt_identity_match_validated_assets():
    p = assets.optional_manifest()['packs']['language']
    assert p['model'] == llm.MODEL_ID == 'Qwen/Qwen3.5-4B'
    assert p['revision'] == llm.REVISION
    assert {m['sha256'] for m in p['members'] if m['path'].endswith('.safetensors')} == {
        '26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61',
        'cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188'}
    for d, h in [('compound_prompts','fa09613d1ba89355014867c9d51caa0473f8e70cb34a84daa87c1f744a4fadcf'),
                 ('agent_prompts','45961de007faa47f3695b919b9cd2638f0134aa83796d5b3835f9eb10ec0ce57')]:
        assert assets.sha256_file(assets.locate(f'data/prepared/{d}/prompts.json')) == h


def test_language_without_consent_never_opens_network(tmp_path):
    def forbidden(*a):
        raise AssertionError('Network before consent')
    with pytest.raises(PermissionError):
        assets.download_pack('language', SimpleNamespace(open=forbidden), dest_root=tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('flag,expected', [('--accept-download', False), ('--accept-model-download', True)])
def test_guided_noninteractive_separates_model_consent(monkeypatch, flag, expected):
    monkeypatch.setattr(cli.sys.stdin, 'isatty', lambda: False)
    seen = []
    monkeypatch.setattr(cli, 'OptionalAssetJob', lambda d,p: SimpleNamespace(start=lambda: seen.extend(p)))
    cli.guided_asset_countdown(cli.parser().parse_args(['guided', flag]), Config())
    assert ('language' in seen) is expected
    assert ('singlecell' in seen) is (not expected)


def test_guided_interactive_no_model_consent_excludes_language(monkeypatch):
    monkeypatch.setattr(cli.sys.stdin, 'isatty', lambda: True)
    monkeypatch.setattr(cli, 'yes', lambda *a,**k: False)
    monkeypatch.setattr(assets, 'pack_status', lambda *a: {'installed': False})
    seen = []
    monkeypatch.setattr(cli, 'OptionalAssetJob', lambda d,p: SimpleNamespace(start=lambda: seen.extend(p)))
    cli.guided_asset_countdown(cli.parser().parse_args(['guided','--accept-download']), Config())
    assert seen == ['singlecell','literature']


def test_all_scenarios_data_consent_cannot_download_model(monkeypatch):
    calls=[]
    monkeypatch.setattr(assets,'status',lambda n: dict(ready=False,missing=['weights'],integrity_error=None))
    def download(*a,**kw): calls.append(a)
    download.release_url=''
    result=selection.choose(explicit=['llm_biomed_tables','agent_tools'],large='include',interactive=False,
        ask=lambda *a,**k:False,say=lambda x:None,accept_download=True,downloader=download)
    assert calls==[] and len(result['excluded'])==2
    assert all(x['reason']=='DOWNLOAD_DECLINED' for x in result['excluded'])


def test_offline_model_consent_still_no_network(monkeypatch):
    monkeypatch.setattr(cli,'make_downloader',lambda *a:pytest.fail('Offline requested network'))
    args=cli.parser().parse_args(['guided','--offline','--accept-model-download'])
    assert cli.guided_asset_countdown(args,Config()) is None


@pytest.mark.parametrize('body', [b'bad', b'excess-size'])
def test_raw_download_requires_exact_size_and_hash(tmp_path,body):
    entry=dict(url='https://example.invalid/data',bytes=4,sha256=hashlib.sha256(b'good').hexdigest())
    with pytest.raises(ValueError):
        upstream.fetch(entry,tmp_path/'raw',SimpleNamespace(open=lambda url:io.BytesIO(body)))


def test_cancelled_raw_download(tmp_path):
    cancel=threading.Event();cancel.set()
    with pytest.raises(TimeoutError):
        upstream.fetch(dict(url='https://example.invalid/data',bytes=4,sha256='a'*64),tmp_path/'raw',
                       SimpleNamespace(open=lambda url:io.BytesIO(b'good')),cancel)


def test_other_model_not_accepted(tmp_path):
    (tmp_path/'config.json').write_text('{"model_type":"other"}')
    with pytest.raises(ValueError):llm.verify_model(tmp_path)


def test_legacy_language_zip_refused_before_read(tmp_path):
    with pytest.raises(ValueError,match='Qwen3.5-4B'):
        assets.install_pack('language',tmp_path/'not-present.zip')


def test_guided_declined_model_not_mislabeled_as_data_download_failure(monkeypatch):
    monkeypatch.setattr(assets,'status',lambda n:dict(ready=False,integrity_error=None))
    job=SimpleNamespace(allowed_packs={'singlecell','literature'},errors=[{'pack':'singlecell','error':'URLError'}])
    _, excluded=cli.guided_optional_scope(job)
    reasons={e['scenario']:e['reason'] for e in excluded}
    assert reasons['singlecell_neighbors']=='DOWNLOAD_FAILED'
    assert reasons['llm_biomed_tables']==reasons['agent_tools']=='DOWNLOAD_DECLINED'
