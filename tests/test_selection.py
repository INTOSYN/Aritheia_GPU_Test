"""Eight built-in scenarios by default; four optional ones asked one by one; downloads need
pinned digests, a release URL and separate consent. Nothing silent in either direction."""
import pytest
from agrel_public import selection
from agrel_public.scenarios import assets
from agrel_public.scenarios.registry import BUILTIN, OPTIONAL

READY = {"drug_screen","drug_finetune","molecule_neighbors","cytology","ocr_cache"}

@pytest.fixture
def quiet(monkeypatch):
    out = []
    monkeypatch.setattr(assets, "status", lambda n: {"scenario":n,"ready":n in READY,"missing":[] if n in READY else ["x"],"integrity_error":None,"tier":"x"})
    return out

def test_unattended_default_is_builtin_only(quiet):
    r = selection.choose(explicit=None, large="skip", interactive=False, ask=lambda *a, **k: True, say=quiet.append, accept_download=False)
    assert r["selected"] == sorted(READY, key=BUILTIN.index)
    assert {e["scenario"] for e in r["excluded"]} == {"genomics_splice","medical_ultrasound","materials_screen"}
    assert all(e["reason"] == "FROZEN_ASSET_MISSING" for e in r["excluded"])

def test_unattended_ask_is_an_error(quiet):
    with pytest.raises(ValueError, match="Unattended"):
        selection.choose(explicit=None, large="ask", interactive=False, ask=lambda *a, **k: True, say=quiet.append, accept_download=False)

def test_interactive_asks_each_optional_scenario_default_no(quiet):
    asked = []
    def ask(prompt, default=False):
        asked.append((prompt, default)); return "单细胞" in prompt
    r = selection.choose(explicit=None, large="ask", interactive=True, ask=ask, say=quiet.append, accept_download=False)
    assert len([a for a in asked if "加入" in a[0]]) == 4 and all(a[1] is False for a in asked)
    # singlecell was chosen; the shipped manifest is pinned but has no release URL yet -> excluded, visibly
    ex = {e["scenario"]: e["reason"] for e in r["excluded"]}
    assert ex["singlecell_neighbors"] in ("ASSET_RELEASE_URL_MISSING", "ASSET_UNPINNED")
    assert "biomed_rag" not in r["selected"] and "biomed_rag" not in ex

def test_explicit_list_and_parse():
    assert selection.parse_scenarios("all") and len(selection.parse_scenarios("all")) == 12
    assert selection.parse_scenarios("builtin") == BUILTIN
    assert selection.parse_scenarios("cytology, drug_screen") == ["drug_screen","cytology"]
    assert selection.parse_scenarios("none") == []
    with pytest.raises(ValueError, match="Unknown"):
        selection.parse_scenarios("cytology,evil")

@pytest.fixture
def pinned(monkeypatch, tmp_path):
    m = {"schema":"aritheia.optional-assets.v1","release_url":"https://example.org/rel","packs":{
        "singlecell":{"file":"singlecell.asset.zip","bytes":10,"sha256":"a"*64,"approx_bytes":10,"license":"x",
                      "members":[{"path":"data/prepared/pbmc3k/data.npz","bytes":5,"sha256":"b"*64}]}}}
    monkeypatch.setattr(assets, "optional_manifest", lambda path=None: m)
    return m

def test_download_requires_consent_then_downloads_once(quiet, pinned, monkeypatch):
    calls = []
    ready = set(READY)
    monkeypatch.setattr(assets, "status", lambda n: {"scenario":n,"ready":n in ready,"missing":[] if n in ready else ["x"],"integrity_error":None,"tier":"x"})
    def downloader(pack, manifest):
        calls.append(pack); ready.add("singlecell_neighbors")
    downloader.release_url = "https://example.org/rel"
    r = selection.choose(explicit=["singlecell_neighbors"], large="skip", interactive=False, ask=lambda *a, **k: False,
                         say=quiet.append, accept_download=False, downloader=downloader)
    assert calls == [] and r["excluded"][0]["reason"] == "DOWNLOAD_DECLINED"
    r = selection.choose(explicit=["singlecell_neighbors"], large="skip", interactive=False, ask=lambda *a, **k: False,
                         say=quiet.append, accept_download=True, downloader=downloader)
    assert calls == ["singlecell"] and r["selected"] == ["singlecell_neighbors"] and r["downloaded"] == ["singlecell"]
    assert any("SHA-256 已钉住" in s for s in quiet)

def test_offline_cannot_download(quiet, pinned):
    r = selection.choose(explicit=["singlecell_neighbors"], large="skip", interactive=False, ask=lambda *a, **k: False,
                         say=quiet.append, accept_download=True, downloader=None)
    assert r["excluded"][0]["reason"] == "DOWNLOAD_UNAVAILABLE"

def test_failed_download_is_visible(quiet, pinned):
    def downloader(pack, manifest):
        raise ValueError("Pack size/SHA-256 mismatch; nothing installed")
    downloader.release_url = "https://example.org/rel"
    r = selection.choose(explicit=["singlecell_neighbors"], large="skip", interactive=False, ask=lambda *a, **k: False,
                         say=quiet.append, accept_download=True, downloader=downloader)
    assert r["excluded"][0]["reason"] == "DOWNLOAD_FAILED" and r["selected"] == []
