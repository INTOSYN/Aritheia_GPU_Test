"""Deep delivery CLI fixtures: no external network, GPU or real binary loading."""
from pathlib import Path

from agrel_public import cli, deep_cases


def test_download_preview_does_not_download_or_enumerate_gpu(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(deep_cases, "download_preview", lambda *a: ({"execute": False}, "1" * 64), raising=False)
    monkeypatch.setattr(deep_cases, "download", lambda *a: calls.append("download"), raising=False)
    monkeypatch.setattr(cli, "inventory", lambda: calls.append("inventory"))
    rc = cli.main(["deep", "preview-download", "run", "--case", "case.json"])
    assert rc == 0 and calls == []
    assert "1" * 64 in capsys.readouterr().out


def test_download_passes_its_own_consent_without_running(monkeypatch):
    calls = []
    def download(run, case, config, confirmed):
        calls.append((run, case, confirmed))
        return Path("verified-cache")
    monkeypatch.setattr(deep_cases, "download", download, raising=False)
    monkeypatch.setattr(deep_cases, "run", lambda *a: calls.append("execute"))
    monkeypatch.setattr(cli, "inventory", lambda: calls.append("inventory"))
    assert cli.main(["deep", "download", "run", "--case", "case.json", "--confirm", "2" * 64]) == 0
    assert calls == [(Path("run"), Path("case.json"), "2" * 64)]


def test_fetch_does_not_download_binary(monkeypatch):
    calls = []
    monkeypatch.setattr(deep_cases, "fetch", lambda *a: [])
    monkeypatch.setattr(deep_cases, "download", lambda *a: calls.append("download"), raising=False)
    assert cli.main(["deep", "fetch", "run"]) == 0
    assert calls == []


def test_guided_cancel_stops_optional_download(monkeypatch):
    from types import SimpleNamespace
    from agrel_public.config import Config
    args = cli.parser().parse_args(["guided"])
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli, "make_downloader", lambda *a: SimpleNamespace(release_url="https://example.invalid/assets"))
    monkeypatch.setattr(cli, "TimedConsoleInput", lambda: SimpleNamespace(poll=lambda timeout: "c"))
    def forbidden(*args):
        raise AssertionError("Cancelled download must not start a job")
    monkeypatch.setattr(cli, "OptionalAssetJob", forbidden)
    assert cli.guided_asset_countdown(args, Config()) is None
    assert args.optional_skip_reason == "OPTIONAL_DOWNLOAD_CANCELLED"
