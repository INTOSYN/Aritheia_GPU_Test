# Aritheia Community — GPU scientific-computing reliability check
# Copyright (C) 2026 The Aritheia authors
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE file
# (GNU GPL v3) shipped with this package or <https://www.gnu.org/licenses/>.
"""Scenario selection before a run: eight built-in scenarios by default, the four
large optional scenarios offered one by one (default: no), and a separate,
size-disclosed consent for each asset pack download. Unattended default selects the eight bundled scenes and skips optional downloads.
Requested but unavailable scenes remain visible as incomplete scope."""
from __future__ import annotations
from .scenarios import assets
from .scenarios.registry import BUILTIN, OPTIONAL, ORDER, PACKS, SCENARIOS, scenario


def parse_scenarios(text: str | None) -> list[str] | None:
    if text is None:
        return None
    text = text.strip()
    if text == "all":
        return list(ORDER)
    if text == "builtin":
        return list(BUILTIN)
    if text == "none":
        return []
    names = [t.strip() for t in text.split(",") if t.strip()]
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        raise ValueError("Unknown scenario(s): " + ", ".join(unknown) + "; choices: " + ", ".join(ORDER))
    return [n for n in ORDER if n in names]


def choose(*, explicit: list[str] | None, large: str, interactive: bool, ask, say,
           accept_download: bool, downloader=None, probes_only=False) -> dict:
    """Return {'selected': [...], 'excluded': [{scenario, reason}], 'downloaded': [...]}."""
    if explicit is not None:
        wanted = list(explicit)
    else:
        wanted = list(BUILTIN)
        if large == "ask" and not interactive:
            raise ValueError("Unattended run: choose --large-scenarios skip|include or --scenarios explicitly")
        if large == "include":
            wanted += OPTIONAL
        elif large == "ask":
            say("四个较大的可选场景逐项选择（默认不加入；缺资产时会再单独确认下载体积）：")
            for name in OPTIONAL:
                s = SCENARIOS[name]
                st = assets.status(name)
                note = "资产已就绪" if st["ready"] else f"需要资产包 {s['pack']}"
                if ask(f"  加入『{s['title']}』（{s['direction']}；{note}）？", default=False):
                    wanted.append(name)
    requested = list(wanted)
    if probes_only:
        return dict(selected=requested, requested=requested, excluded=[], downloaded=[], applications_skipped=True)
    excluded, downloaded = [], []
    manifest = None
    for name in list(wanted):
        st = assets.status(name)
        if st["ready"]:
            continue
        spec = scenario(name)
        if spec["tier"] != "optional":
            wanted.remove(name)
            excluded.append(dict(scenario=name, reason="FROZEN_ASSET_MISSING", missing=st["missing"], integrity=st["integrity_error"]))
            continue
        if st["integrity_error"]:
            wanted.remove(name)
            excluded.append(dict(scenario=name, reason="ASSET_INTEGRITY_ERROR", integrity=st["integrity_error"]))
            continue
        pack = spec["pack"]
        if pack in downloaded:
            if assets.status(name)["ready"]:
                continue
        try:
            manifest = manifest or assets.optional_manifest()
            ps = assets.pack_status(pack, manifest)
        except Exception as e:
            wanted.remove(name)
            excluded.append(dict(scenario=name, reason="ASSET_MANIFEST_INVALID", detail=f"{type(e).__name__}"))
            continue
        if not ps["pinned"]:
            wanted.remove(name)
            excluded.append(dict(scenario=name, reason="ASSET_UNPINNED",
                                 detail="发布者尚未为该资产包钉住 SHA-256/体积；拒绝下载无法核验的数据"))
            continue
        if not (ps["release_url"] or (downloader and downloader.release_url)):
            wanted.remove(name)
            excluded.append(dict(scenario=name, reason="ASSET_RELEASE_URL_MISSING",
                                 detail="未配置正式发布地址（config asset_release_url 或 AGREL_ASSET_RELEASE_URL）"))
            continue
        say(f"『{spec['title']}』需要资产包 {pack}（{ps['label']}），约 {assets.human_bytes(ps['bytes'])}，"
            f"SHA-256 已钉住；许可：{ps['license']}")
        agreed = accept_download or (interactive and ask("  现在下载并校验该资产包？", default=False))
        if not agreed:
            wanted.remove(name)
            excluded.append(dict(scenario=name, reason="DOWNLOAD_DECLINED", pack=pack))
            continue
        if downloader is None:
            wanted.remove(name)
            excluded.append(dict(scenario=name, reason="DOWNLOAD_UNAVAILABLE", pack=pack, detail="offline run"))
            continue
        try:
            downloader(pack, manifest)
            downloaded.append(pack)
        except Exception as e:
            wanted.remove(name)
            excluded.append(dict(scenario=name, reason="DOWNLOAD_FAILED", pack=pack, detail=f"{type(e).__name__}"))
            continue
        if not assets.status(name)["ready"]:
            wanted.remove(name)
            excluded.append(dict(scenario=name, reason="ASSET_STILL_MISSING", pack=pack))
    return dict(selected=[n for n in ORDER if n in wanted], requested=requested, excluded=excluded, downloaded=downloaded)
