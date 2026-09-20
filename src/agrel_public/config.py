# Aritheia Community — GPU scientific-computing reliability check
# Copyright (C) 2026 The Aritheia authors
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE file
# (GNU GPL v3) shipped with this package or <https://www.gnu.org/licenses/>.
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from .common import config_home, load_json, strict_keys

@dataclass
class Config:
    telemetry: str = "auto"
    telemetry_url: str = "https://data.intc.ca:8443/sci-test"
    max_parallel_gpus: int = 2
    api_url: str = ""
    public_keys: dict[str, str] = field(default_factory=dict)
    pack_cache: Path = field(default_factory=lambda: config_home() / "packs")
    network_timeout: float = 4.0
    max_pack_bytes: int = 16 * 1024 * 1024
    worker_timeout: float = 240.0
    max_probe_memory_mib: int = 512
    max_probe_repeats: int = 8
    allow_local_http: bool = False
    iterations: int = 8
    memory_roundtrip_mib: int = 128
    scenario_timeout: float = 1800.0
    asset_release_url: str = ""
    native_blocks: int = 4096
    native_iterations: int = 4

    @classmethod
    def read(cls, path: str | Path | None = None) -> "Config":
        p = Path(path) if path else config_home() / "config.json"
        if not p.exists():
            if path:
                raise FileNotFoundError(p)
            return cls()
        data = load_json(p, 64 * 1024)
        strict_keys(data, set(), set(cls.__dataclass_fields__))
        if "pack_cache" in data:
            data["pack_cache"] = Path(data["pack_cache"]).expanduser()
        obj = cls(**data)
        if not 0.2 <= obj.network_timeout <= 30:
            raise ValueError("network_timeout must be 0.2..30 seconds")
        if not 30 <= obj.worker_timeout <= 3600:
            raise ValueError("worker_timeout must be 30..3600 seconds")
        if not 1 <= obj.max_pack_bytes <= 64 * 1024 * 1024:
            raise ValueError("pack size budget invalid")
        if not 16 <= obj.max_probe_memory_mib <= 8192:
            raise ValueError("probe memory budget invalid")
        if not 1 <= obj.max_probe_repeats <= 32:
            raise ValueError("repeat budget invalid")
        if not 1 <= obj.iterations <= 8:
            raise ValueError("iterations must be 1..8 (the frozen reference holds 8 seeds)")
        if not 1 <= obj.memory_roundtrip_mib <= 1024:
            raise ValueError("memory_roundtrip_mib must be 1..1024")
        if not 60 <= obj.scenario_timeout <= 86400:
            raise ValueError("scenario_timeout must be 60..86400 seconds")
        if not 64 <= obj.native_blocks <= 4096 or not 1 <= obj.native_iterations <= 4:
            raise ValueError("native probe budget invalid")
        if obj.telemetry not in ("auto", "off") or not 1 <= obj.max_parallel_gpus <= 32:
            raise ValueError("Invalid telemetry/parallel GPU configuration")
        from .network import validate_url
        validate_url(obj.telemetry_url, obj.allow_local_http)
        if obj.asset_release_url:
            from .network import validate_url
            validate_url(obj.asset_release_url, obj.allow_local_http)
        return obj
