"""Content-addressed delivery of explicitly reviewed native follow-up probes.

The signed case is the trust root. This is not a native-code sandbox: a user
authorizes code from a trusted publisher. No package installer, scripts, source
files, import search path, or dependency resolver participates in this path.
"""
from __future__ import annotations

import hashlib
import importlib.util
import platform
import re
import stat
import sys
import sysconfig
import tempfile
import time
import zipfile
from pathlib import Path

from .common import load_json, save_json, sha256_file, strict_keys
from .network import validate_url

MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_UNPACKED_BYTES = 256 * 1024 * 1024
MODULE = 'agrel_detector_core'
ASSET_SUFFIXES = {'.json', '.bin', '.fatbin', '.cubin', '.npz'}


def runtime_target():
    machine = platform.machine().lower()
    machine = {'amd64': 'x86_64', 'arm64': 'aarch64'}.get(machine, machine)
    return {'os': platform.system(), 'machine': machine,
            'implementation': sys.implementation.name,
            'python_tag': f'cp{sys.version_info.major}{sys.version_info.minor}',
            'abi_tag': sys.implementation.cache_tag,
            'extension_suffix': sysconfig.get_config_var('EXT_SUFFIX')}


def validate_descriptor(desc, *, match_runtime=True, allow_local_http=False):
    strict_keys(desc, {'schema', 'target', 'url', 'sha256', 'bytes', 'unpacked_bytes',
                       'module', 'entrypoint', 'files'})
    if desc['schema'] != 'computeproof.deep-binary.v1' or desc['module'] != MODULE:
        raise ValueError('Unsupported compiled deep-probe format')
    target = desc['target']
    strict_keys(target, set(runtime_target()))
    if target['os'] not in {'Linux', 'Windows', 'Darwin'} or target['machine'] not in {'x86_64', 'aarch64'}:
        raise ValueError('Unsupported deep-probe platform')
    if target['implementation'] != 'cpython' or not re.fullmatch(r'cp3[0-9]{1,2}', target['python_tag']):
        raise ValueError('A platform-specific CPython build is required')
    suffix = target['extension_suffix']
    if not isinstance(suffix, str) or not re.fullmatch(r'\.[A-Za-z0-9_.-]+\.(so|pyd)', suffix):
        raise ValueError('An ABI-specific compiled extension is required')
    if (target['os'] == 'Windows') != suffix.endswith('.pyd'):
        raise ValueError('Compiled extension platform differs')
    if match_runtime and target != runtime_target():
        raise ValueError('Deep-probe OS, architecture or Python ABI differs; obtain a matching build')
    validate_url(desc['url'], allow_local_http)
    if not re.fullmatch(r'[a-f0-9]{64}', desc['sha256']):
        raise ValueError('Invalid deep-probe archive hash')
    for field, limit in (('bytes', MAX_ARCHIVE_BYTES), ('unpacked_bytes', MAX_UNPACKED_BYTES)):
        if type(desc[field]) is not int or not 0 < desc[field] <= limit:
            raise ValueError('Deep-probe download exceeds the client size limit')
    if desc['entrypoint'] != MODULE + suffix:
        raise ValueError('Only the named ABI-specific extension can be loaded')
    files = desc['files']
    if not isinstance(files, list) or not 1 <= len(files) <= 64:
        raise ValueError('Invalid deep-probe file manifest')
    names = set()
    extensions = 0
    total = 0
    for item in files:
        strict_keys(item, {'path', 'bytes', 'sha256', 'kind'})
        name = item['path']
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,180}', name):
            raise ValueError('Deep-probe paths must be plain file names')
        stem = name.split('.')[0].upper()
        if (name.casefold() in names or name.casefold() == 'descriptor.json' or name.endswith('.') or
                stem in {'CON', 'NUL', 'PRN', 'AUX', *(f'COM{i}' for i in range(10)), *(f'LPT{i}' for i in range(10))}):
            raise ValueError('Duplicate or reserved deep-probe file name')
        names.add(name.casefold())
        if not re.fullmatch(r'[a-f0-9]{64}', item['sha256']):
            raise ValueError('Invalid deep-probe file hash')
        if type(item['bytes']) is not int or not 0 < item['bytes'] <= MAX_ARCHIVE_BYTES:
            raise ValueError('Invalid deep-probe file size')
        total += item['bytes']
        if item['kind'] == 'extension':
            extensions += 1
            if name != desc['entrypoint']:
                raise ValueError('Unreviewed compiled module')
        elif item['kind'] != 'asset' or Path(name).suffix.lower() not in ASSET_SUFFIXES:
            raise ValueError('Source, scripts and additional libraries are not permitted')
    if extensions != 1 or total != desc['unpacked_bytes']:
        raise ValueError('The deep-probe manifest must contain exactly one compiled extension')
    return desc


def _no_symlink(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Deep-probe cache cannot contain symlinks')


def verify_cache(root, desc, *, allow_local_http=False):
    validate_descriptor(desc, allow_local_http=allow_local_http)
    root = Path(root)
    _no_symlink(root)
    if not root.is_dir() or root.name != desc['sha256']:
        raise ValueError('Reviewed deep-probe bundle has not been downloaded')
    expected = {'descriptor.json', *(item['path'] for item in desc['files'])}
    if {p.name for p in root.iterdir()} != expected:
        raise ValueError('Deep-probe cache contents differ from the signed manifest')
    _no_symlink(root / 'descriptor.json')
    if load_json(root / 'descriptor.json') != desc:
        raise ValueError('Cached deep-probe descriptor changed')
    for item in desc['files']:
        path = root / item['path']
        _no_symlink(path)
        if not path.is_file() or path.stat().st_size != item['bytes'] or sha256_file(path) != item['sha256']:
            raise ValueError('Cached deep-probe file changed')
    return root / desc['entrypoint']


def _extract(archive, target, desc):
    files = {item['path']: item for item in desc['files']}
    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        if len(infos) != len(files) or {i.filename for i in infos} != set(files):
            raise ValueError('Archive contains files outside the reviewed manifest')
        for info in infos:
            item = files[info.filename]
            mode = info.external_attr >> 16
            kind = stat.S_IFMT(mode)
            if info.is_dir() or info.flag_bits & 1 or kind not in {0, stat.S_IFREG}:
                raise ValueError('Directories, encrypted entries and symlinks are not permitted')
            if info.file_size != item['bytes']:
                raise ValueError('Unpacked file size differs from the signed manifest')
            raw = bundle.read(info)
            if len(raw) != item['bytes'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
                raise ValueError('Unpacked file hash differs from the signed manifest')
            with (target / info.filename).open('xb') as output:
                output.write(raw)


def download_bundle(desc, cache, transport):
    """Internal helper: caller must first verify the case and download consent."""
    validate_descriptor(desc, allow_local_http=transport.local)
    cache = Path(cache)
    _no_symlink(cache)
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    final = cache / desc['sha256']
    if final.exists() or final.is_symlink():
        verify_cache(final, desc, allow_local_http=transport.local)
        return final
    with tempfile.TemporaryDirectory(prefix='.deep-download-', dir=cache) as temporary:
        temporary = Path(temporary)
        archive = temporary / 'bundle.zip'
        done = 0
        digest = hashlib.sha256()
        deadline = time.monotonic() + 180
        with transport.open(desc['url'], headers={'Accept': 'application/zip'}) as response, archive.open('xb') as output:
            if response.status != 200:
                raise ValueError('Unexpected deep-probe download response')
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError('Deep-probe download time budget exceeded')
                block = response.read(min(65536, desc['bytes'] - done + 1))
                if not block:
                    break
                done += len(block)
                if done > desc['bytes']:
                    raise ValueError('Deep-probe archive exceeds its signed size')
                output.write(block)
                digest.update(block)
        if done != desc['bytes'] or digest.hexdigest() != desc['sha256']:
            raise ValueError('Downloaded deep-probe archive size or hash differs')
        extracted = temporary / 'verified'
        extracted.mkdir(mode=0o700)
        _extract(archive, extracted, desc)
        save_json(extracted / 'descriptor.json', desc)
        if final.exists() or final.is_symlink():
            raise FileExistsError('Deep-probe cache appeared during download; retry after inspection')
        extracted.rename(final)
    verify_cache(final, desc, allow_local_http=transport.local)
    return final


def load_reviewed_extension(root, desc, *, allow_local_http=False):
    """Worker-only load by verified absolute path, never from sys.path."""
    path = verify_cache(root, desc, allow_local_http=allow_local_http)
    if MODULE in sys.modules:
        raise ValueError('Deep extension must start in a fresh worker process')
    spec = importlib.util.spec_from_file_location(MODULE, str(path))
    if spec is None or spec.loader is None:
        raise ValueError('Cannot load this reviewed native extension')
    from importlib.machinery import ExtensionFileLoader
    if not isinstance(spec.loader, ExtensionFileLoader):
        raise ValueError('Only a compiled extension loader is permitted')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, 'run_pack', None)):
        raise ValueError('Deep extension does not implement run_pack')
    return module
