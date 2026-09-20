"""Pinned original-source downloads and explicitly identified local preprocessing."""
from __future__ import annotations
import csv
import io
import json
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

from ..common import load_json, save_json, sha256_file

PREPARATION = 'original-source-retrieval-v1'


def fetch(entry, target, transport, cancel=None, progress=lambda n, total: None):
    from urllib.error import URLError
    for attempt in range(3):
        try:
            return _fetch_once(entry, target, transport, cancel, progress)
        except (URLError, ConnectionError, TimeoutError):
            if attempt == 2 or (cancel and cancel.is_set()):
                raise
            if cancel:
                if cancel.wait(attempt + 1):
                    raise TimeoutError('Download cancelled')
            else:
                time.sleep(attempt + 1)


def _fetch_once(entry, target, transport, cancel=None, progress=lambda n, total: None):
    import hashlib
    expected = entry['bytes']
    h = hashlib.sha256()
    done = 0
    deadline = time.monotonic() + 7200
    with transport.open(entry['url']) as response, target.open('wb') as f:
        while True:
            if (cancel and cancel.is_set()) or time.monotonic() > deadline:
                raise TimeoutError('Original-source download cancelled or timed out')
            block = response.read(min(1 << 20, expected - done + 1))
            if not block:
                break
            done += len(block)
            if done > expected:
                raise ValueError('Original-source file exceeds pinned size')
            h.update(block)
            f.write(block)
            progress(done, expected)
    if done != expected or h.hexdigest() != entry['sha256']:
        raise ValueError('Original-source size/SHA-256 mismatch')


def _zip_text(z, suffix):
    names = [n for n in z.namelist() if n == suffix or n.endswith('/' + suffix)]
    if len(names) != 1 or z.getinfo(names[0]).file_size > 100_000_000:
        raise ValueError('Unexpected source archive member')
    return z.read(names[0]).decode('utf-8')


def prepare(pack, source, output, entry):
    import numpy as np
    import importlib.metadata
    output.mkdir(parents=True, exist_ok=True)
    versions = {'numpy': np.__version__}
    if pack == 'singlecell':
        import anndata as ad
        from scipy import sparse
        a = ad.read_h5ad(source)
        x = a.X.toarray() if sparse.issparse(a.X) else np.asarray(a.X)
        if x.shape != (2638, 1838):
            raise ValueError('Expected PBMC3k processed matrix 2638 x 1838')
        x = x.astype('float32')
        labels = a.obs['louvain'].astype('category').cat.codes.to_numpy(dtype='int64')
        ids = a.obs_names.to_numpy(dtype=str)
        x = (x / np.maximum(np.linalg.norm(x.astype('float64'), axis=1, keepdims=True), 1e-12)).astype('float32')
        arrays = dict(docs=x, queries=x, doc_ids=ids, query_ids=ids, self_doc=np.arange(len(x)),
                      labels=labels, doc_labels=labels, relevance=np.zeros((len(x), 0), dtype=np.int8))
        versions.update({p: importlib.metadata.version(p) for p in ('anndata', 'scipy')})
    elif pack == 'literature':
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize
        with zipfile.ZipFile(source) as z:
            corpus = [json.loads(line) for line in _zip_text(z, 'corpus.jsonl').splitlines() if line]
            queries = [json.loads(line) for line in _zip_text(z, 'queries.jsonl').splitlines() if line]
            rel = list(csv.DictReader(io.StringIO(_zip_text(z, 'qrels/test.tsv')), delimiter='\t'))
        qids = sorted({r['query-id'] for r in rel})
        qmap = {str(q['_id']): q for q in queries}
        corpus = sorted(corpus, key=lambda d: str(d['_id']))
        dids = [str(d['_id']) for d in corpus]
        if len(dids) != 5183 or len(qids) != 300:
            raise ValueError('Expected full SciFact corpus and test queries')
        vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        docs = vec.fit_transform([d['title'] + ' ' + d['text'] for d in corpus])
        queries = vec.transform([qmap[q]['text'] for q in qids])
        svd = TruncatedSVD(n_components=512, n_iter=5, random_state=20260910)
        dx = normalize(svd.fit_transform(docs)).astype('float32')
        qx = normalize(svd.transform(queries)).astype('float32')
        relevance = np.zeros((len(qids), len(dids)), dtype=np.int8)
        qi, di = {v:i for i,v in enumerate(qids)}, {v:i for i,v in enumerate(dids)}
        for r in rel:
            if int(r['score']) > 0:
                relevance[qi[r['query-id']], di[r['corpus-id']]] = 1
        arrays = dict(docs=dx, queries=qx, doc_ids=np.asarray(dids), query_ids=np.asarray(qids),
                      self_doc=np.full(len(qids), -1), labels=np.full(len(qids), -1),
                      doc_labels=np.full(len(dids), -1), relevance=relevance)
        versions.update({p: importlib.metadata.version(p) for p in ('scikit-learn', 'scipy')})
    else:
        raise ValueError('Unknown original-source dataset')
    if not np.isfinite(arrays['docs']).all() or not np.isfinite(arrays['queries']).all():
        raise ValueError('Nonfinite prepared input; not a GPU test result')
    np.savez_compressed(output / 'data.npz', **arrays)
    save_json(output / 'metadata.json', dict(dataset=entry['dataset'], task='retrieval', similarity='cosine', synthetic=False,
        prepared_sha256=sha256_file(output / 'data.npz'), source_url=entry['url'], source_sha256=entry['sha256'],
        preparation=PREPARATION, versions=versions, reference_kind='local_preprocessing_not_archived_bitwise_reference',
        notes=['Preprocessing can differ across library/BLAS versions. Application metrics are descriptive, not a hardware-error oracle.']))


def verify_prepared(name):
    from . import assets
    from .registry import scenario
    spec = scenario(name)
    meta_path = assets.locate(f"data/prepared/{spec['dataset']}/metadata.json")
    if meta_path is None:
        return False
    meta = load_json(meta_path)
    if meta.get('preparation') != PREPARATION:
        return False
    entry = assets.optional_manifest()['packs'][spec['pack']]['upstream']
    if meta.get('source_sha256') != entry['sha256'] or meta.get('source_url') != entry['url']:
        raise ValueError('Prepared input source identity differs')
    assets.verify_frozen_chain(name)
    return True


def download(pack_name, pack, transport, *, cancel=None, progress=lambda n,t: None, dest_root=None):
    from . import assets
    root = Path(dest_root or assets.user_root()).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    relative = f"models/qwen3.5-4b" if pack_name == 'language' else f"data/prepared/{pack['upstream']['dataset']}"
    final = root / relative
    if any(p.is_symlink() for p in (final, *final.parents)):
        raise ValueError('Symlink destination refused')
    if final.exists():
        raise FileExistsError('Destination exists; choose a separate ARITHEIA_ASSETS directory')
    required = sum(e['bytes'] for e in pack['members']) if pack_name == 'language' else 250_000_000
    if shutil.disk_usage(root).free < required + 100_000_000:
        raise OSError('Insufficient disk space for original-source assets')
    temp = Path(tempfile.mkdtemp(prefix='.upstream-', dir=root))
    try:
        prepared = temp / 'prepared'
        if pack_name == 'language':
            prepared.mkdir()
            total = sum(e['bytes'] for e in pack['members'])
            done = 0
            for entry in pack['members']:
                fetch(entry, prepared / Path(entry['path']).name, transport, cancel,
                      lambda n,t,offset=done: progress(offset+n, total))
                done += entry['bytes']
        else:
            # Fail before networking if preparation dependencies are absent.
            if pack_name == 'singlecell':
                import anndata
            else:
                import sklearn
            source = temp / pack['upstream']['filename']
            fetch(pack['upstream'], source, transport, cancel, progress)
            prepare(pack_name, source, prepared, pack['upstream'])
        if cancel and cancel.is_set():
            raise TimeoutError('Download cancelled before installation')
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            raise FileExistsError('Asset appeared concurrently; not overwriting')
        os.rename(prepared, final)
        return root
    finally:
        shutil.rmtree(temp, ignore_errors=True)
