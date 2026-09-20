"""Optional pack download over loopback HTTP: ZIP size, SHA-256, member list and per-member digests."""
import io
import json
import threading
import zipfile
from http.server import HTTPServer, BaseHTTPRequestHandler
import pytest
from agrel_public.common import digest, sha256_file
from agrel_public.network import Transport
from agrel_public.scenarios import assets

def make_zip(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()

@pytest.fixture
def served(tmp_path):
    members = {"data/prepared/pbmc3k/data.npz": b"npz-bytes-1234", "data/prepared/pbmc3k/metadata.json": b"{}"}
    raw = make_zip(members)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            self.send_response(200); self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)
    server = HTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True); t.start()
    manifest = {"schema":"aritheia.optional-assets.v1","release_url":f"http://127.0.0.1:{server.server_port}","packs":{
        "singlecell":{"file":"singlecell.asset.zip","bytes":len(raw),"sha256":digest(raw),"approx_bytes":len(raw),"license":"test",
                      "members":[{"path":k,"bytes":len(v),"sha256":digest(v)} for k, v in members.items()]}}}
    yield manifest, raw, members
    server.shutdown(); server.server_close(); t.join()

def test_download_verifies_and_installs(served, tmp_path):
    manifest, raw, members = served
    root = tmp_path / "assets"
    out = assets.download_pack("singlecell", Transport(2, True), manifest=manifest, dest_root=root)
    assert out == root
    for k, v in members.items():
        assert (root / k).read_bytes() == v
    assert not [p for p in root.iterdir() if p.name.startswith(".pack-")]
    # second call with identical files is a no-op success
    assets.download_pack("singlecell", Transport(2, True), manifest=manifest, dest_root=root)

def test_wrong_zip_digest_installs_nothing(served, tmp_path):
    manifest, raw, members = served
    bad = json.loads(json.dumps(manifest)); bad["packs"]["singlecell"]["sha256"] = "0"*64
    root = tmp_path / "assets"
    with pytest.raises(ValueError, match="SHA-256"):
        assets.download_pack("singlecell", Transport(2, True), manifest=bad, dest_root=root)
    assert not root.exists() or not any(root.rglob("*.npz"))

def test_wrong_member_digest_installs_nothing(served, tmp_path):
    manifest, raw, members = served
    bad = json.loads(json.dumps(manifest)); bad["packs"]["singlecell"]["members"][0]["sha256"] = "0"*64
    root = tmp_path / "assets"
    with pytest.raises(ValueError, match="Member"):
        assets.download_pack("singlecell", Transport(2, True), manifest=bad, dest_root=root)
    assert not any(root.rglob("*.npz"))

def test_unpinned_manifest_refuses(served, tmp_path):
    manifest, raw, members = served
    bad = json.loads(json.dumps(manifest)); bad["packs"]["singlecell"]["sha256"] = None
    with pytest.raises(ValueError, match="pinned"):
        assets.download_pack("singlecell", Transport(2, True), manifest=bad, dest_root=tmp_path)

def test_shipped_manifest_is_pinned_and_uses_original_sources():
    m = assets.optional_manifest()
    assert set(m["packs"]) == {"singlecell","literature","language"}
    assert all(assets.pack_status(p, m)["pinned"] for p in m["packs"])
    assert len(m["packs"]["language"]["members"]) == 12
    assert m['packs']['language']['model'] == 'Qwen/Qwen3.5-4B'
    assert all(assets.pack_status(p, m)['release_url'].startswith('https://') for p in m['packs'])
    assert all(mem["path"].startswith(("data/prepared/", "models/")) for p in m["packs"].values() for mem in p["members"])
    # publish step: release_url is filled only after the ZIPs are uploaded; until then no download is attempted
    assert m["release_url"] is None or m["release_url"].startswith("https://")

@pytest.mark.parametrize("path", ["../evil", "data/prepared/../x", "/abs/path", "models/a/../../b", "other/x"])
def test_unsafe_member_paths_rejected(tmp_path, path):
    from agrel_public.common import save_json
    m = {"schema":"aritheia.optional-assets.v1","release_url":None,"packs":{"singlecell":{"file":"s.zip","bytes":None,"sha256":None,
         "approx_bytes":1,"license":"x","members":[{"path":path,"bytes":None,"sha256":None}]}}}
    p = tmp_path / "m.json"; save_json(p, m)
    with pytest.raises(ValueError):
        assets.optional_manifest(p)

def test_existing_different_file_never_overwritten(served, tmp_path):
    manifest, raw, members = served
    root = tmp_path / "assets"
    target = root / "data/prepared/pbmc3k/data.npz"
    target.parent.mkdir(parents=True); target.write_bytes(b"different")
    with pytest.raises(FileExistsError):
        assets.download_pack("singlecell", Transport(2, True), manifest=manifest, dest_root=root)
    assert target.read_bytes() == b"different"

def test_import_tree_verifies_chain(tmp_path):
    import shutil
    src = tmp_path / "src"
    for rel in assets.required_files("cytology"):
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(assets.PACKAGE_ASSETS / rel, src / rel)
    dest = tmp_path / "dest"
    copied = assets.import_tree(src, ["cytology"], dest)
    assert set(copied) == set(assets.required_files("cytology"))
    with pytest.raises(FileNotFoundError):
        assets.import_tree(src, ["genomics_splice"], dest)


def test_destination_conflict_does_not_partially_install(served,tmp_path):
    manifest,raw,members=served
    root=tmp_path/'assets'
    # A conflict in the last member must be detected before the first is installed.
    paths=list(members)
    conflict=root/paths[-1];conflict.parent.mkdir(parents=True);conflict.write_bytes(b'different')
    with pytest.raises(FileExistsError):assets.download_pack('singlecell',Transport(2,True),manifest=manifest,dest_root=root)
    assert not (root/paths[0]).exists() and conflict.read_bytes()==b'different'


def test_offline_installer_uses_same_pins(served,tmp_path,monkeypatch):
    manifest,raw,members=served
    archive=tmp_path/'offline.zip';archive.write_bytes(raw)
    monkeypatch.setattr(assets,'optional_manifest',lambda:manifest)
    assets.install_pack('singlecell',archive,dest_root=tmp_path/'installed')
    assert (tmp_path/'installed'/next(iter(members))).exists()
    archive.write_bytes(raw+b'changed')
    with pytest.raises(ValueError):assets.install_pack('singlecell',archive,dest_root=tmp_path/'bad')
