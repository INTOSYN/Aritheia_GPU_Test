import json
import threading
from http.server import HTTPServer,BaseHTTPRequestHandler
import pytest
from agrel_public.common import digest
from agrel_public.network import Transport,download_pack,PackJob
from agrel_public.config import Config

@pytest.fixture
def http_pack(signed_pack):
    keys,payload,sign=signed_pack;raw=sign();seen=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            seen.append(dict(self.headers));start=0
            r=self.headers.get("Range")
            if r:start=int(r.split("=")[1].split("-")[0])
            self.send_response(206 if start else 200)
            if start:self.send_header("Content-Range",f"bytes {start}-{len(raw)-1}/{len(raw)}")
            self.send_header("Content-Length",str(len(raw)-start));self.end_headers();self.wfile.write(raw[start:])
    server=HTTPServer(("127.0.0.1",0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url=f"http://127.0.0.1:{server.server_port}/pack"
    yield keys,raw,seen,{"sha256":digest(raw),"bytes":len(raw),"url":url}
    server.shutdown();server.server_close();thread.join()

@pytest.mark.parametrize("resume",[False,True])
def test_download_resume_and_cache(http_pack,tmp_path,resume):
    keys,raw,seen,desc=http_pack
    if resume:(tmp_path/(desc["sha256"]+".part")).write_bytes(raw[:50])
    out=download_pack(Transport(2,True),desc,tmp_path,keys,99999,threading.Event())
    assert out.read_bytes()==raw
    if resume:assert seen[0]["Range"]=="bytes=50-"
    n=len(seen)
    assert download_pack(Transport(2,True),desc,tmp_path,keys,99999,threading.Event())==out
    assert len(seen)==n
    assert "Authorization" not in seen[0]

def test_cancellation_retains_no_fake_pass(http_pack,tmp_path):
    keys,raw,seen,desc=http_pack
    cancel=threading.Event();cancel.set()
    with pytest.raises(TimeoutError):download_pack(Transport(2,True),desc,tmp_path,keys,99999,cancel)
    assert not list(tmp_path.glob("*.agpack"))

def test_wrong_digest(http_pack,tmp_path):
    keys,raw,seen,desc=http_pack;desc=dict(desc,sha256="0"*64)
    with pytest.raises(ValueError,match="hash"):
        download_pack(Transport(2,True),desc,tmp_path,keys,99999,threading.Event())

def test_bad_signature(http_pack,tmp_path):
    _,raw,seen,desc=http_pack
    with pytest.raises(ValueError):download_pack(Transport(2,True),desc,tmp_path,{},99999,threading.Event())
    assert not list(tmp_path.glob("*.agpack"))

def test_background_failure_does_not_raise():
    job=PackJob(Config(api_url="http://not-allowed.example"),{"cards":[]})
    job.start();job.thread.join(2)
    assert job.state.status=="UNAVAILABLE"
    assert job.state.path is None
