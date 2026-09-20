import base64
import json
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
from agrel_public.common import canonical

@pytest.fixture
def signed_pack():
    k=Ed25519PrivateKey.generate()
    pub=base64.b64encode(k.public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw)).decode()
    payload={"schema":"aritheia.pack.v1","pack_id":"test-fixture","backend":"fixed_mm_v1",
             "provenance":"software_fixture","data":{"probes":[]}}
    def sign(p=None):
        data=p or payload
        return canonical({"key_id":"test","payload":data,"signature":base64.b64encode(k.sign(canonical(data))).decode()})
    return {"test":pub},payload,sign

@pytest.fixture(autouse=True)
def no_production_telemetry(monkeypatch):
    from agrel_public.network import Transport
    original=Transport.json
    def guarded(self,url,*args,**kwargs):
        if url.startswith('https://data.intc.ca:8443/'):
            raise AssertionError('Unit tests must never call the production collector')
        return original(self,url,*args,**kwargs)
    monkeypatch.setattr(Transport,'json',guarded)
