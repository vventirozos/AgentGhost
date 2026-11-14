"""§4UN-dark: valid v3 onion labels for tests. The dark-web tools now check the v3 checksum, so a test
address like ``"a" * 56`` is rejected as garbled; ``v3_label("a")`` is a real-shaped, checksum-valid label
(deterministic per seed)."""
import base64
import hashlib


def v3_label(seed: str) -> str:
    pub = hashlib.sha256(("onion-test:" + str(seed)).encode()).digest()
    chk = hashlib.sha3_256(b".onion checksum" + pub + b"\x03").digest()[:2]
    return base64.b32encode(pub + chk + b"\x03").decode().lower()
