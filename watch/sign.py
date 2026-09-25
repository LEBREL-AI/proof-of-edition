"""The watch's own Ed25519 key: every board and every revealed prompt set is signed over its exact bytes.

The registry accepts a publication only with a valid signature from the pinned watch key and serves
the signature beside the bytes, so the router and any reader can check a board without trusting the
bucket or the page. The seed (64 hex characters) lives in WATCH_SIGNING_SEED on the watch machine.
"""
from __future__ import annotations

import base64
import hashlib

from nacl.exceptions import BadSignatureError
from nacl.signing import SigningKey, VerifyKey


def _key(seed_hex: str) -> SigningKey:
    seed = bytes.fromhex(seed_hex.strip())
    if len(seed) != 32:
        raise ValueError("the watch signing seed must be 32 bytes (64 hex characters)")
    return SigningKey(seed)


def public_key_hex(seed_hex: str) -> str:
    return bytes(_key(seed_hex).verify_key).hex()


def key_id(public_hex: str) -> str:
    return hashlib.sha256(bytes.fromhex(public_hex)).hexdigest()


def sign(data: bytes, seed_hex: str) -> tuple[str, str]:
    """(base64 signature over ``data``, key id)."""
    key = _key(seed_hex)
    return base64.b64encode(key.sign(data).signature).decode("ascii"), key_id(bytes(key.verify_key).hex())


def verify(data: bytes, signature_b64: str, public_hex: str) -> bool:
    try:
        VerifyKey(bytes.fromhex(public_hex)).verify(data, base64.b64decode(signature_b64, validate=True))
        return True
    except (BadSignatureError, ValueError):
        return False
