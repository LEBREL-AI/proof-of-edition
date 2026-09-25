"""Serving manifest and per-response receipt: canonical encoding, signing and verification.

Both documents are JSON objects signed with Ed25519 over their canonical form
(sorted keys, no whitespace, UTF-8). The signature and signing key id travel next
to the payload; verifiers pin the publisher's public key.
"""
from __future__ import annotations

import base64
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

MANIFEST_VERSION = 1
RECEIPT_VERSION = 1

MANIFEST_REQUIRED = {
    "version", "edition", "weights", "quantization", "engine", "tokenizer_sha256",
    "chat_template_sha256", "runtime", "attestation", "issued_at", "expires_at", "signing_key_id",
}
RECEIPT_REQUIRED = {
    "version", "request_id", "manifest_sha256", "prompt_sha256", "response_sha256",
    "prompt_tokens", "completion_tokens", "issued_at", "instance_id", "signing_key_id",
}


def canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_revision(value: Any) -> bool:
    """A weights revision is a registry content revision (SHA-256, 64 hex) or a git commit sha (40 hex)."""
    return isinstance(value, str) and len(value) in (40, 64) and all(c in "0123456789abcdef" for c in value)


def manifest_identity(payload: dict[str, Any]) -> str:
    """Digest of the manifest content without its validity window.

    A runtime republishes the same content under a fresh issued_at/expires_at every
    window; receipts reference the content so a verifier can match a receipt against
    any current manifest that describes the same serving configuration.
    """
    identity = {k: v for k, v in payload.items() if k not in ("issued_at", "expires_at")}
    return sha256_hex(canonical(identity))


def key_id(public_key: bytes) -> str:
    return sha256_hex(public_key)


@dataclass(frozen=True)
class Signed:
    payload: dict[str, Any]
    signature_b64: str

    def to_json(self) -> str:
        return json.dumps({"payload": self.payload, "signature": self.signature_b64}, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> "Signed":
        raw = json.loads(text)
        return cls(payload=raw["payload"], signature_b64=raw["signature"])


def sign(payload: dict[str, Any], private_key_seed: bytes) -> Signed:
    from nacl.signing import SigningKey  # PyNaCl, only needed by publishers
    signer = SigningKey(private_key_seed)
    signature = signer.sign(canonical(payload)).signature
    return Signed(payload=payload, signature_b64=base64.b64encode(signature).decode("ascii"))


def verify_signature(signed: Signed, public_key: bytes) -> bool:
    from nacl.exceptions import BadSignatureError
    from nacl.signing import VerifyKey
    try:
        VerifyKey(public_key).verify(canonical(signed.payload), base64.b64decode(signed.signature_b64))
        return True
    except (BadSignatureError, ValueError):
        return False


def check_manifest(signed: Signed, public_key: bytes, now: float | None = None) -> list[str]:
    """Returns a list of problems; empty means the manifest is well-formed, signed and current."""
    problems: list[str] = []
    payload = signed.payload
    missing = MANIFEST_REQUIRED - set(payload)
    if missing:
        problems.append(f"manifest missing fields: {sorted(missing)}")
    if payload.get("version") != MANIFEST_VERSION:
        problems.append("unsupported manifest version")
    if payload.get("signing_key_id") != key_id(public_key):
        problems.append("manifest signed by an unpinned key")
    if not verify_signature(signed, public_key):
        problems.append("manifest signature invalid")
    now = time.time() if now is None else now
    if not (isinstance(payload.get("issued_at"), (int, float)) and isinstance(payload.get("expires_at"), (int, float))):
        problems.append("manifest validity window missing")
    elif not (payload["issued_at"] - 60 <= now <= payload["expires_at"]):
        problems.append("manifest expired or not yet valid")
    weights = payload.get("weights") or {}
    files = weights.get("files") or {}
    if not isinstance(files, dict) or not files or any(not isinstance(v, str) or len(v) != 64 for v in files.values()):
        problems.append("manifest weights.files must map file names to sha256 digests")
    if not is_revision(weights.get("revision")):
        problems.append("manifest weights.revision must be a registry content revision (64 hex) or a git commit sha (40 hex)")
    return problems


def check_receipt(signed: Signed, public_key: bytes, manifest: Signed, prompt: bytes | None = None, response: bytes | None = None) -> list[str]:
    problems: list[str] = []
    payload = signed.payload
    missing = RECEIPT_REQUIRED - set(payload)
    if missing:
        problems.append(f"receipt missing fields: {sorted(missing)}")
    if payload.get("version") != RECEIPT_VERSION:
        problems.append("unsupported receipt version")
    if payload.get("signing_key_id") != key_id(public_key):
        problems.append("receipt signed by an unpinned key")
    if not verify_signature(signed, public_key):
        problems.append("receipt signature invalid")
    if payload.get("manifest_sha256") != manifest_identity(manifest.payload):
        problems.append("receipt does not reference the presented manifest")
    if prompt is not None and payload.get("prompt_sha256") != sha256_hex(prompt):
        problems.append("receipt prompt digest does not match the prompt sent")
    if response is not None and payload.get("response_sha256") != sha256_hex(response):
        problems.append("receipt response digest does not match the response received")
    return problems


# ---- route documents (Proof of Edition kind "route", version 2) ----
#
# A router that forwards to third-party APIs signs a route manifest per model and a
# route receipt per response. They commit to where a request went, what was sent and
# received, what was paid and what the last verification probe of that upstream said.
# They never claim which weights answered; a route receipt is not a serving receipt.

ROUTE_DOCUMENT_VERSION = 2
DOCUMENT_KIND_ROUTE = "route"

ROUTE_MANIFEST_REQUIRED = {
    "version", "kind", "model_id", "level", "family", "upstreams", "pricing", "billing_contract_hash",
    "issued_at", "expires_at", "signing_key_id",
}
ROUTE_RECEIPT_REQUIRED = {
    "version", "kind", "request_id", "model_id", "manifest_sha256", "upstream", "upstream_model", "verification",
    "prompt_sha256", "response_sha256", "prompt_tokens", "completion_tokens", "usage_exact", "price_microusd",
    "issued_at", "instance_id", "signing_key_id",
}
ROUTE_VERIFICATION_STATUSES = {"consistent", "suspect", "drift", "divergent", "disputed", "unverified", "insufficient", "error", "unknown"}


def is_route_document(payload: dict[str, Any]) -> bool:
    return payload.get("kind") == DOCUMENT_KIND_ROUTE


def check_route_manifest(signed: Signed, public_key: bytes, now: float | None = None) -> list[str]:
    problems: list[str] = []
    payload = signed.payload
    missing = ROUTE_MANIFEST_REQUIRED - set(payload)
    if missing:
        problems.append(f"route manifest missing fields: {sorted(missing)}")
    if payload.get("version") != ROUTE_DOCUMENT_VERSION or not is_route_document(payload):
        problems.append("unsupported route manifest version or kind")
    if payload.get("signing_key_id") != key_id(public_key):
        problems.append("route manifest signed by an unpinned key")
    if not verify_signature(signed, public_key):
        problems.append("route manifest signature invalid")
    now = time.time() if now is None else now
    if not (isinstance(payload.get("issued_at"), (int, float)) and isinstance(payload.get("expires_at"), (int, float))):
        problems.append("route manifest validity window missing")
    elif not (payload["issued_at"] - 60 <= now <= payload["expires_at"]):
        problems.append("route manifest expired or not yet valid")
    upstreams = payload.get("upstreams")
    if not isinstance(upstreams, list) or not upstreams or any(not isinstance(u, dict) or not u.get("name") or not u.get("base_url") for u in upstreams):
        problems.append("route manifest upstreams must list name and base_url")
    if payload.get("level") not in (1, 2, 3):
        problems.append("route manifest level must be 1, 2 or 3")
    return problems


def check_route_receipt(signed: Signed, public_key: bytes, manifest: Signed, prompt: bytes | None = None, response: bytes | None = None) -> list[str]:
    problems: list[str] = []
    payload = signed.payload
    missing = ROUTE_RECEIPT_REQUIRED - set(payload)
    if missing:
        problems.append(f"route receipt missing fields: {sorted(missing)}")
    if payload.get("version") != ROUTE_DOCUMENT_VERSION or not is_route_document(payload):
        problems.append("unsupported route receipt version or kind")
    if payload.get("signing_key_id") != key_id(public_key):
        problems.append("route receipt signed by an unpinned key")
    if not verify_signature(signed, public_key):
        problems.append("route receipt signature invalid")
    if payload.get("manifest_sha256") != manifest_identity(manifest.payload):
        problems.append("route receipt does not reference the presented route manifest")
    if payload.get("model_id") != manifest.payload.get("model_id"):
        problems.append("route receipt model does not match the manifest")
    names = {u.get("name") for u in manifest.payload.get("upstreams", []) if isinstance(u, dict)}
    if payload.get("upstream") not in names:
        problems.append("route receipt names an upstream the manifest does not list")
    verification = payload.get("verification")
    if not isinstance(verification, dict) or verification.get("status") not in ROUTE_VERIFICATION_STATUSES:
        problems.append("route receipt verification status missing or unknown")
    if prompt is not None and payload.get("prompt_sha256") != sha256_hex(prompt):
        problems.append("route receipt prompt digest does not match the prompt sent")
    if response is not None and payload.get("response_sha256") != sha256_hex(response):
        problems.append("route receipt response digest does not match the response received")
    return problems
