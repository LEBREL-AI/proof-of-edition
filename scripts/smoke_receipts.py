"""End-to-end smoke test of Proof of Edition against a live Lebrel deployment.

Uses the official encrypted Python client's building blocks so the requests are the
real thing (EHBP-encrypted, edge-authorized) and then checks the standard's documents:

  1. GET /.well-known/lebrel-encryption   -> verified encryption config (pinned signing key)
  2. GET /.well-known/proof-of-edition    -> signed serving manifest, checked with the same key
  3. one non-streaming completion         -> Proof-Of-Edition-Receipt header, X-Request-ID
  4. GET /v1/receipts/{id}                -> same receipt as the header, verified against the
                                             manifest, the plaintext request and the answer
  5. one streaming completion             -> receipt fetched by request id after the stream

Usage:
  PYTHONPATH=<sdk>/src:<proof-of-edition> python3 scripts/smoke_receipts.py --env-file .env.production-canary

Exit 0 when every check passes. Spends two tiny completions on the canary account.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import uuid
from pathlib import Path

import httpx
from lebrel_encrypted.client import MODEL_ID, PRODUCTION_SIGNING_KEY, Lebrel, _plaintext, _response_object

from proof_of_edition.receipts.schema import Signed, check_manifest, check_receipt, manifest_identity

MANIFEST_PATH = "/.well-known/proof-of-edition"
RECEIPTS_PATH = "/v1/receipts/"


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"')
    return values


def signed_get(http: httpx.Client, url: str) -> Signed:
    response = http.get(url, headers={"Accept": "application/json"})
    if response.status_code != 200:
        raise SystemExit(f"{url}: HTTP {response.status_code} {response.text[:200]}")
    return Signed.from_json(response.text)


def encrypted_completion(client: Lebrel, body: dict, request_id: str, stream: bool) -> tuple[bytes, str, httpx.Headers]:
    """Returns (plaintext request bytes, assistant text, response headers)."""
    plaintext = json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    verified = client._metadata()
    encrypted = verified.identity.encrypt_request_body(plaintext)
    request = client._http.build_request("POST", client._origin + "/v1/chat/completions", headers={
        "Authorization": "Bearer " + client._key,
        "Content-Type": "application/octet-stream",
        "Accept": "text/event-stream" if stream else "application/json",
        "Accept-Encoding": "identity",
        "Ehbp-Encapsulated-Key": encrypted.encapsulated_key.hex(),
        "X-Lebrel-Encryption-Key-Id": verified.key_id,
        "X-Lebrel-Request-Id": request_id,
    }, content=encrypted.body)
    response = client._http.send(request, stream=True)
    try:
        if response.status_code != 200 or not response.headers.get("Ehbp-Response-Nonce"):
            raise SystemExit(f"completion failed: HTTP {response.status_code} (nonce present: {bool(response.headers.get('Ehbp-Response-Nonce'))})")
        decrypted = b"".join(_plaintext(response, encrypted, verified.key_id))
    finally:
        response.close()
    if stream:
        text = ""
        for block in decrypted.decode("utf-8").split("\n\n"):
            for line in block.split("\n"):
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data and data != "[DONE]":
                        event = json.loads(data)
                        for choice in event.get("choices", []):
                            text += choice.get("delta", {}).get("content") or ""
        return plaintext, text, response.headers
    result = _response_object(decrypted)
    return plaintext, result["choices"][0]["message"]["content"], response.headers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-file", type=Path, required=True, help="file with LEBREL_PROD_CANARY_API_KEY")
    parser.add_argument("--base-url", default="https://api.lebrel.ai")
    parser.add_argument("--signing-key", default=PRODUCTION_SIGNING_KEY, help="base64 Ed25519 public key pinned in the SDK")
    parser.add_argument("--out", type=Path, default=None, help="directory to save manifest and receipts")
    args = parser.parse_args(argv)

    api_key = read_env(args.env_file).get("LEBREL_PROD_CANARY_API_KEY")
    if not api_key:
        print("LEBREL_PROD_CANARY_API_KEY missing", file=sys.stderr)
        return 2
    public_key = base64.b64decode(args.signing_key)
    client = Lebrel(api_key, base_url=args.base_url, signing_public_key=args.signing_key)
    http = httpx.Client(timeout=30, follow_redirects=False, trust_env=False)
    failures: list[str] = []

    verified = client._metadata()
    print(f"1. encryption config verified: key id {verified.key_id[:16]}…")

    manifest = signed_get(http, args.base_url + MANIFEST_PATH)
    problems = check_manifest(manifest, public_key)
    print(f"2. serving manifest: edition {manifest.payload.get('edition', {}).get('id')} identity {manifest_identity(manifest.payload)[:16]}… problems {problems}")
    failures += [f"manifest: {p}" for p in problems]

    request_id = str(uuid.uuid4())
    body = {"model": MODEL_ID, "messages": [{"role": "user", "content": "Reply with exactly: RECEIPT_OK"}], "max_tokens": 16, "temperature": 0, "stream": False}
    plaintext, answer, headers = encrypted_completion(client, body, request_id, stream=False)
    header_receipt = headers.get("Proof-Of-Edition-Receipt")
    served_id = headers.get("X-Request-ID") or headers.get("X-Request-Id")
    print(f"3. completion answered {answer!r}; X-Request-ID {served_id}; receipt header present: {bool(header_receipt)}")
    if not header_receipt:
        failures.append("no Proof-Of-Edition-Receipt header on the non-streaming completion")
    if served_id != request_id:
        failures.append(f"X-Request-ID {served_id} differs from the client request id {request_id}")
    receipt_from_header = Signed.from_json(base64.b64decode(header_receipt).decode("utf-8")) if header_receipt else None
    fetched = signed_get(http, args.base_url + RECEIPTS_PATH + request_id)
    if receipt_from_header and (receipt_from_header.payload != fetched.payload or receipt_from_header.signature_b64 != fetched.signature_b64):
        failures.append("receipt from header differs from the fetched receipt")
    problems = check_receipt(fetched, public_key, manifest, prompt=plaintext, response=answer.encode("utf-8"))
    print(f"4. receipt {fetched.payload.get('request_id')}: tokens {fetched.payload.get('prompt_tokens')}/{fetched.payload.get('completion_tokens')} problems {problems}")
    failures += [f"receipt: {p}" for p in problems]

    stream_id = str(uuid.uuid4())
    stream_body = dict(body, stream=True, messages=[{"role": "user", "content": "Reply with exactly: STREAM_OK"}])
    stream_plaintext, stream_answer, stream_headers = encrypted_completion(client, stream_body, stream_id, stream=True)
    stream_receipt = signed_get(http, args.base_url + RECEIPTS_PATH + stream_id)
    problems = check_receipt(stream_receipt, public_key, manifest, prompt=stream_plaintext, response=stream_answer.encode("utf-8"))
    print(f"5. streaming answered {stream_answer!r}; receipt problems {problems}")
    failures += [f"stream receipt: {p}" for p in problems]

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "manifest.json").write_text(manifest.to_json(), encoding="utf-8")
        (args.out / f"receipt-{request_id}.json").write_text(fetched.to_json(), encoding="utf-8")
        (args.out / f"receipt-{stream_id}.json").write_text(stream_receipt.to_json(), encoding="utf-8")
    if failures:
        for failure in failures:
            print("FAIL " + failure)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
