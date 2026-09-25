"""Verify a Proof of Edition receipt against a runtime's signed serving manifest.

The verifier pins the runtime's Ed25519 public key (hex), fetches the current
manifest from ``<base-url>/.well-known/proof-of-edition`` and the receipt from
``<base-url>/v1/receipts/<request-id>`` (or reads both from files), and checks:

  1. both documents are signed by the pinned key and the manifest is current;
  2. the receipt references the manifest identity (content without its window);
  3. when the caller supplies the plaintext it sent and the assistant text it
     received, their SHA-256 digests match the receipt.

Exit codes: 0 verified, 1 problems found, 2 could not fetch or parse.

Usage:
  python3 -m proof_of_edition.receipts.verify_receipt --base-url https://runtime.example \
      --public-key <64 hex> --request-id <id> [--prompt-file req.json] [--response-file answer.txt]
  python3 -m proof_of_edition.receipts.verify_receipt --manifest-file m.json --receipt-file r.json --public-key <hex>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from proof_of_edition.receipts.schema import (Signed, check_manifest, check_receipt, check_route_manifest, check_route_receipt, is_route_document,
                             manifest_identity)

MANIFEST_PATH = "/.well-known/proof-of-edition"
RECEIPTS_PATH = "/v1/receipts/"
FETCH_TIMEOUT_SECONDS = 20
MAX_DOCUMENT_BYTES = 256 * 1024


class FetchError(Exception):
    pass


def fetch(url: str, opener=None) -> str:
    opener = opener or urllib.request.build_opener()
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "proof-of-edition-verifier/0.1"})
    try:
        with opener.open(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
            body = response.read(MAX_DOCUMENT_BYTES + 1)
    except urllib.error.HTTPError as error:
        raise FetchError(f"{url}: HTTP {error.code}") from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise FetchError(f"{url}: {error}") from error
    if len(body) > MAX_DOCUMENT_BYTES:
        raise FetchError(f"{url}: document exceeds {MAX_DOCUMENT_BYTES} bytes")
    return body.decode("utf-8")


def load_signed(text: str, what: str) -> Signed:
    try:
        signed = Signed.from_json(text)
    except (ValueError, KeyError, TypeError) as error:
        raise FetchError(f"{what} is not a signed document: {error}") from error
    if not isinstance(signed.payload, dict) or not isinstance(signed.signature_b64, str):
        raise FetchError(f"{what} is not a signed document")
    return signed


def verify(manifest: Signed, receipt: Signed, public_key: bytes, prompt: bytes | None, response: bytes | None, now: float | None = None) -> list[str]:
    if is_route_document(manifest.payload) or is_route_document(receipt.payload):
        problems = [f"route manifest: {item}" for item in check_route_manifest(manifest, public_key, now=now)]
        problems += [f"route receipt: {item}" for item in check_route_receipt(receipt, public_key, manifest, prompt=prompt, response=response)]
        return problems
    problems = [f"manifest: {item}" for item in check_manifest(manifest, public_key, now=now)]
    problems += [f"receipt: {item}" for item in check_receipt(receipt, public_key, manifest, prompt=prompt, response=response)]
    return problems


def describe_route(manifest: Signed, receipt: Signed) -> str:
    """A route receipt proves the route, not the weights; say so in the first line."""
    verification = receipt.payload.get("verification") or {}
    upstreams = ", ".join(f"{u.get('name')} ({u.get('model')})" for u in manifest.payload.get("upstreams", []) if isinstance(u, dict))
    lines = [
        "ROUTE RECEIPT   proves where the request went and what the last probe said; it does not prove which weights answered",
        f"model          {manifest.payload.get('model_id')} (level {manifest.payload.get('level')}, family {manifest.payload.get('family')})",
        f"upstreams      {upstreams}",
        f"served by      {receipt.payload.get('upstream')} as {receipt.payload.get('upstream_model')} (reported model {receipt.payload.get('served_model') or '?'})",
        f"verification   {verification.get('status')} at run {verification.get('run_id') or '?'} ({verification.get('checked_at')})",
        f"manifest id    {manifest_identity(manifest.payload)}",
        f"receipt        request {receipt.payload.get('request_id')} at {receipt.payload.get('issued_at')} on instance {receipt.payload.get('instance_id')}",
        f"tokens         prompt {receipt.payload.get('prompt_tokens')} completion {receipt.payload.get('completion_tokens')} cached {receipt.payload.get('cached_tokens')} exact {receipt.payload.get('usage_exact')}",
        f"price          {receipt.payload.get('price_microusd')} microUSD (fee {receipt.payload.get('fee_bps')} bps)",
    ]
    return "\n".join(lines)


def describe(manifest: Signed, receipt: Signed) -> str:
    edition = manifest.payload.get("edition") or {}
    weights = manifest.payload.get("weights") or {}
    engine = manifest.payload.get("engine") or {}
    quantization = manifest.payload.get("quantization") or {}
    lines = [
        f"edition        {edition.get('name', '?')} ({edition.get('id', '?')})",
        f"weights        {weights.get('repository', weights.get('repo', '?'))} @ {weights.get('revision', '?')} ({len(weights.get('files') or {})} files)",
        f"quantization   {quantization.get('method', '?')} / kv {quantization.get('kv_cache_dtype', quantization.get('kv_cache', '?'))}",
        f"engine         {engine.get('name', '?')} {engine.get('version', '?')} {engine.get('image_digest', '')}",
        f"manifest id    {manifest_identity(manifest.payload)}",
        f"receipt        request {receipt.payload.get('request_id')} at {receipt.payload.get('issued_at')} on instance {receipt.payload.get('instance_id')}",
        f"tokens         prompt {receipt.payload.get('prompt_tokens')} completion {receipt.payload.get('completion_tokens')}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--public-key", required=True, help="pinned Ed25519 public key, 64 hex characters")
    parser.add_argument("--base-url", help="runtime base URL to fetch the manifest and receipt from")
    parser.add_argument("--request-id", help="request id whose receipt to fetch")
    parser.add_argument("--manifest-file", type=Path, help="signed manifest JSON (instead of fetching)")
    parser.add_argument("--receipt-file", type=Path, help="signed receipt JSON (instead of fetching)")
    parser.add_argument("--prompt-file", type=Path, help="exact plaintext request body that was sent")
    parser.add_argument("--response-file", type=Path, help="exact assistant text that was received (UTF-8)")
    parser.add_argument("--now", type=int, default=None, help="override the clock (unix seconds) for manifest validity")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    try:
        public_key = bytes.fromhex(args.public_key)
        if len(public_key) != 32:
            raise ValueError("expected 32 bytes")
    except ValueError as error:
        print(f"invalid public key: {error}", file=sys.stderr)
        return 2

    try:
        if args.manifest_file:
            manifest_text = args.manifest_file.read_text(encoding="utf-8")
        elif args.base_url:
            manifest_text = fetch(args.base_url.rstrip("/") + MANIFEST_PATH)
        else:
            print("need --manifest-file or --base-url", file=sys.stderr)
            return 2
        if args.receipt_file:
            receipt_text = args.receipt_file.read_text(encoding="utf-8")
        elif args.base_url and args.request_id:
            receipt_text = fetch(args.base_url.rstrip("/") + RECEIPTS_PATH + args.request_id)
        else:
            print("need --receipt-file or --base-url with --request-id", file=sys.stderr)
            return 2
        manifest = load_signed(manifest_text, "manifest")
        receipt = load_signed(receipt_text, "receipt")
    except FetchError as error:
        print(str(error), file=sys.stderr)
        return 2

    prompt = args.prompt_file.read_bytes() if args.prompt_file else None
    response = args.response_file.read_bytes() if args.response_file else None
    problems = verify(manifest, receipt, public_key, prompt, response, now=args.now if args.now is not None else time.time())
    if not args.quiet:
        print(describe_route(manifest, receipt) if is_route_document(manifest.payload) else describe(manifest, receipt))
    if problems:
        for problem in problems:
            print(f"FAIL {problem}")
        return 1
    checked = ["signatures", "manifest window", "manifest identity"]
    if prompt is not None:
        checked.append("prompt digest")
    if response is not None:
        checked.append("response digest")
    print("VERIFIED " + ", ".join(checked))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
