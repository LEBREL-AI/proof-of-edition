"""Edition records: what a Lebrel registry publishes for every edition revision.

A record is the registry's signed statement of an edition's bytes: every file with its
SHA-256 and size, a content revision derived from those digests, the edition identity,
license and fingerprint reference. Records are content-addressed: the revision is the
SHA-256 of the canonical files map, so the same bytes always get the same revision on
any mirror, and a mirror cannot substitute files without changing the revision.

The publisher key is the registry's root of trust and is distinct from the runtime's
signing key: the runtime's serving manifest must match a published record
(``registry/check_manifest.py``), and the record must appear in the transparency log
(``registry/log.py``). Canonical encoding and signatures are shared with receipts.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from receipts.schema import Signed, canonical, key_id, sha256_hex, sign, verify_signature

RECORD_VERSION = 1
RECORD_REQUIRED = {"version", "edition", "revision", "files", "total_bytes", "license", "fingerprint_id", "published_at", "publisher_key_id"}
WEIGHT_SUFFIXES = (".safetensors",)


def content_revision(files: dict[str, dict[str, Any]]) -> str:
    """SHA-256 over the canonical {path: sha256} map: identical bytes, identical revision."""
    digests = {path: entry["sha256"] for path, entry in sorted(files.items())}
    return sha256_hex(canonical(digests))


def validate_files(files: Any) -> list[str]:
    problems: list[str] = []
    if not isinstance(files, dict) or not files:
        return ["files must be a non-empty object"]
    for path, entry in files.items():
        if not isinstance(path, str) or not path or path.startswith("/") or ".." in path.split("/"):
            problems.append(f"invalid file path {path!r}")
            continue
        digest = entry.get("sha256") if isinstance(entry, dict) else None
        size = entry.get("size") if isinstance(entry, dict) else None
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            problems.append(f"{path}: sha256 must be 64 lowercase hex characters")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            problems.append(f"{path}: size must be a non-negative integer")
    if not any(path.endswith(WEIGHT_SUFFIXES) for path in files):
        problems.append("files contain no safetensors shards")
    return problems


def build_record(edition: dict[str, Any], weights_manifest: dict[str, Any], *, license_id: str, publisher_public_key: bytes,
                 fingerprint_id: str | None = None, source: dict[str, Any] | None = None, published_at: int | None = None) -> dict[str, Any]:
    """Assemble an unsigned record from an upload-time weights manifest ({"files": {path: {sha256, size}}})."""
    files = {path: {"sha256": entry["sha256"], "size": int(entry["size"])} for path, entry in sorted(weights_manifest["files"].items())}
    problems = validate_files(files)
    for key in ("id", "name", "base_model"):
        if not isinstance(edition.get(key), str) or not edition[key]:
            problems.append(f"edition.{key} is required")
    if problems:
        raise ValueError("; ".join(problems))
    record = {
        "version": RECORD_VERSION,
        "edition": {"id": edition["id"], "name": edition["name"], "base_model": edition["base_model"]},
        "revision": content_revision(files),
        "files": files,
        "total_bytes": sum(entry["size"] for entry in files.values()),
        "license": license_id,
        "fingerprint_id": fingerprint_id,
        "published_at": int(time.time()) if published_at is None else int(published_at),
        "publisher_key_id": key_id(publisher_public_key),
    }
    if source:
        record["source"] = dict(source)
    return record


def check_record(signed: Signed, publisher_public_key: bytes) -> list[str]:
    """Problems with a signed record: structure, content revision, publisher key and signature."""
    problems: list[str] = []
    payload = signed.payload
    missing = RECORD_REQUIRED - set(payload)
    if missing:
        problems.append(f"record missing fields: {sorted(missing)}")
    if payload.get("version") != RECORD_VERSION:
        problems.append("unsupported record version")
    if payload.get("publisher_key_id") != key_id(publisher_public_key):
        problems.append("record signed by an unpinned publisher key")
    if not verify_signature(signed, publisher_public_key):
        problems.append("record signature invalid")
    files = payload.get("files")
    file_problems = validate_files(files)
    problems += [f"record {p}" for p in file_problems]
    if not file_problems:
        if payload.get("revision") != content_revision(files):
            problems.append("record revision does not match its files")
        if payload.get("total_bytes") != sum(entry["size"] for entry in files.values()):
            problems.append("record total_bytes does not match its files")
    return problems


def record_sha256(signed: Signed) -> str:
    """Identity of a record for the transparency log: digest of the canonical payload."""
    return sha256_hex(canonical(signed.payload))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    publish = sub.add_parser("publish", help="build and sign a record from a weights manifest")
    publish.add_argument("--edition", required=True, type=Path, help="edition spec JSON (id, name, base_model, optional license, fingerprint_id)")
    publish.add_argument("--weights-manifest", required=True, type=Path)
    publish.add_argument("--publisher-seed-file", required=True, type=Path, help="file holding the 32-byte Ed25519 seed (hex)")
    publish.add_argument("--license", default=None)
    publish.add_argument("--out", required=True, type=Path)
    check = sub.add_parser("check", help="verify a signed record")
    check.add_argument("--record", required=True, type=Path)
    check.add_argument("--publisher-key", required=True, help="pinned publisher public key, 64 hex characters")
    args = parser.parse_args(argv)

    if args.command == "publish":
        from nacl.signing import SigningKey
        seed = bytes.fromhex(args.publisher_seed_file.read_text().strip())
        public = bytes(SigningKey(seed).verify_key)
        edition = json.loads(args.edition.read_text(encoding="utf-8"))
        weights = json.loads(args.weights_manifest.read_text(encoding="utf-8"))
        record = build_record(edition, weights, license_id=args.license or edition.get("license") or "unspecified",
                              publisher_public_key=public, fingerprint_id=edition.get("fingerprint_id"),
                              source={k: weights[k] for k in ("repository", "revision") if k in weights})
        signed = sign(record, seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(signed.to_json() + "\n", encoding="utf-8")
        print(f"published {record['edition']['id']} revision {record['revision']} ({len(record['files'])} files, {record['total_bytes']} bytes) -> {args.out}")
        return 0
    signed = Signed.from_json(args.record.read_text(encoding="utf-8"))
    problems = check_record(signed, bytes.fromhex(args.publisher_key))
    for problem in problems:
        print("FAIL " + problem)
    if not problems:
        print(f"VERIFIED record {signed.payload['edition']['id']} revision {signed.payload['revision']} sha256 {record_sha256(signed)}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
