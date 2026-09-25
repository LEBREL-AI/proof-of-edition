"""Transparency log: an append-only hash chain of published records with signed checkpoints.

Every publication appends one entry: entry_hash = SHA-256(previous_entry_hash || record_sha256).
The publisher signs checkpoints {size, head, at} with the registry key. Mirrors and
auditors keep checkpoints; a registry that rewrites or removes a record can no longer
produce a chain whose head matches any checkpoint it signed before, so tampering is
evident even to the registry's own operator. The log is small (one line per record)
and is meant to be replicated everywhere the catalog is.

File format: JSON lines, each {"index": n, "record_sha256": ..., "entry_hash": ...}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from proof_of_edition.receipts.schema import Signed, key_id, sign, verify_signature

GENESIS = "0" * 64


def entry_hash(previous: str, record_sha256: str) -> str:
    return hashlib.sha256(bytes.fromhex(previous) + bytes.fromhex(record_sha256)).hexdigest()


def load_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def verify_chain(entries: list[dict[str, Any]]) -> list[str]:
    problems: list[str] = []
    previous = GENESIS
    for expected_index, entry in enumerate(entries):
        if entry.get("index") != expected_index:
            problems.append(f"entry {expected_index}: index mismatch")
        digest = entry.get("record_sha256", "")
        if not isinstance(digest, str) or len(digest) != 64:
            problems.append(f"entry {expected_index}: invalid record digest")
            break
        expected = entry_hash(previous, digest)
        if entry.get("entry_hash") != expected:
            problems.append(f"entry {expected_index}: hash chain broken")
            break
        previous = expected
    return problems


def head(entries: list[dict[str, Any]]) -> str:
    return entries[-1]["entry_hash"] if entries else GENESIS


def append(path: Path, record_sha256: str) -> dict[str, Any]:
    entries = load_entries(path)
    problems = verify_chain(entries)
    if problems:
        raise ValueError("refusing to append to a broken log: " + "; ".join(problems))
    entry = {"index": len(entries), "record_sha256": record_sha256, "entry_hash": entry_hash(head(entries), record_sha256)}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, separators=(",", ":")) + "\n")
    return entry


def checkpoint(entries: list[dict[str, Any]], seed: bytes, at: int | None = None) -> Signed:
    from nacl.signing import SigningKey
    public = bytes(SigningKey(seed).verify_key)
    return sign({"version": 1, "size": len(entries), "head": head(entries), "at": int(time.time()) if at is None else int(at), "publisher_key_id": key_id(public)}, seed)


def check_checkpoint(signed: Signed, entries: list[dict[str, Any]], publisher_public_key: bytes) -> list[str]:
    """A checkpoint is consistent with a log when the log's first `size` entries end at `head`."""
    problems: list[str] = []
    payload = signed.payload
    if payload.get("publisher_key_id") != key_id(publisher_public_key):
        problems.append("checkpoint signed by an unpinned publisher key")
    if not verify_signature(signed, publisher_public_key):
        problems.append("checkpoint signature invalid")
    size = payload.get("size")
    if not isinstance(size, int) or size < 0 or size > len(entries):
        problems.append("checkpoint size exceeds the log")
        return problems
    problems += verify_chain(entries[:size])
    if not problems and head(entries[:size]) != payload.get("head"):
        problems.append("log head does not match the checkpoint: the log was rewritten")
    return problems


def inclusion(entries: list[dict[str, Any]], record_sha256: str) -> int | None:
    for entry in entries:
        if entry.get("record_sha256") == record_sha256:
            return entry["index"]
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("append", help="append a record digest and sign a new checkpoint")
    add.add_argument("--log", required=True, type=Path)
    add.add_argument("--record", required=True, type=Path, help="signed record JSON")
    add.add_argument("--publisher-seed-file", required=True, type=Path)
    add.add_argument("--checkpoint-out", required=True, type=Path)
    verify = sub.add_parser("verify", help="verify the chain and a checkpoint, optionally the inclusion of a record")
    verify.add_argument("--log", required=True, type=Path)
    verify.add_argument("--checkpoint", required=True, type=Path)
    verify.add_argument("--publisher-key", required=True)
    verify.add_argument("--record", type=Path, default=None)
    args = parser.parse_args(argv)

    from proof_of_edition.registry.record import record_sha256
    if args.command == "append":
        seed = bytes.fromhex(args.publisher_seed_file.read_text().strip())
        digest = record_sha256(Signed.from_json(args.record.read_text(encoding="utf-8")))
        entry = append(args.log, digest)
        signed = checkpoint(load_entries(args.log), seed)
        args.checkpoint_out.write_text(signed.to_json() + "\n", encoding="utf-8")
        print(f"appended entry {entry['index']} head {entry['entry_hash']}; checkpoint -> {args.checkpoint_out}")
        return 0
    entries = load_entries(args.log)
    problems = check_checkpoint(Signed.from_json(args.checkpoint.read_text(encoding="utf-8")), entries, bytes.fromhex(args.publisher_key))
    if args.record is not None and not problems:
        index = inclusion(entries, record_sha256(Signed.from_json(args.record.read_text(encoding="utf-8"))))
        if index is None:
            problems.append("record is not in the log")
        else:
            print(f"record included at index {index}")
    for problem in problems:
        print("FAIL " + problem)
    if not problems:
        print(f"VERIFIED log of {len(entries)} entries, head {head(entries)}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
