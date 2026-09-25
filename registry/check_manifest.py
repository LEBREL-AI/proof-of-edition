"""Does the runtime serve what the registry published?

Compares a runtime's signed serving manifest with a signed edition record from the
registry: same edition id, same revision, and every weight file the manifest lists
carries the digest the record published. Both signatures are checked against their
own pinned keys (runtime signing key for the manifest, publisher key for the record).

  poe-check-manifest --manifest manifest.json --runtime-key <hex> \
      --record record.json --publisher-key <hex> [--log log.jsonl --checkpoint cp.json]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from receipts.schema import Signed, check_manifest
from registry.log import check_checkpoint, inclusion, load_entries
from registry.record import check_record, record_sha256


def compare(manifest: dict[str, Any], record: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    edition = (manifest.get("edition") or {}).get("id")
    if edition != (record.get("edition") or {}).get("id"):
        problems.append(f"edition mismatch: manifest {edition!r}, record {(record.get('edition') or {}).get('id')!r}")
    weights = manifest.get("weights") or {}
    if weights.get("revision") != record.get("revision"):
        problems.append(f"revision mismatch: manifest {weights.get('revision')!r}, record {record.get('revision')!r}")
    files = weights.get("files") or {}
    published = record.get("files") or {}
    if not files:
        problems.append("manifest lists no weight files")
    for path, digest in files.items():
        entry = published.get(path)
        if entry is None:
            problems.append(f"{path}: served but not published")
        elif entry.get("sha256") != digest:
            problems.append(f"{path}: digest differs from the published file")
    missing = [path for path in published if path.endswith(".safetensors") and path not in files]
    if missing:
        problems.append(f"published shards not in the manifest: {missing[:5]}{'…' if len(missing) > 5 else ''}")
    manifest_fp = (manifest.get("edition") or {}).get("fingerprint_id")
    if record.get("fingerprint_id") is not None and manifest_fp != record.get("fingerprint_id"):
        problems.append("fingerprint_id differs between manifest and record")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--runtime-key", required=True, help="pinned runtime signing key, 64 hex")
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--publisher-key", required=True, help="pinned registry publisher key, 64 hex")
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--now", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        manifest = Signed.from_json(args.manifest.read_text(encoding="utf-8"))
        record = Signed.from_json(args.record.read_text(encoding="utf-8"))
        runtime_key = bytes.fromhex(args.runtime_key)
        publisher_key = bytes.fromhex(args.publisher_key)
    except (OSError, ValueError, KeyError) as error:
        print(f"cannot start: {error}", file=sys.stderr)
        return 2
    problems = [f"manifest: {p}" for p in check_manifest(manifest, runtime_key, now=args.now if args.now is not None else time.time())]
    problems += [f"record: {p}" for p in check_record(record, publisher_key)]
    problems += compare(manifest.payload, record.payload)
    if args.log is not None and args.checkpoint is not None:
        entries = load_entries(args.log)
        problems += [f"log: {p}" for p in check_checkpoint(Signed.from_json(args.checkpoint.read_text(encoding="utf-8")), entries, publisher_key)]
        if inclusion(entries, record_sha256(record)) is None:
            problems.append("log: record is not in the transparency log")
    for problem in problems:
        print("FAIL " + problem)
    if problems:
        return 1
    print(f"VERIFIED the runtime serves {record.payload['edition']['id']} revision {record.payload['revision']} exactly as published ({len(manifest.payload['weights']['files'])} weight files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
