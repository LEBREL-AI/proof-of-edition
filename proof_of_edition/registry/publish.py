"""Publish an edition to the Lebrel registry bucket (Cloudflare R2 or any S3-compatible store).

The publisher key never leaves the machine running this command. Steps:
  1. verify the signed record against the publisher key;
  2. for every file in the record, check size and SHA-256 of the local copy, then upload it
     to blobs/sha256/<digest> unless the bucket already has it (content-addressed, so a
     shard shared by two revisions is stored once);
  3. write the record and point the tag (default "main") at its revision;
  4. append the record to the transparency log and sign a new checkpoint;
  5. upsert the edition in catalog.json.

Credentials come from the environment (R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY) and the
endpoint from --endpoint (https://<account id>.r2.cloudflarestorage.com). boto3 is only
needed for real uploads; the module is otherwise dependency-free so it can be tested.

  poe-registry-publish --record record.json --files-dir /path/to/edition \
      --publisher-seed-file ~/.lebrel/registry-publisher.seed \
      --bucket lebrel-registry --endpoint https://<account>.r2.cloudflarestorage.com
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Protocol

from proof_of_edition.receipts.schema import Signed
from proof_of_edition.registry.log import append, checkpoint, load_entries, verify_chain
from proof_of_edition.registry.record import check_record, record_sha256

CHUNK = 8 * 1024 * 1024


class ObjectStore(Protocol):
    def exists(self, key: str) -> bool: ...
    def get_text(self, key: str) -> str | None: ...
    def put_text(self, key: str, text: str, content_type: str) -> None: ...
    def upload_file(self, key: str, path: Path, content_type: str) -> None: ...


class S3Store:
    """Thin wrapper over a boto3 client for R2 / S3."""

    def __init__(self, client: Any, bucket: str) -> None:
        self.client, self.bucket = client, bucket

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception as error:  # boto3 raises ClientError with a 404 code
            code = getattr(error, "response", {}).get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def get_text(self, key: str) -> str | None:
        if not self.exists(key):
            return None
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read().decode("utf-8")

    def put_text(self, key: str, text: str, content_type: str) -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=text.encode("utf-8"), ContentType=content_type)

    def upload_file(self, key: str, path: Path, content_type: str) -> None:
        self.client.upload_file(str(path), self.bucket, key, ExtraArgs={"ContentType": content_type})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify_local_files(record: dict[str, Any], files_dir: Path, *, progress=None) -> list[str]:
    problems: list[str] = []
    for path, entry in sorted(record["files"].items()):
        local = files_dir / path
        if not local.is_file():
            problems.append(f"{path}: missing locally")
            continue
        size = local.stat().st_size
        if size != entry["size"]:
            problems.append(f"{path}: size {size} differs from the record's {entry['size']}")
            continue
        digest = sha256_file(local)
        if digest != entry["sha256"]:
            problems.append(f"{path}: sha256 differs from the record")
        if progress:
            progress(path, size)
    return problems


def publish(store: ObjectStore, signed: Signed, files_dir: Path, publisher_seed: bytes, *, tag: str = "main", log=print, skip_verify: bool = False) -> dict[str, Any]:
    from nacl.signing import SigningKey
    public = bytes(SigningKey(publisher_seed).verify_key)
    problems = check_record(signed, public)
    if problems:
        raise ValueError("record rejected: " + "; ".join(problems))
    record = signed.payload
    edition_id, revision = record["edition"]["id"], record["revision"]
    if not skip_verify:
        problems = verify_local_files(record, files_dir, progress=lambda path, size: log(f"verified {path} ({size} bytes)"))
        if problems:
            raise ValueError("local files do not match the record: " + "; ".join(problems))

    uploaded = skipped = 0
    for path, entry in sorted(record["files"].items()):
        key = f"blobs/sha256/{entry['sha256']}"
        if store.exists(key):
            skipped += 1
            continue
        log(f"uploading {path} -> {key}")
        store.upload_file(key, files_dir / path, "application/octet-stream")
        uploaded += 1

    store.put_text(f"editions/{edition_id}/records/{revision}.json", signed.to_json() + "\n", "application/json")
    store.put_text(f"editions/{edition_id}/refs/{tag}", revision + "\n", "text/plain")

    with tempfile.TemporaryDirectory() as directory:
        log_path = Path(directory) / "log.jsonl"
        existing = store.get_text("log/log.jsonl")
        if existing:
            log_path.write_text(existing, encoding="utf-8")
        entries = load_entries(log_path)
        chain_problems = verify_chain(entries)
        if chain_problems:
            raise ValueError("remote transparency log is broken: " + "; ".join(chain_problems))
        digest = record_sha256(signed)
        if any(entry.get("record_sha256") == digest for entry in entries):
            log("record already in the transparency log")
        else:
            append(log_path, digest)
        entries = load_entries(log_path)
        signed_checkpoint = checkpoint(entries, publisher_seed)
        store.put_text("log/log.jsonl", log_path.read_text(encoding="utf-8"), "application/x-ndjson")
        store.put_text("log/checkpoint.json", signed_checkpoint.to_json() + "\n", "application/json")

    catalog_text = store.get_text("catalog.json")
    catalog = json.loads(catalog_text) if catalog_text else {"version": 1, "editions": []}
    summary = {"id": edition_id, "name": record["edition"]["name"], "base_model": record["edition"]["base_model"], "current": revision,
               "license": record["license"], "fingerprint_id": record.get("fingerprint_id"), "total_bytes": record["total_bytes"],
               "files": len(record["files"]), "published_at": record["published_at"]}
    catalog["editions"] = [item for item in catalog["editions"] if item.get("id") != edition_id] + [summary]
    catalog["editions"].sort(key=lambda item: item["id"])
    catalog["updated_at"] = int(time.time())
    store.put_text("catalog.json", json.dumps(catalog, indent=1, ensure_ascii=False) + "\n", "application/json")
    return {"edition": edition_id, "revision": revision, "uploaded": uploaded, "skipped": skipped, "log_size": len(entries), "tag": tag}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--files-dir", required=True, type=Path)
    parser.add_argument("--publisher-seed-file", required=True, type=Path)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--endpoint", required=True, help="S3 endpoint, e.g. https://<account id>.r2.cloudflarestorage.com")
    parser.add_argument("--tag", default="main")
    parser.add_argument("--skip-verify", action="store_true", help="trust local files without re-hashing (not recommended)")
    args = parser.parse_args(argv)
    access_key, secret_key = os.environ.get("R2_ACCESS_KEY_ID"), os.environ.get("R2_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
        print("R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY are required", file=sys.stderr)
        return 2
    import boto3
    from boto3.s3.transfer import TransferConfig
    client = boto3.client("s3", endpoint_url=args.endpoint, aws_access_key_id=access_key, aws_secret_access_key=secret_key, region_name="auto")
    store = S3Store(client, args.bucket)
    original_upload = store.upload_file
    config = TransferConfig(multipart_threshold=64 * 1024 * 1024, multipart_chunksize=64 * 1024 * 1024, max_concurrency=8)

    def upload_file(key: str, path: Path, content_type: str) -> None:
        client.upload_file(str(path), args.bucket, key, ExtraArgs={"ContentType": content_type}, Config=config)

    store.upload_file = upload_file  # type: ignore[method-assign]
    del original_upload
    signed = Signed.from_json(args.record.read_text(encoding="utf-8"))
    seed = bytes.fromhex(args.publisher_seed_file.read_text().strip())
    try:
        result = publish(store, signed, args.files_dir, seed, tag=args.tag, skip_verify=args.skip_verify)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
