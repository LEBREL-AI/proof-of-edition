"""Publish the DeepSeek V4 Flash edition from the production weights volume to the registry bucket.

Runs on Modal (CPU only) next to the weights, so the 167 GB never pass through a laptop:
hash every file against the signed record, upload the blobs to R2 with multipart S3, then
write the record, tag, transparency log entry, checkpoint and catalog. R2 credentials live
in the Modal secret ``lebrel-registry-r2`` (R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY,
R2_ENDPOINT); the publisher seed is passed as a Modal secret too (``lebrel-registry-publisher``,
key REGISTRY_PUBLISHER_SEED_HEX) only for signing the log checkpoint.

    uvx modal run registry/modal_publish.py --record registry/published/<edition>.record.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WEIGHTS_VOLUME = "raw-deepseek-v4-weights"
WEIGHTS_MOUNT = Path("/weights")
EDITION_DIR = WEIGHTS_MOUNT / "deepseek-v4-flash-0731-crack"
BUCKET = "lebrel-registry"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pynacl==1.5.0", "boto3==1.35.99")
    .add_local_dir(str(ROOT / "receipts"), "/root/proof/receipts", copy=True)
    .add_local_dir(str(ROOT / "registry"), "/root/proof/registry", copy=True)
)
app = modal.App("lebrel-registry-publish")


@app.function(image=image, cpu=8, memory=32 * 1024, timeout=6 * 60 * 60,
              volumes={str(WEIGHTS_MOUNT): modal.Volume.from_name(WEIGHTS_VOLUME)},
              secrets=[modal.Secret.from_name("lebrel-registry-r2"), modal.Secret.from_name("lebrel-registry-publisher")])
def publish_edition(record_json: str, tag: str = "main", skip_verify: bool = False) -> dict:
    sys.path.insert(0, "/root/proof")
    import boto3
    from boto3.s3.transfer import TransferConfig
    from receipts.schema import Signed
    from registry.publish import S3Store, publish

    started = time.monotonic()
    client = boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT"], aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                          aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"], region_name="auto")
    store = S3Store(client, BUCKET)
    config = TransferConfig(multipart_threshold=64 * 1024 * 1024, multipart_chunksize=128 * 1024 * 1024, max_concurrency=8)

    def upload_file(key: str, path: Path, content_type: str) -> None:
        client.upload_file(str(path), BUCKET, key, ExtraArgs={"ContentType": content_type}, Config=config)

    store.upload_file = upload_file  # type: ignore[method-assign]
    signed = Signed.from_json(record_json)
    seed = bytes.fromhex(os.environ["REGISTRY_PUBLISHER_SEED_HEX"])
    if not EDITION_DIR.is_dir():
        raise RuntimeError(f"{EDITION_DIR} is not mounted")
    result = publish(store, signed, EDITION_DIR, seed, tag=tag, skip_verify=skip_verify, log=lambda message: print(message, flush=True))
    result["minutes"] = round((time.monotonic() - started) / 60, 1)
    return result


@app.local_entrypoint()
def main(record: str, tag: str = "main", skip_verify: bool = False) -> None:
    record_json = Path(record).read_text(encoding="utf-8")
    print(json.dumps(publish_edition.remote(record_json, tag=tag, skip_verify=skip_verify), indent=1))
