import hashlib
import io
import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from nacl.signing import SigningKey

from receipts.schema import Signed, sign
from registry.log import append, checkpoint, load_entries
from registry.mirror import mirror
from registry.record import build_record, record_sha256

SEED = bytes([3]) * 32
PUBLIC = bytes(SigningKey(SEED).verify_key)
ID = "lebrel/edition"


class FakeResponse:
    def __init__(self, data: bytes, status: int, headers=None):
        self.data, self.status, self.pos = data, status, 0
        self.headers = headers or {}

    def close(self):
        pass

    def read(self, n=-1):
        if n is None or n < 0:
            chunk, self.pos = self.data[self.pos:], len(self.data)
        else:
            chunk = self.data[self.pos:self.pos + n]
            self.pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class FakeFetch:
    """Serves a registry from memory; counts range requests to prove resumption. With
    truncate_first, the first transfer of each blob stops after that many bytes."""

    def __init__(self, texts: dict, blobs: dict, truncate_first: int | None = None):
        self.texts, self.blobs, self.ranges = texts, blobs, []
        self.truncate_first, self.truncated = truncate_first, set()

    def text(self, path):
        return self.texts[path]

    def stream(self, path, offset):
        data = self.blobs[path]
        self.ranges.append(offset)
        if offset >= len(data):
            return 416, None
        body = data[offset:]
        if self.truncate_first is not None and path not in self.truncated and len(body) > self.truncate_first:
            self.truncated.add(path)
            body = body[:self.truncate_first]
        if offset:
            return 206, FakeResponse(body, 206, {"Content-Range": f"bytes {offset}-{len(data) - 1}/{len(data)}"})
        return 200, FakeResponse(body, 200)


def build_registry(tmp: Path):
    files = {"model-00001-of-00001.safetensors": b"W" * 20000, "config.json": b'{"a":1}', "tokenizer.json": b"{}"}
    manifest = {"files": {name: {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)} for name, data in files.items()}}
    record = sign(build_record({"id": ID, "name": "Edition", "base_model": "base"}, manifest, license_id="MIT", publisher_public_key=PUBLIC, published_at=1_790_000_000), SEED)
    revision = record.payload["revision"]
    log_path = tmp / "log.jsonl"
    append(log_path, record_sha256(record))
    cp = checkpoint(load_entries(log_path), SEED, at=1_790_000_001)
    texts = {
        "/v1/editions": json.dumps({"version": 1, "editions": [{"id": ID, "current": revision, "total_bytes": record.payload["total_bytes"]}]}),
        "/v1/log": log_path.read_text(),
        "/v1/log/checkpoint": cp.to_json(),
        f"/v1/editions/{ID}/records/{revision}": record.to_json(),
    }
    blobs = {f"/{ID}/resolve/{revision}/{name}": data for name, data in files.items()}
    return texts, blobs, record, revision, manifest


class MirrorTest(unittest.TestCase):
    def test_mirror_downloads_verifies_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            texts, blobs, record, revision, manifest = build_registry(tmp)
            roots = [tmp / "vol1", tmp / "vol2"]
            for root in roots:
                root.mkdir()
            fetch = FakeFetch(texts, blobs)
            summary = mirror(fetch, roots, PUBLIC, log=lambda *_: None)
            self.assertEqual(summary["problems"], [])
            self.assertEqual(summary["blobs_stored"], 3)
            for name, meta in manifest["files"].items():
                self.assertEqual((roots[0] / "blobs" / "sha256" / meta["sha256"]).stat().st_size, meta["size"])
            self.assertEqual((roots[0] / "editions" / ID / "refs" / "main").read_text().strip(), revision)
            self.assertTrue((roots[1] / "log" / "checkpoint.json").exists())
            index = json.loads((roots[1] / "mirror-index.json").read_text())
            self.assertEqual(index["revisions"][revision], str(roots[0]))
            again = mirror(fetch, roots, PUBLIC, log=lambda *_: None)
            self.assertEqual(again["blobs_stored"], 0)
            self.assertEqual(again["problems"], [])

    def test_mirror_resumes_partial_downloads_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            texts, blobs, record, revision, manifest = build_registry(tmp)
            root = tmp / "vol1"
            root.mkdir()
            shard = "model-00001-of-00001.safetensors"
            digest = manifest["files"][shard]["sha256"]
            part = root / "blobs" / "sha256" / (digest + ".part")
            part.parent.mkdir(parents=True)
            part.write_bytes(b"W" * 5000)
            fetch = FakeFetch(texts, blobs)
            summary = mirror(fetch, [root], PUBLIC, log=lambda *_: None)
            self.assertEqual(summary["problems"], [])
            self.assertIn(5000, fetch.ranges)
            self.assertEqual((root / "blobs" / "sha256" / digest).stat().st_size, 20000)
            # A corrupted origin file is refused, not stored, and reported without aborting the run.
            blobs[f"/{ID}/resolve/{revision}/config.json"] = b'{"a":2}'
            config_blob = root / "blobs" / "sha256" / manifest["files"]["config.json"]["sha256"]
            config_blob.unlink()
            corrupted = mirror(FakeFetch(texts, blobs), [root], PUBLIC, log=lambda *_: None)
            self.assertFalse(config_blob.exists())
            self.assertEqual(len(corrupted["problems"]), 1)
            self.assertIn("1 file(s) not mirrored", corrupted["problems"][0])
            self.assertIn("digest", corrupted["problems"][0])

    def test_truncated_transfers_are_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            texts, blobs, record, revision, manifest = build_registry(tmp)
            root = tmp / "vol1"
            root.mkdir()
            fetch = FakeFetch(texts, blobs, truncate_first=7000)
            with unittest.mock.patch("registry.mirror.time.sleep"):
                summary = mirror(fetch, [root], PUBLIC, log=lambda *_: None, parallel=1)
            self.assertEqual(summary["problems"], [])
            self.assertEqual(summary["blobs_stored"], 3)
            self.assertIn(7000, fetch.ranges, "the shard must have been resumed at the truncation point")
            for meta in manifest["files"].values():
                self.assertEqual((root / "blobs" / "sha256" / meta["sha256"]).stat().st_size, meta["size"])

    def test_foreign_publisher_key_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            texts, blobs, *_ = build_registry(tmp)
            root = tmp / "vol1"
            root.mkdir()
            summary = mirror(FakeFetch(texts, blobs), [root], bytes(SigningKey(bytes([9]) * 32).verify_key), log=lambda *_: None)
            self.assertEqual(summary["revisions"], 0)
            self.assertTrue(any("unpinned" in p for p in summary["problems"]))


if __name__ == "__main__":
    unittest.main()
