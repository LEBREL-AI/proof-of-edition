import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from nacl.signing import SigningKey

from proof_of_edition.receipts.schema import Signed, sign
from proof_of_edition.registry.log import check_checkpoint, load_entries
from proof_of_edition.registry.publish import publish, verify_local_files
from proof_of_edition.registry.record import build_record

SEED = bytes([3]) * 32
PUBLIC = bytes(SigningKey(SEED).verify_key)


class MemoryStore:
    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.uploads: list[str] = []

    def exists(self, key):
        return key in self.objects

    def get_text(self, key):
        return self.objects[key].decode("utf-8") if key in self.objects else None

    def put_text(self, key, text, content_type):
        self.objects[key] = text.encode("utf-8")

    def upload_file(self, key, path, content_type):
        self.objects[key] = Path(path).read_bytes()
        self.uploads.append(key)


def make_edition(directory: Path) -> tuple[Signed, dict]:
    files = {"model-00001-of-00002.safetensors": b"A" * 5000, "model-00002-of-00002.safetensors": b"B" * 300, "config.json": b'{"x":1}', "tokenizer.json": b"{}"}
    manifest = {"files": {}}
    for name, content in files.items():
        (directory / name).write_bytes(content)
        manifest["files"][name] = {"sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}
    record = build_record({"id": "lebrel/edition", "name": "Edition", "base_model": "base"}, manifest, license_id="MIT", publisher_public_key=PUBLIC, published_at=1_790_000_000)
    return sign(record, SEED), manifest


class PublishTest(unittest.TestCase):
    def test_publish_uploads_blobs_once_and_maintains_log_and_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            signed, _ = make_edition(base)
            store = MemoryStore()
            result = publish(store, signed, base, SEED, log=lambda *_: None)
            self.assertEqual(result["uploaded"], 4)
            self.assertEqual(result["log_size"], 1)
            revision = signed.payload["revision"]
            self.assertIn(f"editions/lebrel/edition/records/{revision}.json", store.objects)
            self.assertEqual(store.get_text("editions/lebrel/edition/refs/main").strip(), revision)
            catalog = json.loads(store.get_text("catalog.json"))
            self.assertEqual(catalog["editions"][0]["current"], revision)
            with tempfile.TemporaryDirectory() as logdir:
                log_path = Path(logdir) / "log.jsonl"
                log_path.write_text(store.get_text("log/log.jsonl"))
                entries = load_entries(log_path)
                self.assertEqual(check_checkpoint(Signed.from_json(store.get_text("log/checkpoint.json")), entries, PUBLIC), [])
            # Republishing the same revision uploads nothing and does not duplicate the log entry.
            again = publish(store, signed, base, SEED, log=lambda *_: None)
            self.assertEqual(again["uploaded"], 0)
            self.assertEqual(again["skipped"], 4)
            self.assertEqual(again["log_size"], 1)
            # A second revision sharing shards only uploads the changed file and appends to the log.
            (base / "config.json").write_bytes(b'{"x":2}')
            manifest = {"files": {name: {"sha256": hashlib.sha256((base / name).read_bytes()).hexdigest(), "size": (base / name).stat().st_size} for name in ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "config.json", "tokenizer.json"]}}
            record2 = sign(build_record({"id": "lebrel/edition", "name": "Edition", "base_model": "base"}, manifest, license_id="MIT", publisher_public_key=PUBLIC, published_at=1_790_000_001), SEED)
            third = publish(store, record2, base, SEED, tag="main", log=lambda *_: None)
            self.assertEqual(third["uploaded"], 1)
            self.assertEqual(third["log_size"], 2)
            self.assertEqual(store.get_text("editions/lebrel/edition/refs/main").strip(), record2.payload["revision"])
            self.assertEqual(len(json.loads(store.get_text("catalog.json"))["editions"]), 1)

    def test_local_mismatch_and_bad_record_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            signed, _ = make_edition(base)
            (base / "model-00002-of-00002.safetensors").write_bytes(b"C" * 300)
            problems = verify_local_files(signed.payload, base)
            self.assertEqual(problems, ["model-00002-of-00002.safetensors: sha256 differs from the record"])
            with self.assertRaises(ValueError):
                publish(MemoryStore(), signed, base, SEED, log=lambda *_: None)
            foreign = sign(signed.payload, bytes([9]) * 32)
            with self.assertRaises(ValueError):
                publish(MemoryStore(), foreign, base, SEED, log=lambda *_: None)


if __name__ == "__main__":
    unittest.main()
