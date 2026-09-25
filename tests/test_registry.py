import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from nacl.signing import SigningKey

from proof_of_edition.receipts.schema import Signed, key_id, sign
from proof_of_edition.registry.check_manifest import compare, main as check_main
from proof_of_edition.registry.log import append, check_checkpoint, checkpoint, inclusion, load_entries, verify_chain
from proof_of_edition.registry.record import build_record, check_record, content_revision, record_sha256, main as record_main

NOW = 1_900_000_123
PUBLISHER_SEED = bytes([3]) * 32
RUNTIME_SEED = bytes([7]) * 32
PUBLISHER_PUBLIC = bytes(SigningKey(PUBLISHER_SEED).verify_key)
RUNTIME_PUBLIC = bytes(SigningKey(RUNTIME_SEED).verify_key)


def weights_manifest():
    return {"repository": "dealignai/Repo", "revision": "c" * 40, "files": {
        "model-00001-of-00002.safetensors": {"sha256": "1" * 64, "size": 100},
        "model-00002-of-00002.safetensors": {"sha256": "2" * 64, "size": 50},
        "model.safetensors.index.json": {"sha256": "3" * 64, "size": 10},
        "config.json": {"sha256": "4" * 64, "size": 5},
        "tokenizer.json": {"sha256": "5" * 64, "size": 7},
    }}


def edition():
    return {"id": "lebrel/deepseek-v4-flash-uncensored", "name": "Lebrel DeepSeek V4 Flash Uncensored", "base_model": "deepseek-ai/DeepSeek-V4-Flash", "license": "MIT"}


def signed_record():
    record = build_record(edition(), weights_manifest(), license_id="MIT", publisher_public_key=PUBLISHER_PUBLIC, published_at=NOW)
    return sign(record, PUBLISHER_SEED)


def signed_manifest(record_payload, **overrides):
    files = {path: entry["sha256"] for path, entry in record_payload["files"].items() if path.endswith(".safetensors") or path in ("model.safetensors.index.json", "config.json")}
    payload = {
        "version": 1, "edition": {"id": record_payload["edition"]["id"], "fingerprint_id": None},
        "weights": {"repository": "registry.lebrel.ai/" + record_payload["edition"]["id"], "revision": record_payload["revision"], "files": files},
        "quantization": {"method": "nvfp4"}, "engine": {"name": "sglang", "version": "0.5.18", "image_digest": "sha256:" + "b" * 64},
        "tokenizer_sha256": "5" * 64, "chat_template_sha256": "d" * 64,
        "runtime": {"provider": "modal", "gpu": "B300", "instance_id": "inst-1"}, "attestation": None,
        "issued_at": NOW - NOW % 3600, "expires_at": NOW - NOW % 3600 + 3600, "signing_key_id": key_id(RUNTIME_PUBLIC),
    }
    payload.update(overrides)
    return sign(payload, RUNTIME_SEED)


class RecordTest(unittest.TestCase):
    def test_record_is_content_addressed_and_verifies(self):
        signed = signed_record()
        self.assertEqual(check_record(signed, PUBLISHER_PUBLIC), [])
        self.assertEqual(signed.payload["revision"], content_revision(signed.payload["files"]))
        self.assertEqual(signed.payload["total_bytes"], 172)
        reordered = dict(weights_manifest())
        reordered["files"] = dict(reversed(list(reordered["files"].items())))
        again = build_record(edition(), reordered, license_id="MIT", publisher_public_key=PUBLISHER_PUBLIC, published_at=NOW)
        self.assertEqual(again["revision"], signed.payload["revision"])
        tampered = dict(signed.payload)
        tampered["files"] = dict(tampered["files"], **{"model-00001-of-00002.safetensors": {"sha256": "9" * 64, "size": 100}})
        self.assertTrue(any("revision does not match" in p for p in check_record(Signed(tampered, signed.signature_b64), PUBLISHER_PUBLIC)))
        foreign = bytes(SigningKey(bytes([9]) * 32).verify_key)
        self.assertTrue(any("unpinned" in p for p in check_record(signed, foreign)))
        with self.assertRaises(ValueError):
            build_record(edition(), {"files": {"../x.safetensors": {"sha256": "1" * 64, "size": 1}}}, license_id="MIT", publisher_public_key=PUBLISHER_PUBLIC)

    def test_record_cli_publish_and_check(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "edition.json").write_text(json.dumps(edition()))
            (base / "weights.json").write_text(json.dumps(weights_manifest()))
            (base / "seed.hex").write_text(PUBLISHER_SEED.hex())
            out = io.StringIO()
            with redirect_stdout(out):
                code = record_main(["publish", "--edition", str(base / "edition.json"), "--weights-manifest", str(base / "weights.json"), "--publisher-seed-file", str(base / "seed.hex"), "--out", str(base / "record.json")])
            self.assertEqual(code, 0, out.getvalue())
            with redirect_stdout(out):
                self.assertEqual(record_main(["check", "--record", str(base / "record.json"), "--publisher-key", PUBLISHER_PUBLIC.hex()]), 0)
            self.assertIn("VERIFIED record", out.getvalue())
            signed = Signed.from_json((base / "record.json").read_text())
            self.assertEqual(signed.payload["source"], {"repository": "dealignai/Repo", "revision": "c" * 40})


class LogTest(unittest.TestCase):
    def test_chain_checkpoint_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "log.jsonl"
            first, second = signed_record(), signed_record()
            second = sign(dict(second.payload, published_at=NOW + 1), PUBLISHER_SEED)
            append(log, record_sha256(first))
            cp1 = checkpoint(load_entries(log), PUBLISHER_SEED, at=NOW)
            append(log, record_sha256(second))
            entries = load_entries(log)
            self.assertEqual(verify_chain(entries), [])
            cp2 = checkpoint(entries, PUBLISHER_SEED, at=NOW + 5)
            self.assertEqual(check_checkpoint(cp1, entries, PUBLISHER_PUBLIC), [])
            self.assertEqual(check_checkpoint(cp2, entries, PUBLISHER_PUBLIC), [])
            self.assertEqual(inclusion(entries, record_sha256(second)), 1)
            self.assertIsNone(inclusion(entries, "0" * 64))
            rewritten = [dict(entries[0], record_sha256="f" * 64), entries[1]]
            self.assertTrue(any("broken" in p for p in verify_chain(rewritten)))
            removed = entries[1:]
            self.assertTrue(check_checkpoint(cp1, [dict(removed[0], index=0)], PUBLISHER_PUBLIC))
            with self.assertRaises(ValueError):
                log.write_text(json.dumps(dict(entries[0], entry_hash="0" * 64)) + "\n")
                append(log, record_sha256(second))


class CheckManifestTest(unittest.TestCase):
    def test_compare_detects_substitution(self):
        record = signed_record()
        manifest = signed_manifest(record.payload)
        self.assertEqual(compare(manifest.payload, record.payload), [])
        swapped = signed_manifest(record.payload, weights=dict(manifest.payload["weights"], files=dict(manifest.payload["weights"]["files"], **{"model-00002-of-00002.safetensors": "e" * 64})))
        self.assertTrue(any("digest differs" in p for p in compare(swapped.payload, record.payload)))
        other = signed_manifest(record.payload, edition={"id": "lebrel/other", "fingerprint_id": None})
        self.assertTrue(any("edition mismatch" in p for p in compare(other.payload, record.payload)))
        partial = signed_manifest(record.payload, weights=dict(manifest.payload["weights"], files={"model-00001-of-00002.safetensors": "1" * 64}))
        self.assertTrue(any("published shards not in the manifest" in p for p in compare(partial.payload, record.payload)))

    def test_cli_end_to_end_with_log(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            record = signed_record()
            manifest = signed_manifest(record.payload)
            (base / "record.json").write_text(record.to_json())
            (base / "manifest.json").write_text(manifest.to_json())
            log = base / "log.jsonl"
            append(log, record_sha256(record))
            (base / "cp.json").write_text(checkpoint(load_entries(log), PUBLISHER_SEED, at=NOW).to_json())
            common = ["--manifest", str(base / "manifest.json"), "--runtime-key", RUNTIME_PUBLIC.hex(), "--record", str(base / "record.json"), "--publisher-key", PUBLISHER_PUBLIC.hex(), "--now", str(NOW)]
            out = io.StringIO()
            with redirect_stdout(out):
                code = check_main(common + ["--log", str(log), "--checkpoint", str(base / "cp.json")])
            self.assertEqual(code, 0, out.getvalue())
            self.assertIn("exactly as published", out.getvalue())
            unlisted = sign(dict(record.payload, published_at=NOW + 9), PUBLISHER_SEED)
            (base / "record2.json").write_text(unlisted.to_json())
            manifest2 = signed_manifest(unlisted.payload)
            (base / "manifest2.json").write_text(manifest2.to_json())
            out = io.StringIO()
            with redirect_stdout(out):
                code = check_main(["--manifest", str(base / "manifest2.json"), "--runtime-key", RUNTIME_PUBLIC.hex(), "--record", str(base / "record2.json"), "--publisher-key", PUBLISHER_PUBLIC.hex(), "--now", str(NOW), "--log", str(log), "--checkpoint", str(base / "cp.json")])
            self.assertEqual(code, 1)
            self.assertIn("not in the transparency log", out.getvalue())


if __name__ == "__main__":
    unittest.main()
