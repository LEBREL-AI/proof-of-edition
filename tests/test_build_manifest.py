import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from proof_of_edition.manifest.build_manifest import build_manifest, chat_template_digest, main, validate_edition

ROOT = Path(__file__).resolve().parents[1]
EDITION = ROOT / "proof_of_edition" / "manifest" / "editions" / "lebrel-deepseek-v4-flash-uncensored.json"


def weights_manifest(extra: dict | None = None) -> dict:
    files = {
        "model-00001-of-00002.safetensors": {"sha256": "1" * 64, "size": 100},
        "model-00002-of-00002.safetensors": {"sha256": "2" * 64, "size": 50},
        "model.safetensors.index.json": {"sha256": "3" * 64, "size": 10},
        "config.json": {"sha256": "4" * 64, "size": 5},
        "tokenizer.json": {"sha256": "5" * 64, "size": 7},
        "tokenizer_config.json": {"sha256": "6" * 64, "size": 3},
        "README.md": {"sha256": "7" * 64, "size": 1},
        "encoding/encoding_dsv4.py": {"sha256": "8" * 64, "size": 9},
        "encoding/test_encoding_dsv4.py": {"sha256": "9" * 64, "size": 9},
    }
    files.update(extra or {})
    return {"schema_version": 1, "repository": "dealignai/Repo", "revision": "c" * 40, "files": files}


class BuildManifestTest(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads(EDITION.read_text(encoding="utf-8"))

    def test_production_edition_spec_is_valid(self):
        self.assertEqual(validate_edition(self.spec), [])
        self.assertEqual(self.spec["id"], "lebrel/deepseek-v4-flash-uncensored")
        self.assertEqual(self.spec["engine"]["image_digest"], "sha256:9e148f5ac788e856a06166bd6347a831831eb9fcfab4d1770874823a7c29a1a1")

    def test_manifest_selects_weight_files_and_digests(self):
        manifest = build_manifest(self.spec, weights_manifest())
        self.assertEqual(sorted(manifest["weights"]["files"]), [
            "config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "model.safetensors.index.json",
        ])
        self.assertEqual(manifest["weights"]["total_bytes"], 165)
        self.assertEqual(manifest["weights"]["revision"], "c" * 40)
        self.assertEqual(manifest["tokenizer_sha256"], "5" * 64)
        self.assertEqual(manifest["chat_template_sha256"], "8" * 64)
        self.assertEqual(manifest["chat_template_source"], "encoding/encoding_dsv4.py")
        self.assertEqual(manifest["engine"]["arguments_sha256"], hashlib.sha256("\n".join(self.spec["engine"]["arguments"]).encode()).hexdigest())
        self.assertEqual(manifest["runtime"]["sidecar"]["binary_sha256"], self.spec["sidecar"]["binary_sha256"])
        self.assertNotIn("issued_at", manifest)
        self.assertNotIn("signing_key_id", manifest)
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def test_chat_template_preference_order(self):
        files = weights_manifest({"chat_template.jinja": {"sha256": "e" * 64, "size": 1}})["files"]
        self.assertEqual(chat_template_digest(files, None), ("e" * 64, "chat_template.jinja"))
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8")
            digest, source = chat_template_digest(weights_manifest()["files"], repo)
            self.assertEqual(source, "tokenizer_config.json#chat_template")
            self.assertEqual(digest, hashlib.sha256(b"{{ messages }}").hexdigest())
        files = weights_manifest()["files"]
        for name in list(files):
            if name.startswith("encoding/"):
                del files[name]
        with self.assertRaises(ValueError):
            chat_template_digest(files, None)

    def test_rejects_broken_inputs(self):
        broken = weights_manifest()
        broken["revision"] = "short"
        with self.assertRaises(ValueError):
            build_manifest(self.spec, broken)
        no_shards = weights_manifest()
        for name in list(no_shards["files"]):
            if name.endswith(".safetensors"):
                del no_shards["files"][name]
        with self.assertRaises(ValueError):
            build_manifest(self.spec, no_shards)
        floaty = json.loads(json.dumps(self.spec))
        floaty["runtime"]["load"] = 0.5
        self.assertTrue(any("floating point" in problem for problem in validate_edition(floaty)))
        bad_digest = json.loads(json.dumps(self.spec))
        bad_digest["engine"]["image_digest"] = "9e148f5a"
        self.assertTrue(any("image_digest" in problem for problem in validate_edition(bad_digest)))

    def test_cli_writes_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "weights.json").write_text(json.dumps(weights_manifest()), encoding="utf-8")
            out = base / "out" / "manifest.json"
            self.assertEqual(main(["--edition", str(EDITION), "--weights-manifest", str(base / "weights.json"), "--out", str(out)]), 0)
            written = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(written["edition"]["id"], "lebrel/deepseek-v4-flash-uncensored")


if __name__ == "__main__":
    unittest.main()
