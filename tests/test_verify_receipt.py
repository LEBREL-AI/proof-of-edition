import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from nacl.signing import SigningKey

from receipts.schema import key_id, manifest_identity, sign
from receipts.verify_receipt import main, verify

NOW = 1_900_000_123


def make_manifest_payload(signer_public: bytes) -> dict:
    return {
        "version": 1,
        "edition": {"name": "Edition", "id": "lebrel/edition"},
        "weights": {"repository": "org/repo", "revision": "0" * 40, "files": {"model.safetensors": "a" * 64}},
        "quantization": {"method": "nvfp4", "kv_cache_dtype": "fp8_e4m3"},
        "engine": {"name": "sglang", "version": "0.5.18", "image_digest": "sha256:" + "b" * 64},
        "tokenizer_sha256": "c" * 64,
        "chat_template_sha256": "d" * 64,
        "runtime": {"provider": "modal", "gpu": "B300", "instance_id": "inst-1"},
        "attestation": None,
        "issued_at": NOW - NOW % 3600,
        "expires_at": NOW - NOW % 3600 + 3600,
        "signing_key_id": key_id(signer_public),
    }


class VerifyReceiptTest(unittest.TestCase):
    def setUp(self):
        self.seed = bytes([7]) * 32
        self.public = bytes(SigningKey(self.seed).verify_key)
        self.prompt = b'{"model":"lebrel/edition","messages":[{"role":"user","content":"hi"}]}'
        self.response = "private-answer ✓".encode("utf-8")
        self.manifest = sign(make_manifest_payload(self.public), self.seed)
        self.receipt = sign({
            "version": 1, "request_id": "req-1", "manifest_sha256": manifest_identity(self.manifest.payload),
            "prompt_sha256": hashlib.sha256(self.prompt).hexdigest(),
            "response_sha256": hashlib.sha256(self.response).hexdigest(),
            "prompt_tokens": None, "completion_tokens": 7, "issued_at": NOW,
            "instance_id": "inst-1", "signing_key_id": key_id(self.public),
        }, self.seed)

    def test_verify_accepts_matching_documents_and_plaintext(self):
        self.assertEqual(verify(self.manifest, self.receipt, self.public, self.prompt, self.response, now=NOW), [])

    def test_verify_reports_each_mismatch(self):
        problems = verify(self.manifest, self.receipt, self.public, b"other prompt", b"other answer", now=NOW)
        self.assertEqual(len(problems), 2)
        self.assertTrue(all(problem.startswith("receipt:") for problem in problems))
        foreign = bytes(SigningKey(bytes([9]) * 32).verify_key)
        problems = verify(self.manifest, self.receipt, foreign, None, None, now=NOW)
        self.assertTrue(any("unpinned key" in problem for problem in problems))
        problems = verify(self.manifest, self.receipt, self.public, None, None, now=NOW + 7200)
        self.assertEqual(problems, ["manifest: manifest expired or not yet valid"])

    def test_receipt_matches_manifest_republished_in_a_later_window(self):
        later = dict(self.manifest.payload, issued_at=NOW + 3600, expires_at=NOW + 7200)
        republished = sign(later, self.seed)
        self.assertEqual(verify(republished, self.receipt, self.public, None, None, now=NOW + 3700), [])

    def test_cli_exit_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "manifest.json").write_text(self.manifest.to_json(), encoding="utf-8")
            (base / "receipt.json").write_text(self.receipt.to_json(), encoding="utf-8")
            (base / "prompt.bin").write_bytes(self.prompt)
            (base / "response.bin").write_bytes(self.response)
            (base / "wrong.bin").write_bytes(b"tampered")
            common = ["--public-key", self.public.hex(), "--manifest-file", str(base / "manifest.json"),
                      "--receipt-file", str(base / "receipt.json"), "--now", str(NOW)]
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(common + ["--prompt-file", str(base / "prompt.bin"), "--response-file", str(base / "response.bin")])
            self.assertEqual(code, 0, out.getvalue())
            self.assertIn("VERIFIED signatures, manifest window, manifest identity, prompt digest, response digest", out.getvalue())
            self.assertIn("lebrel/edition", out.getvalue())
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(common + ["--response-file", str(base / "wrong.bin")])
            self.assertEqual(code, 1)
            self.assertIn("FAIL receipt: receipt response digest does not match", out.getvalue())
            err = io.StringIO()
            with redirect_stderr(err), redirect_stdout(io.StringIO()):
                code = main(["--public-key", "zz", "--manifest-file", str(base / "manifest.json"), "--receipt-file", str(base / "receipt.json")])
            self.assertEqual(code, 2)
            (base / "broken.json").write_text("{}", encoding="utf-8")
            err = io.StringIO()
            with redirect_stderr(err), redirect_stdout(io.StringIO()):
                code = main(["--public-key", self.public.hex(), "--manifest-file", str(base / "broken.json"), "--receipt-file", str(base / "receipt.json")])
            self.assertEqual(code, 2)
            self.assertIn("not a signed document", err.getvalue())


if __name__ == "__main__":
    unittest.main()
