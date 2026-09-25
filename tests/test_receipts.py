import time
import unittest

try:
    from nacl.signing import SigningKey
except ImportError:  # pragma: no cover
    SigningKey = None

from receipts.schema import manifest_identity, Signed, canonical, check_manifest, check_receipt, key_id, sha256_hex, sign


def manifest_payload(public_key: bytes, now: float) -> dict:
    return {
        "version": 1, "edition": "lebrel/qwen3.8-27b-lebrel-uncensored",
        "weights": {"repository": "lebrel/qwen3.8-27b-lebrel-uncensored", "revision": "a" * 40, "files": {"model-00001-of-00012.safetensors": "b" * 64}},
        "quantization": {"method": "bf16", "config_sha256": "c" * 64},
        "engine": {"name": "vllm", "version": "0.11", "dtype": "bfloat16", "kv_cache_dtype": "auto", "tensor_parallel": 1, "max_context": 4096},
        "tokenizer_sha256": "d" * 64, "chat_template_sha256": "e" * 64,
        "runtime": {"image_sha256": "f" * 64, "instance_id": "0" * 32},
        "attestation": None, "issued_at": now, "expires_at": now + 3600, "signing_key_id": key_id(public_key),
    }


@unittest.skipIf(SigningKey is None, "PyNaCl not installed")
class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.signer = SigningKey.generate()
        self.seed = bytes(self.signer)
        self.public = bytes(self.signer.verify_key)
        self.now = time.time()
        self.manifest = sign(manifest_payload(self.public, self.now), self.seed)

    def test_valid_manifest_and_receipt(self):
        self.assertEqual(check_manifest(self.manifest, self.public, self.now), [])
        prompt, response = b'{"messages":[{"role":"user","content":"hi"}]}', b"Hello."
        receipt = sign({
            "version": 1, "request_id": "req-1", "manifest_sha256": manifest_identity(self.manifest.payload),
            "prompt_sha256": sha256_hex(prompt), "response_sha256": sha256_hex(response), "prompt_tokens": 5,
            "completion_tokens": 2, "issued_at": self.now, "instance_id": "0" * 32, "signing_key_id": key_id(self.public),
        }, self.seed)
        self.assertEqual(check_receipt(receipt, self.public, self.manifest, prompt, response), [])
        self.assertIn("response digest", " ".join(check_receipt(receipt, self.public, self.manifest, prompt, b"tampered")))

    def test_tampered_or_foreign_signature_is_rejected(self):
        tampered = Signed(payload={**self.manifest.payload, "edition": "other"}, signature_b64=self.manifest.signature_b64)
        self.assertIn("manifest signature invalid", check_manifest(tampered, self.public, self.now))
        other = bytes(SigningKey.generate().verify_key)
        self.assertIn("manifest signed by an unpinned key", check_manifest(self.manifest, other, self.now))

    def test_expired_manifest(self):
        self.assertIn("manifest expired or not yet valid", check_manifest(self.manifest, self.public, self.now + 7200))


if __name__ == "__main__":
    unittest.main()
