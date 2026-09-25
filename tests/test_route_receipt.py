import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from proof_of_edition.receipts.schema import Signed, check_route_manifest, check_route_receipt, manifest_identity, sha256_hex
from proof_of_edition.receipts.verify_receipt import main, verify

FIXTURES = Path(__file__).parent / "fixtures" / "route"
NOW = 1_900_000_120


class RouteReceiptTests(unittest.TestCase):
    """The fixture was signed by the Go router (lebrel-router, TestWriteRouteFixture); the Python verifier must accept it."""

    def setUp(self):
        self.public_key = bytes.fromhex((FIXTURES / "public_key.txt").read_text().strip())
        self.manifest = Signed.from_json((FIXTURES / "manifest.json").read_text())
        self.receipt = Signed.from_json((FIXTURES / "receipt.json").read_text())
        self.prompt = (FIXTURES / "request.json").read_bytes()
        self.response = (FIXTURES / "response.txt").read_bytes()

    def test_go_signed_documents_verify(self):
        self.assertEqual(check_route_manifest(self.manifest, self.public_key, now=NOW), [])
        self.assertEqual(check_route_receipt(self.receipt, self.public_key, self.manifest, prompt=self.prompt, response=self.response), [])
        self.assertEqual(self.receipt.payload["manifest_sha256"], manifest_identity(self.manifest.payload))
        self.assertEqual(self.receipt.payload["prompt_sha256"], sha256_hex(self.prompt))

    def test_tampering_is_detected(self):
        tampered = Signed(payload=dict(self.receipt.payload, upstream="somewhere-else"), signature_b64=self.receipt.signature_b64)
        problems = check_route_receipt(tampered, self.public_key, self.manifest)
        self.assertTrue(any("signature invalid" in p for p in problems))
        self.assertTrue(any("upstream the manifest does not list" in p for p in problems))
        wrong_prompt = check_route_receipt(self.receipt, self.public_key, self.manifest, prompt=b"other prompt")
        self.assertTrue(any("prompt digest" in p for p in wrong_prompt))
        expired = check_route_manifest(self.manifest, self.public_key, now=NOW + 10 * 3600)
        self.assertTrue(any("expired" in p for p in expired))
        other_key = bytes(32)
        self.assertTrue(any("unpinned" in p for p in check_route_manifest(self.manifest, other_key, now=NOW)))

    def test_verify_dispatches_on_kind_and_cli_passes(self):
        self.assertEqual(verify(self.manifest, self.receipt, self.public_key, self.prompt, self.response, now=NOW), [])
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(["--public-key", self.public_key.hex(), "--manifest-file", str(FIXTURES / "manifest.json"), "--receipt-file", str(FIXTURES / "receipt.json"),
                         "--prompt-file", str(FIXTURES / "request.json"), "--response-file", str(FIXTURES / "response.txt"), "--now", str(NOW)])
        self.assertEqual(code, 0, out.getvalue())
        text = out.getvalue()
        self.assertIn("ROUTE RECEIPT", text)
        self.assertIn("does not prove which weights answered", text)
        self.assertIn("VERIFIED signatures, manifest window, manifest identity, prompt digest, response digest", text)

    def test_serving_receipt_path_untouched(self):
        # a route receipt presented against a serving manifest must fail loudly, never pass by accident
        from proof_of_edition.receipts.schema import check_receipt
        problems = check_receipt(self.receipt, self.public_key, self.manifest)
        self.assertTrue(any("unsupported receipt version" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
