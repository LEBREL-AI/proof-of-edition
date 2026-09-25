import hashlib
import io
import json
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from nacl.signing import SigningKey

from proof_of_edition.audit.reexecute import audit, hamming_kernel, is_deterministic, load_samples, main, paired_permutation_test, prefix_agreement
from proof_of_edition.receipts.schema import Signed, key_id, manifest_identity, sign

NOW = 1_900_000_123
SEED = bytes([7]) * 32
PUBLIC = bytes(SigningKey(SEED).verify_key)


def manifest_signed() -> Signed:
    return sign({
        "version": 1, "edition": {"name": "Edition", "id": "lebrel/edition"},
        "weights": {"repository": "org/repo", "revision": "0" * 40, "files": {"model.safetensors": "a" * 64}},
        "quantization": {"method": "nvfp4"}, "engine": {"name": "sglang", "version": "0.5.18", "image_digest": "sha256:" + "b" * 64},
        "tokenizer_sha256": "c" * 64, "chat_template_sha256": "d" * 64,
        "runtime": {"provider": "modal", "gpu": "B300", "instance_id": "inst-1"}, "attestation": None,
        "issued_at": NOW - NOW % 3600, "expires_at": NOW - NOW % 3600 + 3600, "signing_key_id": key_id(PUBLIC),
    }, SEED)


def make_sample(manifest: Signed, index: int, request: dict, response_text: str, tamper: bool = False) -> dict:
    raw = json.dumps(request, separators=(",", ":"), ensure_ascii=False)
    receipt = sign({
        "version": 1, "request_id": f"req-{index}", "manifest_sha256": manifest_identity(manifest.payload),
        "prompt_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "response_sha256": hashlib.sha256((response_text + ("!" if tamper else "")).encode("utf-8")).hexdigest(),
        "prompt_tokens": 10, "completion_tokens": 20, "issued_at": NOW, "instance_id": "inst-1", "signing_key_id": key_id(PUBLIC),
    }, SEED)
    return {"request": request, "request_raw": raw, "response_text": response_text, "receipt": {"payload": receipt.payload, "signature": receipt.signature_b64}}


class ReexecuteUnitTest(unittest.TestCase):
    def test_similarity_helpers(self):
        self.assertEqual(prefix_agreement("abc", "abc"), 1.0)
        self.assertEqual(prefix_agreement("abcd", "abxy"), 0.5)
        self.assertEqual(prefix_agreement("", ""), 1.0)
        self.assertEqual(hamming_kernel(["a", "b", "c"], ["a", "x", "c"], 4), 0.5)
        self.assertEqual(hamming_kernel([], [], 4), 0.0)
        self.assertTrue(is_deterministic({"temperature": 0}))
        self.assertTrue(is_deterministic({"top_k": 1, "temperature": 1}))
        self.assertFalse(is_deterministic({"temperature": 0.7}))
        self.assertFalse(is_deterministic({}))

    def test_permutation_test_detects_shift_and_accepts_null(self):
        rng = __import__("random").Random(1)
        same = [rng.uniform(0.4, 0.6) for _ in range(40)]
        also_same = [rng.uniform(0.4, 0.6) for _ in range(40)]
        # Both seeds are fixed, so this is a deterministic draw from the null (p = 0.038 for these seeds);
        # the test guards against the statistic flagging exchangeable pairs at the audit's alpha of 0.01.
        _, p_null = paired_permutation_test(same, also_same, 500)
        self.assertGreater(p_null, 0.01)
        statistic, p_identical = paired_permutation_test(same, list(same), 500)
        self.assertEqual(statistic, 0.0)
        self.assertEqual(p_identical, 1.0)
        worse = [value - 0.3 for value in same]
        statistic, p_shift = paired_permutation_test(worse, also_same, 500)
        self.assertGreater(statistic, 0.2)
        self.assertLess(p_shift, 0.01)


class ReexecuteAuditTest(unittest.TestCase):
    def setUp(self):
        self.manifest = manifest_signed()

    def test_consistent_deterministic_and_sampled(self):
        samples = []
        answers = {}
        for index in range(12):
            request = {"model": "lebrel/edition", "messages": [{"role": "user", "content": f"q{index}"}], "temperature": 0 if index % 2 == 0 else 0.8}
            answer = " ".join(f"tok{index}_{position}" for position in range(30))
            answers[f"q{index}"] = answer
            samples.append(make_sample(self.manifest, index, request, answer))

        def reference(request):
            return answers[request["messages"][0]["content"]]

        report = audit(samples, self.manifest, PUBLIC, reference, permutations=200, now=NOW)
        self.assertEqual(report.verdict, "consistent", report.reasons)
        self.assertEqual(report.deterministic["exact_rate"], 1.0)
        self.assertEqual(report.sampled["count"], 6)
        self.assertGreater(report.sampled["p_value"], 0.05)

    def test_swapped_model_is_inconsistent_and_bad_receipts_are_excluded(self):
        samples = []
        for index in range(20):
            request = {"model": "lebrel/edition", "messages": [{"role": "user", "content": f"q{index}"}], "temperature": 0 if index < 10 else 0.8}
            recorded = " ".join(f"other{index}_{position}" for position in range(40))
            samples.append(make_sample(self.manifest, index, request, recorded, tamper=(index == 3)))

        def reference(request):
            name = request["messages"][0]["content"]
            return " ".join(f"ref{name}_{position}" for position in range(40))

        report = audit(samples, self.manifest, PUBLIC, reference, permutations=300, now=NOW)
        self.assertEqual(report.verdict, "inconsistent")
        self.assertIn("1 receipt(s) failed verification", report.reasons)
        self.assertTrue(any("deterministic answers diverge" in reason for reason in report.reasons))
        self.assertTrue(any("sampled answers are less similar" in reason for reason in report.reasons))
        self.assertEqual(report.deterministic["count"], 9)
        self.assertEqual(report.deterministic["exact_rate"], 0.0)
        self.assertLess(report.sampled["p_value"], 0.01)

    def test_reference_failures_yield_error_verdict(self):
        request = {"model": "lebrel/edition", "messages": [{"role": "user", "content": "q"}], "temperature": 0}
        samples = [make_sample(self.manifest, 0, request, "answer")]

        def reference(_request):
            raise ValueError("reference down")

        report = audit(samples, self.manifest, PUBLIC, reference, now=NOW)
        self.assertEqual(report.verdict, "error")
        self.assertIn("1 sample(s) could not be re-executed", report.reasons)

    def test_cli_against_mock_reference(self):
        answers = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length))
                assert self.path == "/v1/chat/completions"
                assert body["stream"] is False and body["model"] == "reference-model"
                content = answers[body["messages"][0]["content"]]
                payload = json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                lines = []
                for index in range(6):
                    request = {"model": "lebrel/edition", "messages": [{"role": "user", "content": f"q{index}"}], "temperature": 0}
                    answer = f"answer {index} " * 5
                    answers[f"q{index}"] = answer
                    lines.append(json.dumps(make_sample(self.manifest, index, request, answer), ensure_ascii=False))
                (base / "samples.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
                (base / "manifest.json").write_text(self.manifest.to_json(), encoding="utf-8")
                self.assertEqual(len(load_samples(base / "samples.jsonl")), 6)
                out = io.StringIO()
                with redirect_stdout(out):
                    code = main(["--samples", str(base / "samples.jsonl"), "--manifest", str(base / "manifest.json"), "--public-key", PUBLIC.hex(),
                                 "--reference-url", f"http://127.0.0.1:{server.server_port}/v1", "--reference-model", "reference-model",
                                 "--now", str(NOW), "--out", str(base / "report.json")])
                self.assertEqual(code, 0, out.getvalue())
                report = json.loads((base / "report.json").read_text(encoding="utf-8"))
                self.assertEqual(report["verdict"], "consistent")
                self.assertEqual(report["deterministic"]["exact_rate"], 1.0)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
