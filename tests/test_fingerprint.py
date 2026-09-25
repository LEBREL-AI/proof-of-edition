"""watch.fingerprint: first-word distributions, token counts, receipts, the weekly secret set and the board."""
from __future__ import annotations

import base64
import json
import math
import tempfile
import time
import unittest
from pathlib import Path

from nacl.signing import SigningKey

from watch import fingerprint as fp
from watch.board import build_board, classify
from watch.client import Target, load_targets
from watch.publish import post_signed, reveal_documents
from watch.sign import key_id, public_key_hex, sign, verify

FIXTURES = Path(__file__).parent / "fixtures" / "route"


def dist(**probs: float) -> dict[str, float]:
    return {name.encode().hex(): math.log(p) for name, p in probs.items()}


def target(name: str, model: str = "m", **kw) -> Target:
    return Target(name=name, model=model, base_url="https://example.invalid/v1", upstream_model="up", fingerprint=True, **kw)


class FakeResponse:
    def __init__(self, body: bytes, headers: dict[str, str] | None = None, status: int = 200):
        self.body, self.headers, self.status = body, headers or {}, status

    def read(self) -> bytes:
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeOpener:
    def __init__(self, responder):
        self.responder, self.requests = responder, []

    def open(self, request, timeout=None):
        self.requests.append(request)
        return self.responder(request)


class DistributionTests(unittest.TestCase):
    def test_first_position_by_bytes_ignores_degenerate_alternatives(self):
        document = {"choices": [{"logprobs": {"content": [{"token": "A", "logprob": -0.1, "top_logprobs": [
            {"token": "A", "logprob": -0.1, "bytes": [65]}, {"token": "�", "logprob": -2.0, "bytes": [240, 159]},
            {"token": "�", "logprob": -3.0, "bytes": [240, 160]}, {"token": "B", "logprob": -9999.0, "bytes": [66]}]}]}}]}
        first = fp.first_distribution(document)
        self.assertEqual(set(first), {"41", "f09f", "f0a0"}, "byte keys keep distinct tokens that print alike; -9999 is dropped")
        self.assertIsNone(fp.first_distribution({"choices": [{"message": {"content": "x"}}]}))

    def test_log_gap(self):
        p = dist(a=0.6, b=0.3, c=0.1)
        self.assertEqual(fp.log_gap(p, p), 0.0)
        self.assertAlmostEqual(fp.log_gap(p, dist(a=0.3, b=0.6, c=0.1)), (math.log(2) * 2) / 3, places=6)
        self.assertEqual(fp.log_gap(dist(a=0.9), dist(b=0.9)), fp.MAX_GAP, "no shared token above the floor")
        self.assertIsNone(fp.log_gap({}, {}))
        self.assertAlmostEqual(fp.log_gap(dist(a=0.995, z=0.005), dist(a=0.995, y=0.005)), 0.0, msg="tokens under 1% are not compared")


def summary(prompts: dict[str, tuple[dict[str, float] | None, int | None]], **extra) -> dict:
    return {"model": "m", "prompts": {pid: {"first": first, "prompt_tokens": tokens, "error": None} for pid, (first, tokens) in prompts.items()}, **extra}


class CompareTests(unittest.TestCase):
    def base(self, n: int = 20, shift: float = 0.0, offset=lambda i: 0, logprobs: bool = True):
        out = {}
        for i in range(n):
            a = 0.5 + 0.02 * (i % 5)
            b = 1 - a
            d = dist(x=a, y=b)
            if shift:
                d = {k: v + (shift if j == 0 else -shift) for j, (k, v) in enumerate(d.items())}
            out[f"p{i}"] = (d if logprobs else None, 30 + i + offset(i))
        return out

    def test_match_differs_mismatch_by_similarity(self):
        anchor = summary(self.base())
        self.assertEqual(fp.compare(anchor, summary(self.base(shift=0.002)))["verdict"], "match")
        differs = fp.compare(anchor, summary(self.base(shift=0.05)))
        self.assertEqual(differs["verdict"], "differs", differs)
        mismatch = fp.compare(anchor, summary(self.base(shift=0.3)))
        self.assertEqual(mismatch["verdict"], "mismatch")
        self.assertLess(mismatch["similarity_percent"], fp.MISMATCH_PERCENT)

    def test_token_counts(self):
        anchor = summary(self.base())
        same = fp.compare(anchor, summary(self.base()))
        self.assertEqual(same["tokens"], {"offsets": [0], "same_count": True, "same_tokenizer": True})
        template = fp.compare(anchor, summary(self.base(offset=lambda i: 7)))
        self.assertEqual(template["tokens"], {"offsets": [7], "same_count": False, "same_tokenizer": True}, "a constant offset: another template")
        tokenizer = fp.compare(anchor, summary(self.base(offset=lambda i: i % 3)))
        self.assertEqual(tokenizer["verdict"], "mismatch", "an offset that varies is another tokenizer, whatever the probabilities say")

    def test_no_logprobs_and_insufficient(self):
        anchor = summary(self.base())
        self.assertEqual(fp.compare(anchor, summary(self.base(logprobs=False)))["verdict"], "no_logprobs")
        self.assertEqual(fp.compare(summary(self.base(n=5)), summary(self.base(n=5)))["verdict"], "insufficient")


class SecretSetTests(unittest.TestCase):
    def test_weekly_secret_set_is_deterministic_committed_and_distinct(self):
        seed, pool = b"s" * 32, fp.default_secret_pool()
        week = fp.iso_week(1_790_215_857)
        self.assertEqual(week, "2026-W39")
        first = fp.secret_prompts(seed, week, pool)
        self.assertEqual(first, fp.secret_prompts(seed, week, pool))
        self.assertEqual(len(first), fp.SECRET_PER_WEEK)
        self.assertNotEqual(first, fp.secret_prompts(seed, "2026-W40", pool))
        self.assertNotEqual(first, fp.secret_prompts(b"t" * 32, week, pool))
        items, committed = fp.prompt_set(1_790_215_857, seed=seed, pool=pool)
        self.assertEqual(len(items), len(fp.PUBLIC_PROMPTS) + fp.SECRET_PER_WEEK)
        document = json.dumps({"week": week, "pool": fp.pool_id(pool), "prompts": first}, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        import hashlib
        self.assertEqual(committed["sha256"], hashlib.sha256(document).hexdigest(), "the reveal document hashes to the commitment")
        other_pool_items, _ = fp.prompt_set(1_790_215_857, seed=seed, pool=pool[:-1])
        self.assertFalse({i for i, _ in items if i.startswith("sec-")} & {i for i, _ in other_pool_items if i.startswith("sec-")}, "two pools never share a prompt id")
        self.assertIsNone(fp.prompt_set(0, seed=None, pool=None)[1])


class ProbeTests(unittest.TestCase):
    def test_request_shape_and_extraction(self):
        seen = {}

        def respond(request):
            seen["body"] = json.loads(request.data)
            seen["ua"] = request.get_header("User-agent")
            return FakeResponse(json.dumps({"model": "served", "system_fingerprint": "fp1", "usage": {"prompt_tokens": 11},
                                            "choices": [{"message": {"content": "A"}, "logprobs": {"content": [{"token": "A", "logprob": -0.1,
                                                        "top_logprobs": [{"token": "A", "logprob": -0.1, "bytes": [65]}]}]}}]}).encode())
        t = target("x", body={"thinking": {"type": "disabled"}, "temperature": 0})
        result = fp.probe_once(t, "pub-00", "hello", opener=FakeOpener(respond))
        self.assertEqual(seen["body"]["max_tokens"], 1)
        self.assertEqual(seen["body"]["top_logprobs"], fp.TOP_LOGPROBS)
        self.assertNotIn("temperature", seen["body"], "the default temperature: temperature 0 degenerates the distribution")
        self.assertEqual(seen["body"]["thinking"], {"type": "disabled"})
        self.assertTrue(seen["ua"].startswith("proof-of-edition-watch"))
        self.assertEqual((result.prompt_tokens, result.system_fingerprint, result.served_model), (11, "fp1", "served"))
        self.assertEqual(result.first, {"41": -0.1})

    def test_route_receipts_verify_against_the_manifest(self):
        manifest = (FIXTURES / "manifest.json").read_bytes()
        receipt = (FIXTURES / "receipt.json").read_bytes()
        request_body = (FIXTURES / "request.json").read_bytes()
        answer = (FIXTURES / "response.txt").read_text()
        public = (FIXTURES / "public_key.txt").read_text().strip()
        t = target("lebrel/route", route_model_id="deepseek/deepseek-v4.1-flash", receipt_public_key=public)
        issued = json.loads(manifest)["payload"]["issued_at"]
        check = fp.route_receipt_checker(t, opener=FakeOpener(lambda request: FakeResponse(manifest)), now=issued + 60)
        header = base64.b64encode(receipt).decode()
        self.assertEqual(check(request_body, answer, header), "verified")
        self.assertTrue(check(request_body + b" ", answer, header).startswith("failed"), "another request body does not verify")
        self.assertTrue(check(request_body, answer, "bm90IGpzb24=").startswith("failed"))


class RunAndBoardTests(unittest.TestCase):
    def fake_prober(self, shift_for: dict[str, float], tokens_for: dict[str, int] | None = None, receipt: str | None = None):
        def prober(t, prompt_id, text, timeout=None, receipt_check=None):
            i = int(prompt_id.split("-")[-1])
            a = 0.4 + 0.01 * (i % 20)
            d = dist(x=a, y=1 - a)
            shift = shift_for.get(t.name, 0.0)
            d = {k: v + (shift if j == 0 else -shift) for j, (k, v) in enumerate(d.items())}
            return fp.ProbeResult(prompt_id, 200, d, 20 + len(text) % 7 + (tokens_for or {}).get(t.name, 0), "x", "fp1", "served", None, 0.2, None,
                                  receipt if t.receipt_public_key else None)
        return prober

    def test_run_writes_private_prompts_and_public_summary_and_board_uses_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            targets = [
                target("lab", fingerprint_anchor=True),
                target("route", route_model_id="lab/model", receipt_public_key="ab" * 32),
                target("fp8-host", declared_quantization="fp8"),
                target("near-host"),
                Target(name="battery-only", model="m", base_url="https://example.invalid/v1", upstream_model="up"),
            ]
            base = 1_790_215_000.0
            clock = iter([base + h * 3600 + m for h in range(3) for m in (0.0, 60.0)])
            # another implementation far from the lab (0.6 nats on every word) and one close to it (0.1 nats)
            prober = self.fake_prober({"fp8-host": 0.6, "near-host": 0.1}, receipt="verified")
            for _ in range(2):
                run_dir = fp.run(targets, root / "fp", seed=b"k" * 32, pool=None, prober=prober, clock=lambda: next(clock), log=lambda line: None)
            meta = json.loads((run_dir / "run.json").read_text())
            self.assertEqual(meta["targets"], ["lab", "route", "fp8-host", "near-host"], "only fingerprinted targets run")
            self.assertEqual(meta["commitment"]["week"], "2026-W39")
            pid = fp.pool_id(fp.default_secret_pool())
            self.assertIn(f"sec-2026W39-{pid}-00", json.loads((run_dir / "prompts.json").read_text()), "secret ids carry their week and pool")
            self.assertNotIn("Name", (run_dir / "summary.json").read_text().split('"prompts"')[0], "summary keeps ids, not texts")
            early = build_board(root / "runs", fingerprints_dir=root / "fp", now=base + 3700.0)
            early_entries = {e["target"]: e for e in early["entries"]}
            self.assertEqual(early_entries["fp8-host"]["fingerprint"]["verdict"], "mismatch", "one call is not identical to the lab")
            self.assertEqual(early_entries["fp8-host"]["fingerprint"]["margin"]["verdict"], "insufficient")
            self.assertEqual(early_entries["fp8-host"]["status"], "insufficient", "two hours are not a cloud yet")
            run_dir = fp.run(targets, root / "fp", seed=b"k" * 32, pool=None, prober=prober, clock=lambda: next(clock), log=lambda line: None)
            board = build_board(root / "runs", fingerprints_dir=root / "fp", now=base + 2 * 3600 + 100.0)
            entries = {e["target"]: e for e in board["entries"]}
            self.assertEqual(entries["lab"]["fingerprint"]["verdict"], "anchor")
            self.assertEqual(entries["lab"]["fingerprint"]["drift"]["verdict"], "match")
            self.assertEqual(entries["lab"]["status"], "consistent")
            self.assertEqual(entries["lab"]["fingerprint"]["regimes"]["median"], 1, "the fake lab always answers the same numbers")
            self.assertEqual([h["hour_utc"] for h in entries["lab"]["fingerprint"]["hours"]], [int(time.strftime("%H", time.gmtime(base + h * 3600))) for h in range(3)])
            self.assertEqual(entries["lab"]["fingerprint"]["hours"][-1]["agreement_percent"], 100.0)
            self.assertEqual(entries["lab"]["fingerprint"]["hours"][-1]["pass_agreement_percent"], 100.0)
            self.assertIsNone(entries["lab"]["fingerprint"]["hours"][0]["agreement_percent"], "the first hour has nothing before it")
            self.assertEqual(entries["route"]["fingerprint"]["verdict"], "match")
            self.assertEqual(entries["route"]["fingerprint"]["receipts"]["verified"], len(fp.PUBLIC_PROMPTS) + fp.SECRET_PER_WEEK)
            self.assertEqual(entries["route"]["route_model_id"], "lab/model")
            self.assertEqual(entries["route"]["status"], "consistent")
            self.assertEqual(entries["fp8-host"]["fingerprint"]["verdict"], "mismatch")
            self.assertEqual(entries["fp8-host"]["fingerprint"]["margin"]["verdict"], "outside_margin")
            self.assertAlmostEqual(entries["fp8-host"]["fingerprint"]["margin"]["median_mean_gap"], 0.6, places=3)
            self.assertEqual(entries["fp8-host"]["status"], "divergent", "a host outside the margin is divergent")
            self.assertEqual(entries["fp8-host"]["declared_quantization"], "fp8")
            self.assertEqual(entries["near-host"]["fingerprint"]["verdict"], "differs", "not identical to the lab's own answers")
            self.assertEqual(entries["near-host"]["fingerprint"]["margin"]["verdict"], "within_margin")
            self.assertEqual(entries["near-host"]["status"], "consistent", "another implementation inside the margin is consistent")
            self.assertEqual(board["fingerprint_commitment"]["week"], "2026-W39")
            self.assertEqual(board["fingerprint_thresholds"]["match_percent"], fp.MATCH_PERCENT)
            self.assertEqual(board["fingerprint_thresholds"]["margin_within_nats"], fp.MARGIN_WITHIN_NATS)
            self.assertEqual(board["references"], {})
            stale = build_board(root / "runs", fingerprints_dir=root / "fp", now=base + 2 * 3600 + 100.0 + 4 * 3600)
            self.assertEqual({e["target"]: e["status"] for e in stale["entries"]}["route"], "insufficient", "an old check speaks for nothing")
            # the lab's API against Lebrel's own run of the published weights
            reference_dir = root / "reference"
            reference_dir.mkdir()
            for verdict, gap, expected in (("within_margin", 0.2, "consistent"), ("outside_margin", 0.9, "divergent")):
                (reference_dir / "lab.json").write_text(json.dumps({"version": 1, "model": "m", "run_id": "ref-1", "measured_at": int(base),
                                                                    "checkpoint": {"repo": "lab/model", "revision": "abc"}, "engine": "vllm 0.30.0", "gpu": "B300:2",
                                                                    "api_vs_reference": {"verdict": verdict, "median_mean_gap": gap, "within_2se": 0.6, "passes": [10, 8]}}))
                with_reference = build_board(root / "runs", fingerprints_dir=root / "fp", reference_dir=reference_dir, now=base + 2 * 3600 + 100.0)
                lab = {e["target"]: e for e in with_reference["entries"]}["lab"]
                self.assertEqual(lab["fingerprint"]["reference"]["verdict"], verdict)
                self.assertEqual(lab["fingerprint"]["reference"]["checkpoint"]["revision"], "abc")
                self.assertEqual(lab["status"], expected)
                self.assertEqual(with_reference["references"]["m"]["run_id"], "ref-1")

    def test_failed_receipt_is_divergent_and_severity_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            targets = [target("lab", fingerprint_anchor=True), target("route", route_model_id="lab/model", receipt_public_key="ab" * 32)]
            fp.run(targets, root / "fp", seed=None, pool=None, prober=self.fake_prober({}, receipt="failed: route receipt signature invalid"),
                   clock=lambda: 1_790_215_000.0, log=lambda line: None)
            board = build_board(root / "runs", fingerprints_dir=root / "fp", now=1_790_215_100.0)
            route = {e["target"]: e for e in board["entries"]}["route"]
            self.assertEqual(route["fingerprint"]["verdict"], "receipt_failed")
            self.assertEqual(route["status"], "divergent")
        self.assertEqual(classify({"counts": {"total": 40, "errors": 0}, "drift": {"verdict": "agree"}, "fingerprint": {"verdict": "differs"}}), "suspect")
        self.assertEqual(classify({"counts": {"total": 40, "errors": 0}, "drift": {"verdict": "disagree"}, "fingerprint": {"verdict": "match"}}), "drift")


class SigningAndPublishTests(unittest.TestCase):
    seed = "11" * 32

    def test_sign_verify(self):
        signature, kid = sign(b"board", self.seed)
        public = public_key_hex(self.seed)
        self.assertEqual(kid, key_id(public))
        self.assertTrue(verify(b"board", signature, public))
        self.assertFalse(verify(b"board!", signature, public))
        self.assertFalse(verify(b"board", signature, bytes(SigningKey.generate().verify_key).hex()))

    def test_post_signed_sends_exact_bytes_and_signature(self):
        seen = {}

        def respond(request):
            seen.update(data=request.data, signature=request.get_header("X-lebrel-watch-signature"), method=request.get_method())
            return FakeResponse(b'{"run_id":"r1","bytes":5}')
        answer = post_signed("https://models.example/v1/watch", b"board", self.seed, opener=FakeOpener(respond))
        self.assertEqual(answer["run_id"], "r1")
        self.assertEqual(seen["method"], "POST")
        self.assertEqual(seen["data"], b"board")
        self.assertTrue(verify(b"board", seen["signature"], public_key_hex(self.seed)))

    def test_reveal_only_finished_weeks_whose_commitment_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            t = [target("lab", fingerprint_anchor=True)]
            prober = RunAndBoardTests().fake_prober({})
            fp.run(t, root, seed=b"k" * 32, pool=None, prober=prober, clock=lambda: 1_790_215_000.0, log=lambda line: None)  # 2026-W39
            fp.run(t, root, seed=b"k" * 32, pool=None, prober=prober, clock=lambda: 1_790_215_000.0 + 7 * 86400, log=lambda line: None)  # W40
            documents = reveal_documents(root, "2026-W40")
            set_id = f"2026-W39-{fp.pool_id(fp.default_secret_pool())}"
            self.assertEqual(list(documents), [set_id], "the running week stays secret")
            revealed = json.loads(documents[set_id])
            self.assertEqual(len(revealed["prompts"]), fp.SECRET_PER_WEEK)
            run39 = sorted(root.iterdir())[0]
            prompts = json.loads((run39 / "prompts.json").read_text())
            prompts[f"sec-2026W39-{fp.pool_id(fp.default_secret_pool())}-00"] = "tampered"
            (run39 / "prompts.json").write_text(json.dumps(prompts))
            self.assertEqual(reveal_documents(root, "2026-W40"), {}, "a set that does not match its commitment is never revealed")


class TargetValidationTests(unittest.TestCase):
    def test_one_anchor_per_model_and_pinned_key_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.json"
            base = {"model": "m", "base_url": "https://x/v1", "upstream_model": "u", "fingerprint": True}
            path.write_text(json.dumps([{"name": "a", "fingerprint_anchor": True, **base}, {"name": "b", "fingerprint_anchor": True, **base}]))
            with self.assertRaisesRegex(ValueError, "already has the anchor"):
                load_targets(str(path))
            path.write_text(json.dumps([{"name": "a", "receipt_public_key": "XYZ", "route_model_id": "a/b", **base}]))
            with self.assertRaisesRegex(ValueError, "64 lowercase hex"):
                load_targets(str(path))



class HistoryTests(unittest.TestCase):
    def test_a_lab_with_two_variants_still_matches_its_own_route_and_not_another_precision(self):
        variant_a = {f"p{i}": {"first": dist(x=0.7, y=0.3), "prompt_tokens": 10, "error": None} for i in range(20)}
        variant_b = {f"p{i}": {"first": dist(x=0.3, y=0.7), "prompt_tokens": 10, "error": None} for i in range(20)}
        # the route hit variant A on half of the prompts and variant B on the rest
        route = {"prompts": {f"p{i}": (variant_a if i % 2 else variant_b)[f"p{i}"] for i in range(20)}}
        other_precision = {"prompts": {f"p{i}": {"first": dist(x=0.5, y=0.5), "prompt_tokens": 10, "error": None} for i in range(20)}}
        self.assertEqual(fp.compare([variant_a], route)["verdict"], "mismatch", "against one call only, a two-variant lab looks unlike itself")
        both = fp.compare([variant_a, variant_b], route)
        self.assertEqual((both["verdict"], both["similarity_percent"]), ("match", 100.0))
        self.assertEqual(fp.compare([variant_a, variant_b], other_precision)["verdict"], "mismatch")

    def test_anchor_is_probed_twice_and_the_reference_is_the_recent_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = {"lab": 0}

            def prober(t, prompt_id, text, timeout=None, receipt_check=None):
                if t.name == "lab":
                    calls["lab"] += 1
                i = int(prompt_id.split("-")[-1])
                return fp.ProbeResult(prompt_id, 200, dist(x=0.6 + 0.01 * (i % 10), y=0.4 - 0.01 * (i % 10)), 12, "x", None, None, None, 0.1, None)
            targets = [target("lab", fingerprint_anchor=True), target("route")]
            clock = iter([1_790_000_000.0, 1_790_000_010.0, 1_790_090_000.0, 1_790_090_010.0])
            for _ in range(2):
                fp.run(targets, root, seed=None, pool=None, prober=prober, clock=lambda: next(clock), log=lambda line: None)
            self.assertEqual(calls["lab"], 2 * 2 * len(fp.PUBLIC_PROMPTS), "two passes of the anchor per run")
            from watch.board import load_run, list_runs
            runs = [load_run(p) for p in list_runs(root)]
            sections = fp.fingerprint_sections(runs)
            self.assertEqual(sections["route"]["history_passes"], 2, "the first run is 25 hours old: outside the reference window")
            self.assertEqual(sections["lab"].get("drift"), None, "nothing within the window to compare the anchor with")



class InformativeTests(unittest.TestCase):
    def test_prompts_the_lab_is_sure_of_do_not_dilute_the_verdict(self):
        sure = {f"s{i}": {"first": dist(yes=0.995, no=0.005), "prompt_tokens": 9, "error": None} for i in range(40)}
        open_ended = {f"o{i}": {"first": dist(red=0.5, blue=0.3, green=0.2), "prompt_tokens": 9, "error": None} for i in range(16)}
        other_precision = {f"o{i}": {"first": dist(red=0.3, blue=0.5, green=0.2), "prompt_tokens": 9, "error": None} for i in range(16)}
        anchor = {"prompts": {**sure, **open_ended}}
        host = {"prompts": {**sure, **other_precision}}
        verdict = fp.compare(anchor, host)
        self.assertEqual(verdict["compared"], 16, "only the prompts where the lab hesitates are compared")
        self.assertEqual(verdict["verdict"], "mismatch", verdict)



class AnchorDriftTests(unittest.TestCase):
    def test_a_lab_that_alternates_variants_is_not_drifting(self):
        a = {f"p{i}": {"first": dist(x=0.6, y=0.4), "prompt_tokens": 9, "error": None} for i in range(20)}
        b = {f"p{i}": {"first": dist(x=0.3, y=0.7), "prompt_tokens": 9, "error": None} for i in range(20)}
        earlier = [a, b]
        now_alternating = {"prompts": b, "passes": [a]}  # this hour's first pass hit variant B, the second variant A
        self.assertEqual(fp.compare(earlier, now_alternating)["verdict"], "match")
        now_new = {"prompts": {f"p{i}": {"first": dist(x=0.9, z=0.1), "prompt_tokens": 9, "error": None} for i in range(20)}, "passes": []}
        self.assertEqual(fp.compare(earlier, now_new)["verdict"], "mismatch", "distributions never seen before are a real change")


if __name__ == "__main__":
    unittest.main()


class CloudTests(unittest.TestCase):
    def passes(self, shift: float, n: int, noise: float = 0.0):
        import random
        rng = random.Random(3)
        out = []
        for _ in range(n):
            entry = {}
            for i in range(20):
                a = 0.4 + 0.01 * i
                d = dist(x=a, y=1 - a)
                d = {k: v + (shift if j == 0 else -shift) + rng.gauss(0, noise) for j, (k, v) in enumerate(d.items())}
                entry[f"pub-{i:02d}"] = {"first": d, "prompt_tokens": 8}
            out.append(entry)
        return out

    def test_cloud_reads_the_margin(self):
        same = fp.cloud(self.passes(0.0, 4, 0.02), self.passes(0.0, 4, 0.02))
        self.assertEqual(same["verdict"], "within_margin")
        self.assertGreaterEqual(same["within_2se"], 0.8)
        far = fp.cloud(self.passes(0.0, 4, 0.02), self.passes(0.6, 4, 0.02))
        self.assertEqual(far["verdict"], "outside_margin")
        self.assertAlmostEqual(far["median_mean_gap"], 0.6, delta=0.05)
        self.assertEqual(fp.cloud(self.passes(0.0, 4), self.passes(0.0, 2))["verdict"], "insufficient", "two passes are not a cloud")
        self.assertEqual(fp.margin_verdict(0.35, 30, (4, 4)), "at_margin")

    def test_regimes_count_distinct_distributions(self):
        steady = self.passes(0.0, 5)
        self.assertEqual(fp.regimes(steady)["median"], 1)
        two = self.passes(0.0, 3) + self.passes(0.3, 3)
        self.assertEqual(fp.regimes(two)["median"], 2)
        self.assertEqual(fp.regimes(two)["passes"], 6)
