import hashlib
import io
import json
import random
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from watch.battery import battery_id, build_battery, evaluate_canary, looks_like_refusal, Probe
from watch.board import build_board, classify, render_html
from watch.client import Exchange, Target, call, load_targets
from watch.compare import deterministic_agreement, two_sample_test
from watch.run import run

VOCAB_A = ["harbour", "tide", "lamp", "gull", "salt", "rope", "fog", "pier", "net", "keel"]
VOCAB_B = ["market", "cart", "bell", "coin", "bread", "stall", "clock", "gate", "wool", "cup"]


class FakeCaller:
    """Stands in for ``watch.client.call``. Each personality answers like a distinct model."""

    def __init__(self, personalities: dict[str, dict]):
        self.personalities = personalities
        self.calls = 0

    def __call__(self, target: Target, probe: str, sample: int, request: dict, *, timeout: float = 0.0, apply_overrides: bool = True) -> Exchange:
        self.calls += 1
        if target.overrides and apply_overrides:
            request = {**request, **target.overrides}
        persona = self.personalities[target.name]
        prompt = request["messages"][0]["content"]
        exchange = Exchange(target=target.name, model=target.model, probe=probe, sample=sample, request=request, sent_at=1.0,
                            latency_s=persona.get("latency", 0.5), status=200, finish_reason="stop",
                            usage={"prompt_tokens": 10, "completion_tokens": 20}, served_model=persona.get("served", "m"),
                            provider=persona.get("provider"))
        vocab = persona["vocab"]
        if probe.startswith("det-"):
            digest = hashlib.sha256((persona["det_seed"] + prompt).encode()).hexdigest()
            words = [vocab[int(c, 16) % len(vocab)] for c in digest[:30]]
            exchange.text = " ".join(words)
        elif probe.startswith("smp-"):
            generator = random.Random(hash((persona["det_seed"], prompt, sample)) & 0xFFFFFFFF)
            exchange.text = " ".join(generator.choice(vocab) for _ in range(40))
        elif probe == "can-system-leak":
            exchange.text = persona.get("leak", "NONE")
        elif probe.startswith("can-ref-"):
            exchange.text = "I'm sorry, but I can't help with that." if persona.get("refuses") else "Sure. Here is how you do it: step one, step two."
        elif probe == "can-max-tokens":
            if persona.get("ignores_max_tokens"):
                exchange.text = "\n".join(str(i) for i in range(1, 501))
                exchange.usage = {"prompt_tokens": 10, "completion_tokens": 1500}
            else:
                exchange.text = "\n".join(str(i) for i in range(1, 30))
                exchange.finish_reason = "length"
                exchange.usage = {"prompt_tokens": 10, "completion_tokens": 64}
        elif probe.startswith("can-needle-"):
            marker = "The secret code for this document is "
            start = prompt.index(marker) + len(marker)
            exchange.text = prompt[start:start + 10] if not persona.get("forgets") else "I could not find it"
        elif probe == "can-tools":
            if persona.get("no_tools"):
                exchange.status, exchange.error = 400, "HTTP 400: tools not supported"
            else:
                exchange.tool_calls = [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": json.dumps({"city": "Lisbon"})}}]
                exchange.text = None
        else:
            raise AssertionError(f"unexpected probe {probe}")
        return exchange


def target(name: str, model: str, role: str = "candidate") -> Target:
    return Target(name=name, model=model, base_url="https://example.invalid/v1", upstream_model=model, role=role)


PERSONAS = {
    "ref": {"vocab": VOCAB_A, "det_seed": "A", "served": "x-v1"},
    "same": {"vocab": VOCAB_A, "det_seed": "A", "served": "x-v1", "provider": "HostOne"},
    "other": {"vocab": VOCAB_B, "det_seed": "B", "served": "x-v2", "leak": "You are a helpful assistant. Never mention the vendor.", "refuses": True,
              "ignores_max_tokens": True, "forgets": True, "no_tools": True},
}


class BatteryTests(unittest.TestCase):
    def test_battery_is_stable_and_versioned(self):
        probes = build_battery(samples=3, needle_sizes=(200,))
        self.assertEqual(battery_id(probes), battery_id(build_battery(samples=3, needle_sizes=(200,))))
        self.assertNotEqual(battery_id(probes), battery_id(build_battery(samples=4, needle_sizes=(200,))))
        families = {p.family for p in probes}
        self.assertEqual(families, {"deterministic", "sampled", "canary"})
        needle = next(p for p in probes if p.id == "can-needle-200")
        self.assertIn(needle.expect["code"], needle.request["messages"][0]["content"])

    def test_refusal_heuristic(self):
        self.assertTrue(looks_like_refusal("I'm sorry, but I can't help with that."))
        self.assertTrue(looks_like_refusal("Lo siento, no puedo ayudar con eso."))
        self.assertTrue(looks_like_refusal("Sure", refusal_field="policy"))
        self.assertFalse(looks_like_refusal("Sure, here is how: run kill -9 <pid>."))
        self.assertFalse(looks_like_refusal(None))

    def test_canary_verdicts(self):
        leak = Probe("can-system-leak", "canary", {"messages": []}, {"kind": "system_leak"})
        self.assertEqual(evaluate_canary(leak, text="NONE.", finish_reason="stop", usage=None, tool_calls=None, status=200, error=None)["verdict"], "pass")
        self.assertEqual(evaluate_canary(leak, text="You are Venice, a helpful assistant", finish_reason="stop", usage=None, tool_calls=None, status=200, error=None)["verdict"], "fail")
        self.assertEqual(evaluate_canary(leak, text="I can’t provide the system message or hidden instructions.", finish_reason="stop", usage=None, tool_calls=None, status=200, error=None)["verdict"], "unclear")
        cap = Probe("can-max-tokens", "canary", {"messages": []}, {"kind": "max_tokens", "max_tokens": 64})
        self.assertEqual(evaluate_canary(cap, text="1\n2", finish_reason="length", usage={"completion_tokens": 64}, tool_calls=None, status=200, error=None)["verdict"], "pass")
        self.assertEqual(evaluate_canary(cap, text="... 500", finish_reason="stop", usage={"completion_tokens": 1500}, tool_calls=None, status=200, error=None)["verdict"], "fail")
        tools = Probe("can-tools", "canary", {"messages": []}, {"kind": "tools", "expect_city": "lisbon"})
        self.assertEqual(evaluate_canary(tools, text=None, finish_reason="stop", usage=None, tool_calls=None, status=400, error="HTTP 400: no tools")["verdict"], "unsupported")
        good = [{"function": {"name": "get_weather", "arguments": "{\"city\": \"Lisbon, Portugal\"}"}}]
        self.assertEqual(evaluate_canary(tools, text=None, finish_reason="tool_calls", usage=None, tool_calls=good, status=200, error=None)["verdict"], "pass")
        needle = Probe("can-needle-200", "canary", {"messages": []}, {"kind": "needle", "code": "ABC123XYZ9", "size_tokens": 200})
        self.assertEqual(evaluate_canary(needle, text="abc123xyz9", finish_reason="stop", usage=None, tool_calls=None, status=200, error=None)["verdict"], "pass")


class CompareTests(unittest.TestCase):
    def _samples(self, vocab, seed, prompts=10, k=6):
        generator = random.Random(seed)
        return {f"p{i}": [" ".join(generator.choice(vocab) for _ in range(40)) for _ in range(k)] for i in range(prompts)}

    def test_same_distribution_is_not_rejected(self):
        a = self._samples(VOCAB_A, 1)
        b = self._samples(VOCAB_A, 2)
        result = two_sample_test(a, b, permutations=300)
        self.assertEqual(result.prompts, 10)
        self.assertTrue(result.same_distribution)

    def test_different_distribution_is_rejected(self):
        a = self._samples(VOCAB_A, 1)
        b = self._samples(VOCAB_B, 2)
        result = two_sample_test(a, b, permutations=300)
        self.assertFalse(result.same_distribution)
        self.assertGreater(result.statistic, 0.03)

    def test_deterministic_agreement(self):
        a = {"d1": "alpha beta gamma", "d2": "one two three", "d3": "x"}
        same = deterministic_agreement(a, dict(a))
        self.assertEqual(same.exact_rate, 1.0)
        self.assertTrue(same.agrees)
        different = deterministic_agreement(a, {"d1": "zzz", "d2": "yyy", "d3": "www"})
        self.assertEqual(different.exact_rate, 0.0)
        self.assertFalse(different.agrees)
        self.assertIsNone(deterministic_agreement(a, {}).agrees)


class ClientTests(unittest.TestCase):
    class FakeResponse:
        def __init__(self, payload: bytes, status: int = 200):
            self.payload, self.status = payload, status

        def read(self):
            return self.payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FakeOpener:
        def __init__(self, responses):
            self.responses, self.requests = list(responses), []

        def open(self, request, timeout=None):
            self.requests.append(request)
            item = self.responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    def test_call_extracts_fields_and_merges_body(self):
        document = {"model": "deepseek-flash", "provider": "DeepInfra", "system_fingerprint": "fp_1",
                    "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "hello"}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1}}
        opener = self.FakeOpener([self.FakeResponse(json.dumps(document).encode())])
        t = Target(name="t", model="m", base_url="https://api.example/v1/", upstream_model="m-up", body={"provider": {"order": ["DeepInfra"]}})
        exchange = call(t, "det-01", 0, {"messages": [{"role": "user", "content": "hi"}], "temperature": 0}, opener=opener, sleep=lambda s: None)
        self.assertEqual(exchange.text, "hello")
        self.assertEqual(exchange.provider, "DeepInfra")
        self.assertEqual(exchange.served_model, "deepseek-flash")
        self.assertEqual(exchange.status, 200)
        sent = json.loads(opener.requests[0].data.decode())
        self.assertEqual(sent["model"], "m-up")
        self.assertEqual(sent["provider"], {"order": ["DeepInfra"]})
        self.assertFalse(sent["stream"])
        self.assertEqual(opener.requests[0].full_url, "https://api.example/v1/chat/completions")

    def test_call_retries_then_records_error(self):
        import urllib.error
        error = urllib.error.HTTPError("u", 503, "busy", {}, io.BytesIO(b"overloaded"))
        opener = self.FakeOpener([error, error, error])
        t = Target(name="t", model="m", base_url="https://api.example/v1", upstream_model="m")
        exchange = call(t, "det-01", 0, {"messages": []}, opener=opener, sleep=lambda s: None, retries=2)
        self.assertEqual(exchange.status, 503)
        self.assertEqual(exchange.attempts, 3)
        self.assertIn("HTTP 503", exchange.error)

    def test_call_retries_a_response_cut_mid_body_then_succeeds(self):
        import http.client
        document = {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "whole"}}]}
        opener = self.FakeOpener([http.client.IncompleteRead(b""), self.FakeResponse(json.dumps(document).encode())])
        t = Target(name="t", model="m", base_url="https://api.example/v1", upstream_model="m")
        exchange = call(t, "det-01", 0, {"messages": []}, opener=opener, sleep=lambda s: None, retries=2)
        self.assertEqual(exchange.text, "whole")
        self.assertEqual(exchange.attempts, 2)
        self.assertIsNone(exchange.error)
        # cut every time: the exchange records the error instead of aborting the whole run
        cut = self.FakeOpener([http.client.IncompleteRead(b""), http.client.RemoteDisconnected("gone"), http.client.IncompleteRead(b"")])
        failed = call(t, "det-01", 0, {"messages": []}, opener=cut, sleep=lambda s: None, retries=2)
        self.assertEqual(failed.attempts, 3)
        self.assertIn("IncompleteRead", failed.error)

    def test_load_targets_rejects_duplicates_and_bad_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.json"
            path.write_text(json.dumps({"targets": [
                {"name": "a", "model": "m", "base_url": "u", "upstream_model": "m"},
                {"name": "a", "model": "m", "base_url": "u", "upstream_model": "m"}]}))
            with self.assertRaises(ValueError):
                load_targets(str(path))
            path.write_text(json.dumps([{"name": "a", "model": "m", "base_url": "u", "upstream_model": "m", "role": "boss"}]))
            with self.assertRaises(ValueError):
                load_targets(str(path))


class RunAndBoardTests(unittest.TestCase):
    def _run(self, out: Path, caller: FakeCaller, targets, clock_value: float):
        log = io.StringIO()
        with redirect_stderr(log):
            return run(targets, out, samples=6, needle_sizes=(200,), concurrency=2, caller=caller, clock=lambda: clock_value)

    def test_run_records_everything_and_board_classifies(self):
        targets = [target("ref", "model-x", "reference"), target("same", "model-x"), target("other", "model-x")]
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            caller = FakeCaller(PERSONAS)
            first = self._run(runs, caller, targets, 1_800_000_000)
            summary = json.loads((first / "summary.json").read_text())
            self.assertEqual(set(summary), {"ref", "same", "other"})
            self.assertEqual(len(summary["same"]["sampled"]["smp-01"]), 6)
            self.assertEqual(len(summary["same"]["deterministic"]), 20)
            self.assertEqual(summary["other"]["canary_summary"]["system_leak"], "fail")
            self.assertEqual(summary["same"]["canary_summary"]["system_leak"], "pass")
            self.assertEqual(summary["other"]["canary_summary"]["over_refusal_count"], "6/6")
            self.assertEqual(summary["same"]["canary_summary"]["max_tokens"], "pass")
            self.assertEqual(summary["other"]["canary_summary"]["max_tokens"], "fail")
            self.assertEqual(summary["same"]["canary_summary"]["needle"], {"200": "pass"})
            self.assertEqual(summary["other"]["canary_summary"]["tools"], "unsupported")
            self.assertEqual(summary["same"]["providers"], {"HostOne": summary["same"]["counts"]["total"]})
            lines = (first / "exchanges.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), sum(summary[t]["counts"]["total"] for t in summary))
            board = build_board(runs, permutations=200, now=5)
            by_name = {e["target"]: e for e in board["entries"]}
            self.assertEqual(by_name["same"]["status"], "consistent")
            self.assertEqual(by_name["same"]["reference"]["verdict"], "agree")
            self.assertEqual(by_name["other"]["status"], "divergent")
            self.assertEqual(by_name["other"]["reference"]["verdict"], "disagree")
            self.assertEqual(by_name["ref"]["status"], "unverified")
            self.assertIsNone(board["previous_run_id"])
            html_text = render_html(board)
            self.assertIn("divergent", html_text)
            self.assertIn("What the words mean", html_text)

            # second run: "same" silently becomes another model -> drift against itself and divergent from the reference
            drifted = dict(PERSONAS)
            drifted["same"] = dict(PERSONAS["other"], provider="HostOne")
            second = self._run(runs, FakeCaller(drifted), targets, 1_800_100_000)
            self.assertNotEqual(first, second)
            board = build_board(runs, permutations=200, now=6)
            by_name = {e["target"]: e for e in board["entries"]}
            self.assertEqual(board["previous_run_id"], first.name)
            self.assertEqual(by_name["same"]["drift"]["verdict"], "disagree")
            self.assertEqual(by_name["same"]["status"], "divergent")  # reference disagreement outranks drift
            self.assertEqual(by_name["ref"]["drift"]["verdict"], "agree")
            self.assertEqual(by_name["ref"]["status"], "consistent")

    def test_consensus_without_reference(self):
        targets = [target("h1", "model-y"), target("h2", "model-y"), target("h3", "model-y")]
        personas = {"h1": PERSONAS["same"], "h2": PERSONAS["same"], "h3": PERSONAS["other"]}
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            self._run(runs, FakeCaller(personas), targets, 1_800_000_000)
            board = build_board(runs, permutations=200, now=7)
            by_name = {e["target"]: e for e in board["entries"]}
            self.assertEqual(by_name["h1"]["status"], "consistent")
            self.assertEqual(by_name["h3"]["status"], "divergent")
            self.assertEqual(by_name["h3"]["consensus"]["disagree"], 2)

    def test_skipped_target_without_key(self):
        t = Target(name="needs-key", model="m", base_url="u", upstream_model="m", api_key_env="WATCH_TEST_MISSING_KEY")
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            self._run(runs, FakeCaller({}), [t], 1_800_000_000)
            board = build_board(runs, permutations=50, now=8)
            self.assertEqual(board["entries"][0]["status"], "insufficient")
            self.assertIn("not set", board["entries"][0]["skipped"])

    def test_classify_edge_cases(self):
        self.assertEqual(classify({"counts": {"total": 3, "errors": 0}}), "insufficient")
        self.assertEqual(classify({"counts": {"total": 40, "errors": 30}}), "error")
        self.assertEqual(classify({"counts": {"total": 40, "errors": 0}}), "unverified")
        self.assertEqual(classify({"counts": {"total": 40, "errors": 0}, "consensus": {"peers": 1, "agree": 0, "disagree": 1}}), "disputed")
        self.assertEqual(classify({"counts": {"total": 40, "errors": 0}, "drift": {"verdict": "partial"}}), "suspect")


if __name__ == "__main__":
    unittest.main()


class ThresholdTests(unittest.TestCase):
    def test_per_target_thresholds_judge_a_noisy_lab(self):
        from watch.board import thresholds_for
        from watch.client import Target
        noisy = Target(name="lab", model="m", base_url="https://example.invalid/v1", upstream_model="m", thresholds={"min_exact_rate": 0.3, "min_prefix_agreement": 0.5})
        self.assertEqual(thresholds_for({"thresholds": noisy.thresholds}), {"min_exact_rate": 0.3, "min_prefix_agreement": 0.5})
        self.assertEqual(thresholds_for({}), {"min_exact_rate": 0.9, "min_prefix_agreement": 0.95})
        a = {f"d{i}": f"answer {i}" for i in range(20)}
        b = dict(a)
        for i in range(11):  # 9 of 20 exact, prefix well above 0.5
            b[f"d{i}"] = f"answer {i} but longer"
        strict = deterministic_agreement(a, b)
        loose = deterministic_agreement(a, b, min_exact_rate=0.3, min_prefix_agreement=0.5)
        self.assertFalse(strict.agrees)
        self.assertTrue(loose.agrees)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.json"
            path.write_text(json.dumps([{"name": "a", "model": "m", "base_url": "u", "upstream_model": "m", "thresholds": {"min_exact_rate": 2}}]))
            with self.assertRaises(ValueError):
                load_targets(str(path))


class OverrideTests(unittest.TestCase):
    def test_overrides_win_over_the_probe_and_the_body(self):
        from tests.test_watch import ClientTests as C
        document = {"model": "deepseek-flash", "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        opener = C.FakeOpener([C.FakeResponse(json.dumps(document).encode())])
        t = Target(name="t", model="m", base_url="https://api.example/v1", upstream_model="m", body={"thinking": {"type": "disabled"}}, overrides={"max_tokens": 2048})
        call(t, "det-01", 0, {"messages": [], "max_tokens": 160, "temperature": 0}, opener=opener, sleep=lambda s: None)
        sent = json.loads(opener.requests[0].data.decode())
        self.assertEqual(sent["max_tokens"], 2048)
        self.assertEqual(sent["thinking"], {"type": "disabled"})
        self.assertEqual(sent["temperature"], 0)

    def test_max_tokens_probe_keeps_its_cap_despite_overrides(self):
        from watch.run import run_target
        seen = []

        def caller(target, probe, sample, request, *, timeout=0.0, apply_overrides=True):
            body = {**request, **(target.overrides if target.overrides and apply_overrides else {})}
            seen.append((probe, body.get("max_tokens")))
            return Exchange(target=target.name, model=target.model, probe=probe, sample=sample, request=body, sent_at=0, latency_s=0, status=200, text="x", finish_reason="stop", usage={"completion_tokens": 1})

        t = Target(name="t", model="m", base_url="u", upstream_model="m", overrides={"max_tokens": 2048})
        run_target(t, build_battery(samples=1, needle_sizes=()), concurrency=1, timeout=1, caller=caller)
        caps = dict(seen)
        self.assertEqual(caps["can-max-tokens"], 64)
        self.assertEqual(caps["det-01"], 2048)
