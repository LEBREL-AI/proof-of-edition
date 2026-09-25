"""watch.sampled: the fingerprint by sampling for labs that return no token probabilities."""
from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from watch import fingerprint as fp
from watch.board import build_board, classify
from watch.client import Target
from watch.fingerprint import ProbeResult
from watch.sampled import compare_sampled, first_word


def target(name: str, model: str = "glm-flash", **kw) -> Target:
    return Target(name=name, model=model, base_url="https://example.invalid/v1", upstream_model="up", fingerprint=True, sampled=True, samples=6, **kw)


def passes_from(dist: dict[str, float], n: int, k: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for _ in range(k):
        entry = {}
        for pid in range(20):
            counts: dict[str, int] = {}
            for _ in range(n):
                word = rng.choices(list(dist), weights=list(dist.values()))[0]
                counts[word] = counts.get(word, 0) + 1
            entry[f"p{pid}"] = {"counts": counts, "n": n, "prompt_tokens": 10}
        out.append(entry)
    return out


class FirstWordTest(unittest.TestCase):
    def test_normalises_case_and_punctuation(self):
        self.assertEqual(first_word('  "Blue." is my pick'), "blue")
        self.assertEqual(first_word("¡Azúl!"), "azúl")
        self.assertEqual(first_word("\n\n**Dolphin**"), "dolphin")
        self.assertIsNone(first_word(""))
        self.assertIsNone(first_word(None))
        self.assertIsNone(first_word("..."))


class CompareSampledTest(unittest.TestCase):
    def test_same_distribution_matches_and_a_shift_mismatches(self):
        same = {"blue": 0.5, "red": 0.3, "green": 0.2}
        anchor = passes_from(same, 8, 12, 2)
        match = compare_sampled(anchor, passes_from(same, 8, 1, 3))
        self.assertEqual(match["verdict"], "sampled_match")
        self.assertGreater(match["similarity_percent"], 90)
        self.assertEqual(match["compared"], 20)
        shifted = compare_sampled(anchor, passes_from({"blue": 0.2, "red": 0.2, "green": 0.6}, 8, 1, 4))
        self.assertEqual(shifted["verdict"], "sampled_mismatch")
        self.assertLess(shifted["similarity_percent"], match["similarity_percent"])
        self.assertLess(shifted["p_value"], 0.002)

    def test_another_tokenizer_is_another_model_and_sure_prompts_are_not_counted(self):
        same = {"blue": 0.5, "red": 0.5}
        anchor = passes_from(same, 8, 4, 5)
        other = passes_from(same, 8, 1, 6)
        for index, (pid, result) in enumerate(other[0].items()):
            result["prompt_tokens"] = 10 + index % 3  # offsets that vary prompt to prompt
        self.assertEqual(compare_sampled(anchor, other)["verdict"], "sampled_mismatch")
        sure = compare_sampled(passes_from({"blue": 1.0}, 8, 4, 7), passes_from({"blue": 1.0}, 8, 1, 8))
        self.assertEqual(sure["verdict"], "insufficient", "a lab that never hesitates gives nothing to compare")


class SampledRunTest(unittest.TestCase):
    def fake_prober(self, words_for: dict[str, dict[str, float]], tokens_for: dict[str, int] | None = None):
        # The watch probes from several threads; one shared RNG would hand out draws in whatever order the threads
        # ran, so each (target, prompt, call) seeds its own generator and the run is the same in every environment.
        import threading
        calls: dict[tuple[str, str], int] = {}
        lock = threading.Lock()

        def prober(t, prompt_id, text, timeout=None, receipt_check=None):
            dist = words_for[t.name]
            with lock:
                index = calls[(t.name, prompt_id)] = calls.get((t.name, prompt_id), 0) + 1
            rng = random.Random(f"{t.name}|{prompt_id}|{index}")
            word = rng.choices(list(dist), weights=list(dist.values()))[0]
            return ProbeResult(prompt_id, 200, None, (tokens_for or {}).get(t.name, 10), f"{word.capitalize()}.", "fp", "up", None, 0.2, None)
        return prober

    def test_request_body_asks_for_words_not_probabilities(self):
        t = target("lab", overrides={"temperature": 1, "top_p": 0.95, "max_tokens": 2048}, body={"reasoning_effort": "low"}, sampled_max_tokens=96)
        body = fp._request_body(t, "Name a colour.")
        self.assertNotIn("logprobs", body)
        self.assertEqual((body["max_tokens"], body["temperature"], body["top_p"], body["reasoning_effort"]), (96, 1, 0.95, "low"))
        self.assertEqual(body["messages"][0]["content"], "Name a colour.")

    def test_run_board_and_cadence(self):
        same = {"blue": 0.5, "red": 0.3, "green": 0.2}
        other = {"blue": 0.1, "red": 0.1, "green": 0.8}
        targets = [
            target("lab", fingerprint_anchor=True),
            target("route", route_model_id="z-ai/glm-5.3-flash"),
            target("elsewhere"),
            target("slow-lab", model="kimi", fingerprint_anchor=True, every_hours=6),
            target("slow-route", model="kimi", every_hours=6, route_model_id="moonshotai/kimi-k3"),
        ]
        words = {"lab": same, "route": same, "elsewhere": other, "slow-lab": same, "slow-route": same}
        base = 1_790_000_000.0 - (1_790_000_000.0 % 86400)  # 00:00 UTC
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "runs").mkdir()
            prober = self.fake_prober(words)
            logs: list[str] = []
            fp.run(targets, root / "fp", seed=None, pool=None, prober=prober, clock=lambda: base, log=logs.append)
            fp.run(targets, root / "fp", seed=None, pool=None, prober=prober, clock=lambda: base + 3600, log=logs.append)
            meta = json.loads(sorted((root / "fp").iterdir())[-1].joinpath("run.json").read_text())
            self.assertEqual(meta["targets"], ["lab", "route", "elsewhere"], "the six-hourly targets rest at 01:00")
            self.assertTrue(any("not this hour" in line for line in logs))
            board = build_board(root / "runs", fingerprints_dir=root / "fp", now=base + 3700.0)
            entries = {e["target"]: e for e in board["entries"]}
            self.assertEqual(entries["route"]["fingerprint"]["method"], "sampled")
            self.assertEqual(entries["route"]["fingerprint"]["verdict"], "sampled_match")
            self.assertEqual(entries["route"]["status"], "consistent")
            self.assertEqual(entries["route"]["route_model_id"], "z-ai/glm-5.3-flash")
            self.assertEqual(entries["elsewhere"]["fingerprint"]["verdict"], "sampled_mismatch")
            self.assertEqual(entries["elsewhere"]["status"], "divergent")
            self.assertEqual(entries["lab"]["fingerprint"]["verdict"], "anchor")
            self.assertEqual(entries["lab"]["fingerprint"]["drift"]["verdict"], "sampled_match")
            self.assertIn("hours", entries["lab"]["fingerprint"])
            slow = entries["slow-route"]["fingerprint"]
            self.assertEqual((slow["verdict"], slow["cadence_hours"], slow.get("stale")), ("sampled_match", 6, None), "kept from its own run, not stale on its cadence")
            self.assertEqual(entries["slow-route"]["status"], "consistent")
            late = build_board(root / "runs", fingerprints_dir=root / "fp", now=base + 20 * 3600.0)
            late_entries = {e["target"]: e for e in late["entries"]}
            self.assertTrue(late_entries["slow-route"]["fingerprint"].get("stale"), "six hours of cadence, three of grace: stale after eighteen")


if __name__ == "__main__":
    unittest.main()
