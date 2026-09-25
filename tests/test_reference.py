"""The reference tooling: token bytes, temperature rescaling and the fit, and the summary shape compare() needs."""
import math
import os
from pathlib import Path

import pytest

from watch import fingerprint
from watch.reference import analyze, prepare

CACHE = prepare.CACHE / prepare.REPO.replace("/", "__") / prepare.REVISION


def _tokens():
    if not (CACHE / "tokenizer.json").exists():
        pytest.skip("the edition's tokenizer is not cached (network needed once)")
    return analyze.TokenBytes(CACHE / "tokenizer.json")


def test_token_bytes_round_trip_text():
    tokens = _tokens()
    text = "Name one colour, e.g. 'rojo' — 17×23 日本語"
    ids = tokens.tokenizer.encode(text, add_special_tokens=False).ids
    joined = bytes.fromhex("".join(tokens.hex(i) for i in ids))
    assert joined.decode("utf-8") == text


def test_token_bytes_special_tokens_are_their_text():
    tokens = _tokens()
    bos = tokens.tokenizer.token_to_id("<｜begin▁of▁sentence｜>")
    assert bytes.fromhex(tokens.hex(bos)).decode("utf-8") == "<｜begin▁of▁sentence｜>"


def test_rescale_normalises_and_sharpens():
    top = [(1, math.log(0.6)), (2, math.log(0.3)), (3, math.log(0.1))]
    at_half = dict(analyze.rescale(top, 0.5))
    assert abs(sum(math.exp(v) for v in at_half.values()) - 1.0) < 1e-9
    assert at_half[1] > math.log(0.6)  # colder: the favourite gains
    assert dict(analyze.rescale(top, 1.0))[1] == pytest.approx(math.log(0.6))


def test_temperature_fit_recovers_the_api_temperature():
    class Hex:
        def hex(self, tid):
            return f"{tid:02x}"
    api_passes = []
    result = {"passes": [{"mode": "serial", "results": []}]}
    for pid in range(20):
        raw = [(t, math.log(w)) for t, w in ((1, 0.5), (2, 0.3), (3, 0.15), (4, 0.05))]
        result["passes"][0]["results"].append({"id": f"p{pid}", "top": raw})
    # the API applied T = 0.3 to the same logits
    scaled_pass = {}
    for pid in range(20):
        scaled = analyze.rescale(result["passes"][0]["results"][pid]["top"], 0.3)
        scaled_pass[f"p{pid}"] = {"first": {f"{t:02x}": lp for t, lp in scaled}, "prompt_tokens": 8}
    api_passes.append(scaled_pass)
    original = analyze.TokenBytes
    analyze.TokenBytes = lambda path: Hex()
    analyze.fetch_format = lambda: {"paths": {"tokenizer.json": Path("x")}}
    try:
        fit = analyze.fit_temperature(api_passes, result)
    finally:
        analyze.TokenBytes = original
    assert fit["temperature"] == pytest.approx(0.3, abs=0.01)
    assert fit["pairs"] > 0


def test_reference_summary_compares_with_itself():
    tokens = _tokens()
    ids = tokens.tokenizer.encode("red blue green yellow purple orange black white pink grey brown gold", add_special_tokens=False).ids[:20]
    top = [(tid, math.log(0.3) - 0.2 * i) for i, tid in enumerate(ids)]
    results = [{"id": f"pub-{i:02d}", "top": top, "prompt_tokens": 8, "latency_s": 0.1} for i in range(fingerprint.MIN_COMPARED + 1)]
    result = {"passes": [{"mode": "serial", "results": results}, {"mode": "serial", "results": results}, {"mode": "batched", "results": results}]}
    summary = analyze.reference_summary(result, tokens)
    assert len(summary["passes"]) == 2 and len(summary["prompts"]) == fingerprint.MIN_COMPARED + 1
    verdict = fingerprint.compare({"prompts": summary["prompts"]}, {"prompts": summary["passes"][0]})
    assert verdict["verdict"] == "match" and verdict["similarity_percent"] == 100.0


def test_mean_test_tells_the_same_cloud_from_an_offset_one():
    import random
    rng = random.Random(7)

    def cloud(offset: float, n: int):
        passes = []
        for _ in range(n):
            entry = {}
            for pid in range(16):
                a = 0.5 + rng.gauss(0, 0.03) + offset
                entry[f"p{pid}"] = {"first": {"aa": math.log(min(0.95, max(0.05, a))), "bb": math.log(min(0.95, max(0.05, 1 - a)))}}
            passes.append(entry)
        return passes
    same = analyze.mean_test(cloud(0.0, 6), cloud(0.0, 6))
    shifted = analyze.mean_test(cloud(0.0, 6), cloud(0.25, 6))
    assert same["tokens"] > 0 and same["within_2se"] >= 0.8
    assert shifted["within_2se"] < 0.5 and shifted["median_mean_gap"] > same["median_mean_gap"]
