"""watch.reference.analyze_sampled: the published weights, sampled, against the lab's API as the watch recorded it."""
from __future__ import annotations

import json
import random
from pathlib import Path

from proof_of_edition.watch.reference import analyze_sampled

WORDS = {"blue": 0.5, "red": 0.3, "green": 0.2}
OTHER = {"blue": 0.2, "red": 0.2, "green": 0.6}


def draw(dist: dict[str, float], k: int, rng: random.Random) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _ in range(k):
        word = rng.choices(list(dist), weights=list(dist.values()))[0]
        counts[word] = counts.get(word, 0) + 1
    return counts


# Seeds are fixed so the fixtures are ordinary draws: with seed 0 the two reference passes happen to differ at p = 0.01, a
# one-in-a-hundred fluke of identical distributions that would test the fixture's luck rather than the pipeline.
def reference_result(dist: dict[str, float], *, prompts: int = 12, n: int = 64, passes: int = 2, seed: int = 1) -> dict:
    rng = random.Random(seed)
    out_passes = []
    for _ in range(passes):
        results = []
        for i in range(prompts):
            answers = [{"first": word.capitalize() + ", certainly.", "reasoning": True, "finish": "stop"}
                       for word, count in draw(dist, n, rng).items() for _ in range(count)]
            results.append({"id": f"pub-{i:02d}", "prompt_tokens": 16, "answers": answers})
        out_passes.append({"label": "batched", "results": results, "errors": 0})
    return {"method": "sampled", "model": "glm-5.3-flash", "repo": "zai-org/GLM-5.3-Flash", "revision": "e" * 40, "gpu": "B300:2",
            "vllm_version": "0.30.0", "params": {"temperature": 1.0}, "n": n, "started_at": 1_790_400_000.0, "result_id": "ref1",
            "provenance": {"a.safetensors": {}}, "passes": out_passes}


def api_runs(root: Path, dist: dict[str, float], *, hours: int = 12, prompts: int = 12, samples: int = 8, seed: int = 101, tokens: int = 16) -> Path:
    rng = random.Random(seed)
    runs = root / "runs-fingerprint"
    for h in range(hours):
        started = 1_790_400_000.0 - 3600 * (h + 1)
        run_dir = runs / f"run{h:02d}"
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(json.dumps({"run_id": f"run{h:02d}", "started_at": started, "finished_at": started + 60}))
        per_prompt = {f"pub-{i:02d}": {"counts": draw(dist, samples, rng), "n": samples, "prompt_tokens": tokens} for i in range(prompts)}
        (run_dir / "summary.json").write_text(json.dumps({"zai/api-flash": {"model": "glm-5.3-flash", "sampled": True, "anchor": True, "prompts": per_prompt},
                                                          "lebrel/route-glm-flash": {"model": "glm-5.3-flash", "sampled": True, "prompts": per_prompt}}))
    return runs


def test_same_distribution_on_both_sides_is_a_sampled_match(tmp_path):
    runs = api_runs(tmp_path, WORDS)
    result = reference_result(WORDS)
    api = analyze_sampled.api_passes(runs, "zai/api-flash", since=1_790_400_000.0 - 24 * 3600, until=1_790_400_000.0 + 3600)
    assert len(api) == 12, "one pass per hourly run in the window"
    report = analyze_sampled.compare(result, api, permutations=200)
    assert report["api_vs_reference"]["verdict"] == "sampled_match"
    assert report["api_vs_reference"]["compared"] == 12 and report["api_vs_reference"]["tokens"]["same_count"]
    assert report["reference_vs_itself"]["verdict"] == "sampled_match", "two passes of the same weights agree"
    document = analyze_sampled.export_document(result, report, anchor="zai/api-flash")
    assert document["model"] == "glm-5.3-flash" and document["method"] == "sampled" and document["api_vs_reference"]["method"] == "sampled"
    assert document["api_vs_reference"]["verdict"] == "within_margin" and document["api_vs_reference"]["test_verdict"] == "sampled_match"
    assert document["api_vs_reference"]["margin"]["reference_spread_tv"] >= 0.01 and document["api_vs_reference"]["self_similarity_percent"] is not None
    assert document["checkpoint"] == {"repo": "zai-org/GLM-5.3-Flash", "revision": "e" * 40, "files": 1} and document["engine"] == "0.30.0"


def test_another_distribution_at_the_api_is_a_mismatch_and_another_token_count_is_named(tmp_path):
    runs = api_runs(tmp_path, OTHER)
    # a mismatch needs p < 0.2%, which only 1000 permutations can resolve (the smallest p is 1 / (permutations + 1))
    report = analyze_sampled.compare(reference_result(WORDS), analyze_sampled.api_passes(runs, "zai/api-flash", since=0, until=2e9), permutations=1000)
    assert report["api_vs_reference"]["verdict"] == "sampled_mismatch"
    assert report["api_vs_reference"]["similarity_percent"] < 90
    exported = analyze_sampled.export_document(reference_result(WORDS), report, anchor="zai/api-flash")
    assert exported["api_vs_reference"]["verdict"] == "outside_margin" and exported["api_vs_reference"]["test_verdict"] == "sampled_mismatch"
    offset_runs = api_runs(tmp_path / "offset", WORDS, tokens=21)
    shifted = analyze_sampled.compare(reference_result(WORDS), analyze_sampled.api_passes(offset_runs, "zai/api-flash", since=0, until=2e9), permutations=200)
    tokens = shifted["api_vs_reference"]["tokens"]
    assert tokens["offsets"] == [5] and tokens["same_tokenizer"] and not tokens["same_count"], "one constant offset: the same tokenizer, another prompt format"


def test_window_and_missing_sides(tmp_path):
    runs = api_runs(tmp_path, WORDS, hours=3)
    assert analyze_sampled.api_passes(runs, "zai/api-flash", since=1_790_400_000.0 - 2 * 3600 + 1, until=2e9) == [] or len(analyze_sampled.api_passes(runs, "zai/api-flash", since=1_790_400_000.0 - 2 * 3600 + 1, until=2e9)) == 1
    assert analyze_sampled.api_passes(runs, "moonshot/api", since=0, until=2e9) == [], "an anchor with no runs has no passes"
    report = analyze_sampled.compare({"passes": []}, [])
    assert report["api_vs_reference"]["verdict"] == "insufficient"


def test_cli_writes_the_report_and_the_board_document(tmp_path, capsys):
    runs = api_runs(tmp_path, WORDS)
    result_path = tmp_path / "ref.json"
    result_path.write_text(json.dumps(reference_result(WORDS)))
    code = analyze_sampled.main(["--result", str(result_path), "--runs", str(runs), "--anchor", "zai/api-flash", "--permutations", "100",
                                 "--out", str(tmp_path / "report.json"), "--export", str(tmp_path / "references")])
    assert code == 0
    exported = json.loads((tmp_path / "references" / "glm-5.3-flash.json").read_text())
    assert exported["api_vs_reference"]["verdict"] == "within_margin" and exported["api_vs_reference"]["test_verdict"] == "sampled_match" and exported["anchor"] == "zai/api-flash"
    assert "sampled_match" in capsys.readouterr().out


def test_margin_judges_the_api_against_the_reference_spread_and_publishes_every_number():
    from proof_of_edition.watch import sampled
    same_tokens = {"offsets": [0], "same_count": True, "same_tokenizer": True}
    own = {"tv_observed": 0.1734, "tv_null_mean": 0.1622, "verdict": "sampled_match"}  # the reference's own spread: 0.0112
    api = {"tv_observed": 0.1144, "tv_null_mean": 0.1009, "verdict": "sampled_differs", "tokens": same_tokens}  # excess 0.0135
    judged = sampled.margin_verdict(api, own)
    assert judged["verdict"] == "within_margin" and judged["api_excess_tv"] == 0.0135 and judged["reference_spread_tv"] == 0.0112
    assert sampled.margin_verdict({**api, "tv_observed": 0.1409}, own)["verdict"] == "at_margin", "up to four times the spread"
    assert sampled.margin_verdict({**api, "tv_observed": 0.16}, own)["verdict"] == "outside_margin"
    assert sampled.margin_verdict({**api, "tokens": {"offsets": [0, 3], "same_count": False, "same_tokenizer": False}}, own)["verdict"] == "outside_margin"
    noisy = sampled.margin_verdict({**api, "tv_null_sd": 0.008}, {"tv_observed": 0.10, "tv_null_mean": 0.10})
    assert noisy["reference_spread_tv"] == 0.008 and noisy["verdict"] == "within_margin", "with no own spread, the comparison's sampling noise sets the scale"
    bare = sampled.margin_verdict(api, {"tv_observed": 0.10, "tv_null_mean": 0.10})
    assert bare["reference_spread_tv"] == 0.005 and bare["verdict"] == "at_margin", "the guard alone: a small distance is at the margin, never silently within"
    assert sampled.margin_verdict({"verdict": "insufficient"}, None)["verdict"] == "insufficient"
    assert sampled.margin_verdict(api, None)["reference_passes_compared"] is False
