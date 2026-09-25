"""Fingerprint by sampling: for labs whose APIs return no token probabilities.

Some labs answer without logprobs (Z.ai returns none, Moonshot refuses the parameter) and reason
before every answer, so the first-word distribution cannot be read from one call. It can still be
estimated: the same prompt is asked N times at the lab's own sampling settings and the first word of
each answer is counted. Two endpoints serving the same model draw those words from the same
distribution; the test below asks whether they do.

The statistic is the total variation distance between the two empirical distributions, averaged
over the informative prompts (those where the lab itself hesitates). Its null distribution comes from
permutations: under "same model" the answers of both sides are exchangeable within each prompt, so
labels are shuffled within every prompt and the mean distance recomputed. The p-value is the share
of permutations at least as far apart as the observed pair. Small samples put distance between two
identical distributions by chance alone, so the similarity published is debiased: the distance in
excess of the permutation mean, taken off 100%. Identical deployments sit near 100%; a different
model sits far below, whatever the sample size.

Verdicts: ``sampled_match`` (no evidence of a difference), ``sampled_differs`` (p below 2%: the next
run decides), ``sampled_mismatch`` (p below 0.2%, or another tokenizer). Coarser than the logprob
fingerprint and dearer per run, but it needs nothing from the lab beyond an answer.
"""
from __future__ import annotations

import math
import random
import statistics
import unicodedata
from typing import Any

MIN_SAMPLES_PER_SIDE = 4
MIN_PROMPTS = 8
DIFFERS_ALPHA = 0.02
MISMATCH_ALPHA = 0.002
PERMUTATIONS = 1000
INFORMATIVE_MAX_TOP = 0.9  # the same rule as the logprob fingerprint: a prompt tells models apart only if the lab hesitates

# The implementation margin for a reference by sampling. Two runs of the same weights on different serving stacks do not
# draw first words from exactly the same distribution (kernels and batch composition move it), so the lab's API is not
# asked to be exchangeable with Lebrel's run of the published weights; it is asked to sit no further from it than the
# reference's own passes sit from each other. That spread is measured, never assumed: the excess distance between two
# passes of the reference, or the sampling noise of the comparison itself (the standard deviation of the permutation
# distribution), whichever is larger, and never below a small guard. Within: up to twice that spread. At the margin: up
# to four times. Outside: beyond, or another tokenizer.
MARGIN_FACTOR_WITHIN = 2.0
MARGIN_FACTOR_OUTSIDE = 4.0
MARGIN_FLOOR_TV = 0.005

_STRIP = "\"'`“”‘’«»()[]{}<>.,;:!?¡¿-–—*_~"


def first_word(text: str | None) -> str | None:
    """The first word of an answer, normalised: punctuation and quotes trimmed, case folded, accents kept."""
    if not isinstance(text, str):
        return None
    for token in text.replace(" ", " ").split():
        word = token.strip(_STRIP)
        if word:
            return unicodedata.normalize("NFC", word).casefold()
    return None


def distribution(counts: dict[str, int]) -> dict[str, float] | None:
    """{word utf-8 hex: log frequency}: the shape the logprob fingerprint stores, so the board reads both alike."""
    total = sum(counts.values())
    if total <= 0:
        return None
    return {word.encode("utf-8").hex(): math.log(count / total) for word, count in counts.items() if count > 0}


def pooled(passes: list[dict[str, Any]], prompt_id: str) -> tuple[dict[str, int], int]:
    """Counts of first words for one prompt over every pass that answered it, and the number of answers pooled."""
    counts: dict[str, int] = {}
    n = 0
    for entry in passes:
        result = entry.get(prompt_id) if isinstance(entry, dict) else None
        if not isinstance(result, dict) or result.get("error") or not isinstance(result.get("counts"), dict):
            continue
        for word, count in result["counts"].items():
            if isinstance(count, int) and count > 0:
                counts[word] = counts.get(word, 0) + count
                n += count
    return counts, n


def total_variation(a: dict[str, int], n_a: int, b: dict[str, int], n_b: int) -> float:
    words = set(a) | set(b)
    return 0.5 * sum(abs(a.get(w, 0) / n_a - b.get(w, 0) / n_b) for w in words)


def _tv_of_lists(words_a: list[str], words_b: list[str]) -> float:
    ca: dict[str, int] = {}
    cb: dict[str, int] = {}
    for w in words_a:
        ca[w] = ca.get(w, 0) + 1
    for w in words_b:
        cb[w] = cb.get(w, 0) + 1
    return total_variation(ca, len(words_a), cb, len(words_b))


def permutation_test(pairs: list[tuple[dict[str, int], dict[str, int]]], *, permutations: int = PERMUTATIONS, seed: int = 0) -> dict[str, float]:
    """Observed mean distance over the prompts, its mean under label permutation, and the p-value."""
    observed = statistics.fmean(total_variation(a, sum(a.values()), b, sum(b.values())) for a, b in pairs)
    pools = []
    for a, b in pairs:
        words = [w for w, c in a.items() for _ in range(c)] + [w for w, c in b.items() for _ in range(c)]
        pools.append((words, sum(a.values())))
    rng = random.Random(seed)
    at_least = 0
    means: list[float] = []
    for _ in range(permutations):
        total = 0.0
        for words, n_a in pools:
            rng.shuffle(words)
            total += _tv_of_lists(words[:n_a], words[n_a:])
        mean = total / len(pools)
        means.append(mean)
        if mean >= observed - 1e-12:
            at_least += 1
    null_mean = statistics.fmean(means)
    null_sd = statistics.pstdev(means) if len(means) > 1 else 0.0
    return {"observed": observed, "null_mean": null_mean, "null_sd": null_sd, "p_value": (at_least + 1) / (permutations + 1)}


def excess_distance(comparison: dict[str, Any]) -> float | None:
    """How much further apart the two sides are than exchangeable samples would be: the distance the similarity is debiased by."""
    observed, null_mean = comparison.get("tv_observed"), comparison.get("tv_null_mean")
    if not isinstance(observed, (int, float)) or not isinstance(null_mean, (int, float)):
        return None
    return max(0.0, float(observed) - float(null_mean))


def margin_verdict(api_vs_reference: dict[str, Any], reference_vs_itself: dict[str, Any] | None) -> dict[str, Any]:
    """The published-weights verdict of a sampled reference: the lab's API against Lebrel's run of the weights, judged
    against the spread the reference produces between its own passes (see MARGIN_*). Every number is returned."""
    excess = excess_distance(api_vs_reference)
    own = excess_distance(reference_vs_itself) if reference_vs_itself else None
    noise = api_vs_reference.get("tv_null_sd")
    spread = max(MARGIN_FLOOR_TV, own if own is not None else 0.0, float(noise) if isinstance(noise, (int, float)) else 0.0)
    tokens = api_vs_reference.get("tokens")
    if api_vs_reference.get("verdict") == "insufficient" or excess is None:
        verdict = "insufficient"
    elif tokens and not tokens.get("same_tokenizer"):
        verdict = "outside_margin"  # another tokenizer is another model, whatever the words say
    elif excess <= MARGIN_FACTOR_WITHIN * spread:
        verdict = "within_margin"
    elif excess <= MARGIN_FACTOR_OUTSIDE * spread:
        verdict = "at_margin"
    else:
        verdict = "outside_margin"
    return {"verdict": verdict, "api_excess_tv": None if excess is None else round(excess, 4), "reference_spread_tv": round(spread, 4),
            "reference_own_excess_tv": None if own is None else round(own, 4), "sampling_noise_tv": None if not isinstance(noise, (int, float)) else round(float(noise), 4),
            "reference_passes_compared": reference_vs_itself is not None, "factor_within": MARGIN_FACTOR_WITHIN, "factor_outside": MARGIN_FACTOR_OUTSIDE,
            "floor_tv": MARGIN_FLOOR_TV}


def compare_sampled(anchor_passes: list[dict[str, Any]], other_passes: list[dict[str, Any]], *, permutations: int = PERMUTATIONS,
                    seed: int = 0, min_prompts: int = MIN_PROMPTS) -> dict[str, Any]:
    """The sampled fingerprint of ``other`` against the anchor's pooled answers, prompt by prompt."""
    pairs: list[tuple[dict[str, int], dict[str, int]]] = []
    offsets: set[int] = set()
    samples_anchor = samples_other = 0
    prompt_ids = {pid for entry in other_passes if isinstance(entry, dict) for pid in entry}
    for pid in sorted(prompt_ids):
        a, n_a = pooled(anchor_passes, pid)
        b, n_b = pooled(other_passes, pid)
        if n_a < MIN_SAMPLES_PER_SIDE or n_b < MIN_SAMPLES_PER_SIDE:
            continue
        newest = next((e[pid] for e in anchor_passes if isinstance(e.get(pid), dict) and not e[pid].get("error")), None)
        mine = next((e[pid] for e in other_passes if isinstance(e.get(pid), dict) and not e[pid].get("error")), None)
        if newest and mine and isinstance(newest.get("prompt_tokens"), int) and isinstance(mine.get("prompt_tokens"), int):
            offsets.add(mine["prompt_tokens"] - newest["prompt_tokens"])
        if max(a.values()) / n_a >= INFORMATIVE_MAX_TOP:
            continue  # the lab is sure of its first word here: nothing to learn
        pairs.append((a, b))
        samples_anchor += n_a
        samples_other += n_b
    tokens = {"offsets": sorted(offsets), "same_count": offsets == {0}, "same_tokenizer": len(offsets) == 1} if offsets else None
    result: dict[str, Any] = {"method": "sampled", "compared": len(pairs), "samples": {"anchor": samples_anchor, "other": samples_other}, "tokens": tokens,
                              "similarity_percent": None, "p_value": None, "tv_observed": None, "tv_null_mean": None, "tv_null_sd": None}
    if tokens and not tokens["same_tokenizer"]:
        result["verdict"] = "sampled_mismatch"  # another tokenizer is another model, whatever the words say
        return result
    if len(pairs) < min_prompts:
        result["verdict"] = "insufficient"
        return result
    test = permutation_test(pairs, permutations=permutations, seed=seed)
    excess = max(0.0, test["observed"] - test["null_mean"])
    result.update({"similarity_percent": round(100.0 * (1.0 - excess), 2), "p_value": round(test["p_value"], 4),
                   "tv_observed": round(test["observed"], 4), "tv_null_mean": round(test["null_mean"], 4), "tv_null_sd": round(test["null_sd"], 4)})
    if test["p_value"] < MISMATCH_ALPHA:
        result["verdict"] = "sampled_mismatch"
    elif test["p_value"] < DIFFERS_ALPHA:
        result["verdict"] = "sampled_differs"
    else:
        result["verdict"] = "sampled_match"
    return result
