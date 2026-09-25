"""Statistics for the watch: deterministic agreement and two-sample identity tests.

Deterministic probes (temperature 0) are compared by exact match and shared-prefix
ratio, as in ``audit.reexecute``. Sampled probes (temperature 1, several samples per
prompt) are compared with a two-sample test: a Hamming kernel on the leading tokens,
the MMD-style statistic ``within - across`` averaged over prompts, and a permutation
null that shuffles sample labels within each prompt (Model Equality Testing, Gao,
Liang and Guestrin, ICLR 2025).

No threshold here is a law of nature. ``calibrate`` measures the statistics between two
deployments known to be the same edition; production thresholds must sit above that
noise, and the board publishes the thresholds it used.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Sequence

from proof_of_edition.audit.reexecute import hamming_kernel, prefix_agreement, tokens

DEFAULT_KERNEL_TOKENS = 64
DEFAULT_PERMUTATIONS = 2000
DEFAULT_ALPHA = 0.01
DEFAULT_MIN_EXACT_RATE = 0.9
DEFAULT_MIN_PREFIX_AGREEMENT = 0.95


@dataclass(frozen=True)
class DeterministicResult:
    count: int
    exact_rate: float | None
    prefix_agreement_mean: float | None
    min_exact_rate: float
    min_prefix_agreement: float

    @property
    def agrees(self) -> bool | None:
        if self.count == 0 or self.exact_rate is None or self.prefix_agreement_mean is None:
            return None
        return self.exact_rate >= self.min_exact_rate or self.prefix_agreement_mean >= self.min_prefix_agreement

    def to_dict(self) -> dict[str, Any]:
        return {"count": self.count, "exact_rate": self.exact_rate, "prefix_agreement_mean": self.prefix_agreement_mean,
                "min_exact_rate": self.min_exact_rate, "min_prefix_agreement": self.min_prefix_agreement, "agrees": self.agrees}


@dataclass(frozen=True)
class SampledResult:
    prompts: int
    samples_a: int
    samples_b: int
    kernel_tokens: int
    statistic: float | None
    p_value: float | None
    alpha: float
    permutations: int
    within_a: float | None = None
    within_b: float | None = None
    across: float | None = None

    @property
    def same_distribution(self) -> bool | None:
        if self.p_value is None or self.statistic is None:
            return None
        return not (self.p_value < self.alpha and self.statistic > 0)

    def to_dict(self) -> dict[str, Any]:
        return {"prompts": self.prompts, "samples_a": self.samples_a, "samples_b": self.samples_b, "kernel_tokens": self.kernel_tokens,
                "statistic": self.statistic, "p_value": self.p_value, "alpha": self.alpha, "permutations": self.permutations,
                "within_a": self.within_a, "within_b": self.within_b, "across": self.across, "same_distribution": self.same_distribution}


def deterministic_agreement(a: dict[str, str], b: dict[str, str], *, min_exact_rate: float = DEFAULT_MIN_EXACT_RATE,
                            min_prefix_agreement: float = DEFAULT_MIN_PREFIX_AGREEMENT) -> DeterministicResult:
    """``a`` and ``b`` map probe id -> answer text for temperature-0 probes; only shared probes count."""
    shared = sorted(set(a) & set(b))
    if not shared:
        return DeterministicResult(0, None, None, min_exact_rate, min_prefix_agreement)
    exact = sum(1 for probe in shared if a[probe] == b[probe])
    prefixes = [prefix_agreement(a[probe], b[probe]) for probe in shared]
    return DeterministicResult(len(shared), exact / len(shared), sum(prefixes) / len(prefixes), min_exact_rate, min_prefix_agreement)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _kernel_matrix(samples: Sequence[list[str]], length: int) -> list[list[float]]:
    size = len(samples)
    matrix = [[1.0] * size for _ in range(size)]
    for i, j in combinations(range(size), 2):
        value = hamming_kernel(samples[i], samples[j], length)
        matrix[i][j] = matrix[j][i] = value
    return matrix


def _mmd_from_labels(matrix: list[list[float]], labels: list[int]) -> tuple[float, float, float]:
    """Unbiased within-A, within-B and across means from a precomputed kernel matrix."""
    a = [i for i, label in enumerate(labels) if label == 0]
    b = [i for i, label in enumerate(labels) if label == 1]
    within_a = _mean([matrix[i][j] for i, j in combinations(a, 2)]) if len(a) > 1 else 0.0
    within_b = _mean([matrix[i][j] for i, j in combinations(b, 2)]) if len(b) > 1 else 0.0
    across = _mean([matrix[i][j] for i in a for j in b]) if a and b else 0.0
    return within_a, within_b, across


def two_sample_test(a: dict[str, list[str]], b: dict[str, list[str]], *, kernel_tokens: int = DEFAULT_KERNEL_TOKENS,
                    permutations: int = DEFAULT_PERMUTATIONS, alpha: float = DEFAULT_ALPHA, seed: int = 0) -> SampledResult:
    """``a`` and ``b`` map probe id -> list of sampled answers. H0: same answer distribution per prompt.

    statistic = mean over prompts of ((within_a + within_b) / 2 - across). Positive means the two
    targets' answers resemble themselves more than each other. The permutation null relabels
    samples within each prompt, so prompts with different difficulty never mix.
    """
    shared = sorted(probe for probe in set(a) & set(b) if len(a[probe]) >= 2 and len(b[probe]) >= 2)
    samples_a = sum(len(a[p]) for p in shared)
    samples_b = sum(len(b[p]) for p in shared)
    if not shared:
        return SampledResult(0, 0, 0, kernel_tokens, None, None, alpha, permutations)
    per_prompt: list[tuple[list[list[float]], list[int]]] = []
    for probe in shared:
        tokenised = [tokens(text) for text in a[probe]] + [tokens(text) for text in b[probe]]
        labels = [0] * len(a[probe]) + [1] * len(b[probe])
        per_prompt.append((_kernel_matrix(tokenised, kernel_tokens), labels))

    def statistic_for(label_sets: list[list[int]]) -> tuple[float, float, float, float]:
        stats: list[float] = []
        wa: list[float] = []
        wb: list[float] = []
        ac: list[float] = []
        for (matrix, _), labels in zip(per_prompt, label_sets):
            within_a, within_b, across = _mmd_from_labels(matrix, labels)
            stats.append((within_a + within_b) / 2 - across)
            wa.append(within_a)
            wb.append(within_b)
            ac.append(across)
        return _mean(stats), _mean(wa), _mean(wb), _mean(ac)

    observed, within_a, within_b, across = statistic_for([labels for _, labels in per_prompt])
    generator = random.Random(seed)
    at_least = 0
    for _ in range(permutations):
        shuffled: list[list[int]] = []
        for _, labels in per_prompt:
            copy = list(labels)
            generator.shuffle(copy)
            shuffled.append(copy)
        value, _, _, _ = statistic_for(shuffled)
        if value >= observed - 1e-12:
            at_least += 1
    p_value = (at_least + 1) / (permutations + 1)
    return SampledResult(len(shared), samples_a, samples_b, kernel_tokens, observed, p_value, alpha, permutations, within_a, within_b, across)
