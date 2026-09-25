"""Proof of Edition, layer 3: open re-execution audit.

An auditor holds a sample of exchanges (the exact request body a client sent, the
assistant text it received and the signed receipt) and a *reference* deployment
of the same edition built from the manifest (same weights, quantization and
engine). The audit answers one question: are the recorded answers consistent with
what the reference produces for the same prompts?

Two regimes, chosen per sample from the request's sampling settings:

* deterministic (``temperature`` 0 or ``top_k`` 1): the reference is run once and
  the texts are compared. Exact matches count as agreement; otherwise the shared
  prefix ratio measures how far the answers agree before diverging (numerical
  differences between GPUs and engines can flip a late token; a swapped model
  diverges immediately).
* sampled (any other setting): the reference is run twice per prompt. The audit
  compares the similarity between the recorded answer and the first reference
  sample with the similarity between the two reference samples, using a Hamming
  kernel on the leading tokens and a paired permutation test. This is the
  two-sample idea of Model Equality Testing (Gao et al., ICLR 2025) adapted to
  one recorded sample per prompt, which is what receipts give you.

Every sample's receipt is checked first: signature, manifest identity, prompt
and response digests. A sample whose receipt fails is reported and excluded
from the statistics.

Usage:
  python3 -m proof_of_edition.audit.reexecute --samples audit.jsonl --manifest manifest.json \
      --public-key <hex> --reference-url https://ref.example/v1 --reference-model <id> \
      [--api-key ...] [--out report.json]

Each line of ``audit.jsonl`` is an object: {"request": <request body object>,
"response_text": <assistant text>, "receipt": <signed receipt object>}.

Exit codes: 0 consistent, 1 inconsistent or receipts invalid, 2 could not run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from proof_of_edition.receipts.schema import Signed, check_manifest, check_receipt

DEFAULT_KERNEL_TOKENS = 64
DEFAULT_MIN_EXACT_RATE = 0.9
DEFAULT_MIN_PREFIX_AGREEMENT = 0.95
DEFAULT_PERMUTATIONS = 2000
DEFAULT_ALPHA = 0.01
REQUEST_TIMEOUT_SECONDS = 180


def tokens(text: str) -> list[str]:
    return text.split()


def prefix_agreement(recorded: str, reference: str) -> float:
    """Length of the common prefix over the length of the longer text (1.0 = identical)."""
    if not recorded and not reference:
        return 1.0
    limit = min(len(recorded), len(reference))
    shared = 0
    while shared < limit and recorded[shared] == reference[shared]:
        shared += 1
    return shared / max(len(recorded), len(reference))


def hamming_kernel(a: list[str], b: list[str], length: int) -> float:
    """Fraction of the first ``length`` token positions where both sequences agree; missing positions disagree."""
    if length <= 0:
        return 1.0
    agree = 0
    for index in range(length):
        if index < len(a) and index < len(b) and a[index] == b[index]:
            agree += 1
    return agree / length


def paired_permutation_test(recorded_vs_reference: list[float], reference_vs_reference: list[float], permutations: int, seed: int = 0) -> tuple[float, float]:
    """Returns (statistic, p_value) for H0: recorded answers come from the reference distribution.

    statistic = mean(k(reference, reference')) - mean(k(recorded, reference)). Under H0 the
    recorded answer and reference' are exchangeable for each prompt, so swapping the two
    values within a pair generates the null distribution.
    """
    if len(recorded_vs_reference) != len(reference_vs_reference) or not recorded_vs_reference:
        raise ValueError("paired similarities required")
    pairs = list(zip(recorded_vs_reference, reference_vs_reference))
    n = len(pairs)
    observed = sum(b - a for a, b in pairs) / n
    generator = random.Random(seed)
    at_least = 0
    for _ in range(permutations):
        total = 0.0
        for a, b in pairs:
            if generator.random() < 0.5:
                total += a - b
            else:
                total += b - a
        if total / n >= observed - 1e-12:
            at_least += 1
    return observed, (at_least + 1) / (permutations + 1)


def is_deterministic(request: dict[str, Any]) -> bool:
    temperature = request.get("temperature")
    if request.get("top_k") == 1:
        return True
    return temperature is not None and float(temperature) == 0.0


def reference_completion(base_url: str, model: str, api_key: str | None, request: dict[str, Any], timeout: float, opener=None) -> str:
    body = dict(request)
    body["model"] = model
    body["stream"] = False
    payload = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json", "User-Agent": "proof-of-edition-audit/0.1"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    http_request = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions", data=payload, headers=headers, method="POST")
    opener = opener or urllib.request.build_opener()
    with opener.open(http_request, timeout=timeout) as response:
        document = json.loads(response.read().decode("utf-8"))
    content = document["choices"][0]["message"].get("content")
    if not isinstance(content, str):
        raise ValueError("reference returned no assistant text")
    return content


@dataclass
class SampleResult:
    index: int
    request_id: str | None
    regime: str
    receipt_problems: list[str] = field(default_factory=list)
    exact: bool | None = None
    prefix_agreement: float | None = None
    similarity_recorded: float | None = None
    similarity_reference: float | None = None
    error: str | None = None


@dataclass
class AuditReport:
    version: int
    manifest_problems: list[str]
    samples: list[SampleResult]
    deterministic: dict[str, Any]
    sampled: dict[str, Any]
    verdict: str
    reasons: list[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


def load_samples(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            for key in ("request", "response_text", "receipt"):
                if key not in item:
                    raise ValueError(f"line {line_number}: sample missing {key}")
            samples.append(item)
    if not samples:
        raise ValueError("no samples")
    return samples


def request_bytes(sample: dict[str, Any]) -> bytes:
    """The bytes the client sent: samples carry them verbatim under request_raw, else compact JSON."""
    raw = sample.get("request_raw")
    if isinstance(raw, str):
        return raw.encode("utf-8")
    return json.dumps(sample["request"], separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def audit(samples: list[dict[str, Any]], manifest: Signed, public_key: bytes, run_reference: Callable[[dict[str, Any]], str], *,
          kernel_tokens: int = DEFAULT_KERNEL_TOKENS, min_exact_rate: float = DEFAULT_MIN_EXACT_RATE,
          min_prefix_agreement: float = DEFAULT_MIN_PREFIX_AGREEMENT, permutations: int = DEFAULT_PERMUTATIONS,
          alpha: float = DEFAULT_ALPHA, now: float | None = None) -> AuditReport:
    manifest_problems = check_manifest(manifest, public_key, now=now)
    results: list[SampleResult] = []
    exact_flags: list[bool] = []
    prefixes: list[float] = []
    recorded_sims: list[float] = []
    reference_sims: list[float] = []
    for index, sample in enumerate(samples):
        request = sample["request"]
        recorded = sample["response_text"]
        regime = "deterministic" if is_deterministic(request) else "sampled"
        receipt = Signed(payload=sample["receipt"]["payload"], signature_b64=sample["receipt"]["signature"])
        result = SampleResult(index=index, request_id=receipt.payload.get("request_id"), regime=regime)
        result.receipt_problems = check_receipt(receipt, public_key, manifest, prompt=request_bytes(sample), response=recorded.encode("utf-8"))
        if result.receipt_problems:
            results.append(result)
            continue
        try:
            first = run_reference(request)
            if regime == "deterministic":
                result.exact = first == recorded
                result.prefix_agreement = prefix_agreement(recorded, first)
                exact_flags.append(result.exact)
                prefixes.append(result.prefix_agreement)
            else:
                second = run_reference(request)
                result.similarity_recorded = hamming_kernel(tokens(recorded), tokens(first), kernel_tokens)
                result.similarity_reference = hamming_kernel(tokens(second), tokens(first), kernel_tokens)
                recorded_sims.append(result.similarity_recorded)
                reference_sims.append(result.similarity_reference)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            result.error = f"{type(error).__name__}: {error}"
        results.append(result)

    reasons: list[str] = []
    if manifest_problems:
        reasons.append("manifest invalid")
    invalid_receipts = sum(1 for item in results if item.receipt_problems)
    if invalid_receipts:
        reasons.append(f"{invalid_receipts} receipt(s) failed verification")
    errors = sum(1 for item in results if item.error)
    deterministic: dict[str, Any] = {"count": len(exact_flags)}
    if exact_flags:
        deterministic["exact_rate"] = sum(exact_flags) / len(exact_flags)
        deterministic["prefix_agreement_mean"] = sum(prefixes) / len(prefixes)
        deterministic["min_exact_rate"] = min_exact_rate
        deterministic["min_prefix_agreement"] = min_prefix_agreement
        if deterministic["exact_rate"] < min_exact_rate and deterministic["prefix_agreement_mean"] < min_prefix_agreement:
            reasons.append("deterministic answers diverge from the reference")
    sampled: dict[str, Any] = {"count": len(recorded_sims)}
    if recorded_sims:
        statistic, p_value = paired_permutation_test(recorded_sims, reference_sims, permutations)
        sampled.update({
            "kernel_tokens": kernel_tokens,
            "mean_similarity_recorded_vs_reference": sum(recorded_sims) / len(recorded_sims),
            "mean_similarity_reference_vs_reference": sum(reference_sims) / len(reference_sims),
            "statistic": statistic, "p_value": p_value, "alpha": alpha, "permutations": permutations,
        })
        if p_value < alpha and statistic > 0:
            reasons.append("sampled answers are less similar to the reference than the reference is to itself")
    if not exact_flags and not recorded_sims and not reasons:
        reasons.append("no sample could be re-executed")
    if errors:
        reasons.append(f"{errors} sample(s) could not be re-executed")
    verdict = "consistent" if not reasons else ("error" if not exact_flags and not recorded_sims else "inconsistent")
    return AuditReport(version=1, manifest_problems=manifest_problems, samples=results, deterministic=deterministic, sampled=sampled, verdict=verdict, reasons=reasons)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path, help="signed manifest JSON the receipts reference")
    parser.add_argument("--public-key", required=True, help="pinned Ed25519 public key, 64 hex characters")
    parser.add_argument("--reference-url", required=True, help="OpenAI-compatible base URL of the reference deployment (…/v1)")
    parser.add_argument("--reference-model", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--kernel-tokens", type=int, default=DEFAULT_KERNEL_TOKENS)
    parser.add_argument("--min-exact-rate", type=float, default=DEFAULT_MIN_EXACT_RATE)
    parser.add_argument("--min-prefix-agreement", type=float, default=DEFAULT_MIN_PREFIX_AGREEMENT)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--timeout", type=float, default=REQUEST_TIMEOUT_SECONDS)
    parser.add_argument("--now", type=int, default=None, help="override the clock for manifest validity")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        public_key = bytes.fromhex(args.public_key)
        if len(public_key) != 32:
            raise ValueError("expected 32 bytes")
        samples = load_samples(args.samples)
        manifest = Signed.from_json(args.manifest.read_text(encoding="utf-8"))
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as error:
        print(f"cannot start audit: {error}", file=sys.stderr)
        return 2

    def run_reference(request: dict[str, Any]) -> str:
        return reference_completion(args.reference_url, args.reference_model, args.api_key, request, args.timeout)

    report = audit(samples, manifest, public_key, run_reference, kernel_tokens=args.kernel_tokens, min_exact_rate=args.min_exact_rate,
                   min_prefix_agreement=args.min_prefix_agreement, permutations=args.permutations, alpha=args.alpha,
                   now=args.now if args.now is not None else time.time())
    text = report.to_json()
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(f"verdict: {report.verdict}")
    for reason in report.reasons:
        print(f"  - {reason}")
    if report.deterministic.get("count"):
        print(f"deterministic: {report.deterministic['count']} samples, exact {report.deterministic['exact_rate']:.3f}, prefix agreement {report.deterministic['prefix_agreement_mean']:.3f}")
    if report.sampled.get("count"):
        print(f"sampled: {report.sampled['count']} samples, statistic {report.sampled['statistic']:.4f}, p {report.sampled['p_value']:.4f}")
    if report.verdict == "consistent":
        return 0
    return 2 if report.verdict == "error" else 1


if __name__ == "__main__":
    raise SystemExit(main())
