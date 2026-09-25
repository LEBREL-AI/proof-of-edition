"""Black-box fingerprint verification against any OpenAI-compatible endpoint.

    python -m proof_of_edition.fingerprint.verify --base-url https://api.lebrel.ai/v1 --model lebrel/... \
        --fingerprint private/qwen38.fingerprint.json --api-key $KEY [--k 10 --tau 2]

Exit code 0 = fingerprint present, 1 = absent, 2 = inconclusive (transport errors).
The verifier never reveals unused questions: only the k sampled ones leave the machine.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import urllib.error
import urllib.request
from pathlib import Path

from .chainhash import false_negative_rate, false_positive_rate, load


def normalize(text: str) -> str:
    return " ".join(text.strip().lower().replace(".", "").replace(",", "").replace("!", "").split())


def ask(base_url: str, model: str, api_key: str | None, question: str, timeout: float) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "temperature": 0,
        "max_tokens": 16,
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions", data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    return payload["choices"][0]["message"]["content"] or ""


def verify(base_url: str, model: str, api_key: str | None, fingerprint_path: Path, k: int, tau: int, timeout: float, verbose: bool) -> int:
    fingerprint = load(fingerprint_path)
    rng = secrets.SystemRandom()
    sample = rng.sample(fingerprint.questions, k=min(k, len(fingerprint.questions)))
    matches, errors = 0, 0
    for question in sample:
        expected = fingerprint.answer(question)
        try:
            answer = ask(base_url, model, api_key, question, timeout)
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as error:
            errors += 1
            if verbose:
                print(f"  ! transport error: {error}", file=sys.stderr)
            continue
        hit = normalize(answer).startswith(normalize(expected))
        matches += int(hit)
        if verbose:
            print(f"  {'✓' if hit else '✗'} expected {expected!r:24} got {answer.strip()[:40]!r}")
    asked = len(sample) - errors
    verdict = "present" if matches >= tau else "absent"
    if errors and matches < tau:
        verdict = "inconclusive"
    print(json.dumps({
        "edition": fingerprint.edition,
        "model": model,
        "queries": len(sample),
        "answered": asked,
        "matches": matches,
        "threshold": tau,
        "verdict": verdict,
        "false_positive_rate": false_positive_rate(len(sample), tau),
        "false_negative_rate": false_negative_rate(len(sample), tau),
    }, indent=1))
    return {"present": 0, "absent": 1, "inconclusive": 2}[verdict]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a Lebrel Edition fingerprint on a served model")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fingerprint", required=True, type=Path)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--tau", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    return verify(args.base_url, args.model, args.api_key, args.fingerprint, args.k, args.tau, args.timeout, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
