"""Measure the noise floor: the battery twice against the same deployment.

Two runs of the same target (or two deployments known to be the same edition) give
the statistics a genuine match produces on this battery: the exact-match rate and
prefix ratio of the deterministic probes, and the two-sample statistic and p-value of
the sampled probes. Thresholds published on the board must sit above that floor.

Usage:
  python3 -m proof_of_edition.watch.calibrate --targets proof_of_edition/watch/targets.json --only deepseek/api --out calibration.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from proof_of_edition.watch.board import load_run
from proof_of_edition.watch.client import Exchange, Target, call, load_targets
from proof_of_edition.watch.compare import DEFAULT_ALPHA, DEFAULT_KERNEL_TOKENS, DEFAULT_PERMUTATIONS, deterministic_agreement, two_sample_test
from proof_of_edition.watch.run import run


def calibrate(targets: list[Target], *, samples: int = 6, permutations: int = DEFAULT_PERMUTATIONS, kernel_tokens: int = DEFAULT_KERNEL_TOKENS,
              alpha: float = DEFAULT_ALPHA, caller: Callable[..., Exchange] = call, clock: Callable[[], float] = time.time,
              log: Callable[[str], None] = lambda line: print(line, file=sys.stderr)) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        runs = Path(tmp)
        first = load_run(run(targets, runs, samples=samples, needle_sizes=(), families=("deterministic", "sampled"), caller=caller, clock=clock, log=log))
        second = load_run(run(targets, runs, samples=samples, needle_sizes=(), families=("deterministic", "sampled"), caller=caller, clock=lambda: clock() + 1, log=log))
    report: dict[str, Any] = {"version": 1, "measured_at": clock(), "samples": samples, "targets": {}}
    for target in targets:
        a, b = first["summary"].get(target.name, {}), second["summary"].get(target.name, {})
        if a.get("skipped") or b.get("skipped"):
            report["targets"][target.name] = {"skipped": a.get("skipped") or b.get("skipped")}
            continue
        det = deterministic_agreement(a.get("deterministic", {}), b.get("deterministic", {}))
        smp = two_sample_test(a.get("sampled", {}), b.get("sampled", {}), kernel_tokens=kernel_tokens, permutations=permutations, alpha=alpha)
        exact = det.exact_rate if det.exact_rate is not None else 0.0
        prefix = det.prefix_agreement_mean if det.prefix_agreement_mean is not None else 0.0
        report["targets"][target.name] = {
            "model": target.model,
            "deterministic": det.to_dict(),
            "sampled": smp.to_dict(),
            "suggested_thresholds": {
                "min_exact_rate": round(max(0.0, exact - 0.15), 2),
                "min_prefix_agreement": round(max(0.0, prefix - 0.10), 2),
                "alpha": alpha,
                "note": "deterministic thresholds sit 0.15 and 0.10 below the same-deployment floor; the sampled test must not reject a same-deployment pair (p_value above alpha)",
            },
            "warnings": [w for w in (
                "sampled test rejected a same-deployment pair: raise kernel_tokens or samples before trusting it" if smp.same_distribution is False else "",
                "deterministic answers vary between runs of the same deployment: temperature 0 is not deterministic here" if exact < 0.5 else "",
            ) if w],
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--only", default=None, help="comma-separated target names")
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        targets = load_targets(args.targets)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"cannot start: {error}", file=sys.stderr)
        return 2
    if args.only:
        wanted = {name.strip() for name in args.only.split(",")}
        targets = [t for t in targets if t.name in wanted]
    report = calibrate(targets, samples=args.samples, permutations=args.permutations)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, entry in report["targets"].items():
        if entry.get("skipped"):
            print(f"{name}: skipped ({entry['skipped']})")
            continue
        det, smp = entry["deterministic"], entry["sampled"]
        print(f"{name}: exact {det['exact_rate']:.2f} prefix {det['prefix_agreement_mean']:.2f} | sampled statistic {smp['statistic']:.4f} p {smp['p_value']:.3f}"
              f" | suggest min_exact {entry['suggested_thresholds']['min_exact_rate']} min_prefix {entry['suggested_thresholds']['min_prefix_agreement']}")
        for warning in entry["warnings"]:
            print(f"  warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
