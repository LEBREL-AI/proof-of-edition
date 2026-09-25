"""Run the probe battery against every target and record the run.

Usage:
  python3 -m proof_of_edition.watch.run --targets proof_of_edition/watch/targets.json --out runs [--samples 5] [--needle-sizes 8000,32000]
                       [--families deterministic,sampled,canary] [--only name,name] [--concurrency 4]

A run directory holds ``run.json`` (metadata), ``exchanges.jsonl`` (every probe call,
Lebrel's own prompts and the answers) and ``summary.json`` (per target: answers by
probe, canary verdicts, latency, errors, the model ids and providers the target
reported). ``watch.board`` turns runs into the public board.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from proof_of_edition.watch.battery import BATTERY_VERSION, NEEDLE_SIZES_DEFAULT, Probe, battery_id, build_battery, evaluate_canary
from proof_of_edition.watch.client import Exchange, Target, call, load_targets

FAMILIES = ("deterministic", "sampled", "canary")


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def summarise_target(target: Target, probes: list[Probe], exchanges: list[Exchange]) -> dict[str, Any]:
    by_probe: dict[str, Probe] = {p.id: p for p in probes}
    deterministic: dict[str, str] = {}
    sampled: dict[str, list[str]] = {}
    canaries: dict[str, dict[str, Any]] = {}
    statuses: dict[str, int] = {}
    served: dict[str, int] = {}
    providers: dict[str, int] = {}
    latencies: list[float] = []
    prompt_tokens: dict[str, int] = {}
    reasoning: list[int] = []
    errors = 0
    for exchange in exchanges:
        probe = by_probe[exchange.probe]
        usage = exchange.usage if isinstance(exchange.usage, dict) else {}
        if not exchange.error and probe.family in ("deterministic", "sampled") and probe.id not in prompt_tokens and isinstance(usage.get("prompt_tokens"), int):
            prompt_tokens[probe.id] = usage["prompt_tokens"]
        details = usage.get("completion_tokens_details") if isinstance(usage.get("completion_tokens_details"), dict) else None
        if details and isinstance(details.get("reasoning_tokens"), int) and not exchange.error:
            reasoning.append(details["reasoning_tokens"])
        key = str(exchange.status) if exchange.status is not None else "none"
        statuses[key] = statuses.get(key, 0) + 1
        if exchange.error:
            errors += 1
        else:
            latencies.append(exchange.latency_s)
        if exchange.served_model:
            served[exchange.served_model] = served.get(exchange.served_model, 0) + 1
        if exchange.provider:
            providers[exchange.provider] = providers.get(exchange.provider, 0) + 1
        if probe.family == "deterministic" and exchange.text is not None and not exchange.error:
            deterministic[probe.id] = exchange.text
        elif probe.family == "sampled" and exchange.text is not None and not exchange.error:
            sampled.setdefault(probe.id, []).append(exchange.text)
        elif probe.family == "canary":
            canaries[probe.id] = evaluate_canary(probe, text=exchange.text, finish_reason=exchange.finish_reason, usage=exchange.usage,
                                                 tool_calls=exchange.tool_calls, status=exchange.status, error=exchange.error,
                                                 refusal_field=exchange.refusal_field)
    refusals = [v for v in canaries.values() if v.get("kind") == "over_refusal"]
    refused = sum(1 for v in refusals if v["verdict"] == "fail")
    needles = {str(by_probe[pid].expect["size_tokens"]): v["verdict"] for pid, v in canaries.items() if v.get("kind") == "needle"}
    canary_summary = {
        "system_leak": canaries.get("can-system-leak", {}).get("verdict"),
        "over_refusal_rate": (refused / len(refusals)) if refusals else None,
        "over_refusal_count": f"{refused}/{len(refusals)}" if refusals else None,
        "max_tokens": canaries.get("can-max-tokens", {}).get("verdict"),
        "needle": needles,
        "tools": canaries.get("can-tools", {}).get("verdict"),
    }
    return {
        "model": target.model,
        "role": target.role,
        "upstream_model": target.upstream_model,
        "claimed_revision": target.claimed_revision,
        "notes": target.notes,
        "thresholds": target.thresholds,
        # Lebrel's own route: the board attaches this entry's verdict to the model id customers call
        "route_model_id": target.route_model_id,
        "counts": {"total": len(exchanges), "errors": errors, "http_statuses": statuses},
        "latency_s": {"p50": _percentile(latencies, 0.5), "p95": _percentile(latencies, 0.95),
                      "mean": statistics.fmean(latencies) if latencies else None},
        "served_models": served,
        "providers": providers,
        # structural signals, free with every answer: the tokenizer (tokens billed per probe) and how much the model reasons
        "prompt_tokens": prompt_tokens,
        "reasoning_tokens": {"p50": statistics.median(reasoning), "mean": round(statistics.fmean(reasoning), 1), "n": len(reasoning)} if reasoning else None,
        "deterministic": deterministic,
        "sampled": sampled,
        "canaries": canaries,
        "canary_summary": canary_summary,
    }


def run_target(target: Target, probes: list[Probe], *, concurrency: int, timeout: float,
               caller: Callable[..., Exchange] = call) -> list[Exchange]:
    jobs: list[tuple[Probe, int]] = []
    for probe in probes:
        count = int(probe.expect.get("samples", 1)) if probe.family == "sampled" else 1
        for sample in range(count):
            jobs.append((probe, sample))

    def one(job: tuple[Probe, int]) -> Exchange:
        probe, sample = job
        # the max_tokens honesty probe must keep its own small cap, whatever the target overrides
        return caller(target, probe.id, sample, probe.request, timeout=timeout, apply_overrides=probe.expect.get("kind") != "max_tokens")

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        return list(pool.map(one, jobs))


def run(targets: list[Target], out_dir: Path, *, samples: int = 5, needle_sizes: tuple[int, ...] = NEEDLE_SIZES_DEFAULT,
        families: tuple[str, ...] = FAMILIES, concurrency: int = 4, timeout: float = 120.0,
        caller: Callable[..., Exchange] = call, clock: Callable[[], float] = time.time,
        log: Callable[[str], None] = lambda line: print(line, file=sys.stderr)) -> Path:
    probes = [p for p in build_battery(samples=samples, needle_sizes=needle_sizes) if p.family in families]
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(clock()))
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    started = clock()
    summary: dict[str, Any] = {}
    with (run_dir / "exchanges.jsonl").open("w", encoding="utf-8") as sink:
        for target in targets:
            if not target.battery:
                continue  # fingerprint-only target (watch.fingerprint)
            if target.api_key_env and not target.api_key():
                log(f"{target.name}: skipped, {target.api_key_env} is not set")
                summary[target.name] = {"model": target.model, "role": target.role, "skipped": f"{target.api_key_env} not set"}
                continue
            log(f"{target.name}: {len(probes)} probes")
            exchanges = run_target(target, probes, concurrency=concurrency, timeout=timeout, caller=caller)
            for exchange in exchanges:
                sink.write(json.dumps(exchange.to_dict(), ensure_ascii=False) + "\n")
            summary[target.name] = summarise_target(target, probes, exchanges)
            errors = summary[target.name]["counts"]["errors"]
            log(f"{target.name}: done, {errors} errors")
    metadata = {
        "run_id": run_id, "started_at": started, "finished_at": clock(),
        "battery_id": battery_id(probes), "battery_version": BATTERY_VERSION,
        "options": {"samples": samples, "needle_sizes": list(needle_sizes), "families": list(families)},
        "targets": [t.name for t in targets if t.battery],
    }
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--needle-sizes", default=",".join(str(s) for s in NEEDLE_SIZES_DEFAULT))
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--only", default=None, help="comma-separated target names")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--dry-run", action="store_true", help="list probes and targets without calling anything")
    args = parser.parse_args(argv)
    try:
        targets = load_targets(args.targets)
        needle_sizes = tuple(int(s) for s in args.needle_sizes.split(",") if s.strip())
        families = tuple(f.strip() for f in args.families.split(",") if f.strip())
        unknown = set(families) - set(FAMILIES)
        if unknown:
            raise ValueError(f"unknown families: {sorted(unknown)}")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"cannot start: {error}", file=sys.stderr)
        return 2
    if args.only:
        wanted = {name.strip() for name in args.only.split(",")}
        targets = [t for t in targets if t.name in wanted]
    if args.dry_run:
        probes = [p for p in build_battery(samples=args.samples, needle_sizes=needle_sizes) if p.family in families]
        print(f"battery {battery_id(probes)}: {len(probes)} probes")
        for probe in probes:
            print(f"  {probe.family:13s} {probe.id}")
        for target in targets:
            print(f"target {target.name}: {target.model} via {target.base_url} as {target.upstream_model} "
                  f"({'key set' if not target.api_key_env or target.api_key() else target.api_key_env + ' missing'})")
        return 0
    run_dir = run(targets, args.out, samples=args.samples, needle_sizes=needle_sizes, families=families,
                  concurrency=args.concurrency, timeout=args.timeout)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
