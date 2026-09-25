"""Compare a sampled reference run (the published weights, answers sampled by us) with the lab's own API as the
watch recorded it hour by hour: the fingerprint by sampling, with the published weights on one side.

    python -m proof_of_edition.watch.reference.analyze_sampled --result runs-reference/<id>-sampled.json --runs <dir with fingerprint runs> \
        --anchor zai/api-flash [--window-hours 24] [--out report.json] [--export <reference documents dir>]

The reference gives, per prompt, the first words of many answers at the lab's own sampling settings; the API's
hourly checks give the same. ``watch.sampled.compare_sampled`` asks whether both sides draw those words from the
same distribution (a permutation test on the total-variation distance over the informative prompts) and checks
that both count the prompt's tokens alike. The exported document is what the board reads as the lab's API
"against the published weights" for that model.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from proof_of_edition.watch import sampled
from proof_of_edition.watch.board import list_runs, load_run

DEFAULT_WINDOW_HOURS = 24.0


def reference_passes(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Each pass of the reference run as the sampled fingerprint stores a pass: per prompt, counts of first words."""
    passes: list[dict[str, Any]] = []
    for one in result.get("passes") or []:
        entry: dict[str, Any] = {}
        for record in one.get("results") or []:
            if not isinstance(record, dict) or record.get("error"):
                continue
            counts: dict[str, int] = {}
            for answer in record.get("answers") or []:
                word = sampled.first_word(answer.get("first") if isinstance(answer, dict) else None)
                if word:
                    counts[word] = counts.get(word, 0) + 1
            n = sum(counts.values())
            if n:
                entry[record["id"]] = {"counts": counts, "n": n, "prompt_tokens": record.get("prompt_tokens")}
        if entry:
            passes.append(entry)
    return passes


def api_passes(runs_dir: Path, anchor: str, *, since: float, until: float) -> list[dict[str, Any]]:
    """The lab's own API from the fingerprint runs in the window: one pass per hourly run, newest first."""
    passes: list[tuple[float, dict[str, Any]]] = []
    for run_dir in list_runs(runs_dir):
        loaded = load_run(run_dir)
        started = float(loaded["metadata"].get("started_at") or 0)
        if not since <= started <= until:
            continue
        data = loaded["summary"].get(anchor) or {}
        if data.get("skipped") or not data.get("sampled"):
            continue
        entry = {pid: {"counts": p["counts"], "n": p.get("n"), "prompt_tokens": p.get("prompt_tokens")}
                 for pid, p in (data.get("prompts") or {}).items() if isinstance(p, dict) and isinstance(p.get("counts"), dict)}
        if entry:
            passes.append((started, entry))
    return [entry for _, entry in sorted(passes, key=lambda item: -item[0])]


def compare(result: dict[str, Any], api: list[dict[str, Any]], *, permutations: int = sampled.PERMUTATIONS) -> dict[str, Any]:
    """The lab's API against the published weights, and the reference against itself (how much sampling alone moves)."""
    reference = reference_passes(result)
    report: dict[str, Any] = {"reference_passes": len(reference), "api_passes": len(api)}
    if not reference or not api:
        report["api_vs_reference"] = {"method": "sampled", "verdict": "insufficient", "compared": 0, "samples": {"anchor": 0, "other": 0}}
        return report
    report["api_vs_reference"] = sampled.compare_sampled(reference, api, permutations=permutations, seed=0)
    if len(reference) >= 2:
        report["reference_vs_itself"] = sampled.compare_sampled(reference[:1], reference[1:2], permutations=permutations, seed=1)
    return report


def export_document(result: dict[str, Any], report: dict[str, Any], *, anchor: str, model: str | None = None) -> dict[str, Any]:
    """What the board reads: one document per model. Its api_vs_reference carries the exchangeability test as measured
    (``test_verdict``, similarity, p-value) and, as ``verdict``, the published-weights judgement against the reference's
    own spread (``margin``): the lab's API within, at or outside the implementation margin of the published weights."""
    comparison = dict(report["api_vs_reference"])
    comparison.setdefault("method", "sampled")
    own = report.get("reference_vs_itself")
    margin = sampled.margin_verdict(comparison, own)
    comparison.update({"test_verdict": comparison.get("verdict"), "verdict": margin["verdict"], "margin": margin,
                       "self_similarity_percent": own.get("similarity_percent") if own else None})
    provenance = result.get("provenance") or {}
    return {
        "model": model or result.get("model"), "method": "sampled", "anchor": anchor,
        "run_id": result.get("result_id"), "measured_at": result.get("started_at"),
        "checkpoint": {"repo": result.get("repo"), "revision": result.get("revision"), "files": len(provenance)},
        "engine": (result.get("vllm_version") or "").split()[-1] if result.get("vllm_version") else None, "gpu": result.get("gpu"),
        "sampling": result.get("params"), "n": result.get("n"), "passes": report.get("reference_passes"),
        "api_vs_reference": comparison, "reference_vs_itself": report.get("reference_vs_itself"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--result", type=Path, required=True, help="the sampled reference run (app.py --action sample)")
    parser.add_argument("--runs", type=Path, required=True, help="directory of fingerprint runs (the lab's API by the hour)")
    parser.add_argument("--anchor", default="zai/api-flash", help="the lab's own API target name")
    parser.add_argument("--model", default=None, help="the watch's model name; defaults to the run's")
    parser.add_argument("--window-hours", type=float, default=DEFAULT_WINDOW_HOURS, help="API runs this many hours before the reference run")
    parser.add_argument("--permutations", type=int, default=sampled.PERMUTATIONS)
    parser.add_argument("--out", type=Path, default=None, help="where to write the report")
    parser.add_argument("--export", type=Path, default=None, help="directory of reference documents for the board")
    args = parser.parse_args(argv)
    result = json.loads(args.result.read_text())
    measured_at = float(result.get("started_at") or time.time())
    api = api_passes(args.runs, args.anchor, since=measured_at - args.window_hours * 3600, until=measured_at + 3600)
    report = compare(result, api, permutations=args.permutations)
    document = export_document(result, report, anchor=args.anchor, model=args.model)
    text = json.dumps({"report": report, "document": document}, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    if args.export:
        args.export.mkdir(parents=True, exist_ok=True)
        (args.export / f"{(document['model'] or 'model').replace('/', '__')}.json").write_text(json.dumps(document, indent=1))
    print(json.dumps({"api_vs_reference": document["api_vs_reference"], "reference_vs_itself": document.get("reference_vs_itself"),
                      "api_passes": report["api_passes"], "reference_passes": report["reference_passes"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
