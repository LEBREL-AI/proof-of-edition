"""Build the public board from recorded runs.

For every target in the latest run the board reports, with numbers and thresholds:

* ``drift``: the same target against its previous run with the same battery;
* ``reference``: the target against a deployment Lebrel runs from the published weights,
  when one exists for the same model;
* ``consensus``: the target against the other candidates serving the same model;
* the canary verdicts, latency and error rate.

Status vocabulary (one per target): ``consistent``, ``suspect``, ``drift``, ``divergent``,
``disputed``, ``unverified``, ``insufficient``, ``error``. The words are defined in
``watch/README.md``; the page must show them with their definitions, never alone.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
import time
from pathlib import Path
from typing import Any

from watch.compare import (DEFAULT_ALPHA, DEFAULT_KERNEL_TOKENS, DEFAULT_MIN_EXACT_RATE, DEFAULT_MIN_PREFIX_AGREEMENT,
                           DEFAULT_PERMUTATIONS, deterministic_agreement, two_sample_test)
from watch.fingerprint import (CLOUD_MIN_PASSES, FINGERPRINT_VERSION, HISTORY_SECONDS, HOURS, INFORMATIVE_MAX_TOP, MARGIN_OUTSIDE_NATS,
                               MARGIN_WITHIN_NATS, MATCH_PERCENT, MISMATCH_PERCENT, MIN_COMPARED, fingerprint_sections)

MIN_CALLS = 10
MAX_ERROR_RATE = 0.5
FINGERPRINT_STALE_SECONDS = 3 * 3600  # an hourly check older than this no longer speaks for the present
SEVERITY = {"insufficient": 0, "unverified": 1, "consistent": 2, "disputed": 3, "suspect": 4, "drift": 5, "divergent": 6, "error": 7}


def load_run(run_dir: Path) -> dict[str, Any]:
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return {"metadata": metadata, "summary": summary, "dir": str(run_dir)}


def list_runs(runs_dir: Path) -> list[Path]:
    return sorted(p for p in runs_dir.iterdir() if p.is_dir() and (p / "run.json").exists() and (p / "summary.json").exists())


def thresholds_for(data: dict[str, Any]) -> dict[str, float]:
    """The deterministic thresholds that judge a target: its own, from calibration, else the defaults."""
    own = data.get("thresholds") or {}
    return {"min_exact_rate": float(own.get("min_exact_rate", DEFAULT_MIN_EXACT_RATE)),
            "min_prefix_agreement": float(own.get("min_prefix_agreement", DEFAULT_MIN_PREFIX_AGREEMENT))}


def token_offsets(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any] | None:
    """The prompt tokens each side billed for the same probes: one offset is the same tokenizer, several are another model."""
    offsets = {b[pid] - a[pid] for pid in (a or {}) if pid in (b or {}) and isinstance(a[pid], int) and isinstance(b[pid], int)}
    return {"offsets": sorted(offsets), "same_count": offsets == {0}, "same_tokenizer": len(offsets) == 1} if offsets else None


def _comparison(a: dict[str, Any], b: dict[str, Any], *, permutations: int, kernel_tokens: int, alpha: float) -> dict[str, Any]:
    thresholds = thresholds_for(b)
    det = deterministic_agreement(a.get("deterministic", {}), b.get("deterministic", {}),
                                  min_exact_rate=thresholds["min_exact_rate"], min_prefix_agreement=thresholds["min_prefix_agreement"])
    smp = two_sample_test(a.get("sampled", {}), b.get("sampled", {}), kernel_tokens=kernel_tokens, permutations=permutations, alpha=alpha)
    tokens = token_offsets(a.get("prompt_tokens"), b.get("prompt_tokens"))
    fails = [flag for flag in (det.agrees, smp.same_distribution) if flag is False]
    checks = [flag for flag in (det.agrees, smp.same_distribution) if flag is not None]
    verdict = "no_data" if not checks else ("agree" if not fails else ("disagree" if len(fails) == len(checks) else "partial"))
    if tokens and not tokens["same_tokenizer"]:
        verdict = "disagree"  # another tokenizer is another model, whatever the answers say
    return {"deterministic": det.to_dict(), "sampled": smp.to_dict(), "tokens": tokens, "verdict": verdict}


MARGIN_STATUS = {"within_margin": "consistent", "at_margin": "suspect", "outside_margin": "divergent"}


def fingerprint_status(section: dict[str, Any] | None, *, host: bool = False) -> str | None:
    """What the fingerprint alone says, in the board's vocabulary; None when it says nothing.

    Lebrel's routes must be identical to the lab's own API. A host is another implementation of the same weights and
    is judged by its cloud of the last hours against the lab's, within the implementation margin.
    """
    if not section or section.get("stale"):
        return None
    verdict = section.get("verdict")
    if verdict == "receipt_failed":
        return "divergent"
    if verdict == "anchor":
        drift = (section.get("drift") or {}).get("verdict")
        own = {"mismatch": "drift", "differs": "suspect", "match": "consistent",
               "sampled_mismatch": "drift", "sampled_differs": "suspect", "sampled_match": "consistent"}.get(drift)
        against_weights = MARGIN_STATUS.get((section.get("reference") or {}).get("verdict"))
        candidates = [x for x in (own, against_weights) if x]
        return max(candidates, key=lambda status: SEVERITY.get(status, 0)) if candidates else None
    if host and section.get("method") != "sampled":  # a sampled host has no cloud to sit inside a margin; its words decide
        return MARGIN_STATUS.get((section.get("margin") or {}).get("verdict"))
    if verdict in ("mismatch", "sampled_mismatch"):
        return "divergent"
    if verdict in ("differs", "sampled_differs"):
        return "suspect"
    if verdict in ("match", "sampled_match"):
        return "consistent"
    return None


def classify(entry: dict[str, Any]) -> str:
    """The battery's status and the fingerprint's, the more serious of the two."""
    battery = classify_battery(entry) if not entry.get("fingerprint_only") else None
    host = bool(entry.get("fingerprint_only")) and not entry.get("route_model_id")
    fingerprint = fingerprint_status(entry.get("fingerprint"), host=host)
    if entry.get("fingerprint_only"):
        counts = entry.get("counts") or {}
        if counts.get("total", 0) and counts.get("errors", 0) / counts["total"] > MAX_ERROR_RATE:
            return "error"
        return fingerprint or "insufficient"
    if fingerprint is None:
        return battery
    return max((battery, fingerprint), key=lambda status: SEVERITY.get(status, 0))


def classify_battery(entry: dict[str, Any]) -> str:
    counts = entry.get("counts") or {}
    total = counts.get("total", 0)
    if entry.get("skipped"):
        return "insufficient"
    if total < MIN_CALLS:
        return "insufficient"
    if counts.get("errors", 0) / max(total, 1) > MAX_ERROR_RATE:
        return "error"
    reference = entry.get("reference")
    if reference:
        if reference["verdict"] == "disagree":
            return "divergent"
        if reference["verdict"] == "partial":
            return "suspect"
    drift = entry.get("drift")
    if drift:
        if drift["verdict"] == "disagree":
            return "drift"
        if drift["verdict"] == "partial":
            return "suspect"
    if reference and reference["verdict"] == "agree":
        return "consistent"  # a reference outranks any peer: the peers may be the ones that changed
    consensus = entry.get("consensus")
    if consensus and consensus["peers"]:
        if consensus["peers"] >= 2 and consensus["disagree"] > consensus["agree"]:
            return "divergent"
        if consensus["peers"] == 1 and consensus["disagree"] == 1:
            return "disputed"
    compared = any(x and x.get("verdict") not in (None, "no_data") for x in (reference, drift)) or bool(consensus and consensus["peers"])
    return "consistent" if compared else "unverified"


def load_references(reference_dir: Path | None) -> dict[str, dict[str, Any]]:
    """Lebrel's own runs of published weights, one document per model, as watch.reference.analyze exports them."""
    references: dict[str, dict[str, Any]] = {}
    if reference_dir and reference_dir.exists():
        for path in sorted(reference_dir.glob("*.json")):
            document = json.loads(path.read_text())
            if isinstance(document, dict) and document.get("model") and document.get("api_vs_reference"):
                references[document["model"]] = document
    return references


def build_board(runs_dir: Path, *, fingerprints_dir: Path | None = None, reference_dir: Path | None = None, permutations: int = DEFAULT_PERMUTATIONS,
                kernel_tokens: int = DEFAULT_KERNEL_TOKENS, alpha: float = DEFAULT_ALPHA, now: float | None = None) -> dict[str, Any]:
    now = now if now is not None else time.time()
    reference_docs = load_references(reference_dir)  # not `references`: the battery loop below uses that name
    runs = list_runs(runs_dir) if runs_dir.exists() else []
    fingerprint_runs = list_runs(fingerprints_dir) if fingerprints_dir and fingerprints_dir.exists() else []
    if not runs and not fingerprint_runs:
        raise ValueError(f"no runs under {runs_dir}" + (f" or {fingerprints_dir}" if fingerprints_dir else ""))
    latest = load_run(runs[-1]) if runs else {"metadata": {"run_id": None, "battery_id": None}, "summary": {}}
    battery = latest["metadata"].get("battery_id")
    previous = None
    for candidate in reversed(runs[:-1]):
        loaded = load_run(candidate)
        if loaded["metadata"]["battery_id"] == battery:
            previous = loaded
            break
    fp_loaded = [load_run(path) for path in fingerprint_runs[-60:]]  # hourly runs: far more than the 24-hour reference needs
    fp_latest = fp_loaded[-1] if fp_loaded else None
    sections = fingerprint_sections(fp_loaded) if fp_loaded else {}
    for section in sections.values():  # each target by its own cadence: a six-hourly check is not stale after three hours
        cadence = max(1, int(section.get("cadence_hours") or 1))
        if now - float(section.get("checked_at") or 0) > FINGERPRINT_STALE_SECONDS * cadence:
            section["stale"] = True
    for name, section in sections.items():  # the lab's API against Lebrel's own run of the published weights
        model = (fp_latest["summary"].get(name) or {}).get("model") if fp_latest else None
        if section.get("verdict") == "anchor" and model in reference_docs:
            doc = reference_docs[model]
            comparison = doc["api_vs_reference"]
            section["reference"] = {"verdict": comparison.get("verdict"), "median_mean_gap": comparison.get("median_mean_gap"),
                                    "within_2se": comparison.get("within_2se"), "passes": comparison.get("passes"),
                                    "run_id": doc.get("run_id"), "measured_at": doc.get("measured_at"), "checkpoint": doc.get("checkpoint"),
                                    "engine": doc.get("engine"), "gpu": doc.get("gpu")}
            if comparison.get("method") == "sampled" or doc.get("method") == "sampled":  # the published weights sampled, like the lab's API
                section["reference"].update({"method": "sampled", "similarity_percent": comparison.get("similarity_percent"), "p_value": comparison.get("p_value"),
                                             "compared": comparison.get("compared"), "samples": comparison.get("samples"), "test_verdict": comparison.get("test_verdict"),
                                             "self_similarity_percent": comparison.get("self_similarity_percent"), "margin": comparison.get("margin")})
    summary: dict[str, Any] = latest["summary"]
    entries: list[dict[str, Any]] = []
    for name, data in summary.items():
        entry: dict[str, Any] = {
            "target": name, "model": data.get("model"), "role": data.get("role"), "upstream_model": data.get("upstream_model"),
            "claimed_revision": data.get("claimed_revision"), "notes": data.get("notes", ""), "skipped": data.get("skipped"),
            "counts": data.get("counts"), "latency_s": data.get("latency_s"), "served_models": data.get("served_models"),
            "providers": data.get("providers"), "canaries": data.get("canary_summary"),
            "thresholds_used": thresholds_for(data) if not data.get("skipped") else None,
            "route_model_id": data.get("route_model_id"),
        }
        if not data.get("skipped"):
            if previous and name in previous["summary"] and not previous["summary"][name].get("skipped"):
                entry["drift"] = _comparison(previous["summary"][name], data, permutations=permutations, kernel_tokens=kernel_tokens, alpha=alpha)
                entry["drift"]["against_run"] = previous["metadata"]["run_id"]
            references = [(other, odata) for other, odata in summary.items()
                          if other != name and odata.get("model") == data.get("model") and odata.get("role") == "reference" and not odata.get("skipped")]
            if references and data.get("role") != "reference":
                other, odata = references[0]
                entry["reference"] = _comparison(odata, data, permutations=permutations, kernel_tokens=kernel_tokens, alpha=alpha)
                entry["reference"]["against"] = other
            peers = [(other, odata) for other, odata in summary.items()
                     if other != name and odata.get("model") == data.get("model") and odata.get("role") == "candidate" and not odata.get("skipped")]
            if peers and data.get("role") == "candidate":
                details = []
                agree = disagree = 0
                for other, odata in peers:
                    comparison = _comparison(odata, data, permutations=permutations, kernel_tokens=kernel_tokens, alpha=alpha)
                    if comparison["verdict"] == "agree":
                        agree += 1
                    elif comparison["verdict"] == "disagree":
                        disagree += 1
                    details.append({"peer": other, "verdict": comparison["verdict"],
                                    "deterministic_exact_rate": comparison["deterministic"]["exact_rate"],
                                    "sampled_p_value": comparison["sampled"]["p_value"]})
                entry["consensus"] = {"peers": len(peers), "agree": agree, "disagree": disagree, "details": details}
        if name in sections:
            entry["fingerprint"] = sections[name]
        entry["status"] = classify(entry)
        entries.append(entry)
    # targets that are only fingerprinted (Lebrel's route, other hosts): an entry of their own, from the newest run that probed each
    fp_data: dict[str, dict[str, Any]] = {}
    for run in reversed(fp_loaded):
        for name, data in run["summary"].items():
            if name not in fp_data or (fp_data[name].get("skipped") and not data.get("skipped")):
                fp_data[name] = data
    for name, section in sections.items():
        if name in summary:
            continue
        data = fp_data.get(name) or {}
        entry = {"target": name, "model": data.get("model"), "role": data.get("role"), "upstream_model": data.get("upstream_model"),
                 "notes": data.get("notes", ""), "skipped": data.get("skipped"), "counts": data.get("counts"), "latency_s": data.get("latency_s"),
                 "route_model_id": data.get("route_model_id"), "declared_quantization": data.get("declared_quantization"),
                 "fingerprint_only": True, "fingerprint": section}
        entry["status"] = classify(entry)
        entries.append(entry)
    for entry in entries:  # the route a customer calls, on every entry that belongs to it
        if entry.get("route_model_id") is None and entry["target"] in fp_data:
            entry["route_model_id"] = fp_data[entry["target"]].get("route_model_id")
    return {
        "version": 1,
        "generated_at": now,
        "run_id": latest["metadata"]["run_id"] or (fp_latest["metadata"]["run_id"] if fp_latest else None),
        "previous_run_id": previous["metadata"]["run_id"] if previous else None,
        "battery_id": battery,
        "battery_version": latest["metadata"].get("battery_version"),
        "thresholds": {"min_exact_rate": DEFAULT_MIN_EXACT_RATE, "min_prefix_agreement": DEFAULT_MIN_PREFIX_AGREEMENT,
                       "alpha": alpha, "permutations": permutations, "kernel_tokens": kernel_tokens,
                       "min_calls": MIN_CALLS, "max_error_rate": MAX_ERROR_RATE},
        "fingerprint_run_id": fp_latest["metadata"]["run_id"] if fp_latest else None,
        "fingerprint_version": FINGERPRINT_VERSION if fp_latest else None,
        "fingerprint_thresholds": {"match_percent": MATCH_PERCENT, "mismatch_percent": MISMATCH_PERCENT, "min_compared": MIN_COMPARED,
                                   "informative_max_top": INFORMATIVE_MAX_TOP,
                                   "reference_window_seconds": HISTORY_SECONDS, "stale_after_seconds": FINGERPRINT_STALE_SECONDS,
                                   "margin_within_nats": MARGIN_WITHIN_NATS, "margin_outside_nats": MARGIN_OUTSIDE_NATS,
                                   "cloud_min_passes": CLOUD_MIN_PASSES, "hours": HOURS} if fp_latest else None,
        "fingerprint_commitment": fp_latest["metadata"].get("commitment") if fp_latest else None,
        "references": reference_docs,
        "entries": entries,
    }


FINGERPRINT_TEXT = {
    "match": "the first word's probabilities match the lab's own API within the published threshold",
    "differs": "the probabilities differ from the lab's own API more than noise; needs another run",
    "mismatch": "the probabilities or the token counts do not match the lab's own API",
    "receipt_failed": "a signed route receipt did not verify",
    "anchor": "the lab's own API, compared with its previous check",
}

STATUS_TEXT = {
    "consistent": "consistent with the compared deployments within the published thresholds",
    "suspect": "one of the two tests disagrees; needs another run before any conclusion",
    "drift": "this target changed against its own previous run",
    "divergent": "disagrees with the reference or with the other hosts of the same model",
    "disputed": "two hosts disagree and there is no third to break the tie",
    "unverified": "probed, but nothing to compare against yet",
    "insufficient": "too few calls to say anything",
    "error": "most probes failed",
}


def render_html(board: dict[str, Any]) -> str:
    rows = []
    for e in board["entries"]:
        canaries = e.get("canaries") or {}
        needle = ", ".join(f"{k}: {v}" for k, v in (canaries.get("needle") or {}).items()) or "-"
        latency = e.get("latency_s") or {}
        p50 = f"{latency['p50']:.2f}s" if latency.get("p50") is not None else "-"
        counts = e.get("counts") or {}
        drift = e.get("drift", {}).get("verdict", "-") if e.get("drift") else "-"
        reference = e.get("reference", {}).get("verdict", "-") if e.get("reference") else "-"
        consensus = e.get("consensus")
        consensus_text = f"{consensus['agree']} agree / {consensus['disagree']} disagree of {consensus['peers']}" if consensus else "-"
        fp = e.get("fingerprint") or {}
        fp_text = "-" if not fp else (f"{fp.get('verdict')} {fp.get('similarity_percent')}%" if fp.get("similarity_percent") is not None else str(fp.get("verdict")))
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in (
            e["target"], e.get("model"), e.get("role"), e["status"], fp_text, drift, reference, consensus_text,
            canaries.get("system_leak", "-"), canaries.get("over_refusal_count", "-"), canaries.get("max_tokens", "-"), needle,
            canaries.get("tools", "-"), p50, f"{counts.get('errors', 0)}/{counts.get('total', 0)}")) + "</tr>")
    legend = "".join(f"<li><b>{html.escape(k)}</b>: {html.escape(v)}</li>" for k, v in STATUS_TEXT.items())
    thresholds = html.escape(json.dumps(board["thresholds"]))
    generated = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(board["generated_at"]))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Lebrel watch board</title>
<style>body{{font-family:system-ui,sans-serif;margin:2rem;color:#111}}table{{border-collapse:collapse;font-size:14px}}
td,th{{border:1px solid #ccc;padding:4px 8px;text-align:left}}th{{background:#f3f3f3}}</style></head><body>
<h1>Lebrel watch board</h1>
<p>Run {html.escape(str(board['run_id']))}, battery {html.escape(str(board['battery_id']))}, generated {generated}.
Every row is Lebrel's own probes against a public endpoint; no client traffic is involved.</p>
<table><thead><tr><th>target</th><th>model</th><th>role</th><th>status</th><th>fingerprint</th><th>drift</th><th>reference</th><th>consensus</th>
<th>system leak</th><th>over-refusal</th><th>max_tokens</th><th>needle</th><th>tools</th><th>latency p50</th><th>errors</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>What the words mean</h2><ul>{legend}</ul>
<p>Thresholds used: <code>{thresholds}</code>. Statistical, sampled, black-box. See the method notes before quoting a status.</p>
</body></html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--fingerprints", type=Path, default=None, help="directory of watch.fingerprint runs")
    parser.add_argument("--reference", type=Path, default=None, help="directory of reference documents (watch.reference.analyze --export)")
    parser.add_argument("--out", required=True, type=Path, help="board.json path")
    parser.add_argument("--html", type=Path, default=None)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--kernel-tokens", type=int, default=DEFAULT_KERNEL_TOKENS)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    args = parser.parse_args(argv)
    try:
        board = build_board(args.runs, fingerprints_dir=args.fingerprints, reference_dir=args.reference, permutations=args.permutations, kernel_tokens=args.kernel_tokens, alpha=args.alpha)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"cannot build board: {error}", file=sys.stderr)
        return 2
    args.out.write_text(json.dumps(board, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.html:
        args.html.write_text(render_html(board), encoding="utf-8")
    for entry in board["entries"]:
        fp = entry.get("fingerprint") or {}
        extra = f" fingerprint {fp.get('verdict')} {fp.get('similarity_percent')}" if fp else ""
        print(f"{entry['status']:12s} {entry['target']} ({entry.get('model')}){extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
