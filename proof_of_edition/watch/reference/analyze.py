"""Read a reference run and compare it with what the watch recorded: the lab's API (all its passes) and every host.

    python -m proof_of_edition.watch.reference.analyze --result runs-reference/<id>.json --prompts runs-reference/prompts-<week>.json \
        --runs <dir with fingerprint runs> [--anchor deepseek/api] [--out report.json]

The reference gives raw log-probabilities (before temperature). The lab's API may apply a temperature of its own,
so the report also fits one: the ratio of log-probability gaps between the two, over the words both rank, is the
temperature the API applied relative to the raw model. Distributions are then re-scaled to that temperature before
the second comparison.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

from proof_of_edition.watch import fingerprint
from proof_of_edition.watch.reference.prepare import fetch_format

TOP = fingerprint.TOP_LOGPROBS


def _byte_map():
    """Inverse of the byte-level BPE alphabet (GPT-2 style)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class TokenBytes:
    """Token id -> the UTF-8 bytes the lab's API reports for that token."""

    def __init__(self, tokenizer_path: Path):
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.inverse = _byte_map()
        added = self.tokenizer.get_added_tokens_decoder()
        self.special = {int(i): t.content for i, t in added.items()}

    def hex(self, token_id: int) -> str:
        if token_id in self.special:
            return self.special[token_id].encode("utf-8").hex()
        s = self.tokenizer.id_to_token(token_id)
        if s is None:
            return f"id{token_id}".encode("utf-8").hex()
        try:
            return bytes(self.inverse[ch] for ch in s).hex()
        except KeyError:
            return s.encode("utf-8").hex()


def _logsumexp(values: list[float]) -> float:
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def rescale(top: list[tuple[int, float]], temperature: float) -> list[tuple[int, float]]:
    """Raw log-probabilities at temperature T, renormalised over the words we have (the rest of the vocabulary
    carries negligible mass for the prompts that matter)."""
    scaled = [(tid, lp / temperature) for tid, lp in top]
    z = _logsumexp([lp for _, lp in scaled])
    return [(tid, lp - z) for tid, lp in scaled]


def as_first(top: list[tuple[int, float]], tokens: TokenBytes, temperature: float = 1.0) -> dict[str, float]:
    """A fingerprint-style {token bytes hex: logprob} of the top-20 words, like the API's top_logprobs."""
    ranked = rescale(top, temperature) if temperature != 1.0 else list(top)
    out: dict[str, float] = {}
    for tid, lp in sorted(ranked, key=lambda kv: -kv[1])[:TOP]:
        key = tokens.hex(int(tid))
        out[key] = max(out.get(key, -math.inf), lp)
    return out


def result_passes(result: dict[str, Any], config: int = 0) -> list[dict[str, Any]]:
    """The passes of one server configuration of a run (older runs kept them at the top level)."""
    configs = result.get("configs")
    if configs:
        return list((configs[config] or {}).get("passes") or [])
    return list(result.get("passes") or [])


def reference_summary(result: dict[str, Any], tokens: TokenBytes, temperature: float = 1.0, *, modes: tuple[str, ...] = ("serial", "batched", "busy"),
                      labels: tuple[str, ...] | None = None, config: int = 0) -> dict[str, Any]:
    """The reference run as a fingerprint summary entry: first pass under ``prompts``, the rest under ``passes``."""
    passes = []
    for p in result_passes(result, config):
        if p.get("mode") not in modes or (labels is not None and p.get("label") not in labels):
            continue
        entry: dict[str, Any] = {}
        for r in p.get("results") or []:
            if not r or r.get("error"):
                entry[r["id"]] = {"error": r.get("error", "no result"), "first": None, "prompt_tokens": None} if r else None
                continue
            entry[r["id"]] = {"status": 200, "first": as_first(r["top"], tokens, temperature), "prompt_tokens": r.get("prompt_tokens"),
                              "error": None, "latency_s": r.get("latency_s")}
        passes.append(entry)
    if not passes:
        return {"prompts": {}, "passes": []}
    return {"model": "reference", "prompts": passes[0], "passes": passes[1:]}


def fit_temperature(api_passes: list[dict[str, Any]], result: dict[str, Any], floor: float = 0.01, config: int = 0) -> dict[str, Any]:
    """T such that the API's log-gaps equal the raw model's log-gaps divided by T, over words both rank above the floor."""
    ratios: list[float] = []
    per_prompt: dict[str, float] = {}
    raw_by_id: dict[str, list[tuple[int, float]]] = {}
    for p in result_passes(result, config):
        for r in p.get("results") or []:
            if r and not r.get("error") and r["id"] not in raw_by_id:
                raw_by_id[r["id"]] = r["top"]
    lf = math.log(floor)
    tokens = TokenBytes(fetch_format()["paths"]["tokenizer.json"])
    for pid, raw in raw_by_id.items():
        raw_hex = {}
        for tid, lp in raw:
            k = tokens.hex(int(tid))
            raw_hex[k] = max(raw_hex.get(k, -math.inf), lp)
        seen = [p.get(pid) for p in api_passes if isinstance(p.get(pid), dict) and p[pid].get("first")]
        if not seen:
            continue
        api = seen[0]["first"]
        shared = [k for k, v in api.items() if v >= lf and k in raw_hex]
        if len(shared) < 2:
            continue
        top = max(shared, key=lambda k: api[k])
        local = []
        for k in shared:
            if k == top:
                continue
            d_api = api[top] - api[k]
            d_raw = raw_hex[top] - raw_hex[k]
            if d_api > 0.05 and d_raw > 0:
                local.append(d_raw / d_api)
        if local:
            per_prompt[pid] = round(statistics.median(local), 4)
            ratios.extend(local)
    return {"temperature": round(statistics.median(ratios), 4) if ratios else None, "pairs": len(ratios), "per_prompt": per_prompt}


def mean_test(a_passes: list[dict[str, Any]], b_passes: list[dict[str, Any]], *, floor: float = fingerprint.CLOUD_FLOOR, min_n: int = fingerprint.CLOUD_MIN_PASSES) -> dict[str, Any]:
    """The cloud comparison the board uses (watch.fingerprint.cloud)."""
    return fingerprint.cloud(a_passes, b_passes, floor=floor, min_n=min_n)


def load_runs(runs_dir: Path) -> list[dict[str, Any]]:
    runs = []
    for d in sorted(runs_dir.iterdir()):
        s, m = d / "summary.json", d / "run.json"
        if s.exists() and m.exists():
            runs.append({"meta": json.loads(m.read_text()), "summary": json.loads(s.read_text())})
    return runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--prompts", required=True, type=Path, help="the prompts document prepare.py wrote")
    parser.add_argument("--runs", required=True, type=Path, help="directory of fingerprint runs (run.json + summary.json each)")
    parser.add_argument("--anchor", default="deepseek/api")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--config", type=int, default=0, help="which server configuration of the run to read")
    parser.add_argument("--export", type=Path, default=None, help="write the reference document the board publishes")
    args = parser.parse_args(argv)
    result = json.loads(args.result.read_text())
    tokens = TokenBytes(fetch_format()["paths"]["tokenizer.json"])
    runs = load_runs(args.runs)
    report: dict[str, Any] = {"result": str(args.result), "runs": len(runs), "anchor": args.anchor, "config": args.config,
                              "extra_args": ((result.get("configs") or [{}])[args.config] or {}).get("extra_args") if result.get("configs") else None}
    config = args.config

    raw = reference_summary(result, tokens, config=config)
    # 1. the reference against itself: every pass against every other (the spread batching alone produces)
    passes = result_passes(result, config)
    labels = [p.get("label") or f"{p.get('mode')}-{i}" for i, p in enumerate(passes)]
    singles = [reference_summary(result, tokens, labels=(lab,), config=config) if p.get("label") else None for lab, p in zip(labels, passes)]
    if all(singles):
        matrix = {}
        for la, a in zip(labels, singles):
            matrix[la] = {lb: fingerprint.compare({"prompts": a["prompts"]}, {"prompts": b["prompts"]})["similarity_percent"] for lb, b in zip(labels, singles)}
        report["reference_spread"] = matrix
    serial = reference_summary(result, tokens, modes=("serial",), config=config)
    batched = reference_summary(result, tokens, modes=("batched",), config=config)
    if serial["passes"]:
        report["reference_vs_itself"] = {"serial_vs_serial": fingerprint.compare({"prompts": serial["prompts"]}, {"prompts": serial["passes"][0]})}
    if batched["prompts"]:
        report.setdefault("reference_vs_itself", {})["serial_vs_batched"] = fingerprint.compare({"prompts": serial["prompts"]}, batched)

    # 2. the lab's API (every pass of every run) against the raw reference, then the temperature fit
    api_passes: list[dict[str, Any]] = []
    for run in reversed(runs):  # newest first, as the board's reference window is built
        entry = run["summary"].get(args.anchor)
        if entry and not entry.get("skipped"):
            api_passes.extend(fingerprint.anchor_passes(entry))
    report["api_passes"] = len(api_passes)
    if api_passes:
        # the API against itself, pass by pass: the spread the lab's own serving produces, to set beside ours
        similarities = []
        for i, a in enumerate(api_passes):
            for b in api_passes[i + 1:]:
                v = fingerprint.compare({"prompts": a}, {"prompts": b})["similarity_percent"]
                if v is not None:
                    similarities.append(v)
        report["api_spread"] = {"pairs": len(similarities), "min": min(similarities) if similarities else None,
                                "median": statistics.median(similarities) if similarities else None, "max": max(similarities) if similarities else None}
        report["api_vs_reference_raw"] = fingerprint.compare(api_passes, raw)
        fit = fit_temperature(api_passes, result, config=config)
        report["temperature_fit"] = fit
        if fit["temperature"]:
            scaled = reference_summary(result, tokens, fit["temperature"], config=config)
            report["api_vs_reference_scaled"] = fingerprint.compare(api_passes, scaled)
            # every pass of the API on its own against the reference: the lab serves several variants, this says
            # which of them the published weights are
            report["api_passes_vs_reference_scaled"] = []
            for run in runs:
                entry = run["summary"].get(args.anchor)
                if not entry or entry.get("skipped"):
                    continue
                for index, one in enumerate(fingerprint.anchor_passes(entry)):
                    verdict = fingerprint.compare(fingerprint.anchor_passes(scaled), {"prompts": one})
                    fps = sorted({(r or {}).get("system_fingerprint") for r in one.values() if isinstance(r, dict)} - {None})
                    report["api_passes_vs_reference_scaled"].append({"run_id": run["meta"]["run_id"], "pass": index, "system_fingerprints": fps,
                                                                     "similarity_percent": verdict["similarity_percent"], "verdict": verdict["verdict"],
                                                                     "compared": verdict["compared"]})
            # 3. every other target of the newest run that claims the same model, against the reference and against the API
            newest = runs[-1]["summary"]
            model = (newest.get(args.anchor) or {}).get("model")
            report["targets_vs_reference_scaled"] = {}
            for name, entry in newest.items():
                if name == args.anchor or not isinstance(entry, dict) or entry.get("skipped") or not entry.get("prompts") or entry.get("model") != model:
                    continue
                report["targets_vs_reference_scaled"][name] = {
                    "vs_reference": fingerprint.compare(fingerprint.anchor_passes(scaled), entry),
                    "vs_api": fingerprint.compare(api_passes, entry), "model": entry.get("model")}
            # 4. clouds, not points: mean log-probabilities over passes, with the reference and the API each split
            # against itself as the null
            # raw log-probabilities: the fitted temperature is 1 within its noise (0.99 and 1.10 on two runs), and a
            # wrong scale would masquerade as an offset
            ref_passes = [reference_summary({"passes": [p]}, tokens)["prompts"] for p in passes]
            batched_passes = [reference_summary({"passes": [p]}, tokens)["prompts"] for p in passes if p.get("mode") in ("batched", "busy")]
            cloud = batched_passes if len(batched_passes) >= 4 else ref_passes
            half = len(cloud) // 2
            report["clouds"] = {"reference_null": mean_test(cloud[:half], cloud[half:]) if half >= 2 else None,
                                "api_null": mean_test(api_passes[: len(api_passes) // 2], api_passes[len(api_passes) // 2:]),
                                "api_vs_reference": mean_test(api_passes, cloud), "targets": {}}
            for name, entry in newest.items():
                if name == args.anchor or not isinstance(entry, dict) or entry.get("skipped") or entry.get("model") != model:
                    continue
                target_passes = []
                for run in runs:
                    e = run["summary"].get(name)
                    if e and not e.get("skipped"):
                        target_passes.extend(fingerprint.anchor_passes(e))
                report["clouds"]["targets"][name] = {"vs_api": mean_test(target_passes, api_passes), "vs_reference": mean_test(target_passes, cloud)}
    text = json.dumps(report, indent=1)
    if args.out:
        args.out.write_text(text)
    if args.export and report.get("clouds"):
        anchor_entry = runs[-1]["summary"].get(args.anchor) or {}
        prompts_document = json.loads(args.prompts.read_text())
        config_doc = (result.get("configs") or [{}])[args.config] if result.get("configs") else {}
        document = {"version": 1, "model": anchor_entry.get("model"), "anchor": args.anchor, "run_id": result.get("result_id"),
                    "measured_at": int(result.get("finished_at") or 0),
                    "checkpoint": {"repo": prompts_document.get("repo"), "revision": prompts_document.get("revision"),
                                   "format_sha256": prompts_document.get("format_sha256")},
                    "engine": result.get("vllm_version") or result.get("image"), "image": result.get("image"), "gpu": result.get("gpu"),
                    "config": config_doc.get("extra_args"), "prompt_ids": result.get("prompt_ids"),
                    "api_runs": [run["meta"]["run_id"] for run in runs if run["summary"].get(args.anchor)],
                    "api_vs_reference": report["clouds"]["api_vs_reference"], "reference_null": report["clouds"]["reference_null"],
                    "api_null": report["clouds"]["api_null"], "set_similarity_percent": (report.get("api_vs_reference_raw") or {}).get("similarity_percent"),
                    "margin": {"within_nats": fingerprint.MARGIN_WITHIN_NATS, "outside_nats": fingerprint.MARGIN_OUTSIDE_NATS}}
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_text(json.dumps(document, indent=1))
        print(f"reference document -> {args.export}: {document['api_vs_reference'].get('verdict')} ({document['api_vs_reference'].get('median_mean_gap')} nats)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
