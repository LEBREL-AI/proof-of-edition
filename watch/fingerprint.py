"""Model fingerprint: the first word's probability distribution, compared with the lab's own API.

Every model hesitates in its own way: for the first word of an answer it gives each candidate
token a probability. The watch asks a set of short prompts with ``max_tokens=1``, ``logprobs`` and
``top_logprobs=20`` at the default temperature (at temperature 0 some labs return a degenerate
0/-9999 distribution, which carries nothing) and keeps the distribution of the first position,
whose context is the prompt alone and therefore the same on every target.

The reference is not one call. A lab's own API does not always answer the same way: on 24 Sep 2026
two identical calls to DeepSeek V4 Pro gave nearly the same distribution on some prompts and a very
different one on others (10 of 24), as if the model were served from more than one variant. So the
anchor (the lab's own API for the model) is probed twice every run and kept for six hours, and a
target matches a prompt when its distribution is close to any distribution the lab itself produced
for that prompt in that window; a longer window would make matching easier by accumulation. Lebrel's route always lands on one of them; a host serving another
precision lands on none.

Three checks per target:

* similarity: per prompt, the smallest mean absolute difference of log-probability, over the tokens
  both give at least 1%, against the anchor's recent distributions; the median over the informative
  prompts is the log gap and ``100 * exp(-gap)`` the similarity in percent. A prompt is informative
  when the lab itself hesitates (its top first word under 90%): "yes or no" and "translate this"
  prompts are answered alike by every precision of a model and would only dilute the median. On 24 Sep 2026 Lebrel's routes measured 99.98%
  (Flash) and 100% (Pro); hosts of the same models through OpenRouter 46 to 90%, one fp8 host of V4 Pro
  100%: the method judges what is served, not who serves it.
* tokens: the prompt tokens each target bills for the same prompt. The same model behind the same
  template bills the same count; a constant offset is another template or a hidden instruction; an
  offset that varies from prompt to prompt is another tokenizer, therefore another model.
* receipts (Lebrel's own route): every answer's signed route receipt is verified against the route
  manifest with the pinned router key, digests of the exact request and answer included.

Part of the prompt set is secret each ISO week: its SHA-256 is on the board during the week and the
prompts are published once the week is over, so nobody can recognise the probes while they run and
anybody can check the old verdicts afterwards.

Usage:
  python3 -m watch.fingerprint --targets targets.json --out runs-fingerprint [--only a,b] [--secret-pool pool.json]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import math
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from watch import sampled as sampled_fp
from watch.client import USER_AGENT, Target, load_targets

FINGERPRINT_VERSION = 1
TOP_LOGPROBS = 20
FLOOR_PROBABILITY = 0.01
MAX_GAP = 5.0            # no shared token above the floor: as different as two distributions get
MATCH_PERCENT = 97.0     # at or above: the fingerprint matches the anchor
MISMATCH_PERCENT = 90.0  # below: it does not; in between: it differs and needs another run
MIN_COMPARED = 12
INFORMATIVE_MAX_TOP = 0.9  # a prompt tells models apart only if the lab itself hesitates: top word under 90%
SECRET_PER_WEEK = 24
ANCHOR_SAMPLES = 2                 # passes of the anchor per run
HISTORY_SECONDS = 6 * 3600         # how long the anchor's distributions stay in the reference (12 passes at the hourly pace)
SAMPLED_HISTORY_SECONDS = 24 * 3600  # sampled answers pool over a day: the estimate needs the numbers
# The implementation margin, from Lebrel's own runs of the published DeepSeek-V4.1-Flash weights (24 Sep 2026): the
# lab's API sits 0.20 nats of mean log-probability from that run, two halves of the run sit 0.18 from each other, and
# the same weights under two kernel sets answer 0.5 nats apart. Another implementation of the same weights is judged
# by its cloud of answers, never by one call.
MARGIN_WITHIN_NATS = 0.30
MARGIN_OUTSIDE_NATS = 0.45
CLOUD_MIN_PASSES = 3          # passes on each side before a cloud says anything
CLOUD_FLOOR = 0.05            # words that matter: at least 5% on average on either side
REGIME_SAME_GAP = 0.03        # two distributions closer than this are the same numbers
HOURS = 24                    # the hourly series the board shows

PUBLIC_PROMPTS = [
    "Name one color.", "Pick a number between 1 and 10. Reply with the number only.", "Write one word that comes to mind.",
    "Name a fruit.", "Name a European capital.", "Say a random animal.", "Give me a name for a cat.", "Name a programming language.",
    "Complete: The quick brown fox", "Continue the sequence: 2, 4, 8, 16,", "Translate 'good morning' into Spanish.", "What is 17 times 23?",
    "Write a haiku about rain.", "Describe the ocean in five words.", "请用一个词形容春天。", "Écris une phrase sur Paris.",
    "def fibonacci(n):", "SELECT * FROM users WHERE", "What year did the Berlin Wall fall?", "List three prime numbers.",
    "Give a one-line motto for a bakery.", "Name a chemical element.", "Finish the proverb: A penny saved is", "Name a musical instrument.",
]

_TEMPLATES = [
    "Name a {c}.", "Pick one {c}.", "Give me an example of a {c}.", "Say the first {c} that comes to mind.", "Suggest a {c}.",
    "Tell me one {c} and nothing else.", "Choose a {c} at random.", "What is your favourite {c}? One answer.",
    "Nombra un {c_es}.", "Dis-moi un {c_fr}.",
]
_CATEGORIES = [
    ("color", "color", "couleur"), ("fruit", "fruto", "fruit"), ("animal", "animal", "animal"), ("city", "ciudad", "ville"),
    ("country", "país", "pays"), ("planet", "planeta", "planète"), ("metal", "metal", "métal"), ("gemstone", "piedra preciosa", "pierre précieuse"),
    ("flower", "flor", "fleur"), ("bird", "pájaro", "oiseau"), ("fish", "pez", "poisson"), ("tree", "árbol", "arbre"),
    ("vegetable", "vegetal", "légume"), ("spice", "especia", "épice"), ("dessert", "postre", "dessert"), ("drink", "bebida", "boisson"),
    ("sport", "deporte", "sport"), ("board game", "juego de mesa", "jeu de société"), ("musical instrument", "instrumento musical", "instrument de musique"),
    ("programming language", "lenguaje de programación", "langage de programmation"), ("river", "río", "fleuve"), ("mountain", "monte", "montagne"),
    ("island", "isla", "île"), ("language", "idioma", "langue"), ("currency", "moneda", "monnaie"), ("painter", "pintor", "peintre"),
    ("composer", "compositor", "compositeur"), ("scientist", "científico", "scientifique"), ("constellation", "constelación", "constellation"),
    ("tool", "herramienta", "outil"), ("dog breed", "perro de raza", "chien de race"), ("cheese", "queso", "fromage"), ("dance", "baile", "danse"),
    ("film genre", "género de cine", "genre de film"), ("emotion", "sentimiento", "sentiment"), ("unit of measurement", "unidad de medida", "unité de mesure"),
]


def default_secret_pool() -> list[str]:
    """Used only when no private pool is configured; the private pool on the server is the real one."""
    pool = []
    for template in _TEMPLATES:
        for en, es, fr in _CATEGORIES:
            pool.append(template.format(c=en, c_es=es, c_fr=fr))
    return pool


def iso_week(at: float) -> str:
    return time.strftime("%G-W%V", time.gmtime(at))


def secret_prompts(seed: bytes, week: str, pool: list[str], count: int = SECRET_PER_WEEK) -> list[str]:
    """The week's secret prompts: a deterministic sample of the private pool keyed by the secret seed."""
    if len(pool) < count:
        raise ValueError(f"secret pool has {len(pool)} prompts, needs at least {count}")
    digest = hmac.new(seed, week.encode("utf-8"), hashlib.sha256).digest()
    return random.Random(digest).sample(sorted(set(pool)), count)


def pool_id(pool: list[str]) -> str:
    """Eight hex characters naming a secret pool: two pools never share a prompt id."""
    return hashlib.sha256("\n".join(sorted(set(pool))).encode("utf-8")).hexdigest()[:8]


def reveal_document(week: str, pool: str, prompts: list[str]) -> bytes:
    """The canonical bytes revealed once the week is over; their SHA-256 is the week's commitment."""
    return json.dumps({"week": week, "pool": pool, "prompts": prompts}, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def commitment(week: str, prompts: list[str], pool: str = "") -> str:
    """SHA-256 of the canonical reveal document; published on the board while the week runs."""
    return hashlib.sha256(reveal_document(week, pool, prompts)).hexdigest()


def prompt_set(at: float, *, seed: bytes | None, pool: list[str] | None) -> tuple[list[tuple[str, str]], dict[str, Any] | None]:
    """[(prompt id, text)] for a run, and the week's commitment (None without a secret seed)."""
    items = [(f"pub-{i:02d}", text) for i, text in enumerate(PUBLIC_PROMPTS)]
    if not seed:
        return items, None
    week = iso_week(at)
    chosen_pool = pool or default_secret_pool()
    pid = pool_id(chosen_pool)
    secret = secret_prompts(seed, week, chosen_pool)
    # the week and the pool are in the id: another week's or another pool's sec-00 is another prompt
    items += [(f"sec-{week.replace('-', '')}-{pid}-{i:02d}", text) for i, text in enumerate(secret)]
    return items, {"week": week, "pool": pid, "sha256": commitment(week, secret, pid), "count": len(secret)}


# ---------------------------------------------------------------- measurement

def first_distribution(document: dict[str, Any]) -> dict[str, float] | None:
    """{token bytes as hex: logprob} of the first generated position, or None when the target gave no logprobs."""
    choices = document.get("choices") or []
    content = ((choices[0] if choices else {}).get("logprobs") or {}).get("content") if choices else None
    if not isinstance(content, list) or not content:
        return None
    alternatives = content[0].get("top_logprobs")
    if not isinstance(alternatives, list) or not alternatives:
        return None
    out: dict[str, float] = {}
    for item in alternatives:
        if not isinstance(item, dict) or not isinstance(item.get("logprob"), (int, float)):
            continue
        raw = item.get("bytes")
        key = bytes(raw).hex() if isinstance(raw, list) and all(isinstance(b, int) and 0 <= b < 256 for b in raw) else str(item.get("token", "")).encode("utf-8").hex()
        logprob = float(item["logprob"])
        if logprob < -1000:  # a degenerate distribution (temperature 0 on some labs): nothing to compare
            continue
        out[key] = max(out.get(key, -math.inf), logprob)
    return out or None


def log_gap(p: dict[str, float], q: dict[str, float], floor: float = FLOOR_PROBABILITY) -> float | None:
    """Mean |log p - log q| over the tokens both give at least ``floor``; MAX_GAP when none is shared."""
    lf = math.log(floor)
    above_p = {k for k, v in p.items() if v >= lf}
    above_q = {k for k, v in q.items() if v >= lf}
    if not above_p and not above_q:
        return None
    shared = [k for k in above_p & above_q]
    if not shared:
        return MAX_GAP
    return statistics.fmean(abs(p[k] - q[k]) for k in shared)


def similarity_percent(gaps: list[float]) -> float | None:
    return round(100.0 * math.exp(-statistics.median(gaps)), 2) if gaps else None


@dataclass
class ProbeResult:
    prompt_id: str
    status: int | None
    first: dict[str, float] | None
    prompt_tokens: int | None
    text: str | None
    system_fingerprint: str | None
    served_model: str | None
    provider: str | None
    latency_s: float
    error: str | None
    receipt: str | None = None  # "verified", "failed: …", "missing" for targets whose receipts are checked

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "first": self.first, "prompt_tokens": self.prompt_tokens, "system_fingerprint": self.system_fingerprint,
                "served_model": self.served_model, "provider": self.provider, "latency_s": round(self.latency_s, 3), "error": self.error, "receipt": self.receipt}


def _request_body(target: Target, text: str) -> dict[str, Any]:
    body = dict(target.body)
    if target.sampled:  # no probabilities asked for: the first word of a short answer, at the lab's own sampling settings
        if target.overrides:
            body.update({k: v for k, v in target.overrides.items() if k in ("temperature", "top_p", "reasoning_effort")})
        body.update({"model": target.upstream_model, "messages": [{"role": "user", "content": text}], "max_tokens": target.sampled_max_tokens, "stream": False})
        return body
    body.update({"model": target.upstream_model, "messages": [{"role": "user", "content": text}], "max_tokens": 1,
                 "logprobs": True, "top_logprobs": TOP_LOGPROBS, "stream": False})
    if target.overrides:  # e.g. a host that accepts at most 5 alternatives
        body.update({k: v for k, v in target.overrides.items() if k in ("top_logprobs",)})
    body.pop("temperature", None)  # the default temperature: temperature 0 degenerates the distribution on some labs
    return body


def probe_once(target: Target, prompt_id: str, text: str, *, timeout: float = 60.0, retries: int = 2, opener=None,
               receipt_check: Callable[[bytes, str, str], str] | None = None, sleep=time.sleep) -> ProbeResult:
    body = _request_body(target, text)
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT, **target.headers}
    key = target.api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    opener = opener or urllib.request.build_opener()
    attempt = 0
    while True:
        attempt += 1
        started = time.perf_counter()
        try:
            request = urllib.request.Request(f"{target.base_url.rstrip('/')}/chat/completions", data=payload, headers=headers, method="POST")
            with opener.open(request, timeout=timeout) as response:
                raw = response.read()
                status = getattr(response, "status", 200)
                receipt_header = response.headers.get("Proof-Of-Edition-Receipt") if response.headers else None
            document = json.loads(raw.decode("utf-8"))
            choices = document.get("choices") or [{}]
            message = (choices[0] or {}).get("message") or {}
            text_out = message.get("content") if isinstance(message.get("content"), str) else ""
            usage = document.get("usage") or {}
            result = ProbeResult(prompt_id, status, first_distribution(document),
                                 usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), int) else None,
                                 text_out, document.get("system_fingerprint") if isinstance(document.get("system_fingerprint"), str) else None,
                                 document.get("model") if isinstance(document.get("model"), str) else None,
                                 document.get("provider") if isinstance(document.get("provider"), str) else None,
                                 time.perf_counter() - started, None)
            if receipt_check is not None:
                result.receipt = receipt_check(payload, text_out, receipt_header or "") if receipt_header else "missing"
            return result
        except urllib.error.HTTPError as error:
            detail = ""
            try:
                detail = error.read().decode("utf-8", "replace")[:200]
            except Exception:  # noqa: BLE001 - best effort
                pass
            if error.code in (408, 409, 425, 429, 500, 502, 503, 504) and attempt <= retries:
                sleep(min(2.0 * attempt, 8.0))
                continue
            return ProbeResult(prompt_id, error.code, None, None, None, None, None, None, time.perf_counter() - started, f"HTTP {error.code}: {detail}".strip())
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as error:
            if attempt <= retries:
                sleep(min(2.0 * attempt, 8.0))
                continue
            return ProbeResult(prompt_id, None, None, None, None, None, None, None, time.perf_counter() - started, f"{type(error).__name__}: {error}")


def route_receipt_checker(target: Target, *, opener=None, now: float | None = None) -> Callable[[bytes, str, str], str] | None:
    """For Lebrel's route: verify each answer's receipt against the route manifest with the pinned router key."""
    if not target.receipt_public_key or not target.route_model_id:
        return None
    from receipts.schema import Signed, check_route_manifest, check_route_receipt  # local import: optional dependency path
    public_key = bytes.fromhex(target.receipt_public_key)
    origin = (target.receipt_origin or "https://router.lebrel.ai").rstrip("/")
    slug = target.route_model_id.replace("/", "--")
    request = urllib.request.Request(f"{origin}/m/{slug}/.well-known/proof-of-edition", headers={"Accept": "application/json", "User-Agent": USER_AGENT})
    try:
        with (opener or urllib.request.build_opener()).open(request, timeout=30) as response:
            manifest_doc = json.loads(response.read().decode("utf-8"))
        manifest = Signed(payload=manifest_doc["payload"], signature_b64=manifest_doc["signature"])
        manifest_problems = check_route_manifest(manifest, public_key, now=now)
    except Exception as error:  # noqa: BLE001 - reported per answer
        manifest, manifest_problems = None, [f"manifest unavailable: {type(error).__name__}"]

    def check(prompt: bytes, answer: str, header: str) -> str:
        if manifest is None or manifest_problems:
            return "failed: " + "; ".join(manifest_problems)
        try:
            document = json.loads(base64.b64decode(header))
            signed = Signed(payload=document["payload"], signature_b64=document["signature"])
        except Exception:  # noqa: BLE001
            return "failed: unreadable receipt header"
        problems = check_route_receipt(signed, public_key, manifest, prompt=prompt, response=answer.encode("utf-8"))
        return "verified" if not problems else "failed: " + "; ".join(problems)

    return check


def _one_pass(target: Target, prompts: list[tuple[str, str]], *, concurrency: int, timeout: float, prober, checker, sleep) -> list[ProbeResult]:
    if target.pace_seconds:
        results = []
        for index, item in enumerate(prompts):  # hosts that limit requests per minute: one at a time, spaced
            if index:
                sleep(target.pace_seconds)
            results.append(prober(target, item[0], item[1], timeout=timeout, receipt_check=checker))
        return results
    with ThreadPoolExecutor(max_workers=max(1, target.concurrency or concurrency)) as pool:
        return list(pool.map(lambda item: prober(target, item[0], item[1], timeout=timeout, receipt_check=checker), prompts))


def measure_sampled(target: Target, prompts: list[tuple[str, str]], *, concurrency: int = 4, timeout: float = 60.0,
                    prober: Callable[..., ProbeResult] = probe_once, sleep=time.sleep) -> dict[str, Any]:
    """A sampled target: ``samples`` passes over the prompts; per prompt, the count of each first word."""
    checker = route_receipt_checker(target) if target.receipt_public_key else None
    passes = [_one_pass(target, prompts, concurrency=concurrency, timeout=timeout, prober=prober, checker=checker, sleep=sleep)
              for _ in range(max(2, target.samples))]
    fingerprints: dict[str, int] = {}
    receipts = {"verified": 0, "failed": 0, "missing": 0}
    failures: set[str] = set()
    latencies: list[float] = []
    errors = 0
    per_prompt: dict[str, dict[str, Any]] = {}
    for pid, _ in prompts:
        results = [r for one in passes for r in one if r.prompt_id == pid]
        counts: dict[str, int] = {}
        statuses: dict[str, int] = {}
        tokens: dict[int, int] = {}
        mine: list[float] = []
        for r in results:
            if r.status is not None:
                statuses[str(r.status)] = statuses.get(str(r.status), 0) + 1
            if r.system_fingerprint:
                fingerprints[r.system_fingerprint] = fingerprints.get(r.system_fingerprint, 0) + 1
            if r.receipt is not None:
                key = "verified" if r.receipt == "verified" else ("missing" if r.receipt == "missing" else "failed")
                receipts[key] += 1
                if key == "failed":
                    failures.add(r.receipt)
            word = None if r.error else sampled_fp.first_word(r.text)
            if word is None:
                errors += 1
                continue
            counts[word] = counts.get(word, 0) + 1
            mine.append(r.latency_s)
            latencies.append(r.latency_s)
            if isinstance(r.prompt_tokens, int):
                tokens[r.prompt_tokens] = tokens.get(r.prompt_tokens, 0) + 1
        n = sum(counts.values())
        enough = n >= max(sampled_fp.MIN_SAMPLES_PER_SIDE, len(results) // 2)
        per_prompt[pid] = {"status": max(statuses, key=statuses.get) if statuses else None, "first": sampled_fp.distribution(counts) if enough else None,
                           "counts": counts if enough else None, "n": n, "prompt_tokens": max(tokens, key=tokens.get) if tokens else None,
                           "latency_s": round(statistics.median(mine), 3) if mine else None,
                           "error": None if enough else f"{n} of {len(results)} answers usable",
                           "receipt": ("failed" if any(r.receipt and r.receipt.startswith("failed") for r in results)
                                       else "missing" if any(r.receipt == "missing" for r in results) else "verified") if checker is not None else None}
    return {
        "model": target.model, "role": target.role, "upstream_model": target.upstream_model, "notes": target.notes,
        "anchor": target.fingerprint_anchor, "route_model_id": target.route_model_id, "declared_quantization": target.declared_quantization,
        "sampled": True, "samples": max(2, target.samples), "cadence_hours": target.every_hours,
        "counts": {"total": len(prompts) * len(passes), "errors": errors},
        "latency_s": {"p50": statistics.median(latencies) if latencies else None},
        "system_fingerprints": fingerprints,
        "receipts": receipts | {"failures": sorted(failures)[:3]} if checker is not None else None,
        "prompts": per_prompt,
        "passes": [],
    }


def measure_target(target: Target, prompts: list[tuple[str, str]], *, concurrency: int = 4, timeout: float = 60.0,
                   prober: Callable[..., ProbeResult] = probe_once, sleep=time.sleep) -> dict[str, Any]:
    if target.sampled:
        return measure_sampled(target, prompts, concurrency=concurrency, timeout=timeout, prober=prober, sleep=sleep)
    checker = route_receipt_checker(target) if target.receipt_public_key else None
    results = _one_pass(target, prompts, concurrency=concurrency, timeout=timeout, prober=prober, checker=checker, sleep=sleep)
    extra_passes = []
    for _ in range(max(0, (target.anchor_samples or ANCHOR_SAMPLES) - 1) if target.fingerprint_anchor else 0):
        extra = _one_pass(target, prompts, concurrency=concurrency, timeout=timeout, prober=prober, checker=None, sleep=sleep)
        extra_passes.append({r.prompt_id: r.to_dict() for r in extra})
    fingerprints: dict[str, int] = {}
    receipts = {"verified": 0, "failed": 0, "missing": 0}
    latencies = []
    errors = 0
    for result in results:
        if result.error:
            errors += 1
        else:
            latencies.append(result.latency_s)
        if result.system_fingerprint:
            fingerprints[result.system_fingerprint] = fingerprints.get(result.system_fingerprint, 0) + 1
        if result.receipt is not None:
            key = "verified" if result.receipt == "verified" else ("missing" if result.receipt == "missing" else "failed")
            receipts[key] += 1
    failures = sorted({r.receipt for r in results if r.receipt and r.receipt.startswith("failed")})
    return {
        "model": target.model, "role": target.role, "upstream_model": target.upstream_model, "notes": target.notes,
        "anchor": target.fingerprint_anchor, "route_model_id": target.route_model_id, "declared_quantization": target.declared_quantization,
        "counts": {"total": len(results), "errors": errors},
        "latency_s": {"p50": statistics.median(latencies) if latencies else None},
        "system_fingerprints": fingerprints,
        "receipts": receipts | {"failures": failures[:3]} if checker is not None else None,
        "prompts": {r.prompt_id: r.to_dict() for r in results},
        "passes": extra_passes,
    }


def run(targets: list[Target], out_dir: Path, *, seed: bytes | None, pool: list[str] | None, concurrency: int = 4,
        timeout: float = 60.0, prober: Callable[..., ProbeResult] = probe_once, clock: Callable[[], float] = time.time,
        log: Callable[[str], None] = lambda line: print(line, file=sys.stderr), ignore_cadence: bool = False) -> Path:
    started = clock()
    prompts, week_commitment = prompt_set(started, seed=seed, pool=pool)
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(started))
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {}
    hour = int(time.strftime("%H", time.gmtime(started)))
    selected = []
    for t in targets:
        if not t.fingerprint:
            continue
        if not ignore_cadence and t.every_hours > 1 and hour % t.every_hours != 0:
            log(f"{t.name}: not this hour (every {t.every_hours} hours)")
            continue
        selected.append(t)
    with ThreadPoolExecutor(max_workers=max(1, len(selected))) as pool_targets:
        futures = {}
        for target in selected:
            if target.api_key_env and not target.api_key():
                log(f"{target.name}: skipped, {target.api_key_env} is not set")
                summary[target.name] = {"model": target.model, "role": target.role, "anchor": target.fingerprint_anchor, "skipped": f"{target.api_key_env} not set"}
                continue
            futures[target.name] = pool_targets.submit(measure_target, target, prompts, concurrency=concurrency, timeout=timeout, prober=prober)
        for name, future in futures.items():
            summary[name] = future.result()
            counts = summary[name]["counts"]
            log(f"{name}: {counts['total'] - counts['errors']}/{counts['total']} answered")
    metadata = {"run_id": run_id, "kind": "fingerprint", "fingerprint_version": FINGERPRINT_VERSION, "started_at": started, "finished_at": clock(),
                "prompt_ids": [pid for pid, _ in prompts], "commitment": week_commitment, "targets": [t.name for t in selected]}
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False) + "\n", encoding="utf-8")
    # the prompt texts stay on this machine: secret ones are revealed only after their week
    (run_dir / "prompts.json").write_text(json.dumps(dict(prompts), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return run_dir


# ---------------------------------------------------------------- comparison

def anchor_passes(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Every pass of an anchor in one run, each a {prompt id: result} map."""
    return [summary.get("prompts") or {}] + [p for p in (summary.get("passes") or []) if isinstance(p, dict)]


def compare(anchor: dict[str, Any] | list[dict[str, Any]], other: dict[str, Any]) -> dict[str, Any]:
    """Fingerprint of ``other`` against the anchor's distributions: per prompt, the closest one the anchor produced.

    ``anchor`` is one summary or a list of passes ({prompt id: result}), newest first; token offsets
    are taken against the newest pass that answered the prompt.
    """
    passes = anchor if isinstance(anchor, list) else anchor_passes(anchor)
    # every pass of ``other`` counts: a lab compared with its own past must be judged on all it produced now,
    # since two calls seconds apart can land on two variants
    other_passes = anchor_passes(other)
    gaps: list[float] = []
    offsets: set[int] = set()
    with_logprobs = 0
    for pid, b in (other.get("prompts") or {}).items():
        if not b or b.get("error"):
            continue
        seen = [p[pid] for p in passes if isinstance(p.get(pid), dict) and not p[pid].get("error")]
        if not seen:
            continue
        a = seen[0]
        if isinstance(a.get("prompt_tokens"), int) and isinstance(b.get("prompt_tokens"), int):
            offsets.add(b["prompt_tokens"] - a["prompt_tokens"])
        firsts = [x["first"] for x in seen if x.get("first")]
        mine = [p[pid]["first"] for p in other_passes if isinstance(p.get(pid), dict) and not p[pid].get("error") and p[pid].get("first")]
        if firsts and mine:
            with_logprobs += 1
            if max(firsts[0].values()) >= math.log(INFORMATIVE_MAX_TOP):
                continue  # the lab is sure of its first word here: every precision agrees, nothing to learn
            candidates = [g for g in (log_gap(x, y) for x in firsts for y in mine) if g is not None]
            if candidates:
                gaps.append(min(candidates))
    similarity = similarity_percent(gaps)
    tokens = {"offsets": sorted(offsets), "same_count": offsets == {0}, "same_tokenizer": len(offsets) == 1} if offsets else None
    if tokens and not tokens["same_tokenizer"]:
        verdict = "mismatch"  # another tokenizer is another model, whatever the probabilities say
    elif with_logprobs == 0:
        verdict = "no_logprobs"
    elif len(gaps) < MIN_COMPARED:
        verdict = "insufficient"
    elif similarity >= MATCH_PERCENT:
        verdict = "match"
    elif similarity >= MISMATCH_PERCENT:
        verdict = "differs"
    else:
        verdict = "mismatch"
    return {"verdict": verdict, "similarity_percent": similarity, "median_log_gap": round(statistics.median(gaps), 5) if gaps else None,
            "compared": len(gaps), "tokens": tokens}


def collect(passes: list[dict[str, Any]]) -> dict[str, dict[str, list[float]]]:
    """{prompt id: {token hex: [logprob per pass]}} over every pass that answered."""
    obs: dict[str, dict[str, list[float]]] = {}
    for p in passes:
        for pid, r in p.items():
            if isinstance(r, dict) and r.get("first"):
                for k, v in r["first"].items():
                    obs.setdefault(pid, {}).setdefault(k, []).append(v)
    return obs


def margin_verdict(median_mean_gap: float | None, tokens: int, passes: tuple[int, int]) -> str:
    if median_mean_gap is None or tokens < MIN_COMPARED or min(passes) < CLOUD_MIN_PASSES:
        return "insufficient"
    if median_mean_gap <= MARGIN_WITHIN_NATS:
        return "within_margin"
    if median_mean_gap > MARGIN_OUTSIDE_NATS:
        return "outside_margin"
    return "at_margin"


def cloud(a_passes: list[dict[str, Any]], b_passes: list[dict[str, Any]], *, floor: float = CLOUD_FLOOR, min_n: int = CLOUD_MIN_PASSES) -> dict[str, Any]:
    """Do two sets of passes come from the same cloud of distributions?

    Per informative prompt and per word that either side gives at least ``floor`` on average: the difference of the
    mean log-probabilities and its z-score against the pooled standard error. The verdict is the implementation margin
    applied to the median gap of means.
    """
    A, B = collect(a_passes), collect(b_passes)
    zs: list[float] = []
    gaps: list[float] = []
    prompts = 0
    lf = math.log(floor)
    for pid in A:
        if pid not in B:
            continue
        if max(statistics.fmean(v) for v in A[pid].values()) >= math.log(INFORMATIVE_MAX_TOP):
            continue
        prompts += 1
        for k, xa in A[pid].items():
            xb = B[pid].get(k)
            if xb is None or len(xa) < min_n or len(xb) < min_n:
                continue
            ma, mb = statistics.fmean(xa), statistics.fmean(xb)
            if max(ma, mb) < lf:
                continue
            se = math.sqrt(statistics.pvariance(xa) / len(xa) + statistics.pvariance(xb) / len(xb) + 1e-6)
            zs.append((ma - mb) / se)
            gaps.append(abs(ma - mb))
    gap = round(statistics.median(gaps), 4) if gaps else None
    return {"prompts": prompts, "tokens": len(zs), "passes": [len(a_passes), len(b_passes)], "median_mean_gap": gap,
            "within_2se": round(sum(1 for z in zs if abs(z) < 2) / len(zs), 3) if zs else None,
            "median_abs_z": round(statistics.median(abs(z) for z in zs), 3) if zs else None,
            "verdict": margin_verdict(gap, len(zs), (len(a_passes), len(b_passes)))}


def regimes(passes: list[dict[str, Any]]) -> dict[str, Any]:
    """How many distinct distributions the same endpoint gave each informative prompt across these passes."""
    counts: list[int] = []
    for pid, by_token in collect(passes).items():
        firsts = [p[pid]["first"] for p in passes if isinstance(p.get(pid), dict) and p[pid].get("first")]
        if not firsts or max(firsts[0].values()) >= math.log(INFORMATIVE_MAX_TOP):
            continue
        distinct: list[dict[str, float]] = []
        for f in firsts:
            if not any((g := log_gap(d, f)) is not None and g < REGIME_SAME_GAP for d in distinct):
                distinct.append(f)
        counts.append(len(distinct))
    return {"prompts": len(counts), "passes": len(passes), "median": statistics.median(counts) if counts else None,
            "max": max(counts) if counts else None}


def hourly(runs: list[dict[str, Any]], name: str, newest_at: float, *, hours: int = HOURS) -> list[dict[str, Any]]:
    """The last ``hours`` runs of one target, oldest first: each hour against the target's own earlier hours, its two
    passes against each other, its latency and errors. Neutral numbers; the reader draws the line."""
    series: list[dict[str, Any]] = []
    for index, run in enumerate(runs):
        meta = run["metadata"]
        started = float(meta.get("started_at") or 0)
        if newest_at - started > hours * 3600:
            continue
        data = run["summary"].get(name)
        if not data or data.get("skipped"):
            continue
        passes = anchor_passes(data)
        before = _history(runs[:index], name, started)
        point: dict[str, Any] = {"run_id": meta["run_id"], "started_at": int(started), "hour_utc": int(time.strftime("%H", time.gmtime(started))),
                                 "prompts": len(data.get("prompts") or {}), "errors": (data.get("counts") or {}).get("errors", 0),
                                 "latency_p50_s": (data.get("latency_s") or {}).get("p50"),
                                 "system_fingerprints": len(data.get("system_fingerprints") or {}),
                                 "agreement_percent": None, "pass_agreement_percent": None}
        if before:
            point["agreement_percent"] = compare(before, data)["similarity_percent"]
        if len(passes) >= 2:
            point["pass_agreement_percent"] = compare({"prompts": passes[0]}, {"prompts": passes[1]})["similarity_percent"]
        series.append(point)
    return series


def _history(runs: list[dict[str, Any]], anchor_name: str, newest_at: float, *, exclude_run: str | None = None,
             window: float = HISTORY_SECONDS) -> list[dict[str, Any]]:
    """The anchor's passes in the runs of the last ``window`` seconds, newest first."""
    passes: list[dict[str, Any]] = []
    for run in reversed(runs):
        meta = run["metadata"]
        if exclude_run and meta["run_id"] == exclude_run:
            continue
        if newest_at - float(meta.get("started_at") or 0) > window:
            break
        data = run["summary"].get(anchor_name)
        if data and not data.get("skipped"):
            passes.extend(anchor_passes(data))
    return passes


def hourly_sampled(runs: list[dict[str, Any]], name: str, newest_at: float, *, hours: int = HOURS) -> list[dict[str, Any]]:
    """The sampled anchor's last runs, oldest first: each against its own earlier day, with latency and errors."""
    series: list[dict[str, Any]] = []
    for index, run in enumerate(runs):
        meta = run["metadata"]
        started = float(meta.get("started_at") or 0)
        if newest_at - started > hours * 3600:
            continue
        data = run["summary"].get(name)
        if not data or data.get("skipped"):
            continue
        before = _history(runs[:index], name, started, window=SAMPLED_HISTORY_SECONDS)
        point: dict[str, Any] = {"run_id": meta["run_id"], "started_at": int(started), "hour_utc": int(time.strftime("%H", time.gmtime(started))),
                                 "prompts": len(data.get("prompts") or {}), "errors": (data.get("counts") or {}).get("errors", 0),
                                 "latency_p50_s": (data.get("latency_s") or {}).get("p50"),
                                 "system_fingerprints": len(data.get("system_fingerprints") or {}),
                                 "agreement_percent": None, "pass_agreement_percent": None}
        if before:
            point["agreement_percent"] = sampled_fp.compare_sampled(before, anchor_passes(data), permutations=200, seed=index)["similarity_percent"]
        series.append(point)
    return series


def _seed_of(run_id: str) -> int:
    return int(hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:8], 16)


def fingerprint_sections(runs: list[dict[str, Any]] | dict[str, Any], previous: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Per fingerprinted target: its section of the board, from the newest run that probed it, against the anchor's recent passes.

    A target on a slower cadence keeps the section of its own newest run; the board marks it stale by its cadence.
    """
    if isinstance(runs, dict):  # (latest, previous) form
        runs = [r for r in (previous, runs) if r]
    latest_for: dict[str, dict[str, Any]] = {}
    for run in reversed(runs):
        for name, data in run["summary"].items():
            if name not in latest_for or (latest_for[name]["summary"][name].get("skipped") and not data.get("skipped")):
                latest_for[name] = run
    anchors = {run["summary"][name].get("model"): name for name, run in latest_for.items()
               if run["summary"][name].get("anchor") and not run["summary"][name].get("skipped")}
    sections: dict[str, dict[str, Any]] = {}
    for name, run_of in latest_for.items():
        data = run_of["summary"][name]
        meta = run_of["metadata"]
        run_at = float(meta.get("started_at") or 0)
        checked_at = int(meta.get("finished_at") or meta.get("started_at") or 0)
        hour = int(time.strftime("%H", time.gmtime(run_at)))
        if data.get("skipped"):
            sections[name] = {"verdict": "skipped", "run_id": meta["run_id"], "checked_at": checked_at}
            continue
        section: dict[str, Any] = {"run_id": meta["run_id"], "checked_at": checked_at, "hour_utc": hour,
                                   "prompts": len(data.get("prompts") or {}), "system_fingerprints": sorted(data.get("system_fingerprints") or {}),
                                   "receipts": data.get("receipts")}
        anchor_name = anchors.get(data.get("model"))
        if data.get("sampled"):
            section.update({"method": "sampled", "samples": data.get("samples"), "cadence_hours": data.get("cadence_hours") or 1})
            seed = _seed_of(meta["run_id"])
            if data.get("anchor"):
                section["anchor"] = None
                section["verdict"] = "anchor"
                earlier = _history(runs, name, run_at, exclude_run=meta["run_id"], window=SAMPLED_HISTORY_SECONDS)
                if earlier:
                    section["drift"] = sampled_fp.compare_sampled(earlier, anchor_passes(data), seed=seed) | {"history_passes": len(earlier)}
                section["hours"] = hourly_sampled(runs, name, run_at)
            elif anchor_name:
                history = _history(runs, anchor_name, run_at, window=SAMPLED_HISTORY_SECONDS)
                section["anchor"] = anchor_name
                section.update(sampled_fp.compare_sampled(history, anchor_passes(data), seed=seed))
                section["history_passes"] = len(history)
                section["margin"] = None  # a sampled estimate cannot place a host inside the implementation margin
            else:
                section.update({"anchor": None, "verdict": "no_anchor"})
        elif data.get("anchor"):
            section["anchor"] = None
            section["verdict"] = "anchor"
            earlier = _history(runs, name, run_at, exclude_run=meta["run_id"])
            if earlier:
                section["drift"] = compare(earlier, data) | {"history_passes": len(earlier)}
            section["regimes"] = regimes(_history(runs, name, run_at))
            section["hours"] = hourly(runs, name, run_at)
        elif anchor_name:
            history = _history(runs, anchor_name, run_at)
            section["anchor"] = anchor_name
            section.update(compare(history, data))
            section["history_passes"] = len(history)
            # another implementation of the same weights is judged by its cloud of the last hours against the lab's
            section["margin"] = cloud(history, _history(runs, name, run_at))
        else:
            section.update({"anchor": None, "verdict": "no_anchor"})
        receipts = data.get("receipts")
        if receipts and receipts.get("failed"):
            section["verdict"] = "receipt_failed"
        sections[name] = section
    return sections


# ---------------------------------------------------------------- command line

def _read_seed() -> bytes | None:
    raw = os.environ.get("WATCH_SECRET_SEED", "").strip()
    return bytes.fromhex(raw) if raw else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--only", default=None)
    parser.add_argument("--secret-pool", type=Path, default=None, help="JSON list of private prompts; default: the built-in pool")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--ignore-cadence", action="store_true", help="probe every target now, whatever its every_hours")
    args = parser.parse_args(argv)
    try:
        targets = load_targets(args.targets)
        seed = _read_seed()
        pool = json.loads(args.secret_pool.read_text(encoding="utf-8")) if args.secret_pool else None
        if pool is not None and (not isinstance(pool, list) or not all(isinstance(p, str) and p.strip() for p in pool)):
            raise ValueError("the secret pool must be a JSON list of prompts")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"cannot start: {error}", file=sys.stderr)
        return 2
    if args.only:
        wanted = {name.strip() for name in args.only.split(",")}
        targets = [t for t in targets if t.name in wanted]
    args.out.mkdir(parents=True, exist_ok=True)
    print(run(targets, args.out, seed=seed, pool=pool, concurrency=args.concurrency, timeout=args.timeout, ignore_cadence=args.ignore_cadence))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
