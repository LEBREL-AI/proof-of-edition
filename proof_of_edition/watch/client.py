"""OpenAI-compatible probe client: one call, one recorded exchange, nothing hidden."""
from __future__ import annotations

import http.client
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any

USER_AGENT = "proof-of-edition-watch/0.1"
RETRY_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class Target:
    """One place that claims to serve a model.

    ``model`` is the watch's own family key (e.g. ``deepseek-v4-flash``); ``upstream_model``
    is the id the target expects. ``body`` is merged into every request (provider pinning
    on a router, vendor flags such as thinking off). ``role`` is ``reference`` for a
    deployment Lebrel controls from the published weights, else ``candidate``.
    ``thresholds`` overrides the deterministic thresholds for this target, set from
    ``calibrate``: some labs are not deterministic at temperature 0 (DeepSeek: 0.45 exact,
    0.61 prefix against itself), and a global threshold would flag them forever.
    ``overrides`` are applied after the probe's own fields: a target probed in its native
    reasoning mode needs a larger ``max_tokens`` than the battery's, or the answer is empty.
    """
    name: str
    model: str
    base_url: str
    upstream_model: str
    api_key_env: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    body: dict[str, Any] = field(default_factory=dict)
    role: str = "candidate"
    claimed_revision: str | None = None
    notes: str = ""
    thresholds: dict[str, float] | None = None
    overrides: dict[str, Any] | None = None
    # fingerprint (watch.fingerprint): probed every hour when true; the anchor is the lab's own API for the model
    fingerprint: bool = False
    fingerprint_anchor: bool = False
    # Lebrel's own route: the model id customers call, and the router key its receipts are verified with
    route_model_id: str | None = None
    receipt_public_key: str | None = None
    receipt_origin: str | None = None
    declared_quantization: str | None = None
    concurrency: int | None = None
    pace_seconds: float | None = None   # hosts that limit requests per minute: one request at a time, spaced
    anchor_samples: int | None = None   # passes of an anchor per fingerprint run (default 2)
    # the full battery skips targets that are only fingerprinted
    battery: bool = True
    # fingerprint by sampling (watch.sampled): for labs whose APIs return no token probabilities
    sampled: bool = False
    samples: int = 8                 # answers per prompt in a pass
    sampled_max_tokens: int = 64     # room for a short reasoning and the first word of the answer
    every_hours: int = 1             # fingerprint cadence: probed on the UTC hours divisible by this

    def api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        value = os.environ.get(self.api_key_env)
        return value if value else None


@dataclass
class Exchange:
    """Everything the watch keeps about one probe call. Probes are Lebrel's own text."""
    target: str
    model: str
    probe: str
    sample: int
    request: dict[str, Any]
    sent_at: float
    latency_s: float
    status: int | None = None
    text: str | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    served_model: str | None = None
    provider: str | None = None
    system_fingerprint: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    refusal_field: str | None = None
    error: str | None = None
    attempts: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_targets(path: str) -> list[Target]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    items = raw["targets"] if isinstance(raw, dict) else raw
    targets = [Target(**item) for item in items]
    names = [t.name for t in targets]
    if len(names) != len(set(names)):
        raise ValueError("target names must be unique")
    anchors: dict[str, str] = {}
    for target in targets:
        if target.role not in ("reference", "candidate"):
            raise ValueError(f"{target.name}: role must be reference or candidate")
        if target.fingerprint_anchor:
            if not target.fingerprint:
                raise ValueError(f"{target.name}: a fingerprint anchor must be fingerprinted")
            if target.model in anchors:
                raise ValueError(f"{target.name}: {target.model} already has the anchor {anchors[target.model]}")
            anchors[target.model] = target.name
        if target.receipt_public_key is not None and (len(target.receipt_public_key) != 64 or any(c not in "0123456789abcdef" for c in target.receipt_public_key)):
            raise ValueError(f"{target.name}: receipt_public_key must be 64 lowercase hex characters")
        if target.receipt_public_key and not target.route_model_id:
            raise ValueError(f"{target.name}: receipts are checked against a route, set route_model_id")
        if target.sampled and not target.fingerprint:
            raise ValueError(f"{target.name}: a sampled target must be fingerprinted")
        if not 2 <= target.samples <= 50 or not 1 <= target.every_hours <= 24 or not 1 <= target.sampled_max_tokens <= 4096:
            raise ValueError(f"{target.name}: samples 2-50, every_hours 1-24, sampled_max_tokens 1-4096")
        if target.thresholds is not None:
            unknown = set(target.thresholds) - {"min_exact_rate", "min_prefix_agreement"}
            if unknown or any(not isinstance(v, (int, float)) or not 0 <= v <= 1 for v in target.thresholds.values()):
                raise ValueError(f"{target.name}: thresholds accept min_exact_rate and min_prefix_agreement between 0 and 1")
    return targets


def _extract(document: dict[str, Any], exchange: Exchange) -> None:
    choices = document.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, list):  # some vendors return content parts
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    exchange.text = content if isinstance(content, str) else None
    exchange.finish_reason = choice.get("finish_reason")
    exchange.usage = document.get("usage") if isinstance(document.get("usage"), dict) else None
    exchange.served_model = document.get("model") if isinstance(document.get("model"), str) else None
    exchange.provider = document.get("provider") if isinstance(document.get("provider"), str) else None
    exchange.system_fingerprint = document.get("system_fingerprint") if isinstance(document.get("system_fingerprint"), str) else None
    calls = message.get("tool_calls")
    exchange.tool_calls = calls if isinstance(calls, list) else None
    refusal = message.get("refusal")
    exchange.refusal_field = refusal if isinstance(refusal, str) else None


def call(target: Target, probe: str, sample: int, request: dict[str, Any], *, timeout: float = 120.0,
         retries: int = 2, opener=None, sleep=time.sleep, clock=time.time, apply_overrides: bool = True) -> Exchange:
    """POST one chat completion and record the exchange. Never raises on HTTP errors."""
    body = dict(target.body)
    body.update(request)
    if target.overrides and apply_overrides:
        body.update(target.overrides)  # e.g. a larger max_tokens for a target probed in its reasoning mode
    body["model"] = target.upstream_model
    body["stream"] = False
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
    headers.update(target.headers)
    key = target.api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    opener = opener or urllib.request.build_opener()
    exchange = Exchange(target=target.name, model=target.model, probe=probe, sample=sample, request=body, sent_at=clock(), latency_s=0.0)
    attempt = 0
    while True:
        attempt += 1
        exchange.attempts = attempt
        exchange.error = None  # a retry that succeeds must not carry the failed attempt's error
        started = time.perf_counter()
        try:
            http_request = urllib.request.Request(f"{target.base_url.rstrip('/')}/chat/completions", data=payload, headers=headers, method="POST")
            with opener.open(http_request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
                exchange.status = getattr(response, "status", 200)
            exchange.latency_s = time.perf_counter() - started
            document = json.loads(raw)
            if not isinstance(document, dict):
                raise ValueError("response is not a JSON object")
            _extract(document, exchange)
            if exchange.text is None and exchange.tool_calls is None:
                exchange.error = "no assistant text or tool calls in response"
            return exchange
        except urllib.error.HTTPError as error:
            exchange.latency_s = time.perf_counter() - started
            exchange.status = error.code
            detail = ""
            try:
                detail = error.read().decode("utf-8", "replace")[:300]
            except Exception:  # noqa: BLE001 - best effort
                pass
            exchange.error = f"HTTP {error.code}: {detail}".strip()
            if error.code in RETRY_STATUSES and attempt <= retries:
                sleep(min(2.0 * attempt, 8.0))
                continue
            return exchange
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            exchange.latency_s = time.perf_counter() - started
            exchange.error = f"{type(error).__name__}: {error}"
            if attempt <= retries:
                sleep(min(2.0 * attempt, 8.0))
                continue
            return exchange
