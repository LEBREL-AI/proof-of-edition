"""Publish the watch board, signed, where models.lebrel.ai renders it.

Default: POST the board's exact bytes to the registry (``--registry https://models.lebrel.ai``) with
an Ed25519 signature by the watch key (WATCH_SIGNING_SEED); the registry checks it against the pinned
key, refuses anything older than what it serves, and keeps every board under watch/history/. The same
command reveals the secret fingerprint prompts of the weeks that are over (``--reveal runs-fingerprint``).

Legacy: direct upload to the bucket with R2 credentials (``--bucket``/``--endpoint``).

Uploads ``board.json`` to ``watch/board.json`` and keeps the previous boards under
``watch/history/<run_id>.json`` so a status can always be traced to the run that
produced it. Credentials and endpoint as in ``registry.publish`` (R2_ACCESS_KEY_ID,
R2_SECRET_ACCESS_KEY, --endpoint).

Usage:
  python3 -m proof_of_edition.watch.publish --board board.json --bucket lebrel-registry --endpoint https://<account>.r2.cloudflarestorage.com
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from proof_of_edition.registry.publish import ObjectStore, S3Store

BOARD_KEY = "watch/board.json"
HISTORY_PREFIX = "watch/history/"
REQUIRED = {"version", "generated_at", "run_id", "battery_id", "thresholds", "entries"}


def validate_board(board: dict[str, Any]) -> list[str]:
    problems = [f"board missing {key}" for key in sorted(REQUIRED - set(board))]
    if not isinstance(board.get("entries"), list):
        problems.append("board entries must be a list")
    else:
        for entry in board["entries"]:
            if not isinstance(entry, dict) or not entry.get("target") or not entry.get("status"):
                problems.append("every entry needs target and status")
                break
    run_id = board.get("run_id")
    if not isinstance(run_id, str) or not run_id.isalnum() or len(run_id) > 32:
        problems.append("run_id must be alphanumeric, at most 32 characters")
    return problems


def publish(store: ObjectStore, board: dict[str, Any], *, log=print) -> dict[str, str]:
    problems = validate_board(board)
    if problems:
        raise ValueError("; ".join(problems))
    text = json.dumps(board, ensure_ascii=False, separators=(",", ":"))
    history_key = f"{HISTORY_PREFIX}{board['run_id']}.json"
    store.put_text(history_key, text, "application/json")
    store.put_text(BOARD_KEY, text, "application/json")
    log(f"published {BOARD_KEY} (run {board['run_id']}, {len(board['entries'])} entries) and {history_key}")
    return {"board": BOARD_KEY, "history": history_key}


def post_signed(url: str, data: bytes, seed_hex: str, *, opener=None, timeout: float = 30.0) -> dict[str, Any]:
    """POST ``data`` with the watch signature; returns the registry's JSON answer or raises RuntimeError."""
    import urllib.error
    import urllib.request

    from proof_of_edition.watch.client import USER_AGENT
    from proof_of_edition.watch.sign import sign

    signature, key = sign(data, seed_hex)
    request = urllib.request.Request(url, data=data, method="POST", headers={
        "Content-Type": "application/json", "User-Agent": USER_AGENT, "X-Lebrel-Watch-Signature": signature, "X-Lebrel-Watch-Key": key})
    try:
        with (opener or urllib.request.build_opener()).open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"{url} answered {error.code}: {detail}") from error


def reveal_documents(fingerprints_dir: Path, current_week: str) -> dict[str, bytes]:
    """{set id: canonical reveal document} of every finished week found in the fingerprint runs.

    The set id is the week, followed by the pool id when the commitment names one (``2026-W39-1a2b3c4d``).
    """
    from proof_of_edition.watch.fingerprint import commitment, reveal_document
    weeks: dict[str, bytes] = {}
    for run_dir in sorted(p for p in fingerprints_dir.iterdir() if p.is_dir()):
        try:
            meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            prompts = json.loads((run_dir / "prompts.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        week_commitment = meta.get("commitment") or {}
        week = week_commitment.get("week")
        if not week or week >= current_week:
            continue
        pool = week_commitment.get("pool")
        set_id = f"{week}-{pool}" if pool else week
        if set_id in weeks:
            continue
        secret = [prompts[pid] for pid in meta.get("prompt_ids", []) if pid.startswith("sec-") and pid in prompts]
        if pool:
            document = reveal_document(week, pool, secret)
        else:  # the first boards of week 39 committed to {"week", "prompts"} without a pool
            document = json.dumps({"week": week, "prompts": secret}, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        import hashlib
        if hashlib.sha256(document).hexdigest() != week_commitment.get("sha256"):
            continue  # never reveal something that is not what was committed
        weeks[set_id] = document
    return weeks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--board", required=True, type=Path)
    parser.add_argument("--registry", default=None, help="e.g. https://models.lebrel.ai: signed POST (needs WATCH_SIGNING_SEED)")
    parser.add_argument("--reveal", type=Path, default=None, help="fingerprint runs directory: also reveal finished weeks")
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--endpoint", default=None)
    args = parser.parse_args(argv)
    if args.registry:
        seed = os.environ.get("WATCH_SIGNING_SEED", "").strip()
        if not seed:
            print("WATCH_SIGNING_SEED is required to publish to the registry", file=sys.stderr)
            return 2
        try:
            data = args.board.read_bytes()
            problems = validate_board(json.loads(data.decode("utf-8")))
            if problems:
                print("invalid board: " + "; ".join(problems), file=sys.stderr)
                return 1
            answer = post_signed(f"{args.registry.rstrip('/')}/v1/watch", data, seed)
            print(f"published board {answer.get('run_id')} ({answer.get('bytes')} bytes)")
            if args.reveal and args.reveal.exists():
                import time as _time

                from proof_of_edition.watch.fingerprint import iso_week
                for set_id, document in reveal_documents(args.reveal, iso_week(_time.time())).items():
                    result = post_signed(f"{args.registry.rstrip('/')}/v1/watch/sets/{set_id}", document, seed)
                    print(f"revealed {set_id}: {result.get('status')}")
        except (OSError, json.JSONDecodeError, RuntimeError, ValueError) as error:
            print(f"publish failed: {error}", file=sys.stderr)
            return 1
        return 0
    if not args.bucket or not args.endpoint:
        print("either --registry, or --bucket and --endpoint", file=sys.stderr)
        return 2
    access_key, secret_key = os.environ.get("R2_ACCESS_KEY_ID"), os.environ.get("R2_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
        print("R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY are required", file=sys.stderr)
        return 2
    try:
        board = json.loads(args.board.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"cannot read board: {error}", file=sys.stderr)
        return 2
    import boto3
    client = boto3.client("s3", endpoint_url=args.endpoint, aws_access_key_id=access_key, aws_secret_access_key=secret_key, region_name="auto")
    try:
        publish(S3Store(client, args.bucket), board)
    except ValueError as error:
        print(f"invalid board: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
