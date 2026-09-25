"""Render the fingerprint prompts exactly as the lab's published format does, and count their tokens.

    python -m proof_of_edition.watch.reference.prepare --prompts runs-fingerprint/<run>/prompts.json --out prompts-ids.json \
        [--check runs-fingerprint/<run>/summary.json --anchor deepseek/api]

The prompt format (``encoding/encoding.py``) and the tokenizer come from the published edition at the pinned
revision; their SHA-256 lands in the output so a run says which format it used.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

REPO = "deepseek-ai/DeepSeek-V4.1-Flash"
REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
FORMAT_FILES = ("encoding/encoding.py", "tokenizer.json", "tokenizer_config.json")
CACHE = Path(os.environ.get("LEBREL_REFERENCE_CACHE", Path.home() / ".cache" / "lebrel-reference"))


def fetch_format(repo: str = REPO, revision: str = REVISION) -> dict[str, Any]:
    """The format files of the edition, cached by revision; returns their paths and digests."""
    root = CACHE / repo.replace("/", "__") / revision
    root.mkdir(parents=True, exist_ok=True)
    digests: dict[str, str] = {}
    paths: dict[str, Path] = {}
    for name in FORMAT_FILES:
        target = root / Path(name).name
        if not target.exists():
            url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
            with urllib.request.urlopen(url, timeout=120) as response:
                target.write_bytes(response.read())
        digests[name] = hashlib.sha256(target.read_bytes()).hexdigest()
        paths[name] = target
    return {"repo": repo, "revision": revision, "sha256": digests, "paths": paths}


def load_encoding(path: Path):
    spec = importlib.util.spec_from_file_location("deepseek_encoding", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def render(prompts: dict[str, str], *, thinking_mode: str = "chat", system: str | None = None) -> list[dict[str, Any]]:
    """Each prompt as the lab's format renders a one-message conversation, with its token ids."""
    from tokenizers import Tokenizer

    fmt = fetch_format()
    encoding = load_encoding(fmt["paths"]["encoding/encoding.py"])
    tokenizer = Tokenizer.from_file(str(fmt["paths"]["tokenizer.json"]))
    out = []
    for pid, text in prompts.items():
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
        prompt = encoding.encode_messages(messages, thinking_mode=thinking_mode)
        ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        out.append({"id": pid, "text": text, "prompt": prompt, "ids": ids, "count": len(ids)})
    return out


def check_counts(rendered: list[dict[str, Any]], summary_path: Path, anchor: str) -> dict[str, Any]:
    """Token counts of the rendering against what the lab's API billed for the same prompts."""
    summary = json.loads(summary_path.read_text())
    billed = {pid: (r or {}).get("prompt_tokens") for pid, r in ((summary.get(anchor) or {}).get("prompts") or {}).items()}
    diffs = {r["id"]: (r["count"], billed.get(r["id"])) for r in rendered if billed.get(r["id"]) is not None and billed[r["id"]] != r["count"]}
    return {"anchor": anchor, "compared": sum(1 for r in rendered if billed.get(r["id"]) is not None), "mismatches": diffs}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", required=True, type=Path, help="prompts.json of a fingerprint run ({id: text})")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--thinking-mode", default="chat", choices=("chat", "thinking"))
    parser.add_argument("--system", default=None, help="optional system prompt to prepend (an experiment)")
    parser.add_argument("--check", type=Path, default=None, help="summary.json of the same run, to compare token counts")
    parser.add_argument("--anchor", default="deepseek/api")
    args = parser.parse_args(argv)
    prompts = json.loads(args.prompts.read_text())
    rendered = render(prompts, thinking_mode=args.thinking_mode, system=args.system)
    fmt = fetch_format()
    document = {"repo": fmt["repo"], "revision": fmt["revision"], "format_sha256": fmt["sha256"], "thinking_mode": args.thinking_mode,
                "system": args.system, "prompts": rendered}
    if args.check:
        document["count_check"] = check_counts(rendered, args.check, args.anchor)
        print(json.dumps(document["count_check"], indent=1))
    args.out.write_text(json.dumps(document, ensure_ascii=False, indent=1))
    print(f"{len(rendered)} prompts rendered -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
