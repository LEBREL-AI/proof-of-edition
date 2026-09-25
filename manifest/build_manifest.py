"""Build the serving manifest base for an edition from verifiable artifacts.

The manifest base is the operator-provided part of the Proof of Edition serving
manifest (SPEC.md, layer 2). The runtime completes it at serving time with the
signed fields (version, validity window, signing key id, instance id, attestation).

Inputs:
  * an edition spec (JSON) naming the edition, the engine and the quantization;
  * the weights manifest produced when the edition was uploaded: a JSON object
    with "repository", "revision" and "files" mapping file names to
    {"sha256": ..., "size": ...}. This is the same file whose digest the Modal
    service pins (MODEL_MANIFEST_SHA256), so the manifest describes exactly the
    bytes the runtime loads.

Digests derived here:
  * weights.files: sha256 of every *.safetensors shard plus the shard index and
    config.json, so a verifier can compare against the upstream release's file metadata;
  * tokenizer_sha256: sha256 of tokenizer.json;
  * chat_template_sha256: sha256 of chat_template.jinja when present, otherwise
    of the chat_template string inside tokenizer_config.json, otherwise of the
    model's own encoding module (DeepSeek V4 ships encoding/encoding_dsv4.py).

Usage:
  python3 -m manifest.build_manifest --edition manifest/editions/<name>.json \
      --weights-manifest /path/to/model_manifest.json --out manifest/out/<name>.manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from receipts.schema import is_revision

WEIGHT_FILE_SUFFIXES = (".safetensors",)
WEIGHT_METADATA_FILES = ("model.safetensors.index.json", "config.json", "generation_config.json")
CHAT_TEMPLATE_CANDIDATES = ("chat_template.jinja", "chat_template.json")
ENCODING_MODULE_PREFIX = "encoding/"

EDITION_REQUIRED = {"name", "id", "base_model", "quantization", "engine", "runtime"}
ENGINE_REQUIRED = {"name", "version", "image_digest", "arguments"}
QUANTIZATION_REQUIRED = {"method", "weights_dtype", "kv_cache_dtype"}


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path) -> Any:
    with path.open("rb") as handle:
        return json.loads(handle.read())


def validate_edition(spec: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    missing = EDITION_REQUIRED - set(spec)
    if missing:
        problems.append(f"edition spec missing {sorted(missing)}")
        return problems
    engine = spec["engine"]
    if not isinstance(engine, dict) or ENGINE_REQUIRED - set(engine):
        problems.append(f"engine must contain {sorted(ENGINE_REQUIRED)}")
    elif not isinstance(engine["arguments"], list) or not all(isinstance(item, str) for item in engine["arguments"]):
        problems.append("engine.arguments must be a list of strings")
    elif not str(engine["image_digest"]).startswith("sha256:") or len(str(engine["image_digest"])) != 71:
        problems.append("engine.image_digest must be an OCI sha256 digest")
    quantization = spec["quantization"]
    if not isinstance(quantization, dict) or QUANTIZATION_REQUIRED - set(quantization):
        problems.append(f"quantization must contain {sorted(QUANTIZATION_REQUIRED)}")
    if not isinstance(spec["runtime"], dict) or "provider" not in spec["runtime"] or "gpu" not in spec["runtime"]:
        problems.append("runtime must contain provider and gpu")
    if any(isinstance(value, float) for value in _walk(spec)):
        problems.append("edition spec must not contain floating point numbers (canonical JSON)")
    return problems


def _walk(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)
    else:
        yield value


def validate_weights_manifest(weights: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    for key in ("repository", "revision", "files"):
        if key not in weights:
            problems.append(f"weights manifest missing {key}")
    if problems:
        return problems
    if not is_revision(weights["revision"]):
        problems.append("weights manifest revision must be a git commit sha (40 hex) or a registry content revision (64 hex)")
    files = weights["files"]
    if not isinstance(files, dict) or not files:
        problems.append("weights manifest files must be a non-empty object")
        return problems
    for name, entry in files.items():
        digest = entry.get("sha256") if isinstance(entry, dict) else None
        if not isinstance(digest, str) or len(digest) != 64:
            problems.append(f"weights manifest entry {name} lacks a sha256 digest")
    if not any(name.endswith(WEIGHT_FILE_SUFFIXES) for name in files):
        problems.append("weights manifest contains no safetensors shards")
    if "tokenizer.json" not in files:
        problems.append("weights manifest lacks tokenizer.json")
    return problems


def chat_template_digest(files: dict[str, Any], repo_dir: Path | None) -> tuple[str, str]:
    """Returns (digest, source) for the chat template of the edition."""
    for candidate in CHAT_TEMPLATE_CANDIDATES:
        if candidate in files:
            return files[candidate]["sha256"], candidate
    if repo_dir is not None:
        config = repo_dir / "tokenizer_config.json"
        if config.exists():
            template = load_json(config).get("chat_template")
            if isinstance(template, str):
                return sha256_hex(template.encode("utf-8")), "tokenizer_config.json#chat_template"
    encoding_modules = sorted(
        name for name in files
        if name.startswith(ENCODING_MODULE_PREFIX) and name.endswith(".py") and "test" not in name
    )
    if encoding_modules:
        return files[encoding_modules[0]]["sha256"], encoding_modules[0]
    raise ValueError("no chat template found: expected chat_template.jinja, tokenizer_config.json#chat_template or an encoding module")


def build_manifest(spec: dict[str, Any], weights: dict[str, Any], repo_dir: Path | None = None, registry_record: dict[str, Any] | None = None) -> dict[str, Any]:
    """With registry_record (the payload of a signed edition record), the manifest's weights
    point at the registry: repository "registry.lebrel.ai/<edition id>" and the record's content
    revision, and every digest is checked against the record before it is written."""
    problems = validate_edition(spec) + validate_weights_manifest(weights)
    if problems:
        raise ValueError("; ".join(problems))
    files = weights["files"]
    if registry_record is not None:
        published = registry_record.get("files") or {}
        for name, entry in files.items():
            if name in published and published[name].get("sha256") != entry["sha256"]:
                raise ValueError(f"{name}: digest differs from the registry record")
        if registry_record.get("edition", {}).get("id") != spec["id"]:
            raise ValueError("registry record belongs to a different edition")
    weight_files = {
        name: entry["sha256"]
        for name, entry in sorted(files.items())
        if name.endswith(WEIGHT_FILE_SUFFIXES) or name in WEIGHT_METADATA_FILES
    }
    template_digest, template_source = chat_template_digest(files, repo_dir)
    engine = spec["engine"]
    manifest = {
        "edition": {
            "name": spec["name"],
            "id": spec["id"],
            "base_model": spec["base_model"],
            "fingerprint_id": spec.get("fingerprint_id"),
            "recipe": spec.get("recipe"),
        },
        "weights": {
            "repository": f"registry.lebrel.ai/{spec['id']}" if registry_record is not None else weights["repository"],
            "revision": registry_record["revision"] if registry_record is not None else weights["revision"],
            "files": weight_files,
            "total_bytes": sum(int(entry.get("size", 0)) for name, entry in files.items() if name in weight_files),
        },
        "quantization": dict(spec["quantization"]),
        "engine": {
            "name": engine["name"],
            "version": engine["version"],
            "image_digest": engine["image_digest"],
            "arguments": list(engine["arguments"]),
            "arguments_sha256": sha256_hex("\n".join(engine["arguments"]).encode("utf-8")),
        },
        "tokenizer_sha256": files["tokenizer.json"]["sha256"],
        "chat_template_sha256": template_digest,
        "chat_template_source": template_source,
        "runtime": dict(spec["runtime"]),
    }
    if "sidecar" in spec:
        manifest["runtime"]["sidecar"] = dict(spec["sidecar"])
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--edition", required=True, type=Path, help="edition spec JSON")
    parser.add_argument("--weights-manifest", required=True, type=Path, help="upload-time weights manifest JSON")
    parser.add_argument("--repo-dir", type=Path, default=None, help="local checkout of the weights repo (optional, for tokenizer_config.json)")
    parser.add_argument("--registry-record", type=Path, default=None, help="signed edition record from the Lebrel registry; the manifest then references the registry revision")
    parser.add_argument("--out", type=Path, default=None, help="write the manifest base here (stdout otherwise)")
    args = parser.parse_args(argv)
    record = load_json(args.registry_record)["payload"] if args.registry_record else None
    manifest = build_manifest(load_json(args.edition), load_json(args.weights_manifest), args.repo_dir, registry_record=record)
    text = json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out} ({len(manifest['weights']['files'])} weight files, {manifest['weights']['total_bytes']} bytes)")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
