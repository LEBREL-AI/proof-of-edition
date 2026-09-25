"""Mirror a Lebrel registry into local storage roots (the founder's NAS volumes).

Pulls the catalog, verifies every record against the pinned publisher key, downloads
each blob (resumable, verified by SHA-256 on completion) into ``<root>/blobs/sha256/<hex>``
and stores records, refs, catalog, transparency log and checkpoint alongside, so a root
is itself a complete registry mirror. Editions are placed on the first root with enough
free space; ``<root>/mirror-index.json`` records where each revision lives.

  poe-registry-mirror --origin https://registry.lebrel.ai --publisher-key <hex> \
      --root /mnt/mirror-a --root /mnt/mirror-b

Exit 0 when every published revision is fully mirrored and verified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from proof_of_edition.receipts.schema import Signed
from proof_of_edition.registry.log import check_checkpoint, inclusion, load_entries
from proof_of_edition.registry.record import check_record, record_sha256

CHUNK = 8 * 1024 * 1024
TIMEOUT = 120
RESERVE_BYTES = 10 * 1024 ** 3  # never fill a volume completely


class Fetch:
    """Thin HTTP client; tests substitute one backed by a dict."""

    def __init__(self, origin: str, opener=None) -> None:
        self.origin = origin.rstrip("/")
        self.opener = opener or urllib.request.build_opener()

    USER_AGENT = "proof-of-edition-mirror/0.1"

    def text(self, path: str) -> str:
        with self.opener.open(urllib.request.Request(self.origin + path, headers={"Accept": "application/json", "User-Agent": self.USER_AGENT}), timeout=TIMEOUT) as response:
            return response.read().decode("utf-8")

    def stream(self, path: str, offset: int) -> tuple[int, Any]:
        """Returns (status, response) for a Range request starting at offset. A 206 response
        whose Content-Range does not start at the requested offset is reported as status 0."""
        headers = {"User-Agent": self.USER_AGENT}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            response = self.opener.open(urllib.request.Request(self.origin + path, headers=headers), timeout=TIMEOUT)
        except urllib.error.HTTPError as error:
            if error.code == 416:
                return 416, None
            raise
        if response.status == 206:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {offset}-"):
                response.close()
                return 0, None
        return response.status, response


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def free_bytes(root: Path) -> int:
    usage = shutil.disk_usage(root)
    return usage.free


def choose_root(roots: list[Path], needed: int, index: dict[str, Any], revision: str) -> Path:
    placed = index.get("revisions", {}).get(revision)
    if placed and Path(placed) in roots:
        return Path(placed)
    for root in roots:
        if free_bytes(root) - RESERVE_BYTES >= needed:
            return root
    raise RuntimeError(f"no root has {needed} bytes free")


def load_index(root: Path) -> dict[str, Any]:
    path = root / "mirror-index.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"version": 1, "revisions": {}}


def save_index(roots: list[Path], index: dict[str, Any]) -> None:
    text = json.dumps(index, indent=1, sort_keys=True) + "\n"
    for root in roots:
        (root / "mirror-index.json").write_text(text, encoding="utf-8")


class Truncated(Exception):
    """The transfer ended before the file was complete; the partial file is kept for resumption."""


def _transfer(fetch: Fetch, url_path: str, part: Path, size: int) -> None:
    offset = part.stat().st_size if part.exists() else 0
    if offset > size:
        part.unlink()
        offset = 0
    if offset == size:
        return
    status, response = fetch.stream(url_path, offset)
    if status in (416, 0) or (offset and status == 200):
        # The origin would not resume where we are; start over.
        part.unlink(missing_ok=True)
        offset = 0
        if status != 200:
            status, response = fetch.stream(url_path, 0)
    if status not in (200, 206):
        raise RuntimeError(f"HTTP {status}")
    with part.open("ab" if offset else "wb") as handle, response:
        while True:
            block = response.read(CHUNK)
            if not block:
                break
            handle.write(block)
        handle.flush()
        os.fsync(handle.fileno())
    if part.stat().st_size < size:
        raise Truncated(f"{part.stat().st_size} of {size} bytes")


def download_blob(fetch: Fetch, edition_id: str, revision: str, file_path: str, expected: str, size: int, target: Path, log: Callable[[str], None], attempts: int = 6) -> None:
    """Resumable download to target.part, verified by digest, then atomically renamed.

    Truncated transfers are resumed (up to ``attempts`` times, with backoff); a complete
    file whose digest differs from the record is deleted and downloaded again once, and
    reported if it still differs."""
    if target.exists():
        if target.stat().st_size == size:
            return
        target.unlink()
    part = target.with_suffix(target.suffix + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    url_path = f"/{edition_id}/resolve/{revision}/{file_path}"
    digest_retries = 1
    for attempt in range(1, attempts + 1):
        try:
            _transfer(fetch, url_path, part, size)
        except (Truncated, urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
            if isinstance(error, OSError) and not isinstance(error, (urllib.error.URLError, ConnectionError, TimeoutError)) and error.errno not in (None, 54, 60, 104, 110):
                raise
            if attempt == attempts:
                raise RuntimeError(f"{file_path}: transfer kept failing ({error})")
            delay = min(60, 5 * attempt)
            log(f"{file_path}: {type(error).__name__} {error}; resuming in {delay}s (attempt {attempt}/{attempts})")
            time.sleep(delay)
            continue
        actual = sha256_file(part)
        if actual == expected:
            if part.stat().st_size != size:
                part.unlink()
                raise RuntimeError(f"{file_path}: size differs from the record")
            os.replace(part, target)
            log(f"stored {file_path} ({size} bytes)")
            return
        part.unlink()
        if digest_retries:
            digest_retries -= 1
            log(f"{file_path}: digest {actual[:16]}… does not match the record; downloading again")
            continue
        raise RuntimeError(f"{file_path}: digest {actual[:16]}… does not match the record ({expected[:16]}…)")
    raise RuntimeError(f"{file_path}: could not be downloaded")


def mirror(fetch: Fetch, roots: list[Path], publisher_key: bytes, *, log: Callable[[str], None] = print, only: str | None = None, parallel: int = 3) -> dict[str, Any]:
    for root in roots:
        if not root.is_dir():
            raise RuntimeError(f"{root} is not a directory")
    index: dict[str, Any] = {"version": 1, "revisions": {}}
    for root in roots:
        index["revisions"].update(load_index(root).get("revisions", {}))
    catalog = json.loads(fetch.text("/v1/editions"))
    summary = {"editions": 0, "revisions": 0, "blobs_stored": 0, "bytes_stored": 0, "problems": []}
    if not catalog.get("editions"):
        log("registry has no editions yet")
        return summary
    log_text = fetch.text("/v1/log")
    checkpoint = Signed.from_json(fetch.text("/v1/log/checkpoint"))
    for entry in catalog.get("editions", []):
        edition_id, revision = entry["id"], entry["current"]
        if only and edition_id != only:
            continue
        summary["editions"] += 1
        record = Signed.from_json(fetch.text(f"/v1/editions/{edition_id}/records/{revision}"))
        problems = check_record(record, publisher_key)
        if record.payload["revision"] != revision or record.payload["edition"]["id"] != edition_id:
            problems.append("record does not match the catalog entry")
        if problems:
            summary["problems"].append(f"{edition_id}@{revision[:12]}: " + "; ".join(problems))
            continue
        total = record.payload["total_bytes"]
        root = choose_root(roots, total, index, revision)
        log(f"{edition_id} revision {revision[:12]} -> {root} ({total} bytes)")
        stored_before = summary["blobs_stored"]
        failed: list[str] = []
        lock = threading.Lock()

        def safe_log(message: str) -> None:
            with lock:
                log(message)

        def fetch_one(item: tuple[str, dict[str, Any]]) -> None:
            file_path, meta = item
            target = root / "blobs" / "sha256" / meta["sha256"]
            existed = target.exists() and target.stat().st_size == meta["size"]
            try:
                download_blob(fetch, edition_id, revision, file_path, meta["sha256"], meta["size"], target, safe_log)
            except RuntimeError as error:
                with lock:
                    failed.append(str(error))
                safe_log(f"skipping {file_path}: {error}")
                return
            if not existed:
                with lock:
                    summary["blobs_stored"] += 1
                    summary["bytes_stored"] += meta["size"]

        # Small files first so the catalog-level documents are usable early; shards in parallel.
        items = sorted(record.payload["files"].items(), key=lambda item: (item[1]["size"], item[0]))
        with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
            list(pool.map(fetch_one, items))
        if failed:
            summary["problems"].append(f"{edition_id}@{revision[:12]}: {len(failed)} file(s) not mirrored: " + " | ".join(failed[:3]))
            continue
        record_dir = root / "editions" / edition_id / "records"
        record_dir.mkdir(parents=True, exist_ok=True)
        (record_dir / f"{revision}.json").write_text(record.to_json() + "\n", encoding="utf-8")
        refs = root / "editions" / edition_id / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        (refs / "main").write_text(revision + "\n", encoding="utf-8")
        index["revisions"][revision] = str(root)
        summary["revisions"] += 1
        if summary["blobs_stored"] == stored_before:
            log(f"{edition_id} already complete")
    # Registry-wide documents on every root, with the log verified against its checkpoint.
    with_entries = []
    for line in log_text.splitlines():
        if line.strip():
            with_entries.append(json.loads(line))
    log_problems = check_checkpoint(checkpoint, with_entries, publisher_key)
    if log_problems:
        summary["problems"].append("transparency log: " + "; ".join(log_problems))
    for entry in catalog.get("editions", []):
        if only and entry["id"] != only:
            continue
        rec_root = index["revisions"].get(entry["current"])
        if rec_root:
            signed = Signed.from_json((Path(rec_root) / "editions" / entry["id"] / "records" / f"{entry['current']}.json").read_text(encoding="utf-8"))
            if inclusion(with_entries, record_sha256(signed)) is None:
                summary["problems"].append(f"{entry['id']}: record not in the transparency log")
    for root in roots:
        (root / "log").mkdir(exist_ok=True)
        (root / "log" / "log.jsonl").write_text(log_text, encoding="utf-8")
        (root / "log" / "checkpoint.json").write_text(checkpoint.to_json() + "\n", encoding="utf-8")
        (root / "catalog.json").write_text(json.dumps(catalog, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    index["updated_at"] = int(time.time())
    save_index(roots, index)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--origin", default="https://registry.lebrel.ai")
    parser.add_argument("--publisher-key", required=True, help="pinned publisher public key, 64 hex")
    parser.add_argument("--root", action="append", required=True, type=Path, help="storage root (repeatable, in placement order)")
    parser.add_argument("--only", default=None, help="mirror only this edition id")
    parser.add_argument("--parallel", type=int, default=3, help="concurrent file downloads")
    args = parser.parse_args(argv)
    try:
        summary = mirror(Fetch(args.origin), args.root, bytes.fromhex(args.publisher_key), only=args.only, parallel=args.parallel)
    except (RuntimeError, urllib.error.URLError, OSError, ValueError, KeyError) as error:
        print(f"mirror failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary))
    return 1 if summary["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
