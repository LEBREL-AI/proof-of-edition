"""The reference: a lab's published weights, run by us, measured with the fingerprint prompts.

    modal run watch/reference/app.py --action download          # weights into the volume (CPU, once)
    modal run watch/reference/app.py --action measure --prompts prompts.json --out runs-reference/<id>.json

``REFERENCE_MODEL`` picks the checkpoint (``deepseek-flash`` by default, ``glm-flash``); each one has its own Modal
app and volume, so nothing here touches other Modal apps or volumes.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import modal

MODELS = {
    "deepseek-flash": {"repo": "deepseek-ai/DeepSeek-V4.1-Flash", "revision": "dba1be0a40aa45a94ad051997016db3960a90277",
                       "volume": "lebrel-watch-reference-flash", "app": "lebrel-watch-reference", "watch_model": "deepseek-v4.1-flash"},
    # Z.ai's GLM-5.3 Flash as published (FP8, 62 shards, 328 GB): the lab returns no token probabilities, so this
    # reference is measured by sampling answers (``--action sample``), the way the watch measures the lab's API:
    # the same prompts, the lab's own sampling defaults (generation_config.json: temperature 1.0, top_p 0.95),
    # reasoning effort low as the watch sends it, the first word of every answer counted.
    "glm-flash": {"repo": "zai-org/GLM-5.3-Flash", "revision": "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
                  "volume": "lebrel-watch-reference-glm-flash", "app": "lebrel-watch-reference-glm-flash", "watch_model": "glm-5.3-flash",
                  "serve": ["--reasoning-parser", "glm47", "--kv-cache-dtype", "fp8", "--max-num-seqs", "256"],
                  "sampling": {"temperature": 1.0, "top_p": 0.95, "reasoning_effort": "low", "max_tokens": 2048}},
}
MODEL = os.environ.get("REFERENCE_MODEL", "deepseek-flash")
SPEC = MODELS[MODEL]
REPO, REVISION, VOLUME = SPEC["repo"], SPEC["revision"], SPEC["volume"]
MODEL_DIR = Path("/vol/model")
LOCAL_MODEL_DIR = Path("/local/model")
VLLM_IMAGE = os.environ.get("REFERENCE_VLLM_IMAGE", "vllm/vllm-openai:v0.30.0")
GPU = os.environ.get("REFERENCE_GPU", "B300:2")
MINUTE = 60

app = modal.App(SPEC["app"])
weights = modal.Volume.from_name(VOLUME, create_if_missing=True)

download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_transfer]==1.8.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "REFERENCE_MODEL": MODEL})
)


@app.function(image=download_image, cpu=16, memory=32 * 1024, timeout=8 * 60 * MINUTE, volumes={"/vol": weights})
def download(repo: str = REPO, revision: str = REVISION) -> dict:
    """Fetch the published checkpoint at a pinned revision into the volume; returns what landed. The caller names the
    checkpoint: the container does not see the local environment, so nothing here may depend on it."""
    from huggingface_hub import snapshot_download

    started = time.monotonic()
    if MODEL_DIR.exists():  # one checkpoint per volume, never a mixture
        shutil.rmtree(MODEL_DIR)
    snapshot_download(repo_id=repo, revision=revision, local_dir=str(MODEL_DIR), max_workers=16)
    weights.commit()
    files = sorted(p for p in MODEL_DIR.rglob("*") if p.is_file() and ".cache" not in p.parts)
    total = sum(p.stat().st_size for p in files)
    return {"repo": repo, "revision": revision, "files": len(files), "bytes": total,
            "shards": sum(1 for p in files if p.suffix == ".safetensors"), "seconds": round(time.monotonic() - started, 1)}


@app.function(image=download_image, cpu=2, memory=4 * 1024, timeout=10 * MINUTE, volumes={"/vol": weights})
def inventory() -> dict:
    """What the volume holds (no download)."""
    files = sorted(p for p in MODEL_DIR.rglob("*") if p.is_file() and ".cache" not in p.parts) if MODEL_DIR.exists() else []
    return {"files": len(files), "bytes": sum(p.stat().st_size for p in files),
            "shards": sum(1 for p in files if p.suffix == ".safetensors"),
            "small": [p.name for p in files if p.suffix != ".safetensors"]}


# ---------------------------------------------------------------- measurement on the GPU

vllm_image = (
    modal.Image.from_registry(VLLM_IMAGE, add_python="3.12")
    .entrypoint([])  # the image's own entrypoint is `vllm`, which would swallow Modal's runner
    .env({"VLLM_ENGINE_READY_TIMEOUT_S": "3600", "HF_HUB_OFFLINE": "1", "VLLM_LOGGING_LEVEL": "INFO", "REFERENCE_MODEL": MODEL})
)
SERVER = "http://127.0.0.1:8000"
LOG = Path("/local/vllm.log")


def _run(cmd: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _tp() -> int:
    return int(GPU.split(":")[1]) if ":" in GPU else 1


def _provenance() -> dict:
    """What the volume says about where its bytes came from (hub metadata written by the download)."""
    files: dict[str, dict] = {}
    for meta in sorted((MODEL_DIR / ".cache" / "huggingface" / "download").rglob("*.metadata")):
        lines = meta.read_text().splitlines()
        rel = str(meta.relative_to(MODEL_DIR / ".cache" / "huggingface" / "download"))[: -len(".metadata")]
        files[rel] = {"commit": lines[0] if lines else None, "etag": lines[1] if len(lines) > 1 else None}
    return files


def _copy_local(workers: int = 16) -> dict:
    """The checkpoint from the volume to local disk, shard by shard in parallel (the volume reads faster that way)."""
    from concurrent.futures import ThreadPoolExecutor

    if LOCAL_MODEL_DIR.exists():
        shutil.rmtree(LOCAL_MODEL_DIR)
    started = time.monotonic()
    sources = [p for p in MODEL_DIR.rglob("*") if p.is_file() and ".cache" not in p.relative_to(MODEL_DIR).parts]

    def copy(src: Path) -> int:
        dst = LOCAL_MODEL_DIR / src.relative_to(MODEL_DIR)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        return dst.stat().st_size

    with ThreadPoolExecutor(max_workers=workers) as pool:
        sizes = list(pool.map(copy, sources))
    return {"files": len(sources), "bytes": sum(sizes), "shards": sum(1 for p in sources if p.suffix == ".safetensors"),
            "seconds": round(time.monotonic() - started, 1)}


def _server_command(tp: int, top: int, extra: list[str]) -> list[str]:
    return ["vllm", "serve", str(LOCAL_MODEL_DIR), "--served-model-name", "reference", "--host", "127.0.0.1", "--port", "8000",
            "--tensor-parallel-size", str(tp), "--max-logprobs", str(top), "--logprobs-mode", "raw_logprobs",
            "--return-tokens-as-token-ids", "--max-model-len", "8192", "--max-num-seqs", "64", "--max-num-batched-tokens", "8192",
            "--gpu-memory-utilization", "0.90", *extra]


def _http(path: str, body: dict | None = None, timeout: float = 600.0) -> dict:
    import urllib.request
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(SERVER + path, data=data, headers={"Content-Type": "application/json"}, method="POST" if data else "GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_ready(proc: subprocess.Popen, timeout: int = 75 * MINUTE) -> None:
    import urllib.request
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(SERVER + "/health", timeout=5) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError("vllm did not become ready in time")


def _probe(prompt: dict, top: int, temperature: float) -> dict:
    started = time.monotonic()
    body = {"model": "reference", "prompt": prompt["ids"], "max_tokens": 1, "temperature": temperature, "logprobs": top, "echo": False}
    if isinstance(prompt.get("ids"), str):  # a filler request: text in, many tokens out, nothing recorded but the time
        body.update({"prompt": prompt["ids"], "max_tokens": int(prompt.get("max_tokens", 256)), "logprobs": None})
        try:
            _http("/v1/completions", body, timeout=900)
            return {"id": prompt["id"], "filler": True, "latency_s": round(time.monotonic() - started, 3)}
        except Exception as e:
            return {"id": prompt["id"], "filler": True, "error": repr(e)[:200], "latency_s": round(time.monotonic() - started, 3)}
    try:
        document = _http("/v1/completions", body, timeout=600)
    except Exception as e:  # keep going: one failed prompt must not lose the run
        return {"id": prompt["id"], "error": repr(e)[:400], "latency_s": round(time.monotonic() - started, 3)}
    choice = (document.get("choices") or [{}])[0]
    lp = choice.get("logprobs") or {}
    alternatives = (lp.get("top_logprobs") or [{}])[0]
    ranked = sorted(((int(k.split(":")[1]) if k.startswith("token_id:") else k, float(v)) for k, v in alternatives.items()), key=lambda kv: -kv[1])
    token = (lp.get("tokens") or [None])[0]
    return {"id": prompt["id"], "prompt_tokens": (document.get("usage") or {}).get("prompt_tokens"),
            "sampled": int(token.split(":")[1]) if isinstance(token, str) and token.startswith("token_id:") else token,
            "sampled_logprob": (lp.get("token_logprobs") or [None])[0], "top": ranked, "finish": choice.get("finish_reason"),
            "latency_s": round(time.monotonic() - started, 3)}


FILLER_TEXT = ("Write a detailed, multi-paragraph essay about the history of navigation at sea, covering the astrolabe, "
               "the marine chronometer, the sextant, radio beacons and satellite positioning, and explain how each "
               "instrument changed what a ship's officer could know about the ship's position. ")

DEFAULT_PROTOCOL = [
    {"label": "serial-1", "mode": "serial"},
    {"label": "serial-2", "mode": "serial"},
    {"label": "batched-48-a", "mode": "batched", "concurrency": 48},
    {"label": "batched-48-b", "mode": "batched", "concurrency": 48},
    {"label": "batched-8-shuffled-1", "mode": "batched", "concurrency": 8, "shuffle": 1},
    {"label": "batched-8-shuffled-2", "mode": "batched", "concurrency": 8, "shuffle": 2},
    {"label": "batched-16-shuffled-3", "mode": "batched", "concurrency": 16, "shuffle": 3},
    {"label": "batched-48-shuffled-4", "mode": "batched", "concurrency": 48, "shuffle": 4},
    {"label": "busy-16-fillers-8", "mode": "busy", "concurrency": 16, "fillers": 8, "shuffle": 5},
    {"label": "busy-16-fillers-16", "mode": "busy", "concurrency": 16, "fillers": 16, "shuffle": 6},
    {"label": "serial-3", "mode": "serial"},
]


def _run_pass(step: dict, prompts: list[dict], top: int, temperature: float) -> dict:
    """One pass of the protocol: every prompt once, in the order and with the company the step asks for."""
    import random
    import threading
    from concurrent.futures import ThreadPoolExecutor

    order = list(prompts)
    if step.get("shuffle"):
        random.Random(int(step["shuffle"])).shuffle(order)
    started = time.monotonic()
    fillers: list[dict] = []
    if step.get("mode") == "serial":
        results = [_probe(p, top, temperature) for p in order]
    else:
        stop = threading.Event()
        filler_threads = []
        if step.get("mode") == "busy":
            def keep_busy(index: int) -> None:
                n = 0
                while not stop.is_set():
                    fillers.append(_probe({"id": f"filler-{index}-{n}", "ids": FILLER_TEXT * (1 + index % 3), "max_tokens": 200}, top, temperature))
                    n += 1
            filler_threads = [threading.Thread(target=keep_busy, args=(i,), daemon=True) for i in range(int(step.get("fillers", 8)))]
            for t in filler_threads:
                t.start()
            time.sleep(3)  # let the fillers occupy the batch before the probes arrive
        with ThreadPoolExecutor(max_workers=int(step.get("concurrency", 8))) as pool:
            results = list(pool.map(lambda p: _probe(p, top, temperature), order))
        stop.set()
        for t in filler_threads:
            t.join(timeout=900)
    by_id = {r["id"]: r for r in results}
    return {"label": step.get("label"), "mode": step.get("mode"), "step": step, "seconds": round(time.monotonic() - started, 1),
            "results": [by_id[p["id"]] for p in prompts], "fillers": len(fillers),
            "filler_errors": sum(1 for f in fillers if f.get("error"))}


def _serve_and_measure(extra: list[str], prompts: list[dict], protocol: list[dict], top: int, temperature: float) -> dict:
    """Launch the server with these flags, run the protocol, stop the server; everything that happened, as a dict."""
    out: dict = {"extra_args": extra}
    cmd = _server_command(_tp(), top, extra)
    out["command"] = cmd
    started = time.monotonic()
    if LOG.exists():
        LOG.unlink()
    with LOG.open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            _wait_ready(proc)
            out["startup_seconds"] = round(time.monotonic() - started, 1)
            try:
                out["server"] = {"version": _http("/version"), "models": _http("/v1/models")}
            except Exception as e:
                out["server"] = {"error": repr(e)[:200]}
            out["passes"] = []
            for step in protocol:
                out["passes"].append(_run_pass(step, prompts, top, temperature))
        except Exception as e:
            out["error"] = repr(e)[:500]
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=120)
            except Exception:
                proc.kill()
    out["log_tail"] = LOG.read_text(errors="replace")[-12000:] if LOG.exists() else ""
    out["seconds"] = round(time.monotonic() - started, 1)
    return out


@app.function(image=vllm_image, gpu=GPU, cpu=32, memory=(384 * 1024, 768 * 1024), ephemeral_disk=700 * 1024, timeout=4 * 60 * MINUTE,
              volumes={"/vol": weights})
def measure(prompts: list[dict], top: int = 100, temperature: float = 1.0, configs: list[list[str]] | None = None,
            protocol: list[dict] | None = None) -> dict:
    """Serve the published weights under each configuration in turn and record the first-word distribution of every
    prompt under the protocol (serial passes, batches of several sizes and orders, a busy server).
    ``prompts`` are {id, ids} as ``prepare.py`` renders them."""
    out: dict = {"image": VLLM_IMAGE, "gpu": GPU, "started_at": time.time(), "top": top, "temperature": temperature}
    base = ["--engram-config", '{"cpu_offload":true}', "--language-model-only"]
    configs = configs or [base + ["--enforce-eager"], base]
    protocol = protocol or DEFAULT_PROTOCOL
    out["protocol"] = protocol
    out["gpus"] = _run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"]).stdout.strip().splitlines()
    help_text = _run(["vllm", "serve", "--help=all"], timeout=900).stdout
    out["vllm_version"] = _run(["vllm", "--version"], timeout=600).stdout.strip()[-200:]
    wanted = {"--logprobs-mode", "--return-tokens-as-token-ids"} | {a for c in configs for a in c if a.startswith("--")}
    missing = sorted(flag for flag in wanted if flag not in help_text)
    if missing:
        out["error"] = f"flags unknown to this vllm: {missing}"
        return out
    out["provenance"] = _provenance()
    out["copy"] = _copy_local()
    if out["copy"]["shards"] == 0:
        out["error"] = "no safetensors shards on the volume"
        return out
    out["configs"] = [_serve_and_measure(extra, prompts, protocol, top, temperature) for extra in configs]
    out["finished_at"] = time.time()
    out["seconds"] = round(out["finished_at"] - out["started_at"], 1)
    # the result also lands on the volume, so a run survives the local client going away
    results_dir = Path("/vol/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out["result_id"] = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(out["started_at"]))
    (results_dir / f"{out['result_id']}.json").write_text(json.dumps(out))
    weights.commit()
    return out


# ---------------------------------------------------------------- sampling: labs that return no probabilities

ANSWER_PREFIX_CHARS = 160  # enough for the first word and its context; whole answers are not kept


def _chat(prompt: dict, n: int, params: dict) -> dict:
    """One chat request asking for ``n`` answers, at the sampling settings the watch uses against the lab's API."""
    started = time.monotonic()
    body = {"model": "reference", "messages": [{"role": "user", "content": prompt["text"]}], "n": n, "stream": False,
            "max_tokens": int(params.get("max_tokens", 2048)), "temperature": float(params.get("temperature", 1.0)), "top_p": float(params.get("top_p", 0.95))}
    if params.get("reasoning_effort"):
        body["chat_template_kwargs"] = {"reasoning_effort": params["reasoning_effort"]}
    try:
        document = _http("/v1/chat/completions", body, timeout=1800)
    except Exception as e:  # keep going: one failed prompt must not lose the pass
        return {"id": prompt["id"], "error": repr(e)[:400], "latency_s": round(time.monotonic() - started, 3)}
    answers = []
    for choice in document.get("choices") or []:
        message = choice.get("message") or {}
        content = message.get("content") if isinstance(message.get("content"), str) else None
        answers.append({"first": content[:ANSWER_PREFIX_CHARS] if content is not None else None, "chars": len(content) if content is not None else None,
                        "reasoning": bool(message.get("reasoning_content")), "finish": choice.get("finish_reason")})
    usage = document.get("usage") or {}
    return {"id": prompt["id"], "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
            "answers": answers, "latency_s": round(time.monotonic() - started, 3)}


def _sample_pass(index: int, prompts: list[dict], n: int, params: dict, concurrency: int) -> dict:
    """Every prompt once, ``n`` answers each; passes after the first go in a shuffled order so the batches differ."""
    import random
    from concurrent.futures import ThreadPoolExecutor

    order = list(prompts)
    if index:
        random.Random(index).shuffle(order)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(lambda p: _chat(p, n, params), order))
    by_id = {r["id"]: r for r in results}
    return {"label": f"batched-{index}", "seconds": round(time.monotonic() - started, 1), "results": [by_id[p["id"]] for p in prompts],
            "errors": sum(1 for r in results if r.get("error"))}


@app.function(image=vllm_image, gpu=GPU, cpu=32, memory=(384 * 1024, 768 * 1024), ephemeral_disk=700 * 1024, timeout=4 * 60 * MINUTE,
              volumes={"/vol": weights})
def sample(prompts: list[dict], spec: dict, n: int = 32, passes: int = 2, params: dict | None = None, extra: list[str] | None = None,
           concurrency: int = 16) -> dict:
    """Serve the published weights and sample ``n`` answers to every prompt, ``passes`` times in different batch
    compositions: what the watch measures at a lab that returns no probabilities, measured on the weights the lab
    published. ``prompts`` are {id, text}; ``spec`` is the caller's MODELS entry (the container does not see the
    local environment); the result keeps the first characters of every answer, never whole texts."""
    params = dict(spec.get("sampling") or {}) | (params or {})
    extra = list(spec.get("serve") or []) if extra is None else extra
    out: dict = {"image": VLLM_IMAGE, "gpu": GPU, "started_at": time.time(), "method": "sampled", "model": spec.get("watch_model"),
                 "repo": spec.get("repo"), "revision": spec.get("revision"), "n": n, "passes_requested": passes, "params": params, "extra_args": extra}
    out["gpus"] = _run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"]).stdout.strip().splitlines()
    out["vllm_version"] = _run(["vllm", "--version"], timeout=600).stdout.strip()[-200:]
    out["provenance"] = _provenance()
    out["copy"] = _copy_local()
    if out["copy"]["shards"] == 0:
        out["error"] = "no safetensors shards on the volume"
        return out
    cmd = _server_command(_tp(), 1, extra)
    out["command"] = cmd
    started = time.monotonic()
    if LOG.exists():
        LOG.unlink()
    with LOG.open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            _wait_ready(proc)
            out["startup_seconds"] = round(time.monotonic() - started, 1)
            try:
                out["server"] = {"version": _http("/version"), "models": _http("/v1/models")}
            except Exception as e:
                out["server"] = {"error": repr(e)[:200]}
            out["passes"] = [_sample_pass(index, prompts, n, params, concurrency) for index in range(max(1, passes))]
        except Exception as e:
            out["error"] = repr(e)[:500]
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=120)
            except Exception:
                proc.kill()
    out["log_tail"] = LOG.read_text(errors="replace")[-12000:] if LOG.exists() else ""
    out["finished_at"] = time.time()
    out["seconds"] = round(out["finished_at"] - out["started_at"], 1)
    results_dir = Path("/vol/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out["result_id"] = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(out["started_at"]))
    (results_dir / f"{out['result_id']}.json").write_text(json.dumps(out))
    weights.commit()
    return out


def _prompt_texts(path: Path) -> list[dict]:
    """{id, text} prompts from a fingerprint run's prompts.json ({id: text} or [[id, text]]) or a prepared prompts file."""
    document = json.loads(path.read_text())
    if isinstance(document, dict) and isinstance(document.get("prompts"), list):
        return [{"id": p["id"], "text": p["text"]} for p in document["prompts"]]
    if isinstance(document, dict):
        return [{"id": pid, "text": text} for pid, text in document.items() if isinstance(text, str)]
    return [{"id": item[0], "text": item[1]} for item in document]


@app.local_entrypoint()
def main(action: str = "inventory", prompts: str = "", out: str = "", top: int = 100, temperature: float = 1.0,
         configs: str = "", protocol: str = "", n: int = 32, passes: int = 2):
    if action == "download":
        print(json.dumps(download.remote(REPO, REVISION), indent=1))
    elif action == "sample":
        rendered = _prompt_texts(Path(prompts))
        result = sample.remote(rendered, SPEC, n=n, passes=passes)
        result["prompt_ids"] = [p["id"] for p in rendered]
        target = Path(out or f"runs-reference/{result.get('result_id') or time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-sampled.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=1))
        print("log tail:\n" + (result.get("log_tail") or "")[-2000:])
        print(json.dumps({k: v for k, v in result.items() if k not in ("passes", "provenance", "log_tail", "server")}, indent=1))
        print(f"-> {target}")
    elif action == "measure":
        document = json.loads(Path(prompts).read_text())
        rendered = [{"id": p["id"], "ids": p["ids"]} for p in document["prompts"]]
        result = measure.remote(rendered, top=top, temperature=temperature, configs=json.loads(configs) if configs else None,
                                protocol=json.loads(protocol) if protocol else None)
        result["prompts_document"] = {k: v for k, v in document.items() if k != "prompts"}
        result["prompt_ids"] = [p["id"] for p in rendered]
        target = Path(out or f"runs-reference/{result.get('result_id') or time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=1))
        for c in result.get("configs") or []:
            print(json.dumps({k: v for k, v in c.items() if k not in ("passes", "log_tail", "server")}, indent=1))
            print("log tail:\n" + (c.get("log_tail") or "")[-2000:])
        print(json.dumps({k: v for k, v in result.items() if k not in ("configs", "provenance", "protocol")}, indent=1))
        print(f"-> {target}")
    else:
        print(json.dumps(inventory.remote(), indent=1))
