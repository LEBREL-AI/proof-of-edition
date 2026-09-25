# Proof of Edition · v0.1 (draft, 2026-09-22)

An open standard for open-weight model APIs: every served edition carries its own
identity, every response carries a receipt, anyone can re-run the evidence, and
lying costs the operator money. Works on any GPU and any model; hardware
attestation slots in when available.

## What it proves, honestly

| Layer | Question it answers | Nature of the guarantee |
|---|---|---|
| 1. Edition fingerprint | Is this deployment running weights that descend from the named edition? | Behavioral, black-box, statistical (false positive ≈ 4.5e-5 per check) |
| 2. Serving manifest + receipts | Which exact weights, quantization and engine does the operator commit to for this response? | Signed commitment; fraud leaves verifiable evidence |
| 3. Open re-execution audit | Do the served distributions match the committed weights and quantization? | Reproducible by any auditor with the open weights |
| 4. Serving bond | What does lying cost? | Economic; a public USDC bond claimable with layer-3 evidence |
| 5. Hardware attestation (optional) | Did an attested enclave run it? | Cryptographic, when the runtime is on a confidential GPU |

Without layer 5 there is no mathematical proof of which weights ran. Layers 1–4
make substitution detectable, attributable and expensive. Providers that adopt
the standard state this limit in the same words.

## 0. Edition registry

Every edition is published in a registry the publisher operates (for Lebrel,
`registry.lebrel.ai`), never only on a third-party hub. The registry is designed so
that no host, including the publisher, can silently change or remove what was
published:

* **Content-addressed records.** An edition revision is a signed record listing every
  file with its SHA-256 and size; the revision id is the SHA-256 of the canonical
  `{path: sha256}` map, so identical bytes have identical revisions on every mirror
  and a mirror cannot substitute a file without changing the revision. Records also
  carry the edition identity, base model, license and `fingerprint_id`.
* **Publisher key.** Records and log checkpoints are signed with the registry's
  publisher key, a root of trust separate from the runtime's signing key. Verifiers pin
  both.
* **Transparency log.** Each publication appends `SHA-256(previous || record)` to an
  append-only hash chain; the publisher signs checkpoints `{size, head, at}`. Anyone
  who keeps a checkpoint can prove later that the registry rewrote or removed a
  record. The log and the catalog are kilobytes and are replicated on every mirror.
* **Mirrors are interchangeable.** Weights may be served from any number of hosts
  (rented servers, object storage, the publisher's own storage); clients verify file
  digests on download, so the host is never trusted. Removal by one host is a
  replication event, not a loss.
* **Serving manifests reference the registry.** A runtime's manifest names the
  edition, the registry revision and the digests it loads; `poe-check-manifest`
  proves that the runtime serves exactly what the registry published, and that the
  record is in the log.

Tooling: `registry/record.py` (publish and check records), `registry/log.py`
(append, checkpoint, verify, inclusion), `registry/check_manifest.py`. The download
protocol follows the de-facto hub conventions that inference engines (vLLM, SGLang)
already speak, so they pull from the registry without code changes; this is protocol
compatibility, not a dependency on any third-party hub.

## 1. Edition fingerprint (Chain & Hash)

- The edition is fine-tuned so that a private set **Q** of trigger questions
  (random vocabulary tokens, 10 per question) is answered with short phrases from
  a public list **R** of 256 responses. Each answer is bound by
  `SHA-256(q_i ‖ Q ‖ R ‖ salt) mod 256`, so nobody can add or forge a trigger
  without retraining the weights.
- Training mixes the pairs with ordinary data, varied system prompts and random
  prefix/suffix tokens, and regularizes on the base model's outputs so the
  edition's quality is unchanged (published benchmarks within the edition's own
  evaluation contract).
- Verification: sample k = 10 questions, temperature 0, accept when ≥ τ = 2
  answers match. Reference: Russinovich & Salem, *Hey, That's My Model!*,
  ICLR 2026 (arXiv 2407.10887): <0.5 % loss under INT8 quantization, survives
  downstream fine-tuning.
- The edition publisher keeps Q private and issues disjoint subsets to
  verifiers; spent questions are retired. Public verification bodies can hold
  their own subsets.

## 2. Serving manifest and receipts

`GET /.well-known/proof-of-edition` returns a signed manifest (Ed25519, key pinned
in clients, validity ≤ 24 h):

```json
{
  "version": 1,
  "edition": "lebrel/qwen3.8-27b-lebrel-uncensored",
  "weights": {"repository": "…", "revision": "<git sha>", "files": {"model-00001.safetensors": "<sha256>"}},
  "quantization": {"method": "nvfp4", "config_sha256": "…"},
  "engine": {"name": "sglang", "version": "…", "dtype": "…", "kv_cache_dtype": "…", "tensor_parallel": 1, "max_context": 262144},
  "tokenizer_sha256": "…", "chat_template_sha256": "…",
  "runtime": {"image_sha256": "…", "instance_id": "…"},
  "attestation": null,
  "issued_at": 0, "expires_at": 0, "signing_key_id": "…"
}
```

Each response carries a receipt signed by the runtime. Non-streaming responses
return it base64-encoded in the `Proof-Of-Edition-Receipt` header; every response
(streaming or not) can be fetched afterwards at `GET /v1/receipts/{request_id}`,
where `request_id` is the `X-Request-ID` the runtime returned:

```json
{"payload": {"version": 1, "request_id": "…", "manifest_sha256": "…", "prompt_sha256": "…",
             "response_sha256": "…", "prompt_tokens": 0, "completion_tokens": 0,
             "issued_at": 0, "instance_id": "…", "signing_key_id": "…"},
 "signature": "…"}
```

`manifest_sha256` is the manifest identity: SHA-256 of the canonical manifest
payload with `issued_at` and `expires_at` removed, so a receipt matches every
republication of the same serving configuration. `prompt_sha256` is the SHA-256
of the plaintext request body exactly as the client sent it; `response_sha256`
is the SHA-256 of the assistant text (UTF-8) the client received, concatenated
across stream deltas. Token counts are `null` when the engine does not report
them. Receipts contain hashes only; the client keeps the plaintext. A receipt whose
manifest disagrees with the weights actually served is the evidence of layer 3.

### Tooling for layer 2

* `manifest/build_manifest.py` builds the manifest base for an edition from the
  upload-time weights manifest (per-file SHA-256, the same file the runtime pins)
  and an edition spec in `manifest/editions/`. Weight shard digests, tokenizer
  and chat-template digests and the engine arguments digest are derived, never
  typed by hand. The runtime receives the result as `LEBREL_SERVING_MANIFEST_JSON`.
* `receipts/verify_receipt.py` is the client-side verifier: it pins the runtime's
  Ed25519 public key, fetches (or reads) the manifest and a receipt, and checks
  signatures, the manifest window, the manifest identity and, when given the
  plaintext, the prompt and response digests. Exit code 0 means verified.
* The Lebrel encrypted runtime (Go sidecar) implements the manifest endpoint, receipt issuance for plain, streaming and
  agent responses, the `Proof-Of-Edition-Receipt` header and `GET /v1/receipts/{id}`.
  Its canonical JSON is byte-identical to `receipts/schema.py`.

## 3. Open re-execution audit

Open weights make every deployment reproducible. An auditor with the manifest's
weights, quantization and engine re-runs a sample of receipts (prompt, seed,
sampling settings, top-k logprobs supplied by the client) and applies the
two-sample tests of *Model Equality Testing* (Gao et al., ICLR 2025). The
protocol fixes sample sizes, tolerances per engine and GPU family, and the
report format. Anyone may audit; the publisher runs a continuous public monitor
over its own endpoint and over other providers, with the same method.

### Tooling for layer 3

`audit/reexecute.py` is the reference audit tool. It takes a JSONL of exchanges
(request body, assistant text, signed receipt), verifies every receipt against
the manifest, then re-executes the prompts on a reference deployment of the
edition. Deterministic requests (temperature 0 or top_k 1) are compared by exact
match and shared-prefix ratio; sampled requests are compared with a Hamming
kernel over the leading tokens and a paired permutation test (statistic:
similarity of the reference to itself minus similarity of the recorded answer to
the reference; alpha 0.01). The JSON report lists per-sample results and a
verdict: consistent, inconsistent or error.

## 4. Serving bond

The operator locks a public USDC bond per edition. A claim is a set of receipts
plus an audit report showing a mismatch beyond the published tolerance,
adjudicated by re-execution. Successful claims pay the claimant from the bond.

## 5. Hardware attestation

When the runtime runs on a confidential GPU (TDX / SEV-SNP with NVIDIA CC), the
manifest's `attestation` carries the quote binding the runtime image, the
weights digest and the signing key; verifiers check it like Tinfoil, Phala and
NEAR clients do today. Nothing else in the standard changes.

## Verifier

One open-source tool: `poe verify <endpoint> <edition>` runs layers 1 and 2 in
seconds and prints a verdict with error rates; `poe audit` runs layer 3 given the
weights. Reference implementation in this repository.
