# Proof of Edition

An open standard and toolset for one question: **is this API really serving the
model edition it says it is?** Weights, quantization, engine and all.

Providers publish; anyone verifies. Nothing here depends on trusting the provider's
word or on confidential-computing hardware, although hardware attestation can be
layered on top (layer 5).

| Layer | What it is | Tooling |
| --- | --- | --- |
| 0. Edition registry | The publisher's own registry: content-addressed signed records, a transparency log and interchangeable mirrors, so nothing can be swapped or removed silently. | `proof_of_edition/registry/` (records, log, manifest check) |
| 1. Edition fingerprint | The edition itself answers a private set of questions in a way no other model does (Chain & Hash, Russinovich & Salem, ICLR 2026). | `proof_of_edition/fingerprint/` (the statistics and the verifier; the training pipeline and the private question sets stay with the publisherin on Modal, verify black-box) |
| 2. Signed manifest + receipts | The runtime signs what it serves and signs a receipt for every response: hashes of the request and the answer, the manifest, token counts. | `proof_of_edition/manifest/`, `proof_of_edition/receipts/` |
| 3. Open re-execution audit | Anyone re-runs receipts on a reference deployment of the same edition and tests the answers for consistency. | `proof_of_edition/audit/` |
| 4. Serving bond | The provider stakes value against a proven substitution. | specification only, for now |
| 5. Attestation | Optional hardware attestation of the manifest. | specification only, for now |

The full specification is in [SPEC.md](SPEC.md). This repository is the reference
implementation Lebrel runs in production: the receipts and manifests served at
api.lebrel.ai, the registry behind models.lebrel.ai, and the watch behind
models.lebrel.ai/watch. Not here: the fingerprint training pipeline, every private
question set, and Lebrel's operational notes.

## Verify a response you received

```bash
pip install proof-of-edition   # PyPI; or uv tool install proof-of-edition
curl -s https://api.lebrel.ai/.well-known/proof-of-edition > manifest.json
curl -s https://api.lebrel.ai/v1/receipts/$REQUEST_ID > receipt.json
poe-verify-receipt --public-key $PROVIDER_SIGNING_KEY \
  --manifest-file manifest.json --receipt-file receipt.json \
  --prompt-file request.json --response-file answer.txt
```

Exit code 0 means: both documents are signed by the pinned key, the manifest is
current, the receipt references that manifest, and the digests match the request
you sent and the answer you received.

## Check an edition's fingerprint

```bash
poe-verify-fingerprint --base-url https://api.lebrel.ai/v1 --model lebrel/<edition> \
  --fingerprint fingerprint.private.json --k 10 --tau 2
```

The fingerprint file is private to whoever trained the edition; only the sampled
questions are sent to the deployment. With k = 10 questions and a threshold of 2
hits, the false-positive rate against an unrelated model is negligible
(see `proof_of_edition/fingerprint/chainhash.py`).

## Audit a provider

```bash
poe-audit --samples exchanges.jsonl --manifest manifest.json --public-key $KEY \
  --reference-url https://your-reference-deployment/v1 --reference-model <edition> \
  --out report.json
```

Deterministic requests are compared by exact match and shared prefix; sampled
requests use a paired permutation test over a Hamming kernel. See
`proof_of_edition/audit/reexecute.py`.

## Check a runtime against the registry

```bash
poe-check-manifest --manifest manifest.json --runtime-key $RUNTIME_KEY \
  --record record.json --publisher-key $PUBLISHER_KEY --log log.jsonl --checkpoint checkpoint.json
```

Exit code 0 means the runtime serves exactly the files the registry published for
that edition revision, and the record is in the transparency log.

## Build a manifest for your own edition

```bash
poe-build-manifest --edition manifest/editions/<name>.json \
  --weights-manifest /path/to/model_manifest.json --out manifest/out/<name>.manifest.json
```

The output is the manifest base a runtime signs at serving time. The Lebrel
encrypted runtime (Go) reads it from `LEBREL_SERVING_MANIFEST_JSON`.

## Watch the labs

`proof_of_edition/watch/` probes the labs' own APIs and every host of the same model on a schedule,
compares them with each other and with Lebrel's own run of the published weights,
and publishes a signed board at https://models.lebrel.ai/watch: the battery, the
hourly first-word fingerprint (by log-probabilities, or by sampling where a lab
returns none), the published-weights references and every threshold. How each
status is computed, and what it can and cannot mean, is in
[watch/README.md](proof_of_edition/watch/README.md).

## Development

```bash
uv run --group dev --extra reference python -m pytest
```

The fingerprint training pipeline and the private fingerprint files are not part
of this repository; they never leave the publisher.

## Contact

Questions, findings and security reports: contact@lebrel.ai. Lebrel AI LLC.
