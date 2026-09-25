# Production evidence, 22 Sep 2026

First live run of Proof of Edition layer 2 on api.lebrel.ai (platform version
0d856395, runtime deployed with sidecar build f3f5c51d… and pinned serving manifest
64dcb6d3…). Captured by `scripts/smoke_receipts.py` with the canary account:

- `manifest.json`: signed serving manifest as served by `/.well-known/proof-of-edition`
  (edition lebrel/deepseek-v4-flash-uncensored, 51 weight files, SGLang 0.5.18).
- `receipt-6d566ed8….json`: receipt of a non-streaming completion ("RECEIPT_OK"),
  returned both in the `Proof-Of-Edition-Receipt` header and at `/v1/receipts/{id}`;
  12 prompt tokens, 5 completion tokens.
- `receipt-….json` (second file): receipt of a streaming completion ("STREAM_OK")
  fetched by request id after the stream.

All three verify against the Ed25519 key pinned in the encrypted Python SDK
(`beFZtSwt6FnlhIYbX636n7w3/gpaASIkRnIMx52XJwk=`), and both receipts match the
SHA-256 of the plaintext request sent and of the answer received:

    poe-verify-receipt --public-key 6de159b52c2de859e584861b5fadfa9fbc37fe0a5a01222446720cc79d972709 --manifest-file manifest.json --receipt-file receipt-<id>.json
