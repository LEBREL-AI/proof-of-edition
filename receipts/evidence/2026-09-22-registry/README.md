# Production evidence with the registry, 22 Sep 2026 (evening)

Runtime redeployed with a serving manifest that names the Lebrel registry
(`registry.lebrel.ai/lebrel/deepseek-v4-flash-uncensored`, content revision
`67752462…`). Platform version 7ebb9188 (standby off for the test), 04486bb4 (standby on).

Files: `manifest.json` (live signed serving manifest from api.lebrel.ai), two
`receipt-*.json` (non-streaming "RECEIPT_OK" with header + fetch; streaming "STREAM_OK"
fetched by request id), `registry-record.json` (signed edition record from
registry.lebrel.ai), `registry-log.jsonl` and `registry-checkpoint.json` (transparency
log, one entry, signed head).

Checks that passed:

    poe-verify-receipt  … both receipts: signatures, manifest window, manifest identity, prompt and response digests
    poe-registry-record check … record signed by publisher key 65f063c8…, revision equals its content
    poe-registry-log verify … chain valid, checkpoint signed, record included at index 0
    poe-check-manifest … "VERIFIED the runtime serves lebrel/deepseek-v4-flash-uncensored revision 67752462… exactly as published (51 weight files)"

Publisher key 65f063c8… is provisional until a key ceremony with the founder.
