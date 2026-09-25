# The reference

The watch compares every route and host with the lab's own API. That answers "does Lebrel serve what DeepSeek
serves?". The reference answers the other half: "does DeepSeek's API serve the model it published?". We run the
published weights ourselves and measure the same first-word distributions.

## Files

- `app.py` — a Modal app. `download` fetches the checkpoint at a pinned revision into the volume
  `lebrel-watch-reference-flash`; `measure` serves it with vLLM and records the top-100 first-word
  log-probabilities of every prompt (two serial passes and one with everything in flight).
- `prepare.py` — renders the prompts with the lab's own prompt format (`encoding/encoding.py` of the edition,
  thinking off) and tokenizes them; `--check` compares the token counts with what the API billed for the same
  prompts (they must be equal, otherwise the format differs and nothing else can be compared).
- `analyze.py` — turns a run into fingerprint distributions, fits the temperature the API applies, and compares the
  API and every host with the reference.

## A run

```
python -m proof_of_edition.watch.reference.prepare --prompts runs-fingerprint/<run>/prompts.json \
    --out runs-reference/prompts-<week>.json --check runs-fingerprint/<run>/summary.json
modal run proof_of_edition/watch/reference/app.py --action measure --prompts runs-reference/prompts-<week>.json \
    --out runs-reference/<id>.json
python -m proof_of_edition.watch.reference.analyze --result runs-reference/<id>.json --prompts runs-reference/prompts-<week>.json \
    --runs runs-fingerprint --out runs-reference/<id>-report.json
```

Costs: the download once (CPU, minutes), each measurement roughly half an hour of two B300s.
