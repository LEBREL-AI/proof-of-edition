# Watch: continuous verification of third-party inference APIs

The audit in `audit/` lets a client re-execute its own receipts. The watch does the
same job on a schedule for endpoints Lebrel does not operate, using Lebrel's own
probe battery, and publishes a board. It never touches client traffic.

## What a status can and cannot mean

| Claim | Strength |
| --- | --- |
| "Lebrel's own edition served these weights" (level 3) | Signed manifest, fingerprint and per-response receipts; anyone can re-execute and catch a lie. Not a hardware proof. |
| "This endpoint served answers consistent with the published weights when probed" | Statistical, sampled, black-box. Detects version swaps, most quantization changes, hidden system prompts, truncated context, ignored `max_tokens`. Cannot see a trick applied to a fraction of traffic the probes never hit. |
| "Your specific request was answered by those weights" | **Not claimable for routed traffic.** A route receipt says where the request went, when, and what the last probe said. On request Lebrel re-executes a client's own prompt against a reference and reports the likelihood. |

The board must always show the vocabulary below with its definitions.

* `consistent`: agrees with the compared deployments within the published thresholds.
* `suspect`: one of the two tests disagrees; wait for another run.
* `drift`: the target changed against its own previous run (same battery).
* `divergent`: disagrees with the reference, or with the majority of hosts of the same model.
* `disputed`: two hosts disagree and there is no third one.
* `unverified`: probed, nothing to compare against yet.
* `insufficient`: too few calls (or the key was missing).
* `error`: most probes failed.

## Two modes, both probed

A lab that reasons by default (DeepSeek) is probed twice: once in its native mode
(`deepseek/api-thinking`, thinking on, a larger `max_tokens` so the answer is not eaten
by the reasoning) and once with thinking disabled (`deepseek/api`), which is the setting
every host of the same weights is probed with, so hosts and lab compare like for like.
Disabling thinking is a test setting, applied identically to every target of a family
and printed on the board; it is never a change the router makes to a customer's request.
The router forwards the lab's defaults as they are.

## Method

* **Deterministic probes** (20, temperature 0): exact-match rate and shared-prefix ratio
  (`compare.deterministic_agreement`). A different engine or precision usually flips a
  late token; a swapped model diverges at once.
* **Sampled probes** (10 prompts × k samples, temperature 1): two-sample test with a
  Hamming kernel on the first 64 tokens, MMD-style statistic, permutation null within
  each prompt (`compare.two_sample_test`; Model Equality Testing, Gao, Liang and
  Guestrin, ICLR 2025).
* **Canaries**: hidden system prompt, over-refusal on six benign prompts, `max_tokens`
  honesty, needle recall at 8k and 32k tokens, tool calling. The battery never asks the
  model what it is: identity answers are not evidence of anything (measured on 23 Sep
  2026: DeepSeek's own API names itself or other vendors' models depending on model
  and mode). What identifies a deployment is how it answers the fixed prompts, compared
  statistically against the reference and the other hosts.
* **Comparisons**: drift (same target, previous run), reference (a deployment Lebrel
  runs from the published weights, `role: reference`), consensus (other candidates of
  the same model).

### Calibrate before publishing

Measured on 23 Sep 2026: DeepSeek's own API, two runs of the same battery minutes apart,
thinking disabled, agreed exactly on 45% of the temperature-0 prompts with a mean shared
prefix of 0.61; the sampled test did not reject (p 0.72). Temperature 0 is not
deterministic there, so its target carries `thresholds` of 0.3 and 0.5, and a status is
never "drift" for that alone. Every lab needs its own floor; the defaults are for
deployments that really are deterministic.

Thresholds are not laws. Run the battery twice against two deployments known to be the
same edition (or the same deployment twice) and read `compare` numbers from the board
JSON: the noise floor of `exact_rate`, `prefix_agreement_mean` and the two-sample
`p_value`. Production thresholds must sit above that floor. The board publishes the
thresholds it used in `thresholds`.

## Run, calibrate, publish

```bash
export DEEPSEEK_API_KEY=... OPENROUTER_API_KEY=... VENICE_API_KEY=... ORCAROUTER_API_KEY=...
python3 -m watch.run --targets watch/targets.json --out runs            # one run (~100 calls per target)
python3 -m watch.board --runs runs --out board.json --html board.html   # latest run vs previous
python3 -m watch.run --targets watch/targets.json --out runs --dry-run  # list probes and which keys are missing
```

```bash
python3 -m watch.calibrate --targets watch/targets.json --only deepseek/api   # the same-deployment noise floor
python3 -m watch.publish --board board.json --bucket lebrel-registry --endpoint https://<account>.r2.cloudflarestorage.com
```

`calibrate` runs the battery twice against the same target and prints the exact-match
rate, prefix ratio and two-sample statistic a genuine match produces, with suggested
thresholds; it warns when the sampled test rejects a same-deployment pair. `publish`
uploads the board to `watch/board.json` in the registry bucket (and keeps every run under
`watch/history/`); `models.lebrel.ai/watch` renders it and `/v1/watch` serves it raw.

Targets without their key are skipped and reported as `insufficient`, never invented.
Cost per run is dominated by the needle probes (about 40k input tokens per target); at
DeepSeek Flash prices a full run over ten targets is well under a dollar.

## Files

`client.py` one call, one recorded exchange · `battery.py` the versioned probes ·
`compare.py` the statistics · `run.py` runs and records · `board.py` compares and renders ·
`calibrate.py` the noise floor · `publish.py` upload to the registry bucket ·
`targets.example.json` a starting configuration.
