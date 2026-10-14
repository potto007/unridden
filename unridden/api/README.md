# Unridden Decision API v1

This package provides the isolated, full-depth non-generative API. It loads one
Gemma model in one owned native child and reads logits for a validated finite
label alphabet after prefill. It does not create a sampler, append an answer
token, or feed an answer token back into the model.

The API model id is `local-gemma-unridden-v1`. It names this local
deployment only and claims no parity with any hosted service.

## Install the native worker

Either download a published bundle or compile one. They produce the same
directory and are verified identically at startup.

```bash
# Picks cuda13, cuda12 or cpu from the NVIDIA driver and says which, checks the
# download against the release SHA256SUMS and (with gh installed) GitHub's
# build provenance, then unpacks into a directory that must not already exist.
python -m unridden.api.cli worker fetch --output build/api-worker
```

## Build the native worker

The build is create-only and verifies the base runtime before compiling:

```bash
# 1. Produce the llama.cpp base runtime (headers + shared libraries). The
#    default revision is the tested release v0.4.1; --revision takes any other
#    tag or commit. --cuda-architectures keeps a CUDA build to your own GPU.
python scripts/unridden/build_base_runtime.py \
  --out build/llama-base --cuda --cuda-architectures 120

# 2. Compile the worker against it.
python -m unridden.api.native.build \
  --base build/llama-base \
  --output build/api-worker
```

The result is a bundle that names no path outside itself: the executable at
`build/api-worker/build/unridden-worker`, a copy of the shared libraries at
`build/api-worker/runtime/`, and `build/api-worker/build.json`, which records
the worker source, executable, bundled sha256 helper, linked runtime, and
llama.cpp revision hashes, with the executable and runtime directory relative
to the manifest. Startup enforces the executable, runtime files, and runtime
bundle hashes; the source and helper hashes are a build record and are not
rechecked at startup. `python -m unridden.api.native.bundle pack|unpack`
moves a bundle between machines
([ADR 0005](../../docs/decisions/0005-relocatable-prebuilt-worker-bundles.md)).

The llama.cpp revision is provenance, not a gate. Every hash above is enforced
whatever the revision, and a revision other than the tested one only logs a
warning that the published measurements were taken on the tested revision and
may not reproduce. `GET /v1/models` reports it as `runtime.llama_revision` and
`runtime.tested_revision`. The model is pinned separately by
`ApiConfig.model_sha256` (CLI `--model-sha256`): set it to your GGUF's sha256
and startup refuses any other file. Building and running the CPU helper test
does not load the model or start GPU work.

## Create the ASGI app

Importing the package or creating the app does not start a child process. The
ASGI lifespan owns startup and shutdown:

```python
from pathlib import Path

from unridden.api.app import ApiConfig, create_app

app = create_app(
    ApiConfig(
        model_path=Path("models/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"),
        worker_path=Path("build/api-worker/build/unridden-worker"),
        manifest_path=Path("build/api-worker/build.json"),
        gpu=True,
    )
)
```

`ApiConfig` defaults are context 2048, batch and ubatch 256, 8 threads, at
most 32 questions, a 1 MiB HTTP request, a 4 MiB worker response, a 120 second
request timeout, and a 600 second startup timeout. `gpu` defaults to `False` and
must be opted into explicitly. `batched` defaults to `False`; see "Batched
evaluation" below.

Run a loopback server with the same factory:

```bash
python -m unridden.api.cli serve \
  --worker build/api-worker/build/unridden-worker \
  --manifest build/api-worker/build.json \
  --model-path models/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
  --gpu --host 127.0.0.1 --port 8090
```

`serve` binds a loopback port and runs in the foreground; put it behind a
process supervisor before any standing use. Validation and batch work need no
port at all: use the ASGI harness or `run` below. With `--gpu` and the default
2048-token context the worker holds about 18 GiB of device memory (roughly
16 GiB of weights plus KV and compute buffers), so confirm the card has that
much free before starting it.

The supported routes are `POST /v1/decisions`, `GET /v1/models`, and
`GET /health`. FastAPI also exposes machine-readable OpenAPI at `/openapi.json`.
`POST /v1/systemone` is a compatibility alias for `POST /v1/decisions`, offered
as an interoperability path for clients written against that request shape.

## HTTP request and result

```json
{
  "model": "local-gemma-unridden-v1",
  "state": {"message": "The card was charged twice"},
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Choose the best route",
      "criteria": {
        "billing": null,
        "technical": "A software defect"
      }
    },
    "severity": {
      "type": "score",
      "instructions": null,
      "criteria": ["low", "medium", "high"]
    },
    "duplicate": {
      "type": "noul",
      "instructions": "The state describes a duplicate charge"
    }
  }
}
```

For label probabilities `[0.2, 0.3, 0.5]`, Score returns the zero-based
expected value `1.3`. Noul returns only `P(true)`:

```json
{
  "model": "local-gemma-unridden-v1",
  "answers": {
    "route": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {"billing": 0.8, "technical": 0.2},
      "confidence": 0.8
    },
    "severity": {
      "type": "score",
      "score": 1.3,
      "legend": {"0": "low", "1": "medium", "2": "high"},
      "probabilities": {"0": 0.2, "1": 0.3, "2": 0.5},
      "confidence": 0.5
    },
    "duplicate": {"type": "noul", "noul": 0.9}
  },
  "usage": {"input_tokens": 384, "output_tokens": 0}
}
```

Use `?diagnostics=true` to add a top-level map keyed by question id. Each entry
contains `raw_label_logits`, `token_mapping`, `coverage`,
`full_vocabulary_argmax`, `conditional_score_semantics`, `confidence_formula`,
`prompt_sha256`, `prompt_version`, `cache_cleared`, `prompt_tokens`,
`processed_tokens`, `reused_tokens`, `evaluation_mode`, `batch_sequences`,
`timing_ms`, `model_sha256`, `runtime_sha256`, `execution_mode`,
`generated_tokens`, and `callbacks_enabled`.

### Calibrated confidence and the escalation gate

`confidence` is the raw `max_probability_v1` and stays that way. When the
server is started with `--calibration-table table.json` (ADR 0012), each
diagnostics entry also carries:

- `calibration`: `temperature` and `temperature_source` (`bucket`,
  `type_fallback` or `default`), the `bucket` (`choice:2`, `score:5`,
  `noul:2`), `calibrated_probabilities` keyed like the answer,
  `calibrated_confidence`, `calibrated_probability_margin`,
  `scaled_logit_margin`, `calibrated_normalized_entropy`, and the table's
  `table_status` (`fitted_requires_held_out_validation` or
  `validated_on_held_out`).
- `gate`: `action` (`accept_primary`, `run_permutation_probe`, `escalate`)
  and `reason` from the calibrated-margin policy in `unridden.api.gate`.

Add `?probe=true` (with `?diagnostics=true`) and the service re-asks every
question whose action is `run_permutation_probe` with its options reversed,
against the same state and under the same zero-generation contract. The
verdict then rests on both passes and `gate.probe` reports `flipped`,
`permuted_top1`, `total_variation_distance`, `post_probe_margin`, the
realigned probabilities, and the probe's `prompt_sha256` and
`processed_tokens`. Probe tokens are counted in `usage.input_tokens`. The
primary answers are never replaced by the probe.

The table binds to `model_sha256`, `runtime_sha256` and the compiler's
prompt version; startup refuses a table fitted on anything else, and
`GET /v1/models` reports the loaded binding under `calibration`. Fit one from
a `run_usecase_suites.py --diagnostics` run:

```bash
python scripts/unridden/fit_calibration_table.py \
  --observations outputs/suites/<run>/observations.jsonl \
  --out models/calibration/<model>.json
python -m unridden.api.cli serve ... --calibration-table models/calibration/<model>.json
```

The fitter writes `status: fitted_requires_held_out_validation`; run
`scripts/unridden/evaluate_escalation_gate.py` for the held-out numbers. The
gate thresholds default to `high_margin_threshold: 0.60`,
`low_margin_threshold: 0.20`, `min_coverage: 0.10`, `max_tvd_threshold: 0.25`
and `post_probe_margin_threshold: 0.35`; `--gate-config thresholds.json`
overrides any of them.

Native stderr is never copied into HTTP errors. Operators can enable the
`unridden.api.native` logger at DEBUG or inspect the bounded
`app.state.service.backend.stderr_tail`. The exact owned child is available as
`app.state.service.backend.pid`, and its validated handshake as `.profile`.

## Batch CLI

The batch CLI accepts one request object, an array of request objects, or JSONL.
It starts one backend for the complete batch and creates the output JSONL only
when the path does not already exist:

```bash
python -m unridden.api.cli run \
  --input requests.jsonl --output results.jsonl \
  --worker build/api-worker/build/unridden-worker \
  --manifest build/api-worker/build.json \
  --model-path models/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
  --gpu --diagnostics
```

## SemIf interchange

A separate adapter reads and writes SemIf-style JSONL rows: `id`, `state`,
`question`, and 2 to 16 ordered `{id, description}` options. One row maps to one
Choice question and keeps the row id and option order. SemIf is an unrelated
open-source project; this is an interchange convenience, not an integration.

```bash
python -m unridden.api.semif to-api \
  --input semif.jsonl --output requests.jsonl
python -m unridden.api.semif from-api \
  --input requests.jsonl --output semif-roundtrip.jsonl
```

## Limits and semantics

`GET /v1/models` reports the label capacity validated from the loaded
tokenizer. Choice accepts 2 to 255 options at the wire boundary, then rejects a
question above that backend capacity before inference. Score accepts 2 to 10
levels. Noul requires instructions or a true/false rubric. Option ids and order
are preserved; question ids are correlation metadata and are absent from model
prompts.

Each question is compiled into its own prompt from the state plus that
question's own instructions and criteria. It never sees sibling questions, so a
policy or definition that a question depends on must be in the state or repeated
in that question's instructions.

Within one request, the text every question shares (the system instruction and
the state) is prefilled once and reused: the first question clears the context
and prefills its whole prompt, and each later question trims the context back to
the shared prefix and prefills only its own remainder. The split point depends
only on a question's own prompt, and the prefix is always prefilled in the same
chunks, so a question gets the same logits alone, reordered, or beside any
siblings. Nothing is kept between requests. Diagnostics report the accounting
per question: `processed_tokens + reused_tokens == prompt_tokens`, and
`cache_cleared` is true exactly when nothing was reused. `usage.input_tokens`
counts the tokens actually prefilled.

Parallel GPU requests are outside v1. The service permits one
active inference request and immediately returns 429 while busy. Conditional
label probabilities are finite and normalized, but they are not calibrated
correctness probabilities. Choice and Score confidence is
`max_probability_v1`.

State, instructions, and option text must not spell a model control token such
as `<turn|>` or `<|turn>`. The worker parses special tokens so the chat template works,
so such text could forge a turn boundary. It is rejected before inference with
422 `unsupported_content`. Ordinary markup and angle brackets are accepted.

Errors use `{error: {code, message, field?, retryable}}`. Statuses distinguish
malformed input (400), unknown model (404), timeout (408), oversized body
(413), unsupported media (415), schema or model budget failures (422), busy
(429), unavailable backend (529), and internal failure (500). A batch never
returns a successful partial result.

## Batched evaluation

`ApiConfig.batched` (CLI and runner `--batched`) turns on the opt-in mode of
[ADR 0004](../../docs/decisions/0004-optional-batched-question-evaluation.md).
It is chosen when the worker starts, never per request, because it changes how
the context is built. The worker then gives every question of a request its own
KV sequence over one copy of the shared prefix and decodes all the remainders
together. That removes about a third of the fixed per-question cost: measured
over a 1,007-token state, each extra question costs about 27 ms sequentially and
about 18 ms batched.

It is not bit-identical to the default and cannot be: answers on this model move
with the prefill batch shape, and batching several questions changes that shape
by construction. A question's result then depends on how many siblings the
request carries and how long they are. What a sibling *says* did not move it:
each question's tokens carry only their own sequence id, so the attention mask
drops every sibling cell, and the harness measures that with an adversarial
sibling of identical length in both question orders (delta 0.0 in every probe so
far, and the harness fails if an adversarial sibling moves the target further
than a neutral one of the same length). Measurements and the recommendation are
in [the batched-mode results](../../docs/results/batched-mode.md).

`ApiConfig.batched_context` (default 8192) sizes the unified KV cache. Each cell
costs about 0.21 MiB of KV on this model, so the default costs about 1.3 GiB
more VRAM than the sequential context; the value may not exceed
`max_questions * context_size`, the most any request could need. A request needs
the shared prefix plus every question's remainder to fit; the per-question prompt
limit is still the 2048-token context size. A request that does not fit is
evaluated by the sequential path inside the same context instead, and the
response says so. Nothing is truncated.

A batched worker reports the regime it used in the body of every response, so a
caller sees it without `?diagnostics=true`:

```json
{"evaluation": {"mode": "sequential", "fallback": "context"}}
```

`mode` is `batched` normally and `sequential` when the request did not fit;
`fallback` is `context` only in that case and is otherwise absent. A sequential
worker has one regime for its whole life and reports it in `GET /v1/models`
instead, so its responses carry no `evaluation` field at all. Diagnostics add
`evaluation_mode` and `batch_sequences` per question, and the mapping layer
rejects a report that contradicts the handshake.
