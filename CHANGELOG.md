# Changelog

Notable changes to unridden. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the major version is 0, the HTTP contract, the CLI flags, and the native
worker protocol can change between minor versions without a deprecation
period. Each release records which llama.cpp revision and which model file its
published measurements were taken on.

## Unreleased

### Added

- A Tetris demo, `scripts/unridden/play_tetris.py`: Unridden picks every
  placement through `/v2` Choice questions, with heuristic and random
  baselines, a `--trace` JSONL log, and `tetris_replay.html` to step through a
  recorded game. Results in `docs/results/tetris-demo.md`.

- Experimental Rider mode, `POST /v2/rider` (with `--snapshots`): greedy text
  generation that continues from a saved snapshot on the split runtime without
  re-prefilling the snapshot prefix (ADR 0007, named in ADR 0008). The snapshot
  worker gains a `ride` command. This is the one route that generates tokens;
  `/v1` and the decision routes still generate none.
- Snapshot profile `full-v1` (`--snapshot-profile full-v1`, ADR 0009): one
  whole-model context on the stock graph, so `/v2` serves any Gemma 4,
  including E4B with its shared K/V and per-layer inputs. Snapshots exist only
  at the model's block count (42 for E4B), with no promotion. `checkpoints` and
  `readout.completed_blocks` are now integers the running profile validates
  (`capability_unavailable` otherwise); 26B clients sending 18/30 see no
  change. The final residual is tagged `raw_residual_after_final_block` and
  read as `which=final`. The worker bundle manifest lists `profiles`.
  `validate_api_v2.py` reads the block counts from `/v2/models`, and
  `qualify_snapshots.py` takes `--profile full-v1`; E2B and E4B pass its
  release gates with bit-exact restores
  ([results](docs/results/full-depth-snapshots.md)).

### Fixed

- Failures now reach the server log. `serve` sends the `unridden` loggers to
  stderr (`--log-level`, default info), which the systemd units append to the
  llama.cpp log; worker failure lines (`REQUEST_FAILED`, `PREFLIGHT_FAILED`,
  `WORKER_FAILED`) log as warnings instead of debug, and every 5xx logs its
  underlying error. Before, a worker-side failure showed only as a bare 500.
  A refused create or promote no longer sends a cleanup drop for ids the
  worker never created, so its refusal is the only warning logged.
- `persistence: "disk"` no longer fails with a 500 when the store directory
  is relative (the default `build/snapshots`, as the systemd units use): the
  store resolves its root, since the worker only saves to and loads from
  absolute paths.
- `POST /v2/snapshots` with `checkpoints: [30]` alone no longer fails with a
  500: a 30 snapshot created without an 18 keeps H18, which the response
  schema wrongly rejected (#30).
- A dead snapshot worker is restarted instead of leaving `/v2` unavailable
  until the server restarts. The next request (or `/health`, in the background)
  restarts it, with a 10 s backoff after a failed restart; the new worker must
  match the first. Disk snapshots are reloaded on next use and memory-only
  snapshots answer `snapshot_not_found`. A broken pipe to the worker is now a
  retryable 529 rather than a 500.

## 0.4.0 - 2026-09-23

### Added

- Releases publish an experimental `unridden-snapshot-worker` bundle (cuda13,
  cuda12) for the `/v2` snapshot routes, built against llama.cpp with
  `gemma4-layer-range.patch` applied (ADR 0006). Install it with
  `worker fetch --kind snapshot-worker`; it lands in `build/api-snapshot-worker`.
  The split profile still fails qualification gate 5, so it stays
  experimental, and a failed snapshot build does not hold back the v1 bundles.

## 0.3.0 - 2026-09-22

### Changed

- Renamed the community project to Unridden across the Python distribution,
  import package, command line tools, native workers, container image, model
  identifier, protocol identifiers, documentation, and examples. The new
  distribution and imports are `unridden`; the commands are `unridden-api` and
  `unridden-semif`; the model identifier is `local-gemma-unridden-v1`.
- The first two tagged releases used the former name, Riderless. Their published
  assets and existing installations retain their original names and wire IDs.
  Installations moving from v0.2.0 to v0.3.0 must update imports, commands,
  worker bundles, environment variables, and request model IDs. The HTTP route
  paths and JSON field names are unchanged.

The v0.1.0 and v0.2.0 entries below describe the features under their current
Unridden names. Assets attached to those historical GitHub releases keep their
original filenames.

## 0.2.0 - 2026-09-22

### Added

- Prebuilt worker bundles, published from GitHub Releases, so a first answer
  needs no compiler. `python -m unridden.api.cli worker fetch` (also
  `scripts/unridden/fetch_worker.py`) picks `cuda13`, `cuda12` or `cpu` from
  the NVIDIA driver `nvidia-smi` reports and says which, verifies the download
  against the release's `SHA256SUMS`, checks GitHub's build provenance with
  `gh attestation verify` when `gh` is installed (`--require-attestation`
  makes a missing or failing check fatal instead of a warning), and unpacks
  into a `--output` that must not already exist
  ([ADR 0005](docs/decisions/0005-relocatable-prebuilt-worker-bundles.md)).
- `unridden.api.native.bundle`, which defines the install layout
  (`build.json`, `build/unridden-worker`, `runtime/*.so*`) and packs and
  unpacks it. `ApiConfig`'s defaults already point inside it, so a bundle
  unpacked into `build/api-worker` needs no flags.
- A release workflow on `v*` tags building all three flavors in NVIDIA's devel
  containers with `GGML_NATIVE=OFF` and
  `CMAKE_CUDA_ARCHITECTURES=80;86;89;90;120`, attaching the tarballs and a
  `SHA256SUMS` to the release, and attesting them with
  `actions/attest-build-provenance`.
- A CUDA 13 container image, `ghcr.io/potto007/unridden:<version>-cuda13`,
  built from `docker/Dockerfile`. It expects a GGUF mounted at `/models` and
  serves on 8090.
- `build_base_runtime.py --no-native` (`GGML_NATIVE=OFF`), and CUDA builds now
  copy `libcudart`, `libcublas` and `libcublasLt` next to the CUDA backend and
  hash them with the rest, so a machine with only the NVIDIA driver can run
  the result. `--no-cuda-redist` skips that.

### Changed

- **Build manifests are schema 2 and relocatable.** `executable` and
  `runtime_dir` are now relative to the manifest's own directory and resolved
  against it; an absolute path, or a relative one escaping the bundle, is
  refused. `base_build`, which recorded the builder's own path, is gone;
  `base_build_json_sha256` keeps its provenance. Schema 1 manifests are still
  read with their absolute paths, so a pre-0.2.0 install keeps starting, but
  it cannot be packed or moved. Every hash check is unchanged.
- **The worker takes `--runtime-dir`** instead of a compile-time backend
  directory, and the backend passes the directory it just hashed. The worker
  links with an `$ORIGIN/../runtime` run path and no absolute path, so its
  sha256 no longer depends on where it was built. `unridden-so1-probe` takes
  the same argument, and `bench_competitor_so1.py` gained `--runtime-dir`.
- `unridden.api.native.build` copies the base runtime into the bundle and
  re-hashes the copy, builds in a scratch directory it removes on success, and
  leaves `build/` holding only the executable.
- `build_base_runtime.py` configures llama.cpp with `LLAMA_OPENSSL=OFF`. The
  worker never downloads anything, and without it `libllama-common` no longer
  links the build machine's libssl, which a published bundle would otherwise
  require on the user's system. The only system library a bundle needs is
  OpenMP's `libgomp.so.1`. It also passes `GGML_CUDA_NCCL=OFF`: the NVIDIA
  devel containers ship NCCL, and without the flag ggml links `libnccl.so.2`
  into the CUDA backend of a published bundle.
- A worker that dies before its handshake now reports its last stderr lines
  in the `native backend failed to start` error, so a missing shared library
  or a rejected argument is named instead of hidden behind "closed stdout".

## 0.1.0 - 2026-09-22

First tagged release. Every measurement in `docs/results` was taken on
llama.cpp release `v0.4.1` (commit `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`)
with the Unsloth Gemma 4 26B-A4B UD-Q4_K_XL GGUF on one RTX 5090, unless the
page says otherwise.

### Added

- `unridden` Python package: a local, non-generative decision API. One owned
  llama.cpp child process prefills a compiled prompt and reads final-position
  logits over single-token labels, so a request produces zero generated
  tokens.
- Three caller-defined question types, batched in one request: `choice`
  (pick one of a caller-supplied label set), `score` (a bounded numeric
  rating), and `noul` (true or false against caller criteria).
- HTTP surface: `POST /v1/decisions`, `GET /v1/models`, and `GET /health`,
  served by a FastAPI application whose lifespan owns the worker child.
  Importing the package or building the app starts no process.
  `POST /v1/systemone` is a compatibility alias for `POST /v1/decisions`,
  offered as an interoperability path for clients written against that
  request shape.
- `unridden.api.cli` with `run` (evaluate a JSON or JSONL file of requests
  into JSONL, no port needed) and `serve` (run the ASGI app on a loopback
  listener).
- SemIf row import and export helpers in `unridden.api.semif`.
- Native worker under `unridden/api/native`: a C++ worker built against a
  user-provided llama.cpp checkout, plus `unridden.api.native.build`, which
  verifies the base runtime before compiling and records source, executable,
  runtime, and llama.cpp revision hashes in a build manifest. Startup pins the
  worker executable and the runtime files and bundle; the model file's SHA-256
  is always computed and is enforced when `--model-sha256` is configured.
- Shared-prefix reuse within a request. The compiler reports the byte length
  of the prefix the questions of a request have in common, and the worker
  keeps that prefix's KV cells across the questions instead of re-prefilling
  them, with an exact 0.0 isolation check in the validation harness
  ([ADR 0002](docs/decisions/0002-share-state-prefix-within-a-request.md),
  [results](docs/results/prefix-reuse.md)).
- Opt-in batched question evaluation. `ApiConfig.batched` (CLI, runner and
  worker `--batched`) gives every question of a request its own KV sequence over
  one copy of the shared prefix and decodes all the remainders together.
  `ApiConfig.batched_context` (worker `--batched-context`, default 8192 cells)
  sizes the unified KV cache and may not exceed `max_questions * context_size`.
  The default stays sequential and reproduces the recorded v0.4.1 suite run bit
  for bit
  ([ADR 0004](docs/decisions/0004-optional-batched-question-evaluation.md),
  [results](docs/results/batched-mode.md)).
- A batched worker names the regime that answered each request in the public
  response body as `evaluation: {mode, fallback}`, and a request that needs more
  KV cells than `batched_context` is answered by the sequential path with
  `fallback: "context"` rather than being truncated or refused. Diagnostics gain
  `evaluation_mode` and `batch_sequences` per question. `GET /v1/models` reports
  `runtime.batched` and `runtime.batched_context`.
- A tested llama.cpp default instead of a hard pin. The build records release
  `v0.4.1` as `TESTED_LLAMA_TAG` and `TESTED_LLAMA_REVISION`,
  `build_base_runtime.py` fetches that tag by default and verifies that it
  still resolves to the recorded commit, `--revision <tag-or-sha>` builds any
  other revision, and an untested revision logs one warning at startup instead
  of being refused. `GET /v1/models` reports `runtime.llama_revision` and
  `runtime.tested_revision`
  ([ADR 0003](docs/decisions/0003-tested-default-revision-instead-of-a-hard-pin.md),
  [re-validation results](docs/results/llama-v0.4.1-revalidation.md)).
- `build_base_runtime.py --cuda-architectures`, passed through as
  `CMAKE_CUDA_ARCHITECTURES`. Omitted, llama.cpp's own default applies.
- Validation and use-case scripts under `scripts/unridden`, with the
  matching request and expectation fixtures under `unridden/examples`:
  seven hand-written use-case suites (241 cases, 1,286 questions) with frozen
  gold labels, a validation harness with sibling-independence, repeat-identity
  and cross-question contamination probes, and `compare_observations.py` and
  `compare_fallback.py`, which reduce two recorded runs to answer changes,
  probability and raw logit moves, tokens and latency.
- An open-model comparison. `docs/research/open-model-survey.md` reviews
  eight open decision-readout projects for ideas worth adopting, and
  `docs/results/open-model-comparison.md` runs three of them on the seven
  suites with the same scorer: Kev-9B and Laya as shipped, and so1
  (open-alternative-jev) on unridden's own GGUF and llama.cpp build through a
  backend written for the purpose (`unridden/api/native/so1_probe.cpp`), in
  both its separate and packed modes. The drivers
  (`bench_competitor_kev.py`, `bench_competitor_laya.py`,
  `bench_competitor_so1.py`, `compare_competitor_outcomes.py`) and a
  page-cache eviction helper (`evict_file_cache.py`) ship under
  `scripts/unridden`.
- A whitepaper (`docs/whitepaper.md`) covering the design, the measured
  results, and a generative baseline measured with llama-bench on the
  identical GGUF.
- Architecture decision records under `docs/decisions` and measured results
  under `docs/results`.
- Community and governance files: Apache-2.0 license, NOTICE, contribution
  guide with DCO sign-off, Code of Conduct, security policy, issue and pull
  request templates, and a CPU-only CI workflow on Python 3.12 and 3.13 with a
  native helper build test.

### Notes

- Model weights are not distributed with this repository. Users download a
  Gemma GGUF themselves under its own terms. The worker accepts the `gemma4`
  architecture only.
- llama.cpp is not vendored. Users build it themselves, by default at the
  tested release.
- Reported confidences are the raw softmax over label logits and are not
  calibrated: 94.7% of the 1,286 suite answers sit above 0.99, including most
  of the wrong ones. Do not gate a decision on the probability without fitting
  a temperature first.
- Answers are deterministic per configuration only. About 1% of borderline
  answers move with batch size, prefix reuse, or llama.cpp revision.
- The request and response shape follows TypeSafe's publicly documented Jev
  interface. This project is independent and is not affiliated with or
  endorsed by TypeSafe, and no measurement of the hosted service appears in
  this repository.
