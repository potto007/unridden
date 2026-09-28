# Gemma 4 E4B versus 26B-A4B on Unridden use cases

Measured 2026-09-27 on the same RTX 5090. Both models were the Unsloth
UD-Q4_K_XL GGUF, run through the same current Unridden v1 worker and the
same seven frozen authored suites.

## Result

| Measure | 26B-A4B | E4B | E4B change |
| --- | ---: | ---: | ---: |
| Correct questions | 1,193/1,286 (92.8%) | 1,130/1,286 (87.9%) | −4.9 percentage points |
| Cases with every question correct | 185/241 (76.8%) | 157/241 (65.1%) | −11.6 points |
| Request latency, median | 201.7 ms | 141.7 ms | 29.8% lower |
| Request latency, p95 | 567.1 ms | 405.4 ms | 28.5% lower |
| Model file | 17.01 GB | 5.13 GB | 69.9% smaller |
| Approximate added GPU memory at peak | 17.6 GiB | 4.1 GiB | 76.7% less |

E4B is a substantially smaller and faster way to try the API, but it loses
task quality on these authored cases. The largest gaps are candidate/date
extraction and passage search. The tool-selection suite ties at 119/122, but
is already near its ceiling. The Tetris and snapshot checks in
[full-depth-snapshots.md](full-depth-snapshots.md) establish those specific
capabilities; they do not substitute for these task-quality results.

## Matched method and integrity

The runner was scripts/unridden/run_usecase_suites.py with its built-in
scorer. Each model received the same 241 requests, the same question order,
and the same gold labels. All seven suite file SHA-256 values match across
runs and still match the files on disk. Every paired request body and target
was independently compared from the recorded observations and is identical.
Both runs completed 1,286 scored questions, zero errored cases, and zero
generated tokens.

Both used llama.cpp v0.4.1 through the same worker and runtime bundle,
context 2048, batch and ubatch 256, eight threads, sequential question
evaluation, shared-prefix reuse, diagnostics enabled, and the same GPU. The
worker SHA-256 was
e5e44254d865f6096e88c48881566084ded8019b429e294dde8277a9cc83d152;
its build manifest SHA-256 was
762a2828c430f864048b2f8727c2d93bad1f57db91ffac1f890b7ea636b3f8fa.
The 26B model SHA-256 was
ef728c8e0c337fd1067b947af006e38a9ef2419e56feced4fd29b4bf0636e30c;
E4B was
3cf61de12daa015ee0f7b68e7b7c541405bf220e1e942bad8b47cab827d7df80.
The two models use the same quantization family, but differ in architecture
and capacity.

The paired raw records remain in the local, ignored
`outputs/gemma4-e4b-vs-26b-20260927T110243-1094247/` directory and are not
distributed with this source tree. That directory also contains the run
manifests, summaries, GPU samples, preflight, and restoration record. Output
directories were new; no older result was overwritten.

## Quality by suite

| Suite | Questions | 26B correct | E4B correct | E4B gap |
| --- | ---: | ---: | ---: | ---: |
| Candidate span and date extraction | 194 | 179 (92.3%) | 155 (79.9%) | −12.4 points |
| Claim support verification | 96 | 91 (94.8%) | 89 (92.7%) | −2.1 points |
| Entity alignment and record matching | 157 | 157 (100%) | 154 (98.1%) | −1.9 points |
| Guardrail policy gates | 417 | 361 (86.6%) | 344 (82.5%) | −4.1 points |
| Hierarchical classification | 151 | 139 (92.1%) | 136 (90.1%) | −2.0 points |
| Passage rerank and line search | 149 | 147 (98.7%) | 133 (89.3%) | −9.4 points |
| Tool and argument selection | 122 | 119 (97.5%) | 119 (97.5%) | 0 |
| **All seven** | **1,286** | **1,193 (92.8%)** | **1,130 (87.9%)** | **−4.9 points** |

By primitive, Choice was 373/400 versus 342/400, Noul 741/782 versus
716/782, and Score exact level 79/104 versus 72/104 (26B first in each
pair). Score within the suite's tolerance was 95/104 versus 90/104.
The existing ranking measure was 32/32 top-1 for 26B versus 28/32 for
E4B; most rankings specify only a top-1, so pairwise ranking agreement
overstates its independent signal. Accuracy by authored difficulty was
96.6% versus 91.6% on easy, 93.0% versus 88.6% on medium, and 90.1%
versus 84.6% on hard.

Paired question outcomes: both correct 1,095; only 26B correct 98; only
E4B correct 35; neither correct 58. At the case level, both completely
correct 147; only 26B 38; only E4B 10; neither 46. Thus the gap is not
only a handful of questions repeated within one case.

A 20,000-resample paired bootstrap, sampling cases within each suite,
put the 26B-minus-E4B question-accuracy gap at 3.1 to 6.7 percentage
points (95% descriptive interval). For the fraction of cases completely
correct, the gap was 6.2 to 17.0 points. Cases, not individual questions,
are the resampling unit. These intervals describe variation among the
authored cases; the suites are not a random sample of all tasks.

E4B's extra misses include interpreting an ambiguous numeric date as
absolute, selecting the wrong date components, and missing or overcalling
passage relevance. The guardrail suite has repeated related rubrics and
known noisy labels, so its aggregate score should not be read as one
independent safety capability. Entity matching and tool selection are
near ceiling and have little room to reveal a regression.

## Confidence and calibration

Noul Brier score was 0.050 for 26B and 0.074 for E4B (lower is better).
Among predictions with reported confidence at least 0.8, 26B was correct
1,183/1,269 (93.2%) and E4B 1,088/1,199 (90.7%). Their mean reported
confidence in the 0.8–1.0 bin was 99.76% and 98.60%, respectively. Both
are overconfident. E4B had 111 errors in that high-confidence bin versus
86 for 26B. The API's max-label probability is not a calibrated estimate
of correctness, so neither model should use it as an abstention threshold
without task-specific calibration.

## Cost and speed boundary

The GGUF files are 17,010,980,576 bytes (26B) and 5,126,306,944 bytes
(E4B). These sizes establish a roughly 70% smaller model artifact and
download payload for E4B, not an observed network download time.
One startup, including validation and load, took 12.3 seconds for 26B
and 4.2 seconds for E4B; repeatability was not measured.

Latency is wall time around the same ApiService.evaluate call for each
request, with diagnostics on. The runs were sequential in 26B-then-E4B
order, with one run per model. Total measured request time was 60.1 versus
42.0 seconds. E4B processed 241,837 input tokens versus 246,981 for
26B, a 2.1% lower count, so the latency ratio combines model work and
the models' different token counts. Background activity was not fully
isolated; repeat runs would be needed for a tighter performance estimate.

One-second GPU samples showed a minimum whole-card use of 8,415 MiB
with the benchmark model unloaded, and peaks of 26,018 MiB for 26B and
12,516 MiB for E4B. Subtracting that observed baseline gives approximate
model-associated peaks of 17,603 MiB and 4,101 MiB. Other GPU processes
were present throughout, so these are whole-card deltas rather than
per-process allocation measurements.

The previously active 26B snapshot service was paused for the benchmark
and verified active and healthy afterward. This comparison exercises the
v1 decision worker; it does not retest v2 snapshot parity, Rider, or
long-running serving behavior. The older published 26B v0.4.1 suite run
scored 1,197/1,286 with a different worker build. The paired 26B result
here is the proper reference for the E4B run.

## Reproduction

The current model files and worker bundle must be present, and the GPU
must have enough headroom. These commands require a quiet GPU window;
the runner starts its own worker. Each output path must not already exist.

    PYTHONPATH=. .venv/bin/python scripts/unridden/run_usecase_suites.py --suites unridden/examples/usecases --out outputs/compare-26b-new --worker build/api-worker/build/unridden-worker --manifest build/api-worker/build.json --model-path /home/potto/src/local-ai/models/unsloth-gemma-4-26b-a4b-gguf/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf --batch-size 256 --allow-gpu

    PYTHONPATH=. .venv/bin/python scripts/unridden/run_usecase_suites.py --suites unridden/examples/usecases --out outputs/compare-e4b-new --worker build/api-worker/build/unridden-worker --manifest build/api-worker/build.json --model-path /home/potto/src/local-ai/models/unsloth-gemma-4-e4b-gguf/gemma-4-E4B-it-UD-Q4_K_XL.gguf --batch-size 256 --allow-gpu

Use the recorded suite and worker hashes to check that a future run still
matches this one before interpreting score changes.
