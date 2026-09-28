# Request turnaround by use-case suite

Measured in the paired Gemma 4 E4B versus 26B-A4B UD-Q4_K_XL run on
2026-09-27. This is the same run described in
[the quality comparison](gemma4-e4b-vs-26b-usecases.md), with no new inference.

## Fastest to slowest by E4B median

All times are milliseconds per complete API request. Each row groups the
authored cases in one suite. The question-count column is the median and
range of questions bundled into a request.

| Suite | Requests | Questions/request | E4B min | E4B median | E4B p95 | E4B max | 26B min | 26B median | 26B p95 | 26B max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Claim support verification | 36 | 2.5 (1–5) | 29.1 | 67.3 | 144.7 | 170.0 | 42.2 | 104.7 | 203.0 | 231.8 |
| Tool and argument selection | 32 | 4 (3–6) | 75.2 | 106.8 | 155.0 | 159.1 | 111.7 | 157.8 | 223.8 | 235.3 |
| Entity alignment and record matching | 32 | 5 (4–6) | 102.4 | 136.8 | 181.7 | 189.3 | 146.9 | 182.8 | 240.2 | 252.4 |
| Passage rerank and line search | 36 | 5 (2–6) | 94.8 | 137.3 | 180.2 | 200.3 | 119.4 | 199.2 | 250.8 | 263.9 |
| Hierarchical classification | 36 | 4 (3–5) | 121.4 | 164.7 | 216.1 | 220.2 | 176.4 | 238.7 | 300.7 | 339.6 |
| Candidate span and date extraction | 33 | 6 (3–9) | 90.5 | 167.8 | 335.4 | 405.5 | 132.8 | 239.4 | 505.2 | 574.8 |
| Guardrail policy gates | 36 | 11.5 (11–13) | 296.4 | 365.6 | 496.6 | 522.0 | 428.7 | 531.0 | 711.1 | 724.4 |
| **All suites** | **241** | **5 (1–13)** | **29.1** | **141.7** | **405.4** | **522.0** | **42.2** | **201.7** | **567.1** | **724.4** |

Bundle sizes differ substantially: claim-support requests have a median
of 2.5 questions, while guardrail requests have a median of 11.5. A request
can also contain more text per question. These figures should not be read as
the cost of one Choice, Noul, or Score question. Many requests mix those
primitives, so allocating their total latency to an individual primitive
would invent precision the observations do not have. The suite ordering is
descriptive, not a controlled estimate of what each task category alone costs.

## By authored difficulty

| Difficulty | Requests | E4B median | E4B p95 | 26B median | 26B p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Easy | 56 | 121.6 | 363.1 | 173.3 | 535.5 |
| Medium | 110 | 143.8 | 417.2 | 204.2 | 640.6 |
| Hard | 75 | 155.2 | 370.7 | 222.4 | 508.8 |

Medians rise with the authored difficulty, but these groups have different
suite and bundle-size mixes. In particular, medium has a higher p95 than
hard; this table does not isolate the cost of harder reasoning.

## Measurement boundary and source

Each observation's latency_ms was recorded with a monotonic timer immediately
before and after one ApiService.evaluate call with diagnostics enabled. Both
models received the same 241 request bodies in the same order, used the same
worker, batch and context settings, and had zero errored requests. The timer
covers request processing and all bundled questions; it excludes initial
model load and any network hop. The p95 uses linear interpolation at the
95th percentile, matching the suite runner. Minima and maxima are actual
individual requests, so they are especially sensitive to case length and
ordinary run-to-run noise. There was one run per model, 26B first and E4B
second.

The source is the [26B observations](../../outputs/gemma4-e4b-vs-26b-20260927T110243-1094247/26b/observations.jsonl)
and [E4B observations](../../outputs/gemma4-e4b-vs-26b-20260927T110243-1094247/e4b/observations.jsonl),
with a recorded request and latency on every row. Request question counts
come from the corresponding request objects, not from a token-based estimate.

## Why this differs from the 33.3 ms Tetris figure

The 33.3 ms number in [the Tetris snapshot report](full-depth-snapshots.md)
is the E4B average for one single-move, level-18, seed-1 game through the
/v2 snapshot path. That path computes the board prefix once per move and reuses it
across decisions within that move. The use-case figures above are /v1
requests covering varied states and often several separately evaluated
questions, with diagnostics on. Even the
corresponding Tetris /v1 row averaged 47.6 ms, so the 33.3 ms game number
is not a general E4B turnaround promise for complex requests.
