# Optional E4B-to-26B escalation: assessment

Date: September 27, 2026. **Assessment only:** the gateway does not switch models,
load an additional model, or infer correctness from label confidence.

## Recommendation

Investigate an opt-in route from an E4B primary to an already-warm 26B backend
for quality-sensitive workloads. Keep one application API, but make policy,
actual model identity and added latency observable. Do not introduce hidden
confidence-only switching or model unload/reload inside an ordinary request.

The larger model has useful complementary answers, but is not a reliable oracle.
The present evidence justifies calibration and held-out routing experiments,
not a production uncertainty threshold.

Tracking issue: [Investigate calibrated E4B-to-26B quality escalation in the
gateway](https://github.com/potto007/unridden/issues/37).

## What the existing paired evidence says

The [same-worker comparison](../results/gemma4-e4b-vs-26b-usecases.md) evaluated
1,286 questions in 241 authored cases, seven suites, on the same RTX 5090.
E4B got 1,130 correct; 26B got 1,193 correct. Both used ordinary `/v1`, not
snapshot `/v2`.

| Paired outcome | Questions |
| --- | ---: |
| Both correct | 1,095 |
| Only 26B correct | 98 |
| Only E4B correct | 35 |
| Neither correct | 58 |

The larger model rescued 98 of E4B's 156 misses (62.8%), but reversed 35
E4B-correct answers. An impossible oracle that knew which E4B answers were wrong
would reach 1,228/1,286 (95.5%); that is an upper bound on this dataset, not an
implementable confidence policy.

111 of E4B's 156 errors (71.2%) had reported confidence at least 0.8. Its mean
reported confidence in that bin was 98.6%, versus 90.7% actual accuracy. Raw
maximum-label probability therefore misses many confidently wrong answers.

I also paired the recorded question outcomes by suite/case/question and applied
illustrative thresholds to **existing** predictions:

| Gate, strictly below threshold | Escalated questions | Rescued E4B misses | Reversed correct E4B answers | Final correct |
| --- | ---: | ---: | ---: | ---: |
| E4B confidence < 0.8 | 87 | 25 | 13 | 1,142 (88.8%) |
| E4B confidence < 0.9 | 149 | 49 | 19 | 1,160 (90.2%) |
| E4B confidence < 0.95 | 204 | 61 | 20 | 1,171 (91.1%) |
| E4B confidence < 0.99 | 318 | 80 | 27 | 1,183 (92.0%) |

These are retrospective descriptions, not calibrated or held-out policy results.
They assume selected questions retain their recorded answer when rerun alone;
that assumption needs native verification. Picking the best threshold on these
same cases would overfit.

Escalating the **entire bundle** if any E4B question is below 0.8 selects 68 of
241 requests and 519 questions: 61 rescues, 29 reversals, final 1,162 correct
(90.4%). That is a materially different cost/quality policy. Per-question versus
whole-bundle routing should be evaluated explicitly, not hidden in implementation.

Reproduce from the paired observations linked in the source report: join rows
by `(suite, case)`, outcomes by `qid`; select on E4B `confidence`; count
`not small.correct and large.correct` and
`small.correct and not large.correct`. The final count is 1,130 plus rescues
minus reversals. No model inference or service transition was required.

## Memory and latency

Google distinguishes active parameters from total model parameters: MoE's
roughly four-billion active parameters do not make its full weights a four-billion
model allocation. [Official model card](https://ai.google.dev/gemma/docs/core/model_card_4).

Local measured model files are 5.13 GB (E4B) and 17.01 GB (26B). The paired run
observed whole-card baseline 8,415 MiB, E4B peak 12,516 MiB and 26B peak 26,018 MiB.
Adding the two peak increments to that baseline suggests about 30,119 MiB on
a 32,607 MiB card, leaving only about 2.4 GiB. This arithmetic is **not a
co-residency qualification**: runtime graphs, KV state, snapshots, workload
peaks, fragmentation and other GPU users must be measured together. Lower-VRAM
systems should expect a second device or remote backend, not assumed local fit.

The current model units are mutually exclusive and share a port. Co-residency
requires separate backend instances/configuration; it is not enabled by adding
a gateway route. No second model was loaded for this assessment.

The ordinary paired run measured request medians of 141.7 ms for E4B and
201.7 ms for 26B, excluding HTTP and model load. Those medians cannot be added to
predict an escalated-request median. A routed request pays primary work, policy
work, any secondary capture and secondary inference, plus queue/network time.
Observed startup was 4.2 seconds for E4B and 12.3 seconds for 26B in one run,
not repeated qualification. Unloading/reloading on uncertainty could add seconds,
evict all active snapshots and block unrelated callers. Prefer an already-warm
secondary or report its unavailability.

## Snapshot and semantic requirements

E4B's full profile reads at checkpoint 42; the tested 26B split profile reads
at 30. Their model weights, activations and cache layouts differ. Never transfer
a native parent across models. Capture the same original stable state on the
secondary **before** the variable question, and do not append the primary answer.

Maintain separate per-client parents keyed by exact model/runtime/prompt/profile
identity and backend route. The current gateway advertises a generic model alias,
while backend discovery lacks exact weights/runtime hashes. Extend identity and
per-answer model provenance before transparent multi-backend routing. Manual
contexts must remain pinned; switching their model changes their contract.

Snapshot and ordinary paths can differ numerically because of chunking. Existing
ordinary rescues and thresholds must be remeasured on the actual snapshot route,
with the same prompt/question semantics. Cross-model answer confidence values
must not be compared as if calibrated on one shared scale.

## Useful next experiment and acceptance criteria

1. Freeze a target workload and independent held-out cases/families. Compare
   E4B-only, 26B-only and routed policies with a fixed escalation budget.
2. Gather paired snapshot answers and labels. Calibrate correctness or expected
   *rescue benefit*, using task type, score margin/entropy and application
   validity constraints where justified. Check calibration by task, not only
   an aggregate average. Disagreement alone is not proof either answer is right.
3. Measure rescue **and reversal**, calibration, fallback rate, exact payload
   parity, end-to-end median/p95, cold secondary capture, warm reuse, queueing and
   secondary unavailability. Compare against random selection and the descriptive
   oracle bound at matched budgets.
4. Qualify co-resident peak memory with both workers and snapshot budgets, or use
   separate/remote hardware. Measure startup separately; no automatic hot swap.
5. Add explicit policy opt-in and a latency/budget ceiling. Define unavailable
   behavior visibly: return the primary result with routing diagnostics or
   return an explicit policy failure, according to caller policy. Do not invent
   a hidden answer-selection rule.
6. Require a held-out quality improvement that meets a chosen reversal and
   latency budget before enabling automatic escalation. Keep manual override,
   backend isolation, provenance and rollback tests.

This future feature belongs in reusable gateway policy once its evidence and
contract exist. Application-specific validity checks and loss/quality budgets
still belong to the application or its explicit configuration.
