# Benchmark coverage across Unridden and Jev-like systems

Assessment date: **27 September 2026**. Source review and evaluation plan; no new inference was run for this assessment.

## Conclusion

The surveyed systems overlap strongly on **choosing among supplied answers and returning probabilities**. Their benchmarks overlap much less on the tasks, gold labels, uncertainty measures, and surrounding application work. A common Choice/Score/Noul interface does not make reported “intelligence” scores interchangeable.

Unridden has useful evidence for seven authored workflows, API isolation, and snapshot restoration. Its main gaps are independently maintained task sets, calibrated uncertainty, option-order robustness, wider input/option coverage, and controlled interactive comparisons. Existing evidence does **not** establish superiority over hosted Jev or the whole current field.

**Recommended sequence:** unchanged JevBench public evaluation plus a separately arranged sealed evaluation; a new blind uncertainty/robustness holdout; then a declared Decision Index panel. Keep runtime conformance, snapshot fidelity and gameplay as separate reports. None substitutes for the others.

## Scope and provenance

Reviewed the [core ledger](../../outputs/jev-survey-2026-09-27/worker_core.md), [local alternatives ledger](../../outputs/jev-survey-2026-09-27/worker_local.md), [discovery/census ledger](../../outputs/jev-survey-2026-09-27/worker_discovery.md), primary sources and internal reports cited below. These working ledgers contain version/hardware/license details and may not ship in a public checkout. Moving external links describe today's snapshot; future execution should pin commits and dataset hashes.

The [JevBench versioned artifact](https://github.com/fstandhartinger/jevbench/blob/main/results/v1.4.2.1/jevbench-v1.4.2.1-results.json) contains 94 configurations, 90 ranked. The survey groups 69 eligible non-generative configurations into 55 families, three provisional. These are not 55 independent benchmark programs. Checkpoints, wrappers and shared datasets create evidence overlap; generative thinking modes remain separate.

The completed E4B/26B comparison was read after its report became available. This assessment does not modify benchmark outputs, quickstart changes, services or existing reports. No evaluator submission or external publication occurred.

## 1. Benchmark overlap matrix

“Overlap” means related tasks or primitives, not equivalent scores. Project-authored evaluations remain project measurements even when they use public datasets.

| Evidence family | What is tested | Overlap with Unridden | Extra dimension or limit |
| --- | --- | --- | --- |
| Unridden seven suites | Supplied-context Choice/Noul/Score; original labels reviewed blind | Direct product-fit regression | Narrow exposed corpus; candidates/facts supplied; not independent general intelligence evidence |
| Unridden matched E4B/26B | Same authored requests and worker, different checkpoint capacity/architecture | Direct model-selection tradeoff | Quality, memory and request latency on one GPU; not hosted Jev or snapshot-route comparison. [Report](../results/gemma4-e4b-vs-26b-usecases.md) |
| TypeSafe workflows | Four vendor-authored workflows against model-consensus probability labels | Routing, checks and extraction-like decisions | Teacher agreement rather than human adjudication; vendor task selection; remote API economics. [Evaluation](https://evals.typesafe.ai/), [launch caveats](https://typesafe.ai/blog/introducing-system-one-models-and-jev) |
| LocalLLaMA typed-decisions | 400 cases, five questions each: support, agent traces, security, invoices | All three primitives, operational domains | Synthetic teacher targets; specialist training and zero-shot generalization differ. [Dataset](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) |
| JevBench | Public/private typed decisions; accuracy, calibration, speed, cost | Closest emerging shared comparison | Public-only result does not reproduce official score; composite is not pure intelligence. [Method](https://github.com/fstandhartinger/jevbench/blob/main/docs/METHOD-v1.4.md) |
| Decision Index | Public tasks reformulated as Choice/Noul: tools, retrieval/classification, reasoning, language, taste | Static tool choice, entailment, relevance | Broader domains and menus; no Score; static tool decisions are not execution. Edition 0.2.1 includes 38 benchmarks. [Source](https://github.com/apolinario/decision-index) |
| Kev transfer-v4 | New-source records/held-out policies, probability quality and selective prediction | Policy and routing decisions | Unknown-answer confidence, permutations and risk/coverage; fp32 evaluation differs from bf16 serving; Jev compared on development. [Kev](https://github.com/jaredpalmer/kev), [27B card](https://huggingface.co/jaredpalmer/kev-27b) |
| SemIf/Rizzo | Authored decisions, perturbations, missing evidence, external entailment, calibration | Claim support, isolation and reuse | Wrapper/irrelevant-context/option-reversal sensitivity; reused fixtures reduce independence; missing-evidence behavior can worsen despite aggregate gain. [SemIf results](https://github.com/TheoLeeCJ/SemIf-OpenJev/blob/master/docs/RESULTS.md), [Rizzo](https://github.com/Rizzo-AI-Academy/rizzo-flow) |
| PoorJev | Small support-decision set, Banking77, temperature calibration | Support questions and probabilities | Wide menus and domain-shift calibration; author-run crossbench is not third-party replication. [Results](https://github.com/rupeshpoojary9/poorjev/blob/main/RESULTS.md), [crossbench](https://github.com/rupeshpoojary9/poorjev/tree/main/crossbench) |
| JevMLX/Nimble | Public workflow sample/parity diagnostics; held-out reference decisions/contrastive fact changes | Typed choices and readout stability | Public-sample versus private Jev scores cannot rank models; contrastive pairs and numerical parity are useful additions. [JevMLX](https://github.com/bnsd55/jevmlx/blob/main/BENCHMARKING.md), [Nimble](https://github.com/bespokelabsai/nimble) |
| CLM/decider applications | Cached action scoring, games, candidate verification, sometimes task-specific tuning | Tetris and state reuse | Controller, shortlist, cache and training contribute; selecting generated coding candidates is not independently solving the coding task. [CLM](https://github.com/Contrastive-LM/CLM), [decider demos](https://github.com/Mapika/decider/blob/main/docs/DEMOS.md) |
| API/snapshot qualifications | Errors, schemas, isolation, restoration, resource behavior | Runtime guarantees | Engineering behavior, not semantic intelligence; external compatibility is a different contract. [API](../results/api-validation.md), [snapshots](../results/full-depth-snapshots.md), [jevcompat](https://github.com/mandu5/jevcompat) |

so1, Laya and Rizzo reuse typed-decisions or quote an existing Jev row. Repeated citations do not create repeated Jev measurements. Laya's benchmark-specific checkpoint and base checkpoint behave very differently; so1 warns that its roughly one-point advantage over cited Jev is within noise. [Laya methodology](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md), [so1 results/corrections](https://github.com/ikermoel/open-alternative-jev/blob/main/benchmarks/docs/RESULTS.md). Similarly, many JevBench entrants share one evaluator's methodology. No consensus standard was established by this review.

## 2. Internal task coverage

The [specification](../usecase-suites.md), [original results](../results/usecase-suites.md), and [revalidation](../results/llama-v0.4.1-revalidation.md) describe 241 cases/1,286 questions. Separate authors designed families and another agent reviewed labels blind. Two post-result corrections are documented: later reruns are regression evidence on an exposed corpus.

| Suite | Actual task | External overlap | Boundary / useful extension |
| --- | --- | --- | --- |
| Hierarchical classification, 151 questions | Labels/depth with supplied hierarchy and parent context | Classification, taxonomy routing | Gold parent paths do not test executed cascades; add wrong-parent recovery |
| Candidate spans/dates, 194 | Choose spans/none and components | Invoice and structured-field decisions | Not open extraction/date arithmetic; add ambiguous locales, invalid/relative dates |
| Passage/line search, 149 | Relevance and choice from supplied passages | Retrieval and judgment benchmarks | Retrieval outside test; most gold rankings only identify top result; add graded relevance and missing answers |
| Claim support, 96 | Support/contradiction/silence from evidence | Entailment and hallucination checks | No fact retrieval; add conflicting documents and insufficient evidence |
| Tool/arguments, 122 | Closed tool and argument choices | Static BFCL/API-Bank-style decisions | No execution/recovery; add feedback and failed tools |
| Policy gates, 417 | Hazards, severity and route | Security/policy/adversarial tasks | Repeated rubrics dominate aggregate; explicit policy and balanced blind cases needed; not safety certification |
| Entity alignment, 157 | Different/review/same and field comparisons | Pairwise matching/classification | Near saturation; add conflicting fields, missing data and hard near-duplicates |

Observed suite maxima are 17 Choice options, five Score levels, 13 questions/request and about 705 prompt tokens. Quality near the documented 26-option/10-level/32-question/2,048-token limits remains weakly covered. Schema rejection at a boundary does not establish retained quality near it.

### Scoring conventions

The historical v0.4.1 revalidation reports 1,197/1,286 correct (93.1%). Its aggregate uses **exact modal Score-level accuracy** (79/104), while tolerance success (95/104) is separate. Comparison-page prose describing tolerance should not redefine the aggregate. Expected-score MAE is also different from modal-level error. See the [runner](../../scripts/unridden/run_usecase_suites.py) and summaries linked from the result pages.

The per-question-id majority baseline is about 69.4%; the weaker per-primitive baseline is about 42.8%. Report both by name. Add macro averages across families and complete-case success, because repeated questions are correlated. Gold-position buckets are not option-order invariance: that requires permuting the same question's options and mapping predictions back to identities.

### New matched architecture/capacity result

The [completed 27 September run](../results/gemma4-e4b-vs-26b-usecases.md) used the same current worker, quantization family, RTX 5090, and paired requests. 26B achieved 1,193/1,286 (92.8%) versus E4B 1,130/1,286 (87.9%). Complete-case success was 76.8% versus 65.1%. E4B median request latency was 141.7 ms versus 201.7 ms, with a roughly 70% smaller model file. Both completed without errors or generated tokens.

This supports a concrete size/speed/quality tradeoff, not “small model equally intelligent.” It is a stronger matched timing comparison than the older cross-project experiment. The case-resampled accuracy-gap interval is 3.1–6.7 percentage points, descriptive of these authored cases. Both models are overconfident; max-label probability is not an established abstention policy. Do not mix this current 26B result with the historical 93.1% reference as though they were the same run.

## 3. What the older competitor comparison supports

The [21 September experiment](../results/open-model-comparison.md) supports higher aggregate accuracy for its Unridden configuration than tested older Kev-9B, Laya and custom so1 configurations **on those suites**. Hosted Jev was not measured; current Kev-27B was not tested.

- so1's custom llama.cpp backend shares Unridden's model/runtime. Useful for readout design, it is not shipped Transformers/vLLM performance.
- Kev's two correct returned choices become ties in rounded probability vectors. Returned-answer accuracy is about 90.0%, versus 89.8% with a first-option tie-break. Keep both conventions visible.
- There is no winner on every probability metric. Kev performs better on some calibration/probability measures, and so1 separate on correctness discrimination.
- Timing mixes localhost HTTP, in-process calls and CPU-only Laya, with a noncontemporaneous Unridden reference. Kev lacked optimized kernels. It is not a general speed ranking.

The [batching experiment](../results/batched-mode.md) similarly separates guarantees: faster requests, nine changed answers and a small accuracy difference on one run. Repeat determinism, numerical shape effects, sibling leakage and task quality are different questions.

## 4. Dimensions beyond ordinary accuracy

| Dimension | Present evidence | Missing or insufficiently established |
| --- | --- | --- |
| Correctness | Authored tasks; local paired comparisons | Independent broad holdout; counting, dates, long policies, multihop reasoning |
| Uncertainty | Brier/ECE and correctness discrimination | Disjoint calibration; domain shift; unknown answers; risk/coverage |
| Robustness | Sibling reversal, ID changes, repeat checks | Option permutations, paraphrases, distractor position and wrapper changes |
| Capacity | Budget/schema validation | Quality near limits, wide menus, truncation-free external coverage |
| Interactive action | Tetris trajectories | Model/controller ablations, tool feedback, recovery and delayed reward |
| State reuse | Capture, restore, branch isolation | Broad fresh-run equivalence, realistic cache churn and amortized benefit |
| Production | Ownership, rejection, teardown/recovery | Sustained mixed load, tail latency and cost per successful task |
| Languages/modalities | Reviewed corpus is text | Multilingual/image accuracy; accepted input is not measured understanding |

TypeSafe's own [limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13) identify counting, dates, indirect reasoning and distracting context as weaknesses. These are test categories, not evidence that Unridden solves them better. Larger advertised context/option capacities are coverage differences until matched quality is measured.

### Snapshot restoration versus decision equivalence

[Split qualification](../results/state-snapshots-qualification.md) passes several lifecycle gates but fails GPU stock-versus-split residual equivalence; it remains experimental. The [full-depth profile](../results/full-depth-snapshots.md) establishes bit-exact restoration on its tested corpus. It also reports probability differences between branching and fresh uninterrupted execution caused by numerical shape. Restore identity does not imply universal decision equivalence across execution modes.

Branch latency must accompany creation, save/load, memory, lifetime and reuse count. The qualification report records these costs separately. A fast resident branch excludes setup; durable restoration has additional costs. No reviewed external benchmark provided a directly matched snapshot restoration corpus. That is an evidence gap, not proof of unique capability.

### Tetris needs controller controls

The [full-depth game table](../results/full-depth-snapshots.md) shows lower snapshot request times in its paired rows and mixed gameplay changes. E4B lookahead reached the piece cap in five of six settings on each route. Request latency excludes surrounding search/preparation unless explicitly timed. The report does not isolate the model's contribution from pruning, lookahead, encoding and control rules.

Decider's [demo](https://github.com/Mapika/decider/blob/main/docs/DEMOS.md) shortlists eight moves with a board heuristic. Its models beat random shortlist choice but trail the heuristic baseline in that test. This is a useful ablation pattern, not a directly comparable score: Unridden uses different real-time/controller settings. Selected video is weaker evidence than all seeded outcomes.

CLM's [applications](https://github.com/Contrastive-LM/CLM) include games and verification of separately generated coding candidates. Candidate quality, task-specific fitting and cached representations contribute; the result cannot be imported as autonomous coding or general typed-decision ability.

## 5. Prioritized additions

These are proposed work packages, not tests executed here.

### P0: measurement contract and preflight

Freeze model/tokenizer/adapter hashes, calibration, precision, runtime, prompt/schema, hardware, suite/scorer versions, and training/development exposure. Reconcile exact/tolerant Score, modal/expected score, returned/rounded choice and baseline definitions without silently rewriting historical results.

Inventory option counts, levels, tokens and question counts before external inference. Publish unsupported counts; do not silently truncate, discard difficult items or shortlist. Keep quality, uncertainty/robustness, runtime behavior and economics in separate report sections even when a benchmark also requests a composite.

### P1: JevBench public, followed by evaluator-run sealed assessment

Run the **231 public decisions** unchanged through the ordinary isolated endpoint. Preserve requests/responses and denominators; score invalid/missing answers as failures. Report accuracy, probability metrics and measured local latency. This is a public-slice result, not an official rank.

Official scoring also depends on earlier private evaluation and **308 sealed decisions**. Its harmonic composite includes calibration, speed and cost, with penalties and modeled self-hosted economics. Aggregates cannot reproduce private item-level evaluation; an evaluator-run submission is needed for an official score. No submission/outreach was performed here. [Method](https://github.com/fstandhartinger/jevbench/blob/main/docs/METHOD-v1.4.md), [timing/cost disclosures](https://github.com/fstandhartinger/jevbench).

### P1: blind uncertainty and robustness holdout

Before inspecting outcomes, independently label new missing/conflicting-evidence, none-answer, close-distractor, indirect-reference, numeric/date and nested-policy cases. Include one-fact changes that should flip answers and paraphrases that should preserve them.

Cross option permutations, evidence position, distractors and paraphrases. Exercise sibling insertion/removal and all sibling pairs, including adversarial equal-length siblings. Distinguish option order, sibling order, batching noise and information leakage.

Report accuracy, useful macro-F1, per-primitive Brier, reliability/ECE with uncertainty, and abstention risk/coverage. Fit temperatures on separate calibration cases and evaluate transfer to unseen domains. Calibration must accompany discrimination and useful coverage: base-rate prediction can appear calibrated while solving little.

### P2: declared Decision Index panel

Predeclare components spanning tool arguments, entailment, relevance and harder reasoning. Publish components and coverage; a pilot is not the full index. Dataset access/redistribution conditions apply. Menus exceed Unridden's limit, including 32-candidate retrieval; unsupported inputs must remain visible. Local timing is not the maintainer's standardized board measurement. [Decision Index](https://github.com/apolinario/decision-index).

If wider choices are a product goal, compare an expanded readout against a separately labelled multi-stage shortlist pipeline, counting all calls, latency and accuracy losses. A new pipeline is not a transparent adapter.

### P2: interactive/controller and state-reuse experiments

Publish the game harness: seeds, rules, horizon, candidate generator, pruning and stopping criteria. Compare random legal choice, random shortlist choice, heuristic-only, model single-step, and lookahead with/without the model; hold candidate pools fixed. Report every episode, capped outcomes, lines/survival, full move wall time, calls per move and deadline misses.

Separately sweep state length, branches/prefix, branch depth, resident/disk restoration, cache misses and reloads. Measure setup plus all branches, memory and quality versus a declared reference, and the observed reuse break-even. Preserve exact restore gates and test independent-session interleaving and malformed/rejected snapshots. These establish useful reuse without assuming improved underlying intelligence.

### P2: compatibility and service behavior

Run applicable [jevcompat](https://github.com/mandu5/jevcompat) checks alongside internal tests. This unofficial contract is not an intelligence test. Declare intentional differences such as the 26-option backend limit rather than claiming full compatibility. Extend mixed-load, rejection/queue, restart/recovery and tail-latency tests.

Separate model time, local API, remote round-trip and full application time. Report cold/warm and hit/miss paths. State rental/amortization, utilization and energy assumptions and cost per successful task. Local marginal fees, vendor tariffs and modeled hosted-equivalent prices are different quantities.

### P3: conditional scope expansion

If needed by the product, add multilingual/image tasks and larger contexts/options. Measure quality as dimensions grow, not acceptance alone. Verify code, weights, base model and dataset terms independently before distributing reproduction bundles: permissive serving code does not remove research/noncommercial restrictions on other components.

## Public claim boundaries

- Say “on this frozen suite and configuration,” with versions and dates, rather than “more intelligent.”
- Separate local measurements, publisher claims and imported third-party rows. Ratios of unlike published latencies do not establish hardware-normalized speed.
- Show domains, primitives, errors and unsupported cases before aggregates. Resample paired cases/families rather than correlated questions as independent draws.
- Disclose calibration fitting and probability representation; rounding/temperature/ordinal conventions can change metrics without changing choices.
- Keep conformance, isolation, restoration, fresh-run equivalence and task success as distinct conclusions.

**Decision:** prioritize external unchanged evaluation and a blind uncertainty/robustness holdout. These close more consequential evidence gaps than another exposed-suite rerun alone. Retain the seven suites as regression tests and publish runtime/state-reuse evidence alongside with its own boundaries.
