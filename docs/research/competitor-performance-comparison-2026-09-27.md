# Unridden versus Jev-like systems: what the collected results say

Evidence assembled 27 September 2026 from the [survey](jev-landscape-2026-09-27.md), its [core](../../outputs/jev-survey-2026-09-27/worker_core.md), [local-project](../../outputs/jev-survey-2026-09-27/worker_local.md), and [discovery](../../outputs/jev-survey-2026-09-27/worker_discovery.md) ledgers, and Unridden's linked reports. These are **three different comparison levels**, not one leaderboard. Each row below keeps its measurement boundary. The [94-row census](../../outputs/jev-survey-2026-09-27/jevbench_census.csv) preserves the complete board, including configurations without usable measurements; 66 rows are confirmed candidates, three provisional, and 25 excluded by the survey rule. Those are configurations, not 69 independent products.

## 1. Same questions, local comparison

The [September 21 experiment](../results/open-model-comparison.md) used Unridden's 1,286 authored questions and one scorer. It did not use identical hardware, model size, API path, or request timing boundary across every row. Its Unridden reference was not contemporaneous with all competitors.

| Tested configuration | Correct answers | Median request | What the result means |
| --- | ---: | ---: | --- |
| Unridden 26B reference | 93.1% | 179.8 ms | Older v0.4.1 sequential result on RTX 5090; current matched 26B run below is different. |
| Kev-9B | about 90.0% using Kev's returned choice | 153.5 ms | RTX 5090 local HTTP; faster request median, lower correctness on these cases. Its default optimized serving mode was unavailable for this test. |
| Laya typed-decisions | 61.8% | 1,503.7 ms | CPU-only here; not a fair GPU speed comparison or current Laya release. |
| so1 separate | 92.1% | 187.9 ms | Same 26B weights and GPU through a custom backend; slightly slower, close in accuracy. |
| so1 packed | 90.1% | 107.7 ms | Same weights and GPU, faster by packing questions; answer quality and sibling isolation changed. |

**Conclusion from this test:** Unridden led these particular configurations in accuracy on its authored cases. It did **not** lead every measured latency row. The result does not rank today's Kev-27B, hosted Jev, or all JevBench entrants.

## 2. Unridden's current internal reference points

| Workload and path | Quality | Time | Boundary |
| --- | --- | --- | --- |
| [Same-worker seven-suite 26B](../results/gemma4-e4b-vs-26b-usecases.md) | 1,193/1,286 = 92.8% | 201.7 ms median request | 26B, current worker, RTX 5090, one paired run. |
| [Same-worker seven-suite E4B](../results/gemma4-e4b-vs-26b-usecases.md) | 1,130/1,286 = 87.9% | 141.7 ms median request | E4B is about 30% faster at a 4.9-point accuracy cost on these cases. |
| [Historical direct snapshot Tetris](../results/full-depth-snapshots.md) | Game outcomes affected by premature information; do not use them to claim real-time play | E4B game-level request averages 33.3–46.2 ms | Different prompts and API path; not a seven-suite median or a complete move. |
| [Corrected Tetris run](../results/tetris-information-correction.md) | 26B gateway, 12 games: level-18 Choice and lookahead survived both seeds; level-29 lookahead ended at 39/64 pieces | Choice controller medians about 46–49 ms; lookahead about 92–103 ms | Includes controller work and moves with no model call; not comparable to vendor request p50. |

The corrected Tetris run changes the gameplay claim. No new general competitor inference run was performed as part of that fix.

## 3. Public results for surveyed projects

These are **reported figures**, usually from each project's own hardware and task set. Putting them beside Unridden reveals useful possibilities, but does not turn them into measured speed or intelligence wins over Unridden.

| System | Reported speed | Reported quality | Cost or capacity | Comparison to Unridden |
| --- | --- | --- | --- | --- |
| [TypeSafe Jev 1.13](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | Hosted response **70–500 ms** in its launch; JevBench [raw p50 652.4 ms](https://github.com/fstandhartinger/jevbench/blob/main/results/v1.4.2.1/jevbench-v1.4.2.1-results.json) on that evaluator's request set | Vendor-authored workflow results; JevBench hard accuracy **74.1%** on its own set | [Tariff](https://docs.typesafe.ai/models): **$0.042/million input tokens**, output tokens free; board **$0.0399/1,000** decisions | No Unridden hosted-Jev same-request test or JevBench row. Its vendor range overlaps Unridden's decision-suite latencies; neither proves a winner. |
| [Kev family](https://github.com/jaredpalmer/kev) | Publisher: Kev-4B **18.1 ms** model time for six short questions on H100; Kev-9B **24.0 ms** on H100; Kev-27B **46.5 ms** on B200; long text slower | [Kev-27B card](https://huggingface.co/jaredpalmer/kev-27b): **89.6%** on its locked transfer-v4 test; development Jev row **85.7%** versus Kev-27B **84.8%**, different status | Kev-27B ~55 GB resident bf16; cloud rates are illustrative, not a per-call tariff | The older Kev-9B local row above is matched on questions; publisher's 4B/27B figures are not. Some Kev claims are faster than Unridden's suite median but measure model time and different questions. |
| [so1 / open-alternative-jev](https://github.com/ikermoel/open-alternative-jev) | Publisher: **582 ms** per five-decision case locally versus a **710 ms** hosted Jev row on `typed-decisions`; **4.55 questions/s** packed on RACE-H | Publisher: **73.7%** `typed-decisions` accuracy versus cited Jev **72.7%**; author says one-point gap is within noise | Operator supplies GPU, roughly 30–35 GB in cited test; no all-in cost | The local same-question so1 rows above are more informative for Unridden. Published typed-decisions rows are another dataset and hardware. |
| [Laya](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md) | Publisher T4 warmed calls: English **39.5 ms** one question, **84.5 ms** five; multilingual **32.8/40.1 ms** | Fine-tuned `typed-decisions` **76.6%**, with training on that dataset's train split; base English **36.2%** | Open code/weights; no metered API bill or full ownership cost | Our old CPU Laya row cannot be compared to its published T4 timings. Current fine-tuned Laya is not the older local checkpoint. |
| [OpenJev DiffusionGemma](https://github.com/razorback16/openjev) | Publisher RTX PRO 6000, three questions: **94 ms** p50 at concurrency one; **31 ms** p50 with `samples:1` (different setting); **760 ms** p50 at concurrency 64 | No core-readout gold-label accuracy benchmark found | Core NVFP4 route needs at least **24 GB VRAM**; no verified full dollar cost | Fastest cited mode is near Unridden's historical Tetris request average, but neither workload nor GPU matches. Quality ordering unknown. |
| [Rizzo Flow](https://github.com/Rizzo-AI-Academy/rizzo-flow) | Publisher RTX 5060 Ti: base 4B short-state **49 ms** per decision; fine-tuned later run **66 ms**; shared-state throughput **20.99** or **16.25 decisions/s** | Fine-tuned 4B **64.8%** on `typed-decisions` held-out split; worse than base on a separate missing-evidence fixture | ~4.4 GB weights, ~5.6 GiB GPU free-memory drop in one setup; no full ownership cost | Not the same task, model, hardware, or timing boundary as Unridden. Lower per-decision time than Unridden suite median does not prove a faster equal-quality system. |
| [jevmlx](https://github.com/bnsd55/jevmlx) | Publisher M5 Max: **0.6 s/case** for 7B labels, versus **1.5 s** naive generation | **82.1% consensus agreement** on 45 public examples, not gold accuracy or Jev private score | Local/no metered API fee; hardware and electricity omitted | Cannot rank against Unridden's gold-label suites or RTX 5090 speed. |
| [PoorJev](https://github.com/rupeshpoojary9/poorjev) | Publisher CPU: **137 ms** p50 on 154 one-question Banking77 cases | **78.1%** on its 160-decision hand-labelled set; **65.6%** Banking77 with weak calibration | ~400 MB model download; no API fee, CPU/hardware cost unknown | Different CPU task and labels; no matched Unridden row. |
| [laya2typesafeapi](https://github.com/yunhai-dev/laya2typesafeapi) | No wrapper-specific timing found | No wrapper-specific accuracy found | Wrapper uses Laya; code license unresolved | No separate performance comparison is possible. |

## 4. Shared JevBench board: competitor-versus-competitor, not Unridden-versus-competitor

The [census CSV](../../outputs/jev-survey-2026-09-27/jevbench_census.csv) gives all 94 rows' hard accuracy, sealed aggregate, raw p50 where present, hardware, and cost per 1,000 (mostly modeled). **Unridden has no row.** Selected rows illustrate why the board does not support a universal speed or quality claim for Unridden:

| JevBench configuration | Hard accuracy | Raw p50 | Cost per 1,000 |
| --- | ---: | ---: | ---: |
| Plumb-4B | 77.7% | 16.5 ms | $0.0298 estimated |
| decider-4B v2 | 67.3% | 17.5 ms | $0.0201 estimated |
| TypeSafe Jev 1.13 | 74.1% | 652.4 ms | $0.0399 measured |
| OpenJev DiffusionGemma | 65.5% | 241.3 ms | $0.0656 estimated |
| Laya ModernBERT | 34.1% | 787.1 ms | $0.0029 estimated |
| so1 Qwen3.5-4B | 56.8% | 206.9 ms | $0.0222 estimated |

Other confirmed candidate rows report raw p50 values as low as **16.8 ms** (CLM-8B) and **17.5 ms** (decider-4B), while some are multiple seconds. Board hardware and request shapes vary, and its sealed aggregates and estimated self-hosted costs use evaluator choices. The complete table is in the census; do not use these rows to compare Unridden's Tetris requests or authored-suite requests numerically.

## Practical conclusion

Unridden has strong measured quality on **its own authored suites** and an attractive measured E4B/26B size-speed-quality tradeoff. It is **not established as universally fastest, smartest, or cheapest**. Several alternatives report shorter times on different workloads; Kev-9B and so1 packed were faster in the older local suite. Jev's public latency range does not settle a comparison with Unridden, and the corrected Tetris experiment weakens the earlier full-speed gameplay headline without changing the typed-decision suite scores.

The next decision-quality comparison should submit Unridden's existing compatible endpoint to JevBench's **231 public decisions** with pinned model, prompt, option order, runtime, GPU and latency boundary, then seek a sealed evaluator run. Separately run a blind independent holdout and matched hosted-Jev requests. Report speed, accuracy, calibration, and total cost **separately**, rather than turning a composite score into an intelligence ranking. See the [benchmark coverage assessment](benchmark-coverage-2026-09-27.md).
