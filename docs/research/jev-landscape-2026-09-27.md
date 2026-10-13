# Jev-like decision systems: current source survey

Surveyed September 27, 2026. This is a bounded census of publicly findable typed-decision systems and their published evidence, not a universal ranking of intelligence, speed, or cost.

## Coverage

[JevBench v1.4.2.1](https://github.com/fstandhartinger/jevbench/blob/main/results/v1.4.2.1/jevbench-v1.4.2.1-results.json) has 94 configuration rows. Applying this survey's non-generative decision-system rule yields 69 candidate configurations across 55 named project families. Fifty-two families have an independently findable source or service; three are provisional. Twenty-five rows are controls, generic components, or generative modes excluded from this count. A configuration is not a distinct model or project. The [discovery ledger](../../outputs/jev-survey-2026-09-27/worker_discovery.md) explains every screening choice and includes additional projects and wrappers found outside the board.

A project can be open code, open weights, a wrapper over someone else's weights, or a hosted proprietary service. Those are separate license and deployment claims. The [core ledger](../../outputs/jev-survey-2026-09-27/worker_core.md) covers Jev, so1, Kev, and Laya; the [local-systems ledger](../../outputs/jev-survey-2026-09-27/worker_local.md) covers OpenJev, Rizzo Flow, JevMLX, PoorJev, and a Laya wrapper. The [CSV census](../../outputs/jev-survey-2026-09-27/jevbench_census.csv) and [JSON census](../../outputs/jev-survey-2026-09-27/jevbench_census.json) preserve all 94 board rows with their source links and screening status. These local output files are ignored by Git; this page is the tracked entry point.

## What the evidence supports

Unridden's [September 21 local comparison](../results/open-model-comparison.md) used the same authored questions and scorer for Unridden, Kev, Laya, and an adapted so1. Unridden scored 93.1% of 1,286 questions. Kev's native returned choices scored about 90.0%, Laya 61.8%, and so1 92.1% in separate mode or 90.1% in packed mode. The suites were authored for Unridden, the models and runtime paths differ, Laya ran on CPU, and so1 used a custom llama.cpp backend. These results support a bounded task-quality claim on those cases, not general intelligence superiority. Median request latency in the same report is not a uniform hardware or work boundary; Kev and packed so1 have lower reported medians than sequential Unridden.

TypeSafe's [current Jev documentation](https://docs.typesafe.ai/models) describes a hosted proprietary API and its tariff. Unridden has not measured hosted Jev on the same workload. Neither the local 93.1% score nor a JevBench score can be used to rank those systems against each other. The [JevBench methodology](https://github.com/fstandhartinger/jevbench/blob/main/docs/METHOD-v1.4.md) combines task quality, validity, speed, and cost with chosen weights; some self-hosted speed and cost figures are modeled or estimated. Its current board has no Unridden row.

The new [26B versus E4B matched comparison](../results/gemma4-e4b-vs-26b-usecases.md) tests a different question: the local resource/quality tradeoff inside Unridden. On the same worker and seven authored suites, E4B scored 87.9% versus 26B's 92.8%, with about 30% lower median request latency and a roughly 70% smaller model file. It does not compare either model to Jev.

## Benchmark direction

No cross-project suite has become a settled standard. JevBench is the closest emerging typed-decision reference, but its large board is operated by one evaluator. The next defensible external check is to run Unridden's compatible endpoint on JevBench's 231 public decisions, publishing accuracy, validity, calibration, and latency components separately with pinned model, runtime, prompt, option order, and hardware. An official board score requires the evaluator's sealed process. Keep Unridden's seven frozen use-case suites as a separate product-fit measure. [Decision Index](https://github.com/apolinario/decision-index) is a possible broader second check, but its choice/yes-no adapter does not cover Unridden's Score primitive.

Any claim of lower total cost also needs measured deployment demand, concurrency, energy or hosting rates, and hardware amortization. A local model avoids a per-call API fee; that alone is not a full cost comparison.

## Limits and next checks

The broad census verified identity and source availability more widely than it verified licenses, weights, quality, and cost. Project-specific license terms and model cards should be checked before reuse. The three provisional families remain unresolved. The public JevBench set is not its official sealed score and can be tuned against. Unridden's authored suites need independent holdout evaluation, stronger calibration and option-order tests, and a matched hosted-Jev run before comparative superiority claims. The separate [Tetris/snapshot results](../results/full-depth-snapshots.md) demonstrate bounded interactive behavior and state reuse, not a cross-project intelligence benchmark.
