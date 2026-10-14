# 0012. Calibrated confidence and the escalation gate on the wire

- Status: accepted
- Date: 2026-10-13
- Ships proposal 5 of
  [intermediate-thinking-steps-review.md](../research/intermediate-thinking-steps-review.md)
  (priority matrix item 1) as a served feature. The public answer schemas
  are unchanged.

## Context

`confidence` is `max_probability_v1`, the winning label's conditional
probability. It saturates: on the first frozen suite run 94.7% of answers
exceeded 0.99 and the mean was 0.945 when wrong. The README has said since
0.1.0 that it must not gate decisions without task-specific calibration, and
the whitepaper's first "what I would do next" item was to fit a temperature
on held-out data and publish the calibrated confidence beside the raw one.

`unridden/api/gate.py` has had the machinery since the calibrated-margin PR:
per-`(question_type, option_count)` temperature tables bound to
`model_sha256`, `runtime_sha256` and `prompt_version`; margin and entropy
signals; an option-permutation consistency check; and a two-stage
accept / probe / escalate policy. All of it was reachable only from
`scripts/unridden/`. No route imported it, so a caller of the shipped API
got none of it, and the later research (tuned-lens cascade, E4B-to-26B
routing, two-stage deliberation) had no served signal to trigger on.

## Decision

- **`confidence` stays `max_probability_v1`.** Changing the pinned formula
  would silently move every caller's threshold. The calibrated readout is
  additive and lives where ADR-era rules already put uncertainty signals:
  `QuestionDiagnostics`, under `?diagnostics=true`. Two new optional blocks,
  `calibration` and `gate`, are present only when the server has a table
  loaded and are omitted otherwise, so an uncalibrated server's responses
  are byte-identical to before.
- **The table is an operator artifact, bound at startup, refused on
  mismatch.** `--calibration-table` names a `TemperatureCalibrationTable`
  JSON. `ApiService` loads it before the model (a malformed file fails
  without touching the GPU) and after the handshake refuses to start if the
  table's `model_sha256`, `runtime_sha256` or `prompt_version` differ from
  the live backend, or if any of the three is missing. A served temperature
  is always traceable to the logits it was fitted on. The compiler's
  `PROMPT_VERSION` is the binding key for prompts, so a prompt change
  invalidates every table, which is the intent.
- **`/v1/models` advertises the binding.** `capabilities` gains
  `calibrated_confidence` (`temperature_scaled_max_probability_v1` or
  `null`), `escalation_gate` and `permutation_probe`, and a top-level
  `calibration` object reports the table's status, provenance, bucket
  count, warning count and the active gate thresholds. A client can decide
  from the catalog whether the diagnostics it is about to read are
  calibrated, and against what.
- **The table's status travels with every answer.** `calibration.table_status`
  repeats `fitted_requires_held_out_validation` or `validated_on_held_out`
  per question. A freshly fitted table is honest about being unvalidated on
  the wire, not just in a results document.
- **The gate verdict is served, the escalation is not.** `gate` carries
  `action` (`accept_primary`, `run_permutation_probe`, `escalate`) and
  `reason` from `evaluate_question_gate`, using the table's temperature and
  the thresholds in an optional `--gate-config` file (defaults otherwise).
  The server does not route to a second model: it has one backend, and
  which model to escalate to, and whether to, is the caller's policy. The
  gateway is unchanged in this ADR.
- **`?probe=true` spends one more owned pass on borderline questions.** For
  each question whose stage-1 action is `run_permutation_probe`, the
  service re-asks it with its options reversed (Choice keys reversed, Score
  scale reversed, Noul never) against the same state, in the same busy
  slot, under the same non-generation contract. The realigned probabilities
  feed stage 2, and `gate.probe` reports the flip, total variation distance,
  post-probe margin, the probe's prompt hash and processed tokens. The
  probe's tokens are added to `usage.input_tokens`; `output_tokens` stays
  0. The primary answers are never replaced by the probe: a flip is
  evidence for `escalate`, not a second opinion averaged in. `probe=true`
  requires `diagnostics=true` (422 `invalid_request`) and a loaded table
  (422 `capability_unavailable`).
- **One fitter, one input format.** `scripts/unridden/fit_calibration_table.py`
  fits a table from a single model's `observations.jsonl` (the
  `run_usecase_suites.py --diagnostics` output), takes provenance from the
  rows and refuses mixed provenance. `evaluate_escalation_gate.py` remains
  the held-out evaluation; a table does not earn `validated_on_held_out`
  from the fitter.

## Consequences

- A caller can now read a temperature-scaled confidence, a margin that
  means the same thing across option counts, and a gate verdict without
  running the benchmark scripts, and can trust that all three were fitted
  on exactly the model, runtime and prompt that answered. This is the
  signal the cascade, routing and deliberation work needs to trigger on.
- The default thresholds (`high 0.60`, `low 0.20`, `min_coverage 0.10`,
  `max_tvd 0.25`, `post_probe_margin 0.35`) are the ones the offline
  evaluation used on the authored suites. They are not tuned for any
  caller's cost of a wrong accept versus a wasted escalation; operators
  are expected to set them per deployment through `--gate-config`.
- A probe costs a second prefill of the question (prefix reuse does not
  carry across the two passes) and only on questions the gate asks about.
  On the fixture backend the probe exposes a position bias the raw answer
  hides, which is what the gate is for.
- `/v2` decisions are not annotated. The snapshot routes carry no
  per-question diagnostics today, and a probe over a snapshot is a branch
  pair rather than a second request. That is a separate change.
- No calibration table is shipped. The results documents record the
  fitted temperatures per model, but a table is only valid against one
  worker build and one GGUF, so each deployment fits its own.
