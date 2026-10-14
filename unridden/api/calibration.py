"""Serve a fitted temperature table beside the raw readout (ADR 0012).

The public `confidence` stays `max_probability_v1`. When the operator loads a
`TemperatureCalibrationTable` that is bound to the running model, runtime and
prompt version, every `?diagnostics=true` answer also carries the
temperature-scaled readout and the escalation gate's verdict from
`unridden.api.gate`. With `?probe=true` the service re-asks borderline Choice
and Score questions with their options reversed and folds the realigned result
into the verdict. Nothing here samples a token.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError

from unridden.api.gate import (
    EscalationGateConfig,
    GateDecision,
    TemperatureCalibrationTable,
    bucket_key,
    build_permutation_probe_questions,
    evaluate_question_gate,
)
from unridden.api.schema import (
    BackendProfile,
    CalibrationDiagnostics,
    CalibrationStatus,
    DecisionRequest,
    DecisionResponse,
    GateDiagnostics,
    PermutationProbeDiagnostics,
    Question,
    QuestionDiagnostics,
    StrictModel,
    TemperatureSource,
    Usage,
)


class CalibrationError(RuntimeError):
    """The calibration table cannot be loaded or does not fit this backend."""


class CalibrationBinding(StrictModel):
    """What `/v1/models` says about the loaded table."""

    status: CalibrationStatus
    path: str
    buckets: int = Field(ge=0)
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_version: str
    gate: EscalationGateConfig
    warnings: int = Field(ge=0)


def _read_json(path: Path, what: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise CalibrationError(f"cannot read {what} {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise CalibrationError(f"{what} {path} is not valid JSON") from error


def load_calibration_table(path: Path) -> TemperatureCalibrationTable:
    """Load a table written by `fit_temperature_table` (`model_dump()` JSON)."""
    try:
        return TemperatureCalibrationTable.model_validate(
            _read_json(path, "calibration table")
        )
    except ValidationError as error:
        raise CalibrationError(
            f"calibration table {path} does not match the table schema: {error}"
        ) from error


def load_gate_config(path: Path | None) -> EscalationGateConfig:
    """Load gate thresholds, or the defaults when no file is given."""
    if path is None:
        return EscalationGateConfig()
    try:
        return EscalationGateConfig.model_validate(_read_json(path, "gate config"))
    except ValidationError as error:
        raise CalibrationError(
            f"gate config {path} does not match the gate schema: {error}"
        ) from error


class Calibration:
    """A loaded table plus thresholds, bound to one backend profile at start."""

    def __init__(
        self,
        table: TemperatureCalibrationTable,
        *,
        path: Path,
        gate: EscalationGateConfig,
    ) -> None:
        self.table = table
        self.path = path
        self.gate = gate
        self.binding: CalibrationBinding | None = None

    @classmethod
    def load(cls, table_path: Path, gate_path: Path | None) -> Calibration:
        return cls(
            load_calibration_table(table_path),
            path=table_path,
            gate=load_gate_config(gate_path),
        )

    def bind(self, profile: BackendProfile, *, prompt_version: str) -> None:
        """Refuse a table fitted on another model, runtime or prompt version.

        A table that carries no provenance is refused too: served calibration
        has to be traceable to the logits it was fitted on.
        """
        table = self.table
        missing = [
            name
            for name, value in (
                ("model_sha256", table.model_sha256),
                ("runtime_sha256", table.runtime_sha256),
                ("prompt_version", table.prompt_version),
            )
            if value is None
        ]
        if missing:
            raise CalibrationError(
                f"calibration table {self.path} lacks provenance: {', '.join(missing)}"
            )
        mismatched = [
            f"{name} (table {expected!r}, backend {actual!r})"
            for name, expected, actual in (
                ("model_sha256", table.model_sha256, profile.model_sha256),
                ("runtime_sha256", table.runtime_sha256, profile.runtime_sha256),
                ("prompt_version", table.prompt_version, prompt_version),
            )
            if expected != actual
        ]
        if mismatched:
            raise CalibrationError(
                f"calibration table {self.path} was fitted on a different "
                f"configuration: {'; '.join(mismatched)}"
            )
        assert table.model_sha256 is not None
        assert table.runtime_sha256 is not None
        assert table.prompt_version is not None
        self.binding = CalibrationBinding(
            status=table.status,
            path=str(self.path),
            buckets=len(table.buckets),
            model_sha256=table.model_sha256,
            runtime_sha256=table.runtime_sha256,
            prompt_version=table.prompt_version,
            gate=self.gate,
            warnings=len(table.warnings),
        )

    def temperature_source(
        self, question: Question, option_count: int
    ) -> tuple[str, TemperatureSource]:
        key = bucket_key(question.type, option_count)
        if key in self.table.buckets:
            return key, "bucket"
        if question.type in self.table.type_fallbacks:
            return key, "type_fallback"
        return key, "default"

    def stage1(
        self, request: DecisionRequest, response: DecisionResponse
    ) -> dict[str, GateDecision]:
        """Gate every answered question on its raw logits alone."""
        diagnostics = _require_diagnostics(response)
        return {
            question_id: evaluate_question_gate(
                question_id,
                question,
                diagnostics[question_id].raw_label_logits,
                coverage=diagnostics[question_id].coverage,
                config=self.gate,
                calibration=self.table,
            )
            for question_id, question in request.questions.items()
        }

    def probe_request(
        self, request: DecisionRequest, decisions: Mapping[str, GateDecision]
    ) -> DecisionRequest | None:
        """The option-reversed re-ask for the questions the gate wants probed."""
        probes = build_permutation_probe_questions(request.questions, decisions)
        if not probes:
            return None
        return DecisionRequest(
            model=request.model, state=request.state, questions=probes
        )

    def stage2(
        self,
        request: DecisionRequest,
        response: DecisionResponse,
        decisions: dict[str, GateDecision],
        probe: DecisionRequest,
        probe_response: DecisionResponse,
    ) -> dict[str, GateDecision]:
        """Fold each probed answer into its question's verdict."""
        diagnostics = _require_diagnostics(response)
        for question_id, permuted in probe.questions.items():
            decisions[question_id] = evaluate_question_gate(
                question_id,
                request.questions[question_id],
                diagnostics[question_id].raw_label_logits,
                coverage=diagnostics[question_id].coverage,
                config=self.gate,
                calibration=self.table,
                original_answer=response.answers[question_id],
                permuted_question=permuted,
                permuted_answer=probe_response.answers[question_id],
            )
        return decisions

    def annotate(
        self,
        request: DecisionRequest,
        response: DecisionResponse,
        decisions: Mapping[str, GateDecision],
        probe_response: DecisionResponse | None,
    ) -> DecisionResponse:
        """Attach calibration and gate blocks; answers and usage stay as mapped
        except that the probe's processed tokens join `input_tokens`."""
        diagnostics = _require_diagnostics(response)
        probe_diagnostics = (
            _require_diagnostics(probe_response) if probe_response is not None else {}
        )
        annotated: dict[str, QuestionDiagnostics] = {}
        for question_id, question in request.questions.items():
            decision = decisions[question_id]
            uncertainty = decision.uncertainty
            option_ids = list(_canonical_keys(response, question_id))
            bucket, source = self.temperature_source(question, uncertainty.option_count)
            calibration = CalibrationDiagnostics(
                table_status=self.table.status,
                bucket=bucket,
                temperature=uncertainty.temperature,
                temperature_source=source,
                calibrated_probabilities=dict(
                    zip(option_ids, uncertainty.calibrated_probabilities, strict=True)
                ),
                calibrated_confidence=uncertainty.calibrated_confidence,
                calibrated_probability_margin=(
                    uncertainty.calibrated_probability_margin
                ),
                scaled_logit_margin=uncertainty.scaled_logit_margin,
                calibrated_normalized_entropy=(
                    uncertainty.calibrated_normalized_entropy
                ),
            )
            probe_block: PermutationProbeDiagnostics | None = None
            if decision.stability is not None and question_id in probe_diagnostics:
                stability = decision.stability
                raw_probe = probe_diagnostics[question_id]
                probe_block = PermutationProbeDiagnostics(
                    flipped=stability.flipped,
                    permuted_top1=str(stability.permuted_top1),
                    max_probability_delta=stability.max_probability_delta,
                    total_variation_distance=stability.total_variation_distance,
                    post_probe_margin=stability.post_probe_margin,
                    post_probe_entropy=stability.post_probe_entropy,
                    realigned_probabilities=stability.realigned_probabilities,
                    prompt_sha256=raw_probe.prompt_sha256,
                    processed_tokens=raw_probe.processed_tokens,
                )
            annotated[question_id] = diagnostics[question_id].model_copy(
                update={
                    "calibration": calibration,
                    "gate": GateDiagnostics(
                        action=decision.action,
                        reason=decision.reason,
                        probe=probe_block,
                    ),
                }
            )
        usage = response.usage
        if probe_response is not None:
            usage = Usage(
                input_tokens=response.usage.input_tokens
                + probe_response.usage.input_tokens
            )
        return response.model_copy(update={"diagnostics": annotated, "usage": usage})


def _require_diagnostics(
    response: DecisionResponse,
) -> dict[str, QuestionDiagnostics]:
    if response.diagnostics is None:
        raise ValueError("calibration needs the mapped diagnostics")
    return response.diagnostics


def _canonical_keys(response: DecisionResponse, question_id: str) -> list[str]:
    answer = response.answers[question_id]
    if answer.type == "noul":
        return ["true", "false"]
    return list(answer.probabilities)


__all__ = [
    "Calibration",
    "CalibrationBinding",
    "CalibrationError",
    "load_calibration_table",
    "load_gate_config",
]
