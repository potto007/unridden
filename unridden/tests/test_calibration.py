"""Calibrated confidence and the escalation gate on the wire (ADR 0012)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.unridden.fit_calibration_table import fit_from_observations, main
from unridden.api.app import ApiConfig
from unridden.api.calibration import Calibration, CalibrationError
from unridden.api.compiler import PROMPT_VERSION
from unridden.api.gate import (
    BucketCalibration,
    EscalationGateConfig,
    TemperatureCalibrationTable,
    calibrated_probabilities,
)
from unridden.tests.test_api_app import (
    PROFILE,
    REQUEST,
    DeterministicBackend,
    client_for,
)


def table(**overrides: object) -> TemperatureCalibrationTable:
    fields: dict[str, object] = {
        "model_sha256": PROFILE.model_sha256,
        "runtime_sha256": PROFILE.runtime_sha256,
        "prompt_version": PROMPT_VERSION,
        "buckets": {
            "choice:2": BucketCalibration(
                question_type="choice",
                option_count=2,
                sample_count=40,
                raw_temperature=1.5,
                temperature=1.5,
                clamped=False,
                raw_nll=0.4,
                calibrated_nll=0.3,
            )
        },
        "type_fallbacks": {"score": 2.0},
        "default_temperature": 1.0,
    }
    fields.update(overrides)
    return TemperatureCalibrationTable.model_validate(fields)


def write(path: Path, model: TemperatureCalibrationTable) -> Path:
    path.write_text(json.dumps(model.model_dump(mode="json")), encoding="utf-8")
    return path


def test_binding_refuses_missing_or_foreign_provenance(tmp_path: Path) -> None:
    unbound = Calibration(
        table(model_sha256=None), path=tmp_path / "t.json", gate=EscalationGateConfig()
    )
    with pytest.raises(CalibrationError, match="lacks provenance: model_sha256"):
        unbound.bind(PROFILE, prompt_version=PROMPT_VERSION)

    foreign = Calibration(
        table(runtime_sha256="c" * 64),
        path=tmp_path / "t.json",
        gate=EscalationGateConfig(),
    )
    with pytest.raises(CalibrationError, match="runtime_sha256"):
        foreign.bind(PROFILE, prompt_version=PROMPT_VERSION)

    stale_prompt = Calibration(
        table(), path=tmp_path / "t.json", gate=EscalationGateConfig()
    )
    with pytest.raises(CalibrationError, match="prompt_version"):
        stale_prompt.bind(PROFILE, prompt_version="unridden-gemma-choice-v0")

    bound = Calibration(table(), path=tmp_path / "t.json", gate=EscalationGateConfig())
    bound.bind(PROFILE, prompt_version=PROMPT_VERSION)
    assert bound.binding is not None
    assert bound.binding.buckets == 1
    assert bound.binding.status == "fitted_requires_held_out_validation"


def test_load_rejects_malformed_files(tmp_path: Path) -> None:
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    with pytest.raises(CalibrationError, match="not valid JSON"):
        Calibration.load(bad_json, None)
    wrong_shape = tmp_path / "shape.json"
    wrong_shape.write_text(json.dumps({"buckets": 3}), encoding="utf-8")
    with pytest.raises(CalibrationError, match="table schema"):
        Calibration.load(wrong_shape, None)
    with pytest.raises(CalibrationError, match="cannot read"):
        Calibration.load(tmp_path / "missing.json", None)
    good = write(tmp_path / "good.json", table())
    bad_gate = tmp_path / "gate.json"
    bad_gate.write_text(
        json.dumps({"high_margin_threshold": 0.1, "low_margin_threshold": 0.5}),
        encoding="utf-8",
    )
    with pytest.raises(CalibrationError, match="gate schema"):
        Calibration.load(good, bad_gate)


def test_config_requires_a_table_for_gate_thresholds(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="gate_config_path needs"):
        ApiConfig(gate_config_path=tmp_path / "gate.json")


@pytest.mark.asyncio
async def test_startup_refuses_a_table_fitted_elsewhere(tmp_path: Path) -> None:
    backend = DeterministicBackend()
    path = write(tmp_path / "foreign.json", table(model_sha256="f" * 64))
    with pytest.raises(CalibrationError, match="different configuration"):
        async with client_for(backend, calibration_table_path=path):
            pass
    # The model was loaded and then released; nothing stays resident.
    assert backend.started == 1 and backend.closed >= 1 and not backend.ready


@pytest.mark.asyncio
async def test_a_bad_table_fails_before_the_model_loads(tmp_path: Path) -> None:
    backend = DeterministicBackend()
    path = tmp_path / "bad.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(CalibrationError):
        async with client_for(backend, calibration_table_path=path):
            pass
    assert backend.started == 0


@pytest.mark.asyncio
async def test_without_a_table_nothing_changes_and_probe_is_refused() -> None:
    backend = DeterministicBackend()
    async with client_for(backend) as client:
        models = await client.get("/v1/models")
        plain = await client.post("/v1/decisions?diagnostics=true", json=REQUEST)
        probe = await client.post(
            "/v1/decisions?diagnostics=true&probe=true", json=REQUEST
        )
    capabilities = models.json()["models"][0]["capabilities"]
    assert capabilities["calibrated_confidence"] is None
    assert capabilities["escalation_gate"] is False
    assert capabilities["permutation_probe"] is False
    assert models.json()["models"][0]["calibration"] is None
    assert plain.status_code == 200
    for diagnostic in plain.json()["diagnostics"].values():
        assert "calibration" not in diagnostic
        assert "gate" not in diagnostic
    assert probe.status_code == 422
    assert probe.json()["error"]["code"] == "capability_unavailable"
    assert probe.json()["error"]["field"] == "probe"


@pytest.mark.asyncio
async def test_probe_needs_diagnostics() -> None:
    backend = DeterministicBackend()
    async with client_for(backend) as client:
        response = await client.post("/v1/decisions?probe=true", json=REQUEST)
    assert response.status_code == 422
    assert response.json()["error"] == {
        "code": "invalid_request",
        "message": "probe=true requires diagnostics=true",
        "field": "probe",
        "retryable": False,
    }


@pytest.mark.asyncio
async def test_calibration_is_additive_and_diagnostics_only(tmp_path: Path) -> None:
    backend = DeterministicBackend()
    path = write(tmp_path / "table.json", table())
    async with client_for(backend, calibration_table_path=path) as client:
        models = await client.get("/v1/models")
        bare = await client.post("/v1/decisions", json=REQUEST)
        full = await client.post("/v1/decisions?diagnostics=true", json=REQUEST)

    model = models.json()["models"][0]
    assert model["capabilities"]["confidence"] == "max_probability_v1"
    assert (
        model["capabilities"]["calibrated_confidence"]
        == "temperature_scaled_max_probability_v1"
    )
    assert model["capabilities"]["escalation_gate"] is True
    assert model["calibration"]["model_sha256"] == PROFILE.model_sha256
    assert model["calibration"]["prompt_version"] == PROMPT_VERSION
    assert model["calibration"]["buckets"] == 1
    assert model["calibration"]["gate"]["high_margin_threshold"] == 0.6

    assert bare.status_code == 200 and full.status_code == 200
    assert "diagnostics" not in bare.json()
    body = full.json()
    # The public answers are exactly what an uncalibrated server returns.
    assert body["answers"] == bare.json()["answers"]
    assert body["usage"] == bare.json()["usage"]
    assert body["answers"]["route"]["confidence"] == pytest.approx(0.7310585786)

    route = body["diagnostics"]["route"]
    assert route["confidence_formula"] == "max_probability_v1"
    calibration = route["calibration"]
    assert calibration["confidence_formula"] == (
        "temperature_scaled_max_probability_v1"
    )
    assert calibration["table_status"] == "fitted_requires_held_out_validation"
    assert calibration["bucket"] == "choice:2"
    assert calibration["temperature"] == 1.5
    assert calibration["temperature_source"] == "bucket"
    expected = calibrated_probabilities([0.0, 1.0], 1.5)
    assert calibration["calibrated_probabilities"] == {
        "refund": pytest.approx(expected[0]),
        "duplicate": pytest.approx(expected[1]),
    }
    assert calibration["calibrated_confidence"] == pytest.approx(expected[1])
    assert calibration["calibrated_probability_margin"] == pytest.approx(
        expected[1] - expected[0]
    )
    assert calibration["scaled_logit_margin"] == pytest.approx(1.0 / 1.5)
    # Margin 0.32 sits between the default thresholds, so without a probe the
    # gate asks for one rather than deciding.
    assert route["gate"] == {
        "action": "run_permutation_probe",
        "reason": "borderline_margin",
    }

    severity = body["diagnostics"]["severity"]["calibration"]
    assert severity["bucket"] == "score:3"
    assert severity["temperature"] == 2.0
    assert severity["temperature_source"] == "type_fallback"
    assert list(severity["calibrated_probabilities"]) == ["0", "1", "2"]

    noul = body["diagnostics"]["true_check"]
    assert noul["calibration"]["bucket"] == "noul:2"
    assert noul["calibration"]["temperature"] == 1.0
    assert noul["calibration"]["temperature_source"] == "default"
    assert list(noul["calibration"]["calibrated_probabilities"]) == ["true", "false"]
    # A borderline Noul has no option order to permute, so it escalates.
    assert noul["gate"] == {"action": "escalate", "reason": "low_calibrated_margin"}


@pytest.mark.asyncio
async def test_probe_reasks_borderline_questions_with_reversed_options(
    tmp_path: Path,
) -> None:
    backend = DeterministicBackend()
    path = write(tmp_path / "table.json", table())
    async with client_for(backend, calibration_table_path=path) as client:
        plain = await client.post("/v1/decisions?diagnostics=true", json=REQUEST)
        probed = await client.post(
            "/v1/decisions?diagnostics=true&probe=true", json=REQUEST
        )

    assert probed.status_code == 200, probed.text
    before = plain.json()
    after = probed.json()
    # The probe never rewrites the primary answers; it only informs the gate.
    assert after["answers"] == before["answers"]
    assert after["usage"]["output_tokens"] == 0
    # Only the borderline Choice was probed; its reversed re-ask is the first
    # question of its own pass, so it costs 10 processed tokens in the fixture.
    assert after["usage"]["input_tokens"] == before["usage"]["input_tokens"] + 10

    # The fixture backend always prefers the last label, so reversing the
    # options flips the Choice: a position bias the gate now catches.
    route = after["diagnostics"]["route"]["gate"]
    assert route["action"] == "escalate"
    assert route["reason"] == "permutation_flip"
    assert route["probe"]["flipped"] is True
    assert route["probe"]["permuted_top1"] == "refund"
    assert route["probe"]["realigned_probabilities"] == {
        "refund": pytest.approx(0.7310585786),
        "duplicate": pytest.approx(0.2689414214),
    }
    assert route["probe"]["generated_tokens"] == 0
    assert route["probe"]["processed_tokens"] == 10
    assert route["probe"]["prompt_sha256"] == f"{1:064x}"

    # At the Score's fallback temperature of 2.0 its margin drops below the low
    # threshold, so it escalates outright and no probe is spent on it.
    assert after["diagnostics"]["severity"]["gate"] == {
        "action": "escalate",
        "reason": "low_calibrated_margin",
    }

    # The Noul was never probed and keeps its stage-1 verdict.
    assert after["diagnostics"]["true_check"]["gate"] == {
        "action": "escalate",
        "reason": "low_calibrated_margin",
    }
    assert before["diagnostics"]["route"]["gate"] == {
        "action": "run_permutation_probe",
        "reason": "borderline_margin",
    }


@pytest.mark.asyncio
async def test_gate_thresholds_come_from_the_optional_file(tmp_path: Path) -> None:
    backend = DeterministicBackend()
    path = write(tmp_path / "table.json", table())
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps({"high_margin_threshold": 0.3, "low_margin_threshold": 0.1}),
        encoding="utf-8",
    )
    async with client_for(
        backend, calibration_table_path=path, gate_config_path=gate
    ) as client:
        models = await client.get("/v1/models")
        response = await client.post(
            "/v1/decisions?diagnostics=true&probe=true", json=REQUEST
        )
    assert models.json()["models"][0]["calibration"]["gate"][
        "high_margin_threshold"
    ] == pytest.approx(0.3)
    route = response.json()["diagnostics"]["route"]["gate"]
    # Margin 0.32 now clears the high threshold: accepted, nothing probed.
    assert route == {"action": "accept_primary", "reason": "high_calibrated_margin"}
    assert response.json()["usage"]["input_tokens"] == 33


def observation_rows(prompt_version: str = PROMPT_VERSION) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(12):
        logits = [3.0, 0.0] if index % 3 else [0.2, 0.0]
        probs = calibrated_probabilities(logits, 1.0)
        gold = "a" if index % 4 else "b"
        rows.append(
            {
                "suite": "s1" if index < 6 else "s2",
                "case": f"c{index}",
                "difficulty": "medium",
                "status": "ok",
                "request": {
                    "model": "local-gemma-unridden-v1",
                    "state": "x",
                    "questions": {
                        "q": {
                            "type": "choice",
                            "instructions": "pick",
                            "criteria": {"a": None, "b": None},
                        }
                    },
                },
                "response": {
                    "model": "local-gemma-unridden-v1",
                    "answers": {
                        "q": {
                            "type": "choice",
                            "choice": "a",
                            "probabilities": {"a": probs[0], "b": probs[1]},
                            "confidence": max(probs),
                        }
                    },
                    "diagnostics": {
                        "q": {
                            "raw_label_logits": logits,
                            "coverage": 0.9,
                            "model_sha256": PROFILE.model_sha256,
                            "runtime_sha256": PROFILE.runtime_sha256,
                            "prompt_version": prompt_version,
                        }
                    },
                },
                "outcomes": [
                    {"qid": "q", "type": "choice", "expected": gold, "got": "a"}
                ],
            }
        )
    rows.append({"suite": "s2", "case": "failed", "status": "error", "outcomes": []})
    return rows


def test_fit_script_writes_a_bound_table_the_server_accepts(tmp_path: Path) -> None:
    observations = tmp_path / "observations.jsonl"
    observations.write_text(
        "".join(json.dumps(row) + "\n" for row in observation_rows()),
        encoding="utf-8",
    )
    out = tmp_path / "fitted" / "table.json"
    assert main(["--observations", str(observations), "--out", str(out)]) == 0
    loaded = Calibration.load(out, None)
    loaded.bind(PROFILE, prompt_version=PROMPT_VERSION)
    assert loaded.table.model_sha256 == PROFILE.model_sha256
    assert loaded.table.prompt_version == PROMPT_VERSION
    assert set(loaded.table.buckets) == {"choice:2"}
    assert loaded.table.buckets["choice:2"].sample_count == 12
    assert 1.0 <= loaded.table.buckets["choice:2"].temperature <= 5.0
    assert loaded.table.status == "fitted_requires_held_out_validation"
    with pytest.raises(SystemExit):
        main(["--observations", str(observations), "--out", str(out)])


def test_fit_refuses_mixed_provenance_and_missing_diagnostics(
    tmp_path: Path,
) -> None:
    mixed = observation_rows()[:12]
    mixed[3]["response"]["diagnostics"]["q"]["runtime_sha256"] = "d" * 64  # type: ignore[index]
    path = tmp_path / "mixed.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in mixed), encoding="utf-8")
    with pytest.raises(ValueError, match="mix provenance"):
        fit_from_observations(path)

    bare = observation_rows()[:12]
    del bare[0]["response"]["diagnostics"]  # type: ignore[attr-defined]
    path = tmp_path / "bare.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in bare), encoding="utf-8")
    with pytest.raises(ValueError, match="no diagnostics"):
        fit_from_observations(path)
