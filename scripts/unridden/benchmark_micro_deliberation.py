"""Benchmark Two-Stage Snapshot Decomposition vs Bounded Micro-Rider (#4).

Priority matrix item 4 (Proposal 2 of
`docs/research/intermediate-thinking-steps-review.md`). Against Gemma 4 E4B
(`full-v1`) on the two suites with the largest E4B-vs-26B accuracy gaps
(`candidate-span-and-date-extraction` and `passage-rerank-and-line-search`),
this script creates one warm context snapshot per case and evaluates:

1. `single_pass`: single-stage `/v2/decisions` baseline (`output_tokens == 0`).
2. `discrete_state_eval`: zero-token discrete decomposition (`output_tokens == 0`)
   using `decisions -> state_eval(save_result_snapshot=True) -> decisions(followup)`.
3. `discrete_context_instr`: zero-token discrete decomposition (`output_tokens == 0`)
   branching Stage-2 questions from the warm context snapshot with the Stage-1
   verification summary appended to question `instructions`.
4. `micro_rider_8`: bounded 8-token greedy `/v2/rider` scratchpad followed by
   zero-token `/v2/decisions`.
5. `micro_rider_16`: bounded 16-token greedy `/v2/rider` scratchpad followed by
   zero-token `/v2/decisions`.

Requires `--allow-gpu` when launching the native worker on GPU, or `--dry-run`
to validate the suite files and decomposition plans on CPU without touching a
model or GPU.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

from scripts.unridden.run_usecase_suites import (
    Case,
    Suite,
    load_suites,
    score_answer,
)
from unridden.api.snapshots.backend import SnapshotNativeBackend
from unridden.api.snapshots.deliberation import (
    ContinuationMode,
    DeliberationMode,
    plan_two_stage_questions,
    run_two_stage_deliberation,
)
from unridden.api.snapshots.schema import (
    ContextInput,
    SnapshotCreateRequest,
)
from unridden.api.snapshots.service import SnapshotService

DEFAULT_SUITES: tuple[str, ...] = (
    "unridden/examples/usecases/candidate-span-and-date-extraction.json",
    "unridden/examples/usecases/passage-rerank-and-line-search.json",
)
DEFAULT_WORKER = (
    "/home/potto/src/ottomation/unridden/build/api-snapshot-worker"
    "/build/unridden-snapshot-worker"
)
DEFAULT_MANIFEST = (
    "/home/potto/src/ottomation/unridden/build/api-snapshot-worker/build.json"
)
DEFAULT_MODEL = (
    "/home/potto/src/local-ai/models/unsloth-gemma-4-e4b-gguf"
    "/gemma-4-E4B-it-UD-Q4_K_XL.gguf"
)

ARMS: tuple[tuple[str, DeliberationMode, ContinuationMode, int], ...] = (
    ("single_pass", "single_pass", "state_eval_child", 8),
    ("discrete_state_eval", "discrete_two_stage", "state_eval_child", 8),
    ("discrete_context_instr", "discrete_two_stage", "context_instructions", 8),
    ("micro_rider_8", "micro_rider", "context_instructions", 8),
    ("micro_rider_16", "micro_rider", "context_instructions", 16),
)


def _score_case_arm(
    suite_slug: str,
    case: Case,
    answers: dict[str, Any],
) -> tuple[int, int, dict[str, dict[str, Any]]]:
    correct_count = 0
    total_count = 0
    per_question: dict[str, dict[str, Any]] = {}
    for target in case.targets:
        ans = answers[target.qid]
        outcome = score_answer(suite_slug, case, target, ans)
        total_count += 1
        if outcome.correct:
            correct_count += 1
        per_question[target.qid] = {
            "got": outcome.got,
            "expected": outcome.expected,
            "correct": outcome.correct,
            "confidence": round(outcome.confidence, 4),
        }
    return correct_count, total_count, per_question


async def _run_benchmark(
    args: argparse.Namespace, suites: list[Suite]
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="unridden-delib-") as tmp_dir:
        backend = SnapshotNativeBackend(
            worker_path=Path(args.worker).resolve(),
            model_path=Path(args.model).resolve(),
            manifest_path=Path(args.manifest).resolve(),
            gpu=True,
            context_size=args.context_size,
            batch_size=args.batch_size,
            ubatch_size=args.ubatch_size,
            threads=args.threads,
            reference=False,
            snapshot_profile=args.profile,
        )
        service = SnapshotService(
            backend,
            store_root=tmp_dir,
            host_bytes=2 * (1 << 30),
            default_ttl=600,
            request_timeout=120.0,
            startup_timeout=180.0,
        )
        await service.start()
        try:
            assert service.profile is not None
            final_block = service.profile.final_block
            suite_reports: dict[str, Any] = {}
            overall_arm_totals: dict[str, dict[str, Any]] = {
                arm_name: {
                    "correct": 0,
                    "questions": 0,
                    "cases_all_correct": 0,
                    "cases": 0,
                    "rescues_vs_single": 0,
                    "reversals_vs_single": 0,
                    "triggered_stage2_cases": 0,
                    "decision_output_tokens": 0,
                    "rider_generated_tokens": 0,
                    "latencies_ms": [],
                }
                for arm_name, _, _, _ in ARMS
            }

            for suite in suites:
                suite_arm_totals: dict[str, dict[str, Any]] = {
                    arm_name: {
                        "correct": 0,
                        "questions": 0,
                        "cases_all_correct": 0,
                        "cases": 0,
                        "rescues_vs_single": 0,
                        "reversals_vs_single": 0,
                        "triggered_stage2_cases": 0,
                        "decision_output_tokens": 0,
                        "rider_generated_tokens": 0,
                        "latencies_ms": [],
                    }
                    for arm_name, _, _, _ in ARMS
                }
                case_records: list[dict[str, Any]] = []

                for case in suite.cases:
                    created = await service.create(
                        SnapshotCreateRequest(
                            input=ContextInput(
                                kind="context", state=case.request.state
                            ),
                            checkpoints=[final_block],
                            ttl_seconds=600,
                        ),
                        owner="bench",
                    )
                    ctx_id = created.snapshots[0].id
                    try:
                        arm_outcomes: dict[str, Any] = {}
                        single_q_results: dict[str, dict[str, Any]] = {}

                        for arm_name, mode, cont, r_tokens in ARMS:
                            res = await run_two_stage_deliberation(
                                service,
                                context_snapshot_id=ctx_id,
                                questions=case.request.questions,
                                readout_blocks=final_block,
                                mode=mode,
                                continuation=cont,
                                rider_max_tokens=r_tokens,
                                owner="bench",
                            )
                            corr, tot, q_map = _score_case_arm(
                                suite.slug, case, res.response.answers
                            )
                            if arm_name == "single_pass":
                                single_q_results = q_map

                            rescues = 0
                            reversals = 0
                            for qid, q_info in q_map.items():
                                base_ok = single_q_results[qid]["correct"]
                                now_ok = q_info["correct"]
                                if not base_ok and now_ok:
                                    rescues += 1
                                elif base_ok and not now_ok:
                                    reversals += 1

                            for bucket in (
                                suite_arm_totals[arm_name],
                                overall_arm_totals[arm_name],
                            ):
                                bucket["correct"] += corr
                                bucket["questions"] += tot
                                bucket["cases"] += 1
                                if corr == tot:
                                    bucket["cases_all_correct"] += 1
                                bucket["rescues_vs_single"] += rescues
                                bucket["reversals_vs_single"] += reversals
                                if res.deliberation.triggered_stage2:
                                    bucket["triggered_stage2_cases"] += 1
                                bucket["decision_output_tokens"] += (
                                    res.response.usage.output_tokens
                                )
                                bucket["rider_generated_tokens"] += (
                                    res.deliberation.rider_generated_tokens
                                )
                                bucket["latencies_ms"].append(res.deliberation.total_ms)

                            arm_outcomes[arm_name] = {
                                "correct": corr,
                                "total": tot,
                                "rescues": rescues,
                                "reversals": reversals,
                                "triggered_stage2": res.deliberation.triggered_stage2,
                                "stage2_questions": (
                                    res.deliberation.stage2_question_ids
                                ),
                                "note": res.deliberation.deliberation_note,
                                "rider_tokens": res.deliberation.rider_generated_tokens,
                                "total_ms": round(res.deliberation.total_ms, 2),
                                "questions": q_map,
                            }

                        case_records.append(
                            {
                                "case": case.name,
                                "arms": arm_outcomes,
                            }
                        )
                        s_corr = arm_outcomes["single_pass"]["correct"]
                        d_se = arm_outcomes["discrete_state_eval"]["correct"]
                        d_ci = arm_outcomes["discrete_context_instr"]["correct"]
                        mr8 = arm_outcomes["micro_rider_8"]["correct"]
                        mr16 = arm_outcomes["micro_rider_16"]["correct"]
                        tot = arm_outcomes["single_pass"]["total"]
                        if len({s_corr, d_se, d_ci, mr8, mr16}) > 1:
                            print(
                                f"  [{suite.slug}] {case.name}: "
                                f"single={s_corr}/{tot} "
                                f"disc_se={d_se}/{tot} "
                                f"disc_instr={d_ci}/{tot} "
                                f"rider8={mr8}/{tot} "
                                f"rider16={mr16}/{tot}",
                                flush=True,
                            )
                    finally:
                        await service.delete(ctx_id)

                suite_reports[suite.slug] = {
                    "summary": _summarize_arm_Map(suite_arm_totals),
                    "cases": case_records,
                }

            return {
                "model_path": str(Path(args.model).resolve()),
                "model_sha256": service.profile.model_sha256,
                "runtime_sha256": service.profile.runtime_sha256,
                "profile": service.profile.profile,
                "final_block": final_block,
                "overall": _summarize_arm_Map(overall_arm_totals),
                "suites": suite_reports,
            }
        finally:
            await service.close()


def _summarize_arm_Map(raw: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm_name, row in raw.items():
        lats = row["latencies_ms"]
        q_tot = row["questions"]
        c_tot = row["cases"]
        rescues = row["rescues_vs_single"]
        reversals = row["reversals_vs_single"]
        out[arm_name] = {
            "correct": row["correct"],
            "questions": q_tot,
            "accuracy": round(row["correct"] / q_tot, 4) if q_tot else 0.0,
            "cases_all_correct": row["cases_all_correct"],
            "cases": c_tot,
            "case_accuracy": round(row["cases_all_correct"] / c_tot, 4)
            if c_tot
            else 0.0,
            "rescues_vs_single": rescues,
            "reversals_vs_single": reversals,
            "net_gain_vs_single": rescues - reversals,
            "triggered_stage2_cases": row["triggered_stage2_cases"],
            "decision_output_tokens": row["decision_output_tokens"],
            "rider_generated_tokens": row["rider_generated_tokens"],
            "median_ms": round(statistics.median(lats), 2) if lats else 0.0,
            "mean_ms": round(statistics.fmean(lats), 2) if lats else 0.0,
        }
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suites",
        nargs="+",
        default=list(DEFAULT_SUITES),
        help="Suite JSON paths to evaluate.",
    )
    parser.add_argument("--worker", default=DEFAULT_WORKER)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--profile",
        choices=("full-v1", "split18-30-v1"),
        default="full-v1",
    )
    parser.add_argument("--context-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ubatch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-gpu",
        action="store_true",
        help="Explicitly authorize launching the native snapshot worker on GPU.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    suites = load_suites([Path(p) for p in args.suites])
    if args.dry_run:
        for suite in suites:
            families: dict[str, int] = {}
            aux_count = 0
            for case in suite.cases:
                plan = plan_two_stage_questions(case.request.questions)
                families[plan.family] = families.get(plan.family, 0) + 1
                aux_count += len(plan.auxiliary_questions)
            print(
                f"dry-run {suite.slug}: {len(suite.cases)} cases, "
                f"families={families}, total_aux_questions={aux_count}"
            )
        return 0

    if not args.allow_gpu:
        print(
            "Refusing to launch GPU snapshot worker without --allow-gpu "
            "(or pass --dry-run).",
            file=sys.stderr,
        )
        return 2

    report = asyncio.run(_run_benchmark(args, suites))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n=== OVERALL SUMMARY ===")
    for arm_name, stats in report["overall"].items():
        print(
            f"  {arm_name:24s}: {stats['correct']}/{stats['questions']} "
            f"({stats['accuracy'] * 100:.2f}%), "
            f"cases={stats['cases_all_correct']}/{stats['cases']} "
            f"({stats['case_accuracy'] * 100:.1f}%), "
            f"net={stats['net_gain_vs_single']:+d} "
            f"(+{stats['rescues_vs_single']}/-{stats['reversals_vs_single']}), "
            f"stage2={stats['triggered_stage2_cases']}/{stats['cases']}, "
            f"rider_tok={stats['rider_generated_tokens']}, "
            f"median={stats['median_ms']:.1f}ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
