"""Benchmark orchestration script executing all 5 load scenarios and 3 retrieval experiments."""

import os
import json
import asyncio
import pandas as pd
from pathlib import Path
from typing import List
from httpx import ASGITransport
import httpx

from app.config import settings
from app.main import app, state, init_app_state
from app.schemas import BenchmarkRecord
from benchmarks.load_generator import LoadGenerator


def load_questions() -> List[dict]:
    questions = []
    with open(settings.QUESTIONS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


async def run_evaluation_set(retrieval_setting: str = "B") -> List[BenchmarkRecord]:
    """Runs evaluation across the 15 questions in data/questions.jsonl and grades them."""
    questions_data = load_questions()
    generator = LoadGenerator(app)
    records: List[BenchmarkRecord] = []
    transport = ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        for q_item in questions_data:
            q_text = q_item["question"]
            expected_status = q_item["target_status"]

            rec = await generator.execute_request(
                client=client,
                question=q_text,
                scenario=f"eval_retrieval_{retrieval_setting}",
                retrieval_setting=retrieval_setting,
                source="stand-in",
                answer_grade=expected_status,
            )
            records.append(rec)

    return records


async def run_all_benchmarks():
    """Executes the full suite of 5 benchmark scenarios and 3 retrieval evaluations,

    persisting all granular rows directly into results/results.csv.
    """
    print("=" * 60)
    print("STARTING COMPLETE THE RELAY BENCHMARK & EXPERIMENT SUITE")
    print("=" * 60)

    init_app_state()
    rate_limiter = state["rate_limiter"]
    orig_rpm = rate_limiter.rpm_limit
    orig_tpm = rate_limiter.tpm_limit

    questions_data = load_questions()
    sample_questions = [q["question"] for q in questions_data]
    generator = LoadGenerator(app)
    all_records: List[BenchmarkRecord] = []

    # -------------------------------------------------------------
    # PART 1: RETRIEVAL EXPERIMENTS (Settings A, B, and C)
    # -------------------------------------------------------------
    print("\n--> [Part 1/2] Running Retrieval Experiments (Settings A, B, C)...")
    # Provide generous headroom during retrieval accuracy evaluation
    rate_limiter.rpm_limit = 500
    rate_limiter.tpm_limit = 200000

    for setting in ["A", "B", "C"]:
        print(f"  * Evaluating Retrieval Configuration {setting}...")
        recs = await run_evaluation_set(retrieval_setting=setting)
        all_records.extend(recs)
        print(f"    Completed {len(recs)} evaluations for Setting {setting}")

    # -------------------------------------------------------------
    # PART 2: THE FIVE BENCHMARK SCENARIOS
    # -------------------------------------------------------------
    print("\n--> [Part 2/2] Running 5 Benchmark Load Scenarios...")
    rate_limiter.rpm_limit = orig_rpm
    rate_limiter.tpm_limit = orig_tpm

    # Scenario 1: Steady Traffic
    print("  1. Scenario 1: Steady Traffic (concurrency=2, N=20, 100ms spacing)")
    rate_limiter._minute_window.clear()
    s1_recs = await generator.run_scenario(
        scenario_name="steady_traffic",
        questions=sample_questions,
        num_requests=20,
        concurrency=2,
        interval_ms=100,
        retrieval_setting="B",
    )
    all_records.extend(s1_recs)
    print(f"     Recorded {len(s1_recs)} requests.")

    # Scenario 2: Sudden Burst
    print("  2. Scenario 2: Sudden Burst (concurrency=12, N=24, instantaneous)")
    rate_limiter._minute_window.clear()
    s2_recs = await generator.run_scenario(
        scenario_name="sudden_burst",
        questions=sample_questions,
        num_requests=24,
        concurrency=12,
        interval_ms=0,
        retrieval_setting="B",
    )
    all_records.extend(s2_recs)
    print(f"     Recorded {len(s2_recs)} requests.")

    # Scenario 3: Sustained Overload
    print("  3. Scenario 3: Sustained Overload (concurrency=10, N=35, continuous)")
    s3_recs = await generator.run_scenario(
        scenario_name="sustained_overload",
        questions=sample_questions,
        num_requests=35,
        concurrency=10,
        interval_ms=20,
        retrieval_setting="B",
    )
    all_records.extend(s3_recs)
    print(f"     Recorded {len(s3_recs)} requests.")

    # Scenario 4: Per-minute limit halved
    print("  4. Scenario 4: Per-minute limit halved (RPM halved to 15, N=20)")
    rate_limiter.rpm_limit = max(1, orig_rpm // 2)
    try:
        s4_recs = await generator.run_scenario(
            scenario_name="rpm_halved",
            questions=sample_questions,
            num_requests=20,
            concurrency=5,
            interval_ms=50,
            retrieval_setting="B",
        )
        all_records.extend(s4_recs)
        print(f"     Recorded {len(s4_recs)} requests.")
    finally:
        rate_limiter.rpm_limit = orig_rpm

    # Scenario 5: Daily Budget Exhausted
    print("  5. Scenario 5: Daily Budget Exhausted (forcing daily exhaustion, N=15)")
    orig_daily_reqs = rate_limiter._daily_requests
    rate_limiter._daily_requests = rate_limiter.daily_request_budget
    try:
        s5_recs = await generator.run_scenario(
            scenario_name="daily_budget_exhausted",
            questions=sample_questions,
            num_requests=15,
            concurrency=3,
            interval_ms=30,
            retrieval_setting="B",
        )
        all_records.extend(s5_recs)
        print(f"     Recorded {len(s5_recs)} requests.")
    finally:
        rate_limiter._daily_requests = orig_daily_reqs

    # -------------------------------------------------------------
    # WRITE RAW RESULTS TO results/results.csv
    # -------------------------------------------------------------
    rows = [r.dict() for r in all_records]
    df = pd.DataFrame(rows)

    mandatory_cols = [
        "request_id",
        "scenario",
        "source",
        "start_time",
        "first_token_ms",
        "total_ms",
        "status",
        "retries",
        "tokens_in",
        "tokens_out",
        "retrieval_setting",
        "answer_grade",
    ]
    other_cols = [c for c in df.columns if c not in mandatory_cols]
    final_df = df[mandatory_cols + other_cols]

    settings.RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(settings.RESULTS_CSV, index=False)
    print(f"\n[SUCCESS] Successfully written {len(final_df)} raw benchmark records to {settings.RESULTS_CSV}")

    # Run automated analysis script
    from benchmarks.analyze_results import analyze_and_export_reports
    analyze_and_export_reports()


if __name__ == "__main__":
    asyncio.run(run_all_benchmarks())
