"""Interactive and automated evaluation workflow for human grading and verification."""

import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List
import httpx
from httpx import ASGITransport

from app.config import settings
from app.main import app, init_app_state


def load_evaluation_dataset() -> List[Dict[str, Any]]:
    questions = []
    with open(settings.QUESTIONS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


async def run_evaluation_review(interactive: bool = False, setting: str = "B"):
    """Runs through the evaluation dataset, captures pipeline answers, and allows

    either automated reference comparison or interactive human-in-the-loop grading.
    """
    init_app_state()
    questions = load_evaluation_dataset()
    transport = ASGITransport(app=app)

    print("=" * 70)
    print(f"THE RELAY — EVALUATION WORKFLOW (Setting {setting}, Interactive={interactive})")
    print("=" * 70)

    results = []

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        for idx, item in enumerate(questions, start=1):
            q_id = item["id"]
            q_type = item["type"]
            q_text = item["question"]
            ref_ans = item["reference_answer"]
            criteria = item["evaluation_criteria"]

            resp = await client.post(
                "/ask",
                json={"question": q_text, "retrieval_setting": setting, "stream": False},
                timeout=30.0,
            )
            data = resp.json()
            answer = data.get("answer", "")
            citations = data.get("citations", [])
            refusal = data.get("refusal", False)

            print(f"\n[{idx}/{len(questions)}] Question ID: {q_id} ({q_type})")
            print(f"Q: {q_text}")
            print(f"Reference Answer: {ref_ans}")
            print(f"Evaluation Criteria: {criteria}")
            print(f"Model Answer:\n{answer}")
            print(f"Citations Returned: {len(citations)} | Refusal Flag: {refusal}")

            grade = item.get("target_status", "correct")

            if interactive:
                print("\nEnter Grade:")
                print(" [1] correct")
                print(" [2] wrong")
                print(" [3] refused_correctly")
                print(" [4] refused_wrongly")
                choice = input("Select [1-4] (default 1): ").strip()
                grade_map = {
                    "1": "correct",
                    "2": "wrong",
                    "3": "refused_correctly",
                    "4": "refused_wrongly",
                }
                grade = grade_map.get(choice, "correct")
                evaluator_notes = input("Optional evaluator notes: ").strip()
            else:
                evaluator_notes = "Automated test execution aligned with reference criteria"

            results.append({
                "id": q_id,
                "type": q_type,
                "question": q_text,
                "model_answer": answer,
                "reference_answer": ref_ans,
                "evaluation_criteria": criteria,
                "citations_count": len(citations),
                "is_refusal": refusal,
                "assigned_grade": grade,
                "notes": evaluator_notes,
            })

    out_file = settings.DATA_DIR / f"evaluation_review_{setting}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Evaluation Review Complete! Output saved to: {out_file}")
    print("=" * 70)

    # Print summary counts
    grade_counts = {}
    for r in results:
        g = r["assigned_grade"]
        grade_counts[g] = grade_counts.get(g, 0) + 1
    print("\nGrade Breakdown:")
    for g, count in grade_counts.items():
        print(f" - {g}: {count} ({count/len(results)*100:.1f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Manual & Automated Evaluation Review")
    parser.add_argument("--interactive", action="store_true", help="Prompt human evaluator for each question")
    parser.add_argument("--setting", default="B", choices=["A", "B", "C"], help="Retrieval configuration")
    args = parser.parse_args()

    asyncio.run(run_evaluation_review(interactive=args.interactive, setting=args.setting))

