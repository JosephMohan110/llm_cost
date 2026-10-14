"""Run a small paired evaluation and write outputs for manual quality review.

The keyword-coverage fields are weak regression proxies, not semantic quality scores.
Review outputs blind with data/quality_rubric.md before deciding whether a model route is acceptable.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from llm_cost_optimizer import ContextDocument, LLMCostOptimizedFeature, Settings

ROOT = Path(__file__).resolve().parents[1]


def keyword_coverage(answer: str, keywords: list[str]) -> str:
    if not keywords:
        return ""
    text = answer.casefold()
    found = sum(keyword.casefold() in text for keyword in keywords)
    return f"{found}/{len(keywords)}"


def load_eval_cases(path: Path) -> list[dict]:
    cases = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(case, dict):
            raise ValueError(f"{path}:{line_number}: each evaluation case must be a JSON object")
        case_id = case.get("id")
        query = case.get("query")
        context = case.get("context")
        must_include = case.get("must_include", [])
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen_ids:
            raise ValueError(f"{path}:{line_number}: case id must be non-empty and unique")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"{path}:{line_number}: query must be a non-empty string")
        if not isinstance(context, str):
            raise ValueError(f"{path}:{line_number}: context must be a string")
        if not isinstance(must_include, list) or not all(isinstance(item, str) for item in must_include):
            raise ValueError(f"{path}:{line_number}: must_include must be a list of strings")
        case["must_include"] = must_include
        seen_ids.add(case_id)
        cases.append(case)
    if not cases:
        raise ValueError(f"Evaluation dataset is empty: {path}")
    return cases


def main() -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running the live evaluation.")

    settings = Settings.from_env()
    optimizer = LLMCostOptimizedFeature(settings)
    cases_path = ROOT / "data" / "eval_cases.jsonl"
    cases = load_eval_cases(cases_path)
    run_id = uuid.uuid4().hex
    dataset_sha256 = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    run_started_at = datetime.now(timezone.utc).isoformat()
    output_path = ROOT / "data" / "evaluation_results.csv"
    rows = []
    for case in cases:
        docs = [ContextDocument(case["context"], source=f"eval-{case['id']}", score=1.0)]
        # Separate scopes, disabled answer cache: baseline and candidate always call the provider.
        baseline = optimizer.answer(case["query"], docs, cache_scope=f"eval-baseline-{case['id']}", allow_cache=False,
                                    force_model=settings.strong_model)
        candidate = optimizer.answer(case["query"], docs, cache_scope=f"eval-candidate-{case['id']}", allow_cache=False)
        rows.append({
            "run_id": run_id,
            "run_started_at_utc": run_started_at,
            "dataset_sha256": dataset_sha256,
            "case_id": case["id"],
            "query": case["query"],
            "baseline_model": baseline.model,
            "baseline_route_reason": baseline.route_reason,
            "baseline_latency_ms": f"{baseline.latency_ms:.3f}",
            "baseline_retry_count": baseline.retry_count,
            "baseline_usage_estimated": baseline.usage_estimated,
            "baseline_request_id": baseline.request_id,
            "baseline_cost_usd": f"{baseline.cost_usd:.10f}",
            "baseline_input_tokens": baseline.input_tokens,
            "baseline_output_tokens": baseline.output_tokens,
            "baseline_keyword_coverage_proxy": keyword_coverage(baseline.answer, case.get("must_include", [])),
            "baseline_answer": baseline.answer,
            "candidate_model": candidate.model,
            "candidate_route_reason": candidate.route_reason,
            "candidate_latency_ms": f"{candidate.latency_ms:.3f}",
            "candidate_retry_count": candidate.retry_count,
            "candidate_usage_estimated": candidate.usage_estimated,
            "candidate_request_id": candidate.request_id,
            "candidate_cost_usd": f"{candidate.cost_usd:.10f}",
            "candidate_input_tokens": candidate.input_tokens,
            "candidate_output_tokens": candidate.output_tokens,
            "candidate_keyword_coverage_proxy": keyword_coverage(candidate.answer, case.get("must_include", [])),
            "candidate_answer": candidate.answer,
            "human_score_baseline_1_to_5": "",
            "human_score_candidate_1_to_5": "",
            "critical_error_found": "",
            "review_notes": "",
        })
        print(f"Completed {case['id']} ({len(rows)}/{len(cases)})")

    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if len(rows) != len(cases):
        raise RuntimeError(f"Evaluation run incomplete: wrote {len(rows)} of {len(cases)} cases")
    print(f"Wrote: {output_path}")
    print("Next: complete the human-review columns using data/quality_rubric.md before accepting the optimization.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
