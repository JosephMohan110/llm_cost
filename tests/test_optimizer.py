from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from llm_cost_optimizer.config import Settings
from llm_cost_optimizer.pricing import PriceRates, calculate_cost_usd
from llm_cost_optimizer.router import choose_model
from llm_cost_optimizer.service import ContextDocument, LLMCostOptimizedFeature
from llm_cost_optimizer.storage import BudgetExceeded
from scripts.run_evaluation import load_eval_cases


class FakeResponses:
    def __init__(self, answer: str = "Use Forgot password to reset your password."):
        self.answer = answer
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_text=self.answer,
            usage=SimpleNamespace(
                input_tokens=120,
                output_tokens=12,
                input_tokens_details=SimpleNamespace(cached_tokens=20, cache_write_tokens=0),
            ),
        )


class FakeClient:
    def __init__(self, answer: str = "Use Forgot password to reset your password."):
        self.responses = FakeResponses(answer)


class CostOptimizerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp.name) / "test.sqlite3")
        self.settings = Settings(db_path=self.db_path, daily_budget_usd=5.0, cache_ttl_seconds=3600)
        self.client = FakeClient()
        self.feature = LLMCostOptimizedFeature(self.settings, client=self.client)

    def tearDown(self):
        self.temp.cleanup()

    def test_cost_math_splits_cached_tokens(self):
        rates = PriceRates(1.0, 0.1, 1.25, 5.0)
        cost = calculate_cost_usd(
            input_tokens=1000, cached_input_tokens=400, cache_write_input_tokens=100,
            output_tokens=200, rates=rates,
        )
        expected = (500 * 1.0 + 400 * 0.1 + 100 * 1.25 + 200 * 5.0) / 1_000_000
        self.assertAlmostEqual(cost, expected, places=12)

    def test_simple_query_uses_cheap_model(self):
        decision = choose_model("How do I reset my password?", "Short FAQ", self.settings)
        self.assertEqual(decision.model, self.settings.cheap_model)

    def test_complex_query_uses_strong_model(self):
        decision = choose_model("Compare options and design a system architecture", "Short context", self.settings)
        self.assertEqual(decision.model, self.settings.strong_model)

    def test_identical_request_uses_exact_response_cache(self):
        first = self.feature.answer(
            "How do I reset my password?",
            [ContextDocument("Choose Forgot password and use the reset link.", "faq", 1.0)],
            cache_scope="tenant-test",
        )
        second = self.feature.answer(
            "How do I reset my password?",
            [ContextDocument("Choose Forgot password and use the reset link.", "faq", 1.0)],
            cache_scope="tenant-test",
        )
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(len(self.client.responses.calls), 1)
        self.assertEqual(second.cost_usd, 0.0)

    def test_cache_scope_prevents_cross_scope_hit(self):
        docs = [ContextDocument("Use the password reset workflow.", "faq", 1.0)]
        self.feature.answer("How do I reset my password?", docs, cache_scope="tenant-A")
        other = self.feature.answer("How do I reset my password?", docs, cache_scope="tenant-B")
        self.assertFalse(other.cache_hit)
        self.assertEqual(len(self.client.responses.calls), 2)

    def test_route_policy_version_invalidates_cached_answer(self):
        docs = [ContextDocument("Use the password reset workflow.", "faq", 1.0)]
        self.feature.answer("How do I reset my password?", docs, cache_scope="tenant-A")
        other_client = FakeClient()
        other_feature = LLMCostOptimizedFeature(
            Settings(db_path=self.db_path, route_policy_version="support-route-v2"),
            client=other_client,
        )
        result = other_feature.answer(
            "How do I reset my password?", docs, cache_scope="tenant-A"
        )
        self.assertFalse(result.cache_hit)
        self.assertEqual(len(other_client.responses.calls), 1)

    def test_duplicate_context_passages_are_sent_once(self):
        docs = [
            ContextDocument("The policy passage.", "faq", 1.0),
            ContextDocument("The policy passage.", "faq", 0.5),
        ]
        self.feature.answer("What is the policy?", docs, allow_cache=False)
        user_message = self.client.responses.calls[0]["input"][1]["content"]
        self.assertEqual(user_message.count("[faq]"), 1)

    def test_non_finite_document_score_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            self.feature.answer(
                "What is the policy?",
                [ContextDocument("The policy passage.", score=float("nan"))],
                allow_cache=False,
            )
        self.assertEqual(self.client.responses.calls, [])

    def test_evaluation_loader_reads_all_supplied_cases(self):
        cases_path = Path(__file__).resolve().parents[1] / "data" / "eval_cases.jsonl"
        cases = load_eval_cases(cases_path)
        self.assertEqual(len(cases), 10)

    def test_evaluation_loader_rejects_empty_dataset(self):
        cases_path = Path(self.temp.name) / "empty.jsonl"
        cases_path.write_text("\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "empty"):
            load_eval_cases(cases_path)

    def test_budget_blocks_call_before_provider_request(self):
        settings = Settings(db_path=self.db_path, daily_budget_usd=0.00000001)
        client = FakeClient()
        feature = LLMCostOptimizedFeature(settings, client=client)
        with self.assertRaises(BudgetExceeded):
            feature.answer("How do I reset my password?", allow_cache=False)
        self.assertEqual(len(client.responses.calls), 0)

    def test_actual_usage_is_recorded(self):
        result = self.feature.answer("How do I reset my password?", allow_cache=False)
        self.assertEqual(result.input_tokens, 120)
        self.assertEqual(result.cached_input_tokens, 20)
        self.assertEqual(result.output_tokens, 12)
        self.assertGreater(result.cost_usd, 0)
        self.assertFalse(result.usage_estimated)
        summary = self.feature.usage_summary()
        self.assertEqual(summary["api_calls"], 1)
        self.assertEqual(summary["estimated_usage_calls"], 0)

    def test_cache_requires_explicit_scope(self):
        with self.assertRaisesRegex(ValueError, "cache_scope"):
            self.feature.answer("How do I reset my password?")

    def test_query_limit_rejects_before_provider_call(self):
        settings = Settings(db_path=self.db_path, max_query_chars=10)
        client = FakeClient()
        feature = LLMCostOptimizedFeature(settings, client=client)
        with self.assertRaisesRegex(ValueError, "at most 10"):
            feature.answer("This query is too long", allow_cache=False)
        self.assertEqual(client.responses.calls, [])

    def test_routing_kill_switch_uses_strong_model(self):
        settings = Settings(db_path=self.db_path, routing_enabled=False)
        client = FakeClient()
        feature = LLMCostOptimizedFeature(settings, client=client)
        result = feature.answer("How do I reset my password?", allow_cache=False)
        self.assertEqual(result.model, settings.strong_model)
        self.assertEqual(result.route_reason, "Model routing disabled; using the strong model")

    def test_missing_provider_usage_is_marked_as_estimated(self):
        class NoUsageResponses:
            def create(self, **kwargs):
                self.request_options = kwargs
                return SimpleNamespace(output_text="Answer without usage metadata.")

        feature = LLMCostOptimizedFeature(
            Settings(db_path=self.db_path),
            client=SimpleNamespace(responses=NoUsageResponses()),
        )
        result = feature.answer("How do I reset my password?", allow_cache=False)
        self.assertTrue(result.usage_estimated)
        self.assertGreater(result.input_tokens, 0)
        self.assertGreater(result.output_tokens, 0)
        self.assertEqual(feature.usage_summary()["estimated_usage_calls"], 1)

    def test_transient_provider_failure_is_retried_once_and_recorded(self):
        class RetryResponses:
            def __init__(self):
                self.calls = 0

            def create(self, **_kwargs):
                self.request_options = _kwargs
                self.calls += 1
                if self.calls == 1:
                    raise TimeoutError("temporary timeout")
                return SimpleNamespace(
                    output_text="Answer",
                    usage=SimpleNamespace(
                        input_tokens=20,
                        output_tokens=2,
                        input_tokens_details=SimpleNamespace(cached_tokens=0, cache_write_tokens=0),
                    ),
                )

        client = SimpleNamespace(responses=RetryResponses())
        feature = LLMCostOptimizedFeature(
            Settings(db_path=self.db_path, retry_backoff_seconds=0),
            client=client,
        )
        result = feature.answer("How do I reset my password?", allow_cache=False)
        self.assertEqual(client.responses.calls, 2)
        self.assertEqual(result.retry_count, 1)
        self.assertTrue(result.usage_estimated)
        self.assertEqual(feature.usage_summary()["retry_count"], 1)

    def test_retry_count_is_bounded(self):
        class AlwaysTimeout:
            def __init__(self):
                self.calls = 0

            def create(self, **_kwargs):
                self.request_options = _kwargs
                self.calls += 1
                raise TimeoutError("still unavailable")

        client = SimpleNamespace(responses=AlwaysTimeout())
        feature = LLMCostOptimizedFeature(
            Settings(db_path=self.db_path, max_provider_retries=1, retry_backoff_seconds=0),
            client=client,
        )
        with self.assertRaises(TimeoutError):
            feature.answer("How do I reset my password?", allow_cache=False)
        self.assertEqual(client.responses.calls, 2)

    def test_ambiguous_provider_failure_keeps_budget_reservation(self):
        class RaisingResponses:
            def __init__(self):
                self.calls = 0

            def create(self, **_kwargs):
                self.request_options = _kwargs
                self.calls += 1
                raise TimeoutError("provider request timed out")

        settings = Settings(db_path=self.db_path, daily_budget_usd=0.00012, max_provider_retries=0)
        client = SimpleNamespace(responses=RaisingResponses())
        feature = LLMCostOptimizedFeature(settings, client=client)

        with self.assertRaises(TimeoutError):
            feature.answer("How do I reset my password?", allow_cache=False)
        with self.assertRaises(BudgetExceeded):
            feature.answer("How do I reset my password?", allow_cache=False)
        self.assertEqual(client.responses.calls, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
