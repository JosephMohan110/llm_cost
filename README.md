# LLM Cost Optimisation Study — runnable reference implementation

This project demonstrates how to reduce the cost of a customer-support answer feature while placing explicit guardrails around answer quality. It is intentionally small enough to understand, test, and adapt.

## What is included

- The written study is available in `report.md` and `report.docx`.
- Exact-response application caching with TTL and scope-aware cache keys.
- Deterministic routing: short/simple requests go to a cheaper model; longer, complex, or flagged higher-risk requests go to a stronger model.
- Context trimming, exact duplicate passage removal, query-size validation, and an explicit output-token cap.
- Pre-call budget reservation, daily UTC spending cap, and SQLite token/cost/latency ledger.
- Required trusted cache scope, prompt/route-policy-versioned cache keys, and cache/routing rollback switches.
- Explicit provider timeouts, bounded retries for transient failures, and usage-estimate indicators when provider usage is absent.
- A no-API unit test suite using a fake client.
- A paired evaluation runner and sample cases, including fields for blind human review.
- A simple CLI example.

## 1. Requirements

Use Python 3.11 or later. From this folder:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows Command Prompt:
# .venv\Scripts\activate.bat
# macOS/Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Configure

Copy `.env.example` to `.env` and set `OPENAI_API_KEY`. Check that the model names are enabled for your API account. **Update the per-million-token prices to your actual model, processing mode, context tier, and region.** Prices in the example reflect the official OpenAI API pricing page's standard, short-context “All models” entries checked on 2026-10-09; billing can change.

Set `LLM_ROUTING_ENABLED=false` to route requests to the strong model while investigating a quality regression. Set `LLM_CACHE_ENABLED=false` to bypass exact-response caching. Caching requires the caller to provide a trusted tenant/user `cache_scope`; do not pass a scope supplied directly by an unauthenticated user.

For Windows PowerShell, you can also set the key just for the current terminal:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
```

Never put an API key in frontend JavaScript, a Git repository, or a public issue. Do not commit `.env` or the SQLite database.

## 3. Run the example

```bash
python main.py --query "How do I reset my password?" --context-file data/support_faq_example.md --cache-scope "demo-tenant"
```

Run the same command again to exercise exact-response caching. The second call should be served locally until the TTL expires, provided the query, context, prompt version, model route, output cap, and cache scope are unchanged.

The service is a text-only support-answer example. Replace its knowledge input with your existing retrieval pipeline; pass only relevant, access-controlled documents.

## 4. Run tests without API calls

```bash
python -m unittest discover -s tests -v
```

The tests use a fake Responses API client and do not require an API key. They test routing, price arithmetic, scope-aware caching, token-usage accounting, and the daily budget guard.

## 5. Run the evaluation (uses real API calls)

Before running the live evaluation, put representative non-sensitive examples into `data/eval_cases.jsonl` and ensure the daily budget setting is large enough for the baseline plus candidate calls. Then run:

```bash
python -m scripts.run_evaluation
```

The runner calls the strong model for a baseline and the optimized route for a candidate, and writes `data/evaluation_results.csv`. Each row includes a run ID, dataset hash, provider request IDs, route reasons, latency, retry count, and whether usage was estimated. Invalid or empty evaluation datasets stop with a clear error instead of producing a partial report. The keyword-coverage proxy is only a quick regression hint. Complete the human-review columns according to `data/quality_rubric.md` and check the candidate blindly before enabling it in production.

## 6. Key implementation choices

- **Cache key:** query + trimmed context + model + prompt version + route-policy version + context/output caps + `cache_scope`. Pass a stable trusted tenant/customer scope for private data. Change `LLM_PROMPT_VERSION` whenever prompt semantics change, and `LLM_ROUTE_POLICY_VERSION` whenever route semantics change.
- **Model routing:** the included keyword/length heuristic is a starter, not a general reasoning classifier. Replace it with rules learned from evaluation results or a trained complexity classifier; do not accept a route merely because it saved money.
- **Budget:** a request reserves an estimated maximum cost for every configured attempt before calling the provider. The estimate assumes no cached-input discount and includes a modest tokenization margin. Ambiguous failed attempts are conservatively included in settled cost and failed requests remain counted against the daily cap because the provider may have processed a request before a timeout. Daily accounting uses UTC.
- **Cost:** provider input/output usage is used where returned. Missing usage is estimated and exposed as `usage_estimated`; provider fields and cache-write billing can vary, so reconcile against the official usage/billing dashboard.
- **Cache privacy:** SQLite stores answers and a hashed key. Use per-tenant/user cache scopes; set `LLM_CACHE_ENABLED=false` for sensitive or non-repeatable output. Deploy a managed cache/database with access control and encryption for production.
- **Quality:** exact-response cache, smaller-model routing, and shorter context must be gated on a representative evaluation set. The code cannot prove quality equivalence by itself.

## Project tree

```text
llm_cost_optimisation_implementation/
  .env.example
  README.md
  report.md
  main.py
  requirements.txt
  llm_cost_optimizer/
    __init__.py
    config.py
    pricing.py
    router.py
    service.py
    storage.py
    tokens.py
  data/
    eval_cases.jsonl
    quality_rubric.md
    support_faq_example.md
  scripts/run_evaluation.py
  tests/test_optimizer.py
```

## Documentation links

- OpenAI API pricing: https://developers.openai.com/api/docs/pricing
- Prompt caching: https://developers.openai.com/api/docs/guides/prompt-caching
- Model optimization and evaluations: https://developers.openai.com/api/docs/guides/model-optimization
- Batch API FAQ: https://help.openai.com/en/articles/9197833-batch-api-faq
