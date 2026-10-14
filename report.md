# LLM Cost Optimisation Study
## Cut the cost of an LLM feature without materially hurting quality

**Prepared:** 9 October 2026  
**Implementation language:** Python  
**Reference feature:** retrieval-grounded customer-support answer generation  
**Provider used in code:** OpenAI Responses API, with model names and prices configurable

---

## 1. Executive summary

The objective is not simply to spend less per API call. It is to reduce **cost per successful task** while keeping factuality, completeness, safety, and user satisfaction within predefined limits. A cheaper model that causes incorrect answers, repeat contacts, or human escalations can increase total operating cost.

The recommended first implementation combines five controls:

1. **Exact-response caching** for genuinely identical requests, using a cache key that includes the query, relevant context, model, prompt version, output limit, and tenant/customer scope.
2. **Evaluation-gated model routing**: use a low-cost model for short, routine requests and a stronger model for complex requests. The included router is deliberately transparent and must be tuned against representative examples.
3. **Context selection and trimming**: send the few relevant retrieved passages, remove duplicates, and cap the context token budget instead of passing an entire knowledge base or conversation.
4. **Output limits and concise instructions**: control unnecessary response length with a maximum output-token budget and a task-appropriate answer format.
5. **Observability and a budget guard**: record provider usage, estimated USD cost, chosen model, cache hits, response latency, and failures; reserve an estimated maximum cost before starting a call.

The implementation includes working Python modules, a SQLite ledger/cache, a CLI, unit tests with a mocked provider, evaluation examples, and a paired evaluation script. The code provides a safe starting point, not a proof of production readiness. A real launch should be gated on a blind quality evaluation and a canary rollout.

### Proposed quality guardrails

Use a fixed evaluation set with examples representative of real traffic. As a reasonable initial target, reject an optimization if the candidate's task-success or groundedness score falls by more than **0.1 on a 1–5 scale** versus the baseline, if any critical error is introduced, or if important request categories degrade. These are starting thresholds rather than statistical guarantees; choose stricter thresholds and larger samples for high-impact use cases. The quality rubric in `data/quality_rubric.md` provides the review criteria.

---

## 2. Problem framing and success metrics

### 2.1 Define the feature boundary

This study uses a support-answer feature as a concrete example:

**Input:** user question + a small set of retrieved support-policy passages.  
**Output:** a concise answer grounded in that material.  
**Out of scope:** the retrieval system itself, a frontend, autonomous tool execution, and safety-critical decision-making.

If your feature is summarization, coding assistance, extraction, or a multi-step agent, the same measurement framework applies, but routing thresholds, evaluation labels, and safety checks will differ.

### 2.2 The primary metric

Track **cost per successful task**, not only cost per call:

`Cost per successful task = total LLM + relevant infrastructure cost / number of tasks passing the quality criterion`

Also track:

- cost per request and per active user/tenant;
- input, cached-input, and output tokens by route/model;
- answer acceptance or task-success rate;
- groundedness/factuality and critical-error rate;
- cache-hit rate and percentage of provider calls avoided;
- p50 and p95 latency;
- retry, timeout, empty-output, escalation, and human-handoff rates;
- output length and user re-contact rate;
- usage by feature, endpoint, tenant, and prompt/model version.

A low per-call cost can still be a bad change if it increases re-contacts or human review. The business metric should include those effects where you can reliably measure them.

### 2.3 Baseline before changing anything

For at least one representative usage window (ideally a week, or longer for infrequent tasks), record:

- number of requests and distribution of input/output tokens;
- model/endpoint and applicable processing tier;
- actual provider invoice or usage-dashboard cost;
- latency percentiles and error rate;
- a sample of outputs scored against a fixed rubric;
- task outcomes (accepted, corrected, escalated, abandoned, or retried).

Do not infer average token use from character count alone. Use API-reported usage for post-call billing and tokenization only for planning and context limits.

---

## 3. Cost model and illustrative savings

### 3.1 Cost formula

For a text-only request, a useful approximation is:

`request cost = uncached input tokens × input rate + cached-read tokens × cached-input rate + cache-write tokens × cache-write rate + output tokens × output rate`

Here, `uncached input tokens = total input tokens − cached-read tokens − cache-write tokens` (never below zero). Do not add the cached-read tokens on top of the total input count; they are a priced subset of input usage.

Divide each token count by 1,000,000 when the rates are listed per million tokens. Your provider may have extra charges for tools, image/audio input, web search, vector storage, embeddings, batch processing, fine-tuning, data-residency regions, or priority tiers. The ledger in this reference code focuses on text-token charges and should be reconciled with billing records.

### 3.2 Current reference rates used by the example

The example defaults use the official OpenAI API pricing page's **standard / short-context / “All models”** rows checked on 9 October 2026. Rates are USD per 1 million tokens:

| Example model | Input | Cached input | Cache write | Output |
|---|---:|---:|---:|---:|
| `gpt-6-luna` (low-cost route) | $0.05 | $0.005 | $0.0625 | $0.25 |
| `gpt-6.1-sol` (strong baseline/route) | $1.00 | $0.05 | $1.25 | $5.00 |

Source: [OpenAI API pricing](https://developers.openai.com/api/docs/pricing). These defaults are configuration examples, not a guarantee that the model names are enabled for every account or that your final invoice matches these rates. Confirm your account's current models, context tier, processing mode, region, and discounts before a production decision. If you use another provider, replace the rate configuration and usage extraction.

### 3.3 Monthly example: impact of several levers

Assume **10,000 requests/month**, **2,000 input tokens/request**, and **500 output tokens/request**. The table below uses only text-token charges, the rates above, and these assumptions:

- Baseline sends every request to the strong model.
- Routing sends 75% of traffic to the low-cost model and 25% to the strong model.
- The third row trims average billed input from 2,000 to 1,000 tokens and average output from 500 to 300 tokens.
- The final row assumes 10% of remaining candidate requests are exact-cache hits with no provider call.

| Scenario | Approximate monthly token cost | Change vs baseline |
|---|---:|---:|
| All requests use strong model | $45.00 | Baseline |
| Route 75% low-cost / 25% strong | $12.94 | 71.3% lower |
| Routing + 50% fewer input tokens + 40% fewer output tokens | $7.19 | 84.0% lower |
| Previous row + 10% exact-cache hit rate | $6.47 | 85.6% lower |

These are arithmetic scenarios, not measured results or savings promises. They exclude infrastructure, embeddings, tool calls, cache writes, safety layers, re-contact costs, and quality failures. A real cache hit rate depends on how often inputs are genuinely identical; semantic caching is intentionally not enabled in the reference implementation because similar requests are not necessarily interchangeable.

**Important conclusion:** routing all traffic to the cheapest model creates the most obvious token-price saving in this example, but it is not the recommended rollout by itself. The preferred candidate is a mixed route whose task quality has been measured against the strong baseline.

---

## 4. Which optimizations to implement first

### 4.1 Exact-response application cache — high priority

**What it does:** a repeat of the exact same question and context can return a previous response without making another model call. It saves all token charges for that repeat, not just input tokens.

**Implementation rules:**

- Key on normalized question, the exact context that was sent, chosen model, prompt version, output limit, and tenant/user scope.
- Use a TTL; expire entries when source policy changes or when the TTL is reached.
- Increment the prompt version when the prompt meaning or output contract changes.
- Use scope values such as `tenant:123` or a user-specific key for private data. Do not share across tenants by default.
- Turn caching off for sensitive, rapidly changing, non-repeatable, or user-specific outputs where correctness/privacy requires it.
- Cache only successful, validated answers. The example does not cache API errors or empty responses.

**Trade-off:** a low cache hit rate produces little benefit; aggressive normalization can accidentally merge requests that should be different. This code uses conservative exact matching and keeps scope in the key.

### 4.2 Model routing — high priority after evaluation

**What it does:** assign each request to the least expensive model that meets the quality target for that request class.

Starter route:

- Low-cost model: short, single-task, low-ambiguity requests with limited context.
- Strong model: longer prompts, multiple questions, explicit analysis/design/debugging tasks, or requests that match configured higher-risk keywords.

The included heuristic is transparent, deterministic, and easy to test. It is **not** a dependable universal measure of complexity and is not a safety policy. Treat it as a prototype. Use evaluation results to add or remove rules, or train a lightweight request classifier if request volume justifies it. Never use a model's self-reported confidence as the only routing criterion.

**Quality safeguards:**

- test each traffic category separately;
- track failure rate by route and model;
- use the strong model for uncertain categories until evidence supports downgrading them;
- keep a kill switch to route all requests back to the baseline;
- do not use a stronger model as a substitute for domain-specific safety controls.

### 4.3 Context trimming and retrieval quality — high priority

Input tokens include system/developer instructions, conversation history, retrieved documents, tool schemas, and the question. Large prompts can be expensive and can also distract the model.

Implement in this order:

1. Retrieve only passages likely to answer the current question.
2. Deduplicate identical or near-duplicate passages before prompting.
3. Prefer a compact authoritative source over multiple redundant copies.
4. Sort by relevance and keep the best material within a token budget.
5. Remove unrelated conversation history or summarize it into a small, verified state object.
6. Inspect difficult evaluation examples to ensure truncation did not remove a qualification, date, exception, or cited evidence.

The reference implementation accepts ranked `ContextDocument` items and caps the context. It uses `tiktoken` when installed; model-specific tokenizer mismatches and message framing mean its pre-call counts are estimates. Actual billed usage is read from the provider response where available.

### 4.4 Output limits and concise response design — high priority

Set `max_output_tokens` to the smallest value that lets the feature complete its task. Use different limits for short classification/extraction and longer explanations rather than one large cap for every endpoint.

A prompt should specify:

- the exact task and expected audience;
- what evidence it may use;
- what to do when evidence is missing;
- a concise format/length target;
- any required fields, citations, or refusal/escalation behavior.

Measure truncation, completeness, and re-contact rates before lowering the cap. Do not clip arbitrary final characters from a generated answer; limit generation and validate the completed result instead.

### 4.5 Provider prompt caching — useful only when prefixes repeat

Prompt caching is different from application-level exact-response caching. Provider prompt caching reuses a repeated **input prefix** and reduces the price of eligible cached input tokens; it does not return a prior answer and does not eliminate output-token charges. OpenAI's current prompt-caching guide says eligible GPT-5.6-and-later prompts need at least 1,024 visible input tokens; the rate and cache-write cost depend on model and processing mode. See [Prompt caching documentation](https://developers.openai.com/api/docs/guides/prompt-caching).

Practical steps:

- keep stable developer instructions and tool definitions at the beginning of the request;
- avoid changing the static prefix unnecessarily;
- place dynamic data after the reusable prefix where practical;
- inspect actual cached-token usage rather than assuming a hit;
- do not add useless text just to reach the cacheable length threshold.

The code uses a stable developer prompt, but a short static prompt alone may not be long enough for provider-side prompt caching. Exact-answer caching is the main caching optimization demonstrated by this code.

### 4.6 Batch processing — high priority for non-interactive work

If results do not need to be returned immediately (offline classification, document processing, nightly backfills, or evaluation jobs), test the provider's batch mode. The official OpenAI Batch API FAQ currently describes a 50% cost discount for eligible models and processing within a 24-hour window. Check model eligibility and the current service terms at [Batch API FAQ](https://help.openai.com/en/articles/9197833-batch-api-faq).

Do not route a synchronous user chat through batch unless its latency contract permits that delay. Batch is a separate pathway; it is not included in the online reference service.

### 4.7 Later-stage options

- **Semantic cache:** only after a domain-specific similarity threshold and false-hit evaluation. A semantic match can be wrong when dates, user intent, permissions, or account state differ.
- **Fine-tuning or task-specific model:** consider when the task is stable, high-volume, and labelled examples exist. Compare training/inference/maintenance cost with prompt/routing alternatives.
- **Tool-call reduction:** avoid unnecessary multi-step agents, redundant retrieval calls, repeated summarization, and expensive tools when deterministic code or a database query can answer reliably.
- **Prompt compression:** rewrite, do not blindly delete. Keep policy constraints and evidence needed for answer correctness.
- **Fallback/escalation:** use only if the failure signal is validated. A second model call on every request can erase savings; a model saying “confidence 0.9” is not a calibrated failure detector.

---

## 5. Reference architecture and request flow

```text
Incoming question + ranked context documents
        |
        v
Validate input / establish tenant cache scope
        |
        v
Transparent route decision (cheap or strong model)
        |
        v
Trim context to a fixed token budget
        |
        v
Build cache key (question + context + model + prompt version + scope)
        |                         \
        | cache hit                \ cache miss
        v                           v
Return stored answer          Estimate maximum cost
                                    |
                                    v
                           Reserve daily budget (SQLite)
                                    |
                                    v
                           Call Responses API once
                                    |
                                    v
                   Read usage / calculate token charge / log latency
                                    |
                                    v
                        Settle ledger + cache successful answer
                                    |
                                    v
                                 Return
```

The cache lookup occurs before budget reservation. This means an exact cache hit avoids the provider call and should not consume token budget. The daily ledger uses UTC days, and its SQLite reservation uses a transaction to reduce overspend under concurrent requests in a single local database. For a multi-instance production service, replace SQLite with a shared transactional store or centralized quota service.

---

## 6. Code implementation walkthrough

### 6.1 Configuration and model rates

`llm_cost_optimizer/config.py` reads model names, price rates, output cap, context cap, cache TTL, database path, prompt version, and daily budget from environment variables. Pricing belongs in configuration because model and service prices change.

The example price fields are input, cached input, cache-write input, and output rates in USD per million tokens. If you swap models, update all relevant rates. `rates_for()` maps the configured cheap model to the cheap rate set and the configured strong model (or forced evaluation model) to the strong set.

### 6.2 Routing

`llm_cost_optimizer/router.py` returns the model and a reason. The reason is logged with the result for analysis. The current heuristic routes longer/multi-question/complex requests to the strong model and short requests to the cheap model. Its threshold values are configurable.

**Production adaptation:** replay evaluation examples through the router, print the route for each case, and inspect false downgrades. Add a category-specific route when a task fails the quality gate. Do not tune thresholds just to maximize the cheap-model share.

### 6.3 Context budget

`llm_cost_optimizer/tokens.py` counts and truncates tokens using `tiktoken` where available; otherwise it uses a rough character approximation. The approximation is only for context planning. API usage fields, not that approximation, should drive final usage reporting.

The service sorts documents by score, adds source labels, and stops when the context cap is reached. In a real retrieval pipeline, use a relevance score that has been checked on your own documents; the highest numeric score is not necessarily the most trustworthy policy source.

### 6.4 Exact cache

`llm_cost_optimizer/service.py` constructs a SHA-256 cache key from the request data and serving configuration. The database stores the answer and minimal metadata; it does not store the raw query as a separate key. The cache TTL is configurable.

Changing the prompt version or output limit automatically produces a cache miss. Changes to the underlying context also produce a miss. For personalized or tenant-isolated content, pass an explicit scope on every call, for example:

```python
result = feature.answer(
    user_question,
    documents,
    cache_scope=f"tenant:{tenant_id}",
)
```

Never trust a tenant ID supplied by an unauthenticated client; derive it from your authenticated server-side session.

### 6.5 Cost reservation and usage ledger

Before calling the provider, the service estimates prompt tokens, adds a 15% framing margin, and reserves the estimated maximum cost using the configured output limit. This is intentionally conservative and assumes no cached-input discount before the call.

After the response, the service records actual API-reported input/output token counts, cached-input counts when reported, calculated token cost, chosen model, and latency. If the call raises an exception, it is marked failed but its estimated reservation remains counted by the daily budget guard: a timeout can happen after the provider has processed and billed a request. Displayed spend includes settled calls only, so reconcile failed attempts and local estimates with provider billing dashboards. Cache-write usage or special processing surcharges may not be exposed by every SDK response shape.

The daily cap is a local application guard in USD, not a guarantee about the provider account's organization-wide spending. Ambiguous failed calls can conservatively consume the local cap until the UTC day rolls over; this favors safety over availability. Also configure provider-side spend/rate limits and alerting.

### 6.6 Responses API and error path

The request sends a stable developer instruction plus the user question and trimmed context. It caps `max_output_tokens`, checks for empty output, and only writes successful answers to the application cache. The sample is intentionally a single model call per cache miss; it does not blindly call a second model to grade every output.

Add production policies for timeouts, provider retries, rate limits, trace IDs, request cancellation, and fallback behavior. Any retry policy should avoid unbounded duplicate cost, especially when an upstream timeout happens after the provider has already processed a request.

---

## 7. How to implement this in an existing application

### Phase A — establish evidence (1–3 days depending on data access)

1. Instrument the current call site and capture token usage/model/cost/latency from provider responses.
2. Build a representative evaluation dataset from real queries, with personal and confidential data removed or protected.
3. Add labels: expected answer facts, acceptable uncertainty, critical wrong answers, and whether a human should handle the request.
4. Calculate the current cost per request and cost per successful task by traffic category.

### Phase B — add low-risk savings (1–3 days)

1. Put the LLM call behind one server-side service function.
2. Add a stable prompt version and bounded output tokens.
3. Keep only relevant context documents, deduplicate, and impose a token budget.
4. Add exact-response cache with TTL, tenant scope, and a cache-disable switch.
5. Record usage and compare application estimates with provider usage reports.

### Phase C — implement routing (2–5 days depending on feature complexity)

1. Make the strong model the baseline for the evaluation set.
2. Run the candidate router on the same examples and compare outputs.
3. Score blindly using the rubric; inspect every severe regression and each category with meaningful score loss.
4. Tune or restrict routes until the quality gate passes.
5. Add automated tests for route boundaries, long contexts, ambiguous requests, and policy exceptions.

### Phase D — canary and productionize (1–2 weeks of controlled observation)

1. Release to a small traffic slice or internal users.
2. Monitor task success, re-contact/handoff rate, critical errors, token cost, cache hits, p95 latency, and route distribution.
3. Compare with the baseline cohort, controlling for traffic mix where possible.
4. Roll back the model route independently from cache/context changes if a regression appears.
5. Increase traffic only after the guardrails pass; retain a one-flag route-to-strong-model rollback.

The day estimates are planning guidance, not a promise; actual effort depends on dataset readiness, integration architecture, and review requirements.

---

## 8. Evaluation protocol: prove quality before saving money

### 8.1 Dataset design

Create a fixed set with:

- frequent/simple examples;
- uncommon and long-tail queries;
- ambiguous and multi-intent requests;
- missing or contradictory context;
- stale policy/date boundaries;
- prompt-injection content embedded in retrieved documents;
- privacy-sensitive and cross-tenant cases;
- previously escalated requests and known failures.

Split development data from a held-out test set. Do not tune thresholds on the final holdout over and over; that causes overfitting to the evaluation set.

### 8.2 Paired baseline/candidate comparison

Run each evaluation query through:

1. **Baseline:** strong model, existing prompt/context policy.
2. **Candidate:** optimized prompt/context and model route.
3. Optional ablations: cache only, routing only, context trimming only, and output cap only.

Ablations show which change causes a cost or quality difference. Record input/output tokens, model, latency, actual cost, and route reason per case. Cache hits should be measured in a separate replay because an exact hit hides how the candidate model itself would answer a first-time question.

### 8.3 Quality checks

Use a combination of:

- deterministic assertions for required facts, JSON schemas, citations, and policy constraints;
- human blind ratings for task success, completeness, clarity, and groundedness;
- an independent grader where useful, validated against human judgments;
- explicit review of safety, privacy, and critical errors;
- confidence intervals or paired statistical tests when sample size is sufficient.

The included evaluation code performs paired live calls and fills a CSV with outputs and cost. Its keyword coverage is only a weak proxy, not a semantic judge. It leaves human-rating fields blank by design; complete the review rubric before accepting the route change.

### 8.4 Release gate

Accept an optimization only if:

- cost per successful task improves enough to matter;
- task-success and groundedness remain within pre-agreed non-inferiority limits;
- no unacceptable critical-error or safety regression is introduced;
- p95 latency and timeout rates remain acceptable;
- cache isolation tests pass;
- logs and rollback controls work;
- the provider invoice reconciles with local estimates within a defined tolerance.

---

## 9. Testing and operating the included code

### Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Add your API key and review the model/rate settings in `.env`, then run:

```bash
python main.py --query "How do I reset my password?" --context-file data/support_faq_example.md --cache-scope "demo-tenant"
```

### Run unit tests (no API key required)

```bash
python -m unittest discover -s tests -v
```

The mocked tests cover rate arithmetic, model selection, exact cache reuse, cache scope isolation, budget refusal before provider calls, and returned usage accounting.

### Run paired evaluation (real API calls, uses budget)

```bash
python -m scripts.run_evaluation
```

This makes baseline and candidate API calls. Use a dedicated test project/API key and raise the daily budget only as needed for the evaluation. The generated CSV is placed in `data/evaluation_results.csv`. Review it using `data/quality_rubric.md`.

---

## 10. Risks and mitigations

| Risk | Why it matters | Mitigation |
|---|---|---|
| Smaller model misses intent | Lower price can reduce answer quality | Evaluation-gated routing; keep strong route for failed categories |
| Context trim removes a key exception | Can create a confident but incorrect answer | Rank authoritative sources, keep source labels, test dates/exceptions, inspect truncation cases |
| Stale exact-cache response | Policies or state may have changed | TTL, prompt version, source version in context, cache invalidation, disable for volatile state |
| Cross-tenant cache leakage | Cache can expose a prior user's answer | Include authenticated tenant/user scope in cache key; test isolation |
| Heuristic router overfits | Keywords do not represent task difficulty reliably | Route by evaluated categories, measure per-segment regressions, keep rollback |
| Budget estimate is imperfect | Token framing/prices can change | Conservative reservation, provider-side spend limits, periodic rate refresh and invoice reconciliation |
| Database contention or loss | SQLite is not a distributed quota/cache | For multiple instances, use shared Redis/Postgres/centralized quota with transactions and access control |
| Evaluation proxy is weak | Keyword match cannot grade reasoning or factuality | Blind human review, validated graders, critical-error review, held-out tests |
| Cached sensitive content | Stored outputs might contain personal or confidential information | Minimize data, tenant isolation, encrypted storage, retention policy, opt-out for sensitive requests |
| Retries duplicate work | Ambiguous timeouts may cause repeated charges | Bound retries, use request IDs/tracing, monitor timeout-after-success patterns |

---

## 11. Recommended delivery checklist

- [ ] Record baseline cost, quality, and latency before changes.
- [ ] Confirm current model availability and pricing in the account.
- [ ] Create a representative, held-out evaluation dataset.
- [ ] Add input/context limits and a feature-appropriate output cap.
- [ ] Implement exact caching with versioned and tenant-aware keys.
- [ ] Add model routing and log the decision reason.
- [ ] Add usage-ledger metrics and provider-side budget/alert controls.
- [ ] Run unit tests and the paired baseline/candidate evaluation.
- [ ] Blind-review quality and resolve regressions before rollout.
- [ ] Canary deploy with per-route monitoring and a one-step rollback.
- [ ] Reconcile ledger estimates with actual provider billing.
- [ ] Review results monthly or after model/prompt/pricing changes.

## 12. Conclusion

Start with measurement, exact caching, bounded context/output, and a visible cost ledger. Add smaller-model routing only after a paired evaluation shows that routine traffic keeps its quality. Use provider prompt caching and batch processing where their prefix/latency requirements fit the workload. Keep quality guardrails, privacy boundaries, a provider-side spending limit, and a fast rollback path in place.

The intended success criterion is **lower cost per successfully completed task with no material regression in quality**, not the maximum possible percentage of traffic on the cheapest model.

---

## References

1. OpenAI API pricing, accessed 9 October 2026: https://developers.openai.com/api/docs/pricing
2. OpenAI Prompt caching guide, accessed 9 October 2026: https://developers.openai.com/api/docs/guides/prompt-caching
3. OpenAI Model optimization guide, accessed 9 October 2026: https://developers.openai.com/api/docs/guides/model-optimization
4. OpenAI Batch API FAQ, accessed 9 October 2026: https://help.openai.com/en/articles/9197833-batch-api-faq
