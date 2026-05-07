# RAG Worker Technical Documentation

Files covered:
- `rag_agent_service/main.py`
- `rag_agent_service/rag_worker.py`
- `rag_agent_service/context_fetcher.py`
- `rag_agent_service/retrieval_planner.py`
- `rag_agent_service/retrieval_utils.py`
- `rag_agent_service/time_filter_parser.py`
- `rag_agent_service/reranker_client.py`
- `rag_agent_service/prompt.py`
- `rag_agent_service/prompt_store.py`
- `rag_agent_service/prompt_defaults.py`
- `rag_agent_service/settings.py`

Last updated: 2026-04-24

This document describes the behavior in the current repository. Older documentation that described a simpler v4 planner/reranker path, different context-fetcher URLs, or older source-access logic is stale.

---

## 1. What This Service Does

`rag_agent_service` is the Redis-stream worker that:
- consumes routed chat requests from Redis
- fetches chat history, uploaded-file context, or big-data report chunks through `context_fetcher_prod`
- plans retrieval using a dedicated planner LLM call plus deterministic fallbacks
- optionally reranks big-data chunks through an external reranker
- runs a post-rerank evidence-selector stage over the reranked list before judging/final answering
- can short-circuit social/meta prompts before retrieval
- can run an intermediate coverage judge plus one bounded recovery pass
- builds the final RAG prompt
- streams the answer to the client through Redis
- emits source document payloads for the UI/backend
- optionally appends a second "Data From Open Source ( till April 2024 )" section
- persists the final assistant message through the API gateway save endpoint

It is a retrieval-and-generation worker only. It does not own vector storage, Elasticsearch indexing, or direct web browsing.

---

## 2. Runtime Topology

### 2.1 Worker Bootstrap

`main.py` builds:
- one shared `ContextFetcher`
- one `RAGWorker`
- one `AIWorker` Redis consumer

`AIWorker`:
- connects to Redis
- creates or reuses the configured consumer group
- polls the configured stream
- processes jobs concurrently under a semaphore
- acknowledges successful jobs
- pushes failures to the DLQ

### 2.2 Default Runtime Settings

Important defaults from `settings.py`:

| Area | Setting | Default |
|---|---|---|
| Redis | `redis_url` | `redis://:Redis@123@192.168.10.35:6379/0` |
| Redis stream | `stream_name` | `tasks.rag_production` |
| Redis DLQ | `dead_letter_queue` | `tasks.dlq_production` |
| Consumer group | `group_name` | `rag` |
| Worker name | `worker_name` | `rag_01` |
| Max worker concurrency | `rag_worker_max_concurrency` | `3` |
| Context fetch timeout | `context_fetch_timeout_s` | `60.0` |
| Context fetch retries | `context_fetch_retries` | `3` |
| Answer model base URL | `vllm_base_url` | `http://192.168.10.210:8000/v1` |
| Answer model | `model_name` | `openai/gpt-oss-120b` |
| Save response endpoint | `assistant_resp_endpoint` | `http://192.168.10.182:5000/message/save_assistant_response` |

### 2.3 External Retrieval Dependencies

The worker is fully dependent on `context_fetcher_prod` for retrieval:

| Purpose | Setting | Default endpoint |
|---|---|---|
| Structured chat context | `conv_context_url` | `http://192.168.10.67:5010/chat_context_new` |
| Uploaded-file context | `file_context_url` | `http://192.168.10.67:5010/uploaded_file_context_new` |
| Big-data semantic retrieval | `big_data_context_url` | `http://192.168.10.67:5010/big_data_documents_context` |
| Big-data exact retrieval | `big_data_exact_context_url` | `http://192.168.10.67:5010/big_data_documents_exact_context` |

Reranking is separate:

| Purpose | Setting | Default |
|---|---|---|
| External reranker base URL | `rag_reranker_base_url` | `http://192.168.10.210:9082` |
| Reranker timeout | `rag_reranker_timeout_s` | `120.0` |

---

## 3. Redis Task Contract

The worker reads and updates `task:{message_id}` hashes created upstream by the gateway.

### 3.1 Fields Used by RAG

Current active fields:
- `status`
- `current_stage`
- `ui`
- `ui_detail`
- `stream`
- `system_resp_id`
- `updated_at`

### 3.2 Status Behavior

The worker keeps the original gateway semantics:
- `status="queued"` before worker pickup
- `status="answering"` when RAG starts
- `status="completed"` when the first answer token starts streaming, or when a plain fallback response starts
- `status="finished"` after the stream end marker is emitted
- `status="failed"` on unhandled worker failure
- `status="cancelled"` when the upstream cancellation flag is observed

`current_stage` remains `rag_agent` while this worker owns the request.

### 3.3 Cancellation Cleanup

Cancellation is signaled by the Redis key `cancelled:{message_id}`.

When RAG sees that flag before it starts processing, or while it is streaming answer tokens, it:
- resolves the response stream name from `task:{message_id}` if needed
- deletes that Redis response stream key
- sets `task:{message_id}` to `status=cancelled`
- sets `ui=cancelled` and `ui_detail=request cancelled`
- stops without writing a fallback error response or final stream end marker

This prevents stale partial LLM output from remaining in the client-visible stream after a user cancels.

### 3.4 UI Phase Fields

The worker now writes pre-answer progress into:
- `ui`
- `ui_detail`

Current phase values:
- `planning`
- `semantic_fetch`
- `exact_candidates`
- `reranking`
- `sending_to_llm`

Once answer streaming begins, the token stream becomes the user-visible progress signal.

---

## 4. End-to-End Flow

High-level flow:

1. `AIWorker` receives a Redis stream job.
2. `RAGWorker.process()` parses the wrapper and validates the task.
3. `RAGWorker._rag()` may short-circuit social/meta prompts immediately.
4. Otherwise it builds retrieval context.
5. For big-data requests it now runs: reranker -> evidence selector batches -> coverage judge -> bounded recovery (if needed).
6. The worker chooses one of:
   - chat-context-only answer
   - attachment/file-grounded answer
   - big-data answer
   - no-context response
7. The worker streams the answer.
8. The worker resolves source documents.
9. The worker may append a second general-background section.
10. The worker saves the system message through the message API.

---

## 5. Entry Point: `RAGWorker.process()`

`process()` does the following:

1. Parses `data["data"]` as JSON.
2. Accepts either:
   - a wrapped payload under `{"payload": {...}}`
   - or a flat payload with direct `message_id`, `user_id`, `chat_id`
3. Validates:
   - `message_id` exists
   - `task:{message_id}` still exists
   - request is not cancelled
4. Applies defaults:
   - `content=""`
   - `attachments=[]`
   - `has_attachments` from attachments if absent
   - `search_mode="assistant"`
   - `seq=0`
5. Generates a new `system_resp_id`
6. Updates Redis task state:
   - `status=answering`
   - `current_stage=rag_agent`
   - `ui=planning`
   - `ui_detail=building retrieval plan`
7. Resolves the Redis response stream name from the task hash
8. Calls `_rag(payload, stream_name)`
9. Saves the final assistant response through the message API
10. On failure:
   - logs the exception
   - sets `status=failed`
   - streams a generic fallback error message if possible
11. On cancellation:
   - deletes the response stream key
   - sets `status=cancelled`
   - returns without saving an assistant response

---

## 6. Context Fetcher Client (`context_fetcher.py`)

`ContextFetcher` is a thin async HTTP client over `context_fetcher_prod`.

### 6.1 Request Style

It uses one `_post()` helper with retry logic:
- retries `context_fetch_retries + 1` times total
- logs failures
- backs off slightly between attempts

Endpoint request style:
- chat context: query params first, JSON fallback
- uploaded-file context: query params first, JSON fallback
- big-data semantic: JSON body first, query-param fallback
- big-data exact: JSON body first, query-param fallback

The big-data endpoints prefer JSON first to avoid oversized URLs for metadata-heavy retrieval.

### 6.2 Returned Shapes Expected by RAG

Expected response shapes:
- `fetch_conv_context(...)` -> `dict`
- `fetch_file_context(...)` -> `dict` with `file_context`
- `fetch_bigdata_context(...)` -> `list[dict]`
- `fetch_bigdata_exact_context(...)` -> `list[dict]`

---

## 7. Retrieval Planning Stage

## 7.1 First Chat Fetch: Recent-Only

`_build_retrieval_plan()` always begins with a recent, non-semantic chat fetch:

- `enable_semantic_search=False`
- `top_n=max(8, rag_chat_context_top_n)`

With current settings:
- `rag_chat_context_top_n=14`
- so the planner usually starts from `top_n=14`

This is intentional. Older semantic history is only fetched if the planner later asks for it.

## 7.2 Explicit Filter Snapshot

Before planning, the worker builds an `explicit_filters` snapshot from the incoming payload:
- `report_types` / `report_type`
- `branch`
- `doc_id`
- `parent_id`
- `lang`
- `is_attachment`
- `chunk_no`
- `document_date_gte/lte`
- `ingestion_date_gte/lte`

These are passed into the planner prompt as already-known constraints.

## 7.3 `RetrievalPlanner.plan()`

The planner is an LLM-backed JSON planner with deterministic fallback logic.

Planner inputs:
- current user query
- normalized recent chat block
- explicit payload filters
- whether history expansion is allowed
- whether history expansion has already been used

### 7.3.1 Planner Output Schema

Normalized `RetrievalPlan` fields:
- `standalone_query`
- `query_variants`
- `focus_subject`
- `focus_hint`
- `answer_intent`
- `retrieval_action`
- `exact_terms`
- `filters`
- `time_filter`
- `followup`
- `context_dependent`
- `needs_history_expansion`
- `planner_used`
- `raw_plan`

### 7.3.2 Answer Intent

Current planner answer intents:
- `inform`
- `format`
- `summarize`
- `continue`

These are important because they affect:
- whether retrieval can be skipped for chat-only transformations
- whether the general-background section is allowed later

### 7.3.3 Retrieval Action

Current retrieval actions:
- `fresh_retrieval`
- `reuse_previous_topic`

`reuse_previous_topic` is used for context-dependent follow-ups, especially formatting or continuation requests on the same topic.

### 7.3.4 Deterministic Fallback

The planner always builds a deterministic backup payload locally using:
- discourse-reference regexes
- pronoun/follow-up heuristics
- topic extraction from prior turns
- formatting-transform detection

If the model:
- fails
- returns malformed JSON
- misses context dependence
- leaves `standalone_query` equal to the raw unresolved query
- or misses topic/focus fields that deterministic logic found

the deterministic payload can override the model output.

Unit coverage exists in:
- `test_retrieval_planner_deterministic.py`

### 7.3.5 JSON Repair Pass

If the first planner call returns non-JSON output, the planner runs a repair call that asks the model to convert the malformed output into strict JSON.

### 7.3.6 Conservative Time-Filter Policy

Time filters are deliberately conservative.

The planner prompt explicitly tells the model not to turn event dates into metadata filters unless the user is clearly asking for retrieval-time scoping, for example:
- "reports from the database in 2024"
- "documents dated 12 Sept 2012"
- "what is mentioned in reports from last year"
- "last six months"

The post-normalization gate in `time_filter_parser.should_apply_retrieval_time_filter(...)` further suppresses weak time filters, especially:
- single-day event dates
- dates that look like content facts rather than retrieval constraints

This prevents queries like "movement of X on 01 Dec 2019" from over-restricting retrieval by metadata.

## 7.4 Optional History Expansion

If the first planner result sets `needs_history_expansion=True`, the worker:

1. fetches hybrid chat context with `enable_semantic_search=True`
2. uses `top_n=max(recent_top_n, rag_query_planner_hybrid_chat_top_n)`
3. replans using the expanded chat context

This pass is only kept if it actually increases the available chat-turn count.

---

## 8. Effective Filter Resolution

`_build_bigdata_filters()` merges filters from three sources:

Priority order:
1. explicit request payload
2. planner filters
3. regex/inferred filters from the raw query

Resolved fields:
- `report_type`
- `report_types`
- `branch`
- `doc_id`
- `parent_id`
- `lang`
- `is_attachment`
- `chunk_no`
- `document_date_gte/lte`
- `ingestion_date_gte/lte`
- `collection_name`

Time-filter application rules:
- explicit payload date bounds win
- exact payload `document_date` / `ingestion_date` are expanded by `+/- 1 day`
- planner/query-derived time filters apply only if no stronger explicit date bounds are already present

The final effective filter set is logged.

---

## 9. Retrieval Execution (`_fetch_rag_context()`)

## 9.1 Query Variant Finalization

The worker builds `retrieval_queries` from:
- `plan.standalone_query`
- planner `query_variants`
- sometimes the raw `user_query`

Important rule:
- the planner may return up to `rag_retrieval_query_variants=4`
- but only the top `rag_semantic_query_variants=2` are allowed to hit semantic big-data retrieval

This is the current recall-vs-cost balance.

## 9.2 Exact Term Finalization

`RetrievalSupport.fallback_exact_terms(...)` expands exact terms using:
- quoted phrases
- long numeric strings
- ID-like tokens
- derived entity-like phrases
- planner `focus_subject`
- planner `exact_terms`

Generic junk such as "source number 2" is intentionally filtered out.

## 9.3 Social / Meta Short-Circuit Path

Before planner or retrieval work begins, `_rag()` can answer a small class of non-retrieval messages directly.

Current deterministic short-circuit classes:
- greeting
- acknowledgement
- farewell
- simple assistant-meta/help prompts

Examples:
- `hi`
- `hello`
- `thanks`
- `bye`
- `who are you`
- `what can you do`

Behavior:
- planner is skipped
- chat retrieval is skipped
- big-data retrieval is skipped
- reranker is skipped
- no source documents are emitted

This path is implemented by `_classify_social_meta_query(...)` and `_social_meta_response(...)`.

## 9.4 Chat-Context-Only Transform Path

The worker can skip vector retrieval completely when all of the following hold:
- no attachments
- planner `retrieval_action == "reuse_previous_topic"`
- query is context-dependent / follow-up
- chat context already contains an assistant answer
- answer intent is `format`, `summarize`, or `continue`, or the query matches a transform/follow-up regex

Important current detail:
- `search_mode` is still parsed in `_fetch_rag_context(...)`
- but `_should_use_chat_context_only(...)` intentionally ignores it
- this was done because upstream payloads effectively always include `bigdata`, which otherwise suppresses the chat-only transform path for every request

In that case:
- `kb_chunks=[]`
- `retrieval_plan["chat_context_only"]=True`
- the final answer is built from chat context only using `build_chat_context_system_prompt()` and `build_chat_context_answer_text()`
- the hidden used-sources block is still required, but must stay empty because no KB chunks were used

## 9.5 Attachment Path

If the request carries attachments, the worker prefers uploaded-file retrieval over big-data retrieval.

`_fetch_attachment_context(...)` currently does:
- `retrieval_mode="full_file"` when attachments are present
- `top_n=rag_file_chunk_cap`
- `min_score=0.0` in full-file mode

With current settings:
- `rag_file_chunk_cap=320`

If attachment context is used successfully:
- `used_attachment_context=True`
- the big-data reranker is skipped

## 9.6 Big-Data Retrieval Path

If attachment context is not used, the worker runs `_fetch_bigdata_context(...)`.

### 9.5.1 Filter Variants

If `report_types` contains multiple values, the worker expands them into multiple filter variants, one `report_type` per request.

### 9.6.2 Strict Semantic Pass

The strict pass:
- uses only the top semantic query variants
- runs one retrieval call per `(query_variant x filter_variant)`
- uses one explicit per-query fetch budget

Current calculations:
- `rag_semantic_query_variants = 2`
- `rag_bigdata_semantic_top_n_per_query = 200`

So with one filter variant, the strict semantic pass currently does:
- query variant 1 -> `top_n=200`
- query variant 2 -> `top_n=200`

Important current behavior:
- upstream semantic fetch size is no longer derived from `rag_reranker_semantic_pool_size`
- reranker pool sizing and Qdrant semantic fetch sizing are now separate concerns
- this removed the earlier hidden behavior where an 800-document semantic reranker target could inflate semantic retrieval to 400 per query

### 9.6.3 Exact Lane

If exact terms exist and exact retrieval is enabled, the worker launches `_fetch_bigdata_exact_context_multi(...)` in parallel with the strict semantic pass.

Current defaults:
- `rag_exact_match_enabled=True`
- `rag_exact_match_top_n=224`
- `rag_exact_match_max_terms=4`

### 9.6.4 Strict Merge

Semantic and exact rows are merged by `RetrievalSupport.merge_chunks_by_identity(...)`.

Merge behavior:
- dedupes by `chunk_dedupe_key(...)`
- preserves the highest similarity score seen
- preserves the highest exact-match score seen
- merges `_retrieval_queries`
- increments `_query_match_count`
- merges matched-keyword lists

### 9.6.5 No Broad Relaxed First Pass

The worker no longer runs a second broad relaxed semantic sweep during the initial retrieval phase.

Previous stale behavior:
- if strict merged candidates were "too few", the worker would rerun semantic retrieval broadly with a weaker threshold before reranking

Current behavior:
- strict semantic + exact retrieval are merged
- that merged strict set goes directly to reranking
- any weaker-threshold semantic recovery now happens only inside the bounded recovery step after the coverage judge

---

## 10. Heuristic Ranking Before Reranking

Before the external reranker runs, the worker heuristically ranks chunks with `RetrievalSupport.rank_chunks_for_query(...)`.

The pre-rerank score combines:
- raw semantic similarity
- primary query term overlap
- secondary query overlap
- topic overlap
- exact phrase bonus
- multi-query match bonus
- exact-match bonus
- matched-keyword-count bonus

This score is stored in:
- `_heuristic_score`
- `_final_score` before reranking

Important:
- `pre_sim` in logs is the raw semantic similarity
- `pre_final` in logs is this heuristic composite score

These are intentionally different.

---

## 11. Contextual Retry

After heuristic ranking, the worker can run one extra focused retrieval retry for context-dependent queries.

Conditions:
- `rag_contextual_retry_enabled=True`
- planner says the query is context-dependent
- planner has a `focus_subject` or `focus_hint`
- no attachments are involved

The worker computes `topic_alignment_score(...)` over the top ranked chunks. If alignment is below:
- `rag_contextual_retry_topic_overlap_threshold=0.18`

it builds topic-focused retry queries and reruns big-data retrieval. The retry results are merged back into the original candidate set and re-ranked.

---

## 11.5 Evidence Selector Stage

After reranking and before the coverage judge, the worker now runs a selector stage over the reranked chunk list.

This is the current Phase 2a agentic-RAG upgrade.

### 11.5.1 Why It Exists

The selector stage addresses a different problem than Phase 1 recovery:
- too many reranked chunks can still fit into the final 30k prompt budget
- many of those chunks are relevant but not directly evidentiary
- the worker wants to reduce the final prompt to the chunks that most directly answer the query before the judge and final answer model see them

### 11.5.2 Global Stable Source IDs

The worker assigns a stable `global_source_id` over the full reranked list:
- rerank rank 1 -> `global_source_id=1`
- rerank rank 2 -> `global_source_id=2`
- etc.

These IDs are only for the selector stage. The final answer prompt still uses its own local `[source:N]` numbering over the final selected prompt rows.

### 11.5.3 Two Sequential Selector Batches

The worker builds up to two sequential selector batches from the reranked list:
- batch 1 = top reranked rows that fit in `rag_evidence_selector_batch_token_budget`
- batch 2 = next reranked rows that fit in the same batch budget

Current defaults:
- `rag_evidence_selector_enabled=True`
- `rag_evidence_selector_batch_token_budget=30000`
- `rag_evidence_selector_max_batches=2`

The batches are built sequentially over the reranked list, not by document grouping. This means the second selector batch sees the next slice of reranked overflow rather than a random reshuffle.

### 11.5.4 Selector Prompt / Output Contract

Each selector LLM call receives:
- user query
- resolved retrieval query
- retrieval focus
- batch metadata
- reranked chunk rows with:
  - `global_source_id`
  - rerank rank
  - origin
  - metadata
  - content excerpt

Each selector returns strict JSON:
- `evidence_source_ids`
- `ambiguity_detected`
- `insufficient_in_batch`
- `assessment`

Important behavior:
- the selector is not allowed to answer the user
- it only chooses chunk IDs that contain direct evidence
- it works on globally stable source IDs, so batch 1 and batch 2 outputs can be unioned safely

### 11.5.5 Repacking After Selection

The worker unions and dedupes selector IDs across the batches, rehydrates those chunks from the reranked list, and then runs the normal final prompt-budget selector on only that reduced chunk set.

If the selector returns no usable IDs:
- the worker falls back to the baseline prompt-budget selection from the full reranked list

This preserves safety:
- selector failure does not collapse the answer path
- the system still has a non-selector fallback path

### 11.5.6 Important Design Boundary

The selector stage runs before the judge, not after it.

Current big-data order is:
1. retrieve
2. heuristic rank
3. external rerank
4. evidence selector batch 1
5. evidence selector batch 2 (if needed)
6. coverage judge
7. bounded recovery if the judge says evidence is still weak
8. final answer generation

---

## 11.6 Coverage Judge and Bounded Recovery

After reranking and prompt-budget packing, the worker now runs an intermediate coverage / answerability judge.

This is the current Phase 1 agentic-RAG upgrade.

### 11.6.1 Judge Input

The judge receives:
- user query
- resolved retrieval query
- retrieval focus
- exact terms already in use
- selected prompt chunks
- counts of additional reranked and semantic candidates still available

### 11.6.2 Judge Output

The judge returns strict JSON fields including:
- `answerable`
- `ambiguity_detected`
- `should_ask_clarification`
- `clarification_question`
- `missing_aspects`
- `need_exact_retry`
- `need_semantic_rescue`
- `need_next_reranked_window`
- `suggested_queries`
- `suggested_exact_terms`
- `assessment`

### 11.6.3 Recovery Pass

If the selected prompt is only partial or insufficient, the worker performs at most one bounded recovery pass.

Recovery lanes:
- exact retry
- strict semantic rescue from the pre-rerank ranked set
- small relaxed semantic rescue
- next reranked window

### 11.6.4 Relaxed Semantic Rescue

The relaxed semantic lane now exists only here, not in the initial retrieval phase.

Current defaults:
- `rag_recovery_relaxed_semantic_enabled=True`
- `rag_recovery_relaxed_semantic_top_n_per_query=20`
- `rag_recovery_relaxed_semantic_total_chunks=30`
- `rag_relaxed_min_score=0.18`

Behavior:
- relaxed filters remove `doc_id`, `parent_id`, and `chunk_no`
- only a small number of relaxed rows are kept after merge and ranking
- those rows are tagged as `_recovery_source="relaxed_semantic_rescue"`

### 11.6.5 Important Design Boundary

The recovery pass does not run a second external reranker call.

That is deliberate:
- exact retry rows can surface without being immediately re-suppressed
- semantic rescue rows can surface without reranker interference
- recovery remains bounded and cheaper than a full second retrieve+rereank cycle

### 11.6.6 Final Coverage Guidance

After the post-recovery judge, the worker builds a short `coverage_guidance` string and injects it into the final answer prompt.

This tells the answer model whether:
- context is sufficient
- context is only partial
- ambiguity remains
- a clarification question should be asked if evidence cannot disambiguate

---

## 12. External Reranking

## 12.1 `NativeRerankerClient`

The reranker client:
- tries multiple endpoint forms:
  - `/rerank`
  - `/v1/rerank`
- tries multiple payload shapes:
  - `{"query": ..., "documents": [...]}`
  - `{"query": ..., "documents": [{"text": ...}]}`
  - `{"query": ..., "texts": [...]}`
  - `{"query": ..., "input": [...]}`
- optionally injects `model`
- normalizes different response schemas into `[{"index": i, "score": s}]`

## 12.2 Pool Construction

`_build_rerank_pool(...)` builds a mixed pool:
- primarily semantic rows
- with a reserved exact lane
- then backfill from remaining rows

Current defaults:
- `rag_reranker_enabled=True`
- `rag_reranker_candidate_pool_size=1024`
- `rag_reranker_semantic_pool_size=800`
- `rag_reranker_exact_pool_size=224`
- `rag_reranker_min_candidates=8`

Important current distinction:
- `rag_reranker_semantic_pool_size=800` affects only pool composition inside `_build_rerank_pool(...)`
- it no longer controls how many documents are fetched per semantic Qdrant call

## 12.3 What Text Is Sent to the Reranker

For each candidate, the worker prefers:
- `_rerank_content`
- otherwise raw chunk content

It then prefixes metadata when available:
- `DocId`
- `ParentId`
- `ReportType`
- `Branch`
- `Section`
- `SystemPath`
- page range
- chunk number
- quality score
- neighbor-context chunk count

If Qwen3 reranker templating is enabled, the worker wraps:
- the query as a Qwen3-style system/user prompt
- the document as `<Document>: ... <|im_end|> ...`

Current defaults:
- `rag_reranker_use_instruction=True`
- `rag_reranker_qwen3_template_enabled=True`
- `rag_reranker_doc_max_chars=5000`

## 12.4 Reranker Scores vs Final Scores

When the reranker returns a score:
- `_rerank_score` is set
- `_final_score` is overwritten with the rerank score

If a document is missing from the reranker response:
- `_final_score` falls back to `_heuristic_score`

After reranking, ordering is determined by:
1. whether `_rerank_score` exists
2. `_rerank_score`
3. `_heuristic_score`
4. original local index

## 12.5 Important Current Limitation

Neighbor-enriched content is used for reranking when `_rerank_content` is present, but the final answer LLM prompt does not consume `_rerank_content`. The answer prompt uses:
- raw content
- or `_prompt_content` if prompt-budget truncation was applied

So neighbor chunks currently influence:
- which chunks rank higher

but not:
- the exact chunk text shown to the final answer model

---

## 13. Prompt Selection and Budgeting

`_select_prompt_chunks_by_token_budget(...)` is the final context packer.

Current defaults:

| Setting | Default |
|---|---|
| `rag_prompt_token_budget` | `30000` |
| `rag_prompt_chars_per_token` | `4.0` |
| `rag_prompt_static_overhead_tokens` | `1400` |
| `rag_prompt_chunk_overhead_tokens` | `80` |
| `rag_prompt_min_chunk_tokens` | `180` |
| `rag_max_prompt_chunks` | `0` (no hard cap) |
| `rag_exact_match_guaranteed_chunks` | `2` |

Behavior:
- chunks are considered in ranked order
- exact chunks can be force-kept early if the guarantee is enabled
- if a full chunk does not fit, the worker may include a truncated excerpt
- truncated excerpts are stored in `_prompt_content`
- metadata about truncation is written into `_prompt_content_truncated`, `_prompt_content_chars`, `_raw_content_chars`

The prompt-budget summary is added into `retrieval_plan["prompt_budget"]`.

---

## 14. Prompt Builders and Output Contract

## 14.1 Runtime Prompt Source

`build_rag_system_prompt()` does not hardcode the main answer prompt. It reads the active prompt from:
- `rag_agent_service/prompts/system/default/rag_system_prompt.md`

through `prompt_store.py`.

If the file does not exist, it is created from `prompt_defaults.DEFAULT_RAG_SYSTEM_PROMPT`.

## 14.2 Hidden Source-Usage Contract

All answer-generation prompts append a backend contract that requires the model to output:

- a normal visible answer
- followed by exactly one hidden block:

`<<USED_SOURCES_JSON>> {"used_sources":[]} <<END_USED_SOURCES_JSON>>`

Visible text must not contain:
- `[source:N]`
- source tags
- doc IDs
- parent IDs

The hidden block is for backend parsing only.

## 14.3 Main RAG User Prompt

`build_rag_context_text(...)` renders:
- user query
- resolved retrieval query
- retrieval focus
- applied time filter label
- `## Knowledge Base Context`
- `## Chat Context (Oldest To Newest)`
- final answering instructions

Each KB chunk is rendered with a metadata line like:
- `[source:N] id=... score=... report_type=... branch=... lang=... document_date=... ingestion_date=...`

## 14.4 Chat-Only Prompt

`build_chat_context_answer_text(...)` is a dedicated prompt for chat-only transforms/follow-ups.

It explicitly says:
- vector retrieval was intentionally skipped
- answer only from prior chat context
- if the prior answer cannot be resolved, ask a short clarification question

## 14.5 General-Background Prompt

`build_general_background_prompt(...)` receives:
- original user query
- resolved retrieval query
- retrieval focus
- the DB-grounded answer already sent
- the reason why the addendum is being requested

It tells the second LLM call to add only useful public/general context beyond the already-sent grounded answer.

---

## 15. Answer Generation and Source Emission

## 15.1 Main Generation

`_llm_generate(...)`:
- appends the LLM trace entry
- streams the answer from the configured vLLM-compatible endpoint
- separates visible text from the hidden used-sources block
- writes visible answer chunks to the Redis stream

## 15.2 Source Resolution

After streaming finishes, `_llm_generate(...)` resolves source usage by:
1. parsing the hidden JSON block
2. falling back to inline `[source:N]` markers if needed
3. falling back to the top 5 scored prompt chunks if the model provided no usable source numbering

It then calls `build_source_documents(...)`.

## 15.3 `build_source_documents(...)`

Source document payloads are built from selected KB chunks with:
- dedupe key `(system_path, doc_id, parent_id)`
- `excerpt` from chunk content
- `report_type`
- `branch`
- `score`
- `document_date`
- `access`

Access resolution is chunk-ACL based:
- user access codes are fetched once from `rag_user_access_branches_endpoint`
- chunk ACLs come from `access_branches` and `access_groups`
- codes are matched exactly after trim/lowercase
- super-user levels can trigger `access_override=True`

Current default output limit:
- `rag_emit_source_documents_limit=40`

Important current behavior:
- if a chunk has no ACL codes, access resolves false unless override is active
- if `used_sources=[]`, the backend no longer emits every prompt chunk as a source-document fallback; it emits only the top scored 5 prompt chunks

---

## 16. Optional General Background Section

After the grounded answer, the worker may append a second LLM-generated section.

Rendered header:
- `## Data From Open Source ( till April 2024 )`

This path is governed by `_should_add_general_background(...)`.

It is suppressed when:
- the user explicitly disables open-source/general background
- the query is clearly internal-only
- the query requests current/live/medical/legal/financial verified data
- the planner intent is `format`, `summarize`, or `continue`
- the request is clearly a report/formatted-output request

It is allowed when:
- trigger mode is `default_on`
- or the user explicitly asks for broader/general/public background
- or conservative mode allows it for no-context or insufficient-context general queries

Current defaults:
- `rag_general_background_enabled=True`
- `rag_general_background_trigger_mode=default_on`
- `rag_general_background_max_tokens=700`
- `rag_general_background_temperature=0.2`

This section is streamed after the main answer and before the final stream end.

---

## 17. Trace Logs and Debugging Artifacts

## 17.1 Main LLM Trace

`llm_trace_file` default:
- `logs/rag_llm_trace.log`

Contains:
- system prompt
- user prompt
- token estimates
- hidden output block
- result metadata

## 17.2 Planner Trace

`rag_planner_trace_file` default:
- `logs/rag_planner_trace.log`

Contains:
- planner system prompt
- planner user prompt
- raw planner output
- repaired payload if repair was needed
- deterministic payload
- final normalized plan

## 17.3 Selector Trace

Evidence-selector traces now go to:
- `rag_selector_call.log`

This file contains:
- selector system prompt
- selector user prompt
- raw selector output
- parsed selector payload
- normalized selector decision
- error details if a selector call fails

One block is written per selector batch call.

## 17.4 Judge Trace

Coverage-judge traces now go to:
- `rag_judge_call.log`

This file contains:
- judge system prompt
- judge user prompt
- raw judge output
- parsed judge payload
- normalized judge decision
- error details if the judge call fails

The main `rag_llm_trace.log` no longer includes selector or judge prompt/result blocks.

## 17.5 Compact Context Trace

`rag_context_trace_enabled` and `rag_context_trace_file` control a compact retrieval trace.

Current behavior:
- `_append_context_trace(...)` writes selected chunk shape, origin counts, ACL fields, exact retrieval diagnostics, selector metadata, coverage metadata, and compact content excerpts
- `rag_context_trace.log` is the active artifact for debugging retrieval composition
- the main answer prompt is still visible through the main LLM trace path

## 17.6 Reranker Trace

The worker also appends a separate `logs/reranker_trace.log` file containing:
- raw query
- rerank query sent
- whether Qwen3 templating was enabled
- pre-rerank semantic candidates
- rerank body before templating
- final rerank input sent
- score comparison before vs after reranking
- top post-rerank scores

---

## 18. Correlation with `context_fetcher_prod`

The RAG worker depends on the following `context_fetcher_prod` contracts.

## 18.1 `/chat_context_new`

RAG expects:
- `ordered_turns`
- `recent_conversations`
- `conversation_history`

The worker benefits from current chat-fetcher behavior such as:
- recent-tail budgeting
- optional semantic history expansion
- lexical fallback for referential chat queries
- scope annotations on turns

## 18.2 `/big_data_documents_context`

RAG expects a list of chunk dicts with fields such as:
- `content`
- `doc_id`
- `parent_id`
- `report_type`
- `branch`
- `lang`
- `document_date`
- `ingestion_date`
- `system_path`
- `access_branches`
- `access_groups`
- optional debug/retrieval fields such as `_sim_score`

Current correlation detail:
- the semantic big-data service may attach `_rerank_content` with neighbor-enriched text
- the RAG reranker consumes that field if present

## 18.3 `/big_data_documents_exact_context`

RAG expects exact hits to come back as chunk-like rows that can merge with semantic rows using the same dedupe logic.

## 18.4 `/uploaded_file_context_new`

RAG expects:
- `file_context` as the actual chunk list

For attachment-grounded queries, RAG uses this endpoint in full-file mode and bypasses the big-data reranker.

---

## 19. Active Settings on the Current Critical Path

Most important settings actually used on the live path:

### 19.1 Retrieval
- `rag_top_n_docs=16`
- `rag_bigdata_semantic_top_n_per_query=200`
- `rag_min_score=0.3`
- `rag_relaxed_min_score=0.18`
- `rag_semantic_query_variants=2`
- `rag_retrieval_query_variants=4`

### 19.2 Reranker
- `rag_reranker_enabled=True`
- `rag_reranker_candidate_pool_size=1024`
- `rag_reranker_semantic_pool_size=800`
- `rag_reranker_exact_pool_size=224`
- `rag_reranker_doc_max_chars=5000`
- `rag_reranker_qwen3_template_enabled=True`

### 19.3 Prompt Budget
- `rag_prompt_token_budget=30000`
- `rag_prompt_static_overhead_tokens=1400`
- `rag_prompt_chunk_overhead_tokens=80`
- `rag_prompt_min_chunk_tokens=180`
- `rag_max_prompt_chunks=0`
- `rag_max_chunk_chars=0`

### 19.4 Planner / Chat Context
- `rag_chat_context_top_n=14`
- `rag_query_planner_max_turns=6`
- `rag_query_planner_context_chars=10000`
- `rag_query_planner_hybrid_chat_top_n=18`
- `rag_query_planner_allow_history_expansion=True`

### 19.5 Output
- `rag_answer_max_tokens=2048`
- `rag_emit_source_documents_limit=40`
- `rag_general_background_enabled=True`
- `rag_general_background_trigger_mode=default_on`

### 19.6 Evidence Selector
- `rag_evidence_selector_enabled=True`
- `rag_evidence_selector_batch_token_budget=30000`
- `rag_evidence_selector_max_batches=2`
- `rag_evidence_selector_static_overhead_tokens=1000`
- `rag_evidence_selector_chunk_overhead_tokens=80`
- `rag_evidence_selector_min_chunk_tokens=160`
- `rag_evidence_selector_max_chars_per_chunk=2200`
- `rag_evidence_selector_max_evidence_ids_per_batch=18`

### 19.7 Coverage Judge / Recovery
- `rag_coverage_judge_enabled=True`
- `rag_coverage_judge_max_chunks=12`
- `rag_coverage_judge_max_chars_per_chunk=1400`
- `rag_recovery_enabled=True`
- `rag_recovery_max_attempts=1`
- `rag_recovery_semantic_rescue_chunks=8`
- `rag_recovery_next_reranked_window_chunks=8`
- `rag_recovery_relaxed_semantic_enabled=True`
- `rag_recovery_relaxed_semantic_top_n_per_query=20`
- `rag_recovery_relaxed_semantic_total_chunks=30`

### 19.8 Trace Files
- `llm_trace_enabled=True`
- `rag_planner_trace_enabled=True`
- `rag_selector_trace_enabled=True`
- `rag_judge_trace_enabled=True`
- `rag_context_trace_enabled=True`

## 19.9 Legacy or Compatibility Settings

Some settings remain in `settings.py` for compatibility but are not central on the current code path, for example:
- `rag_enable_query_rewrite`
- `rag_enable_contextual_query_planner`
- `rag_enable_exact_term_extractor`
- `rag_file_access_endpoint`

The current worker mainly uses the unified planner and chunk-ACL-based source access flow.

---

## 20. Current Behaviors and Caveats

- Only the top 2 planner query variants currently hit semantic big-data retrieval, even if the planner emits up to 4 variants.
- Chat-only formatting/summarization follow-ups can skip vector retrieval entirely.
- Social/meta prompts can bypass planner and retrieval entirely.
- Weak single-date event mentions do not become retrieval metadata filters.
- Exact rows and semantic rows are merged before reranking; reranker logs are therefore not "semantic-only" lists.
- `pre_sim` and `pre_final` are intentionally different; `pre_final` includes heuristic boosts.
- Neighbor-enriched `_rerank_content` helps reranking but is not yet fed to the final answer LLM prompt.
- If the model does not return a valid hidden `used_sources` block, the backend falls back to only the top 5 scored prompt chunks when building `source_documents`.
- The worker does not browse the web. The second open-source section is still model-generated general background, not live retrieval.
- The open-source addendum is a second LLM call, not part of the main grounded answer.
- `search_mode` is currently ignored by the chat-context-only transform gate because upstream payloads effectively always include `bigdata`.
- The initial big-data semantic pass is now explicit and stable; reranker pool targets do not inflate Qdrant fetch size anymore.
- Relaxed semantic retrieval is now a small recovery-only lane, not a broad first-pass sweep.

---

## 21. Summary

The current RAG worker is a multi-stage retrieval-and-generation pipeline with:
- social/meta short-circuit handling
- structured chat-aware planning
- conservative retrieval filtering
- parallel semantic and exact retrieval
- heuristic ranking plus external reranking
- coverage judging plus one bounded recovery pass
- token-budgeted prompt packing
- hidden machine-readable source attribution
- ACL-aware source document emission
- optional second-pass general background generation

Its main retrieval dependency is `context_fetcher_prod`, and the two services should be read together when debugging retrieval quality.
