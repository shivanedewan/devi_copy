# Context Fetcher Prod - Technical Documentation


the collection chat_messages_production_1 has fields this exactly .... for any code fetching context use these names
content
created_at
message_id
user_id
has_attachments
chat_id
role
seq
attachment_ids



For chat_attachment_chunks_production_1



content
created_at
user_id
file_id
chat_id
bucket_name
title  ( this is the file name)
chunk_no







Files covered:
- `context_fetcher_prod/app.py`
- `context_fetcher_prod/router.py`
- `context_fetcher_prod/context_creator.py`
- `context_fetcher_prod/chat_context_new_creator.py`
- `context_fetcher_prod/big_data_documents_context_creator.py`
- `context_fetcher_prod/uploaded_file_context_creator.py`
- `context_fetcher_prod/uploaded_file_context_new_creator.py`
- `context_fetcher_prod/qdrant_utils.py`

Last updated: 2026-04-26

This document describes the behavior that exists in the current repository. Older documentation that mentioned `chat_file_mapping_v1`, `common.py`, or `migration.py` does not match this repo version.

---

## 1. What This Service Does

`context_fetcher_prod` is the retrieval backend used by other services such as:
- `qa_worker_v2`
- `rag_agent_service`
- `report_generator`

It serves three retrieval domains:
- chat history
- big-data / report chunks
- uploaded file chunks

The service is FastAPI-based and exposes HTTP endpoints only. It does not own any LLM generation.

---

## 2. Runtime Architecture

### 2.1 Startup (`app.py`)

`app.py`:
- configures logging at import time
- creates a FastAPI app
- mounts custom local Swagger and ReDoc assets from `/static`
- enables permissive CORS
- mounts the router from `router.py`

Current code default when run directly:
- host: `192.168.10.67`
- port: `5010`

### 2.2 Singleton Construction (`router.py`)

At module load time the router builds shared singletons:
- one `QdrantClient`
- one chat `ContextFetcher`
- one `ChatContextNewFetcher`
- one `BigDataDocumentsContextFetcher`
- one `UploadedFileContextFetcher`
- one `UploadedFileContextNewFetcher`
- one `QdrantUtils`

Current router defaults:

| Setting | Default |
|---|---|
| `QDRANT_URL` | `http://192.168.10.32:6333` |
| `QDRANT_TIMEOUT_SECONDS` | `60` |
| `EMBED_URL` | `http://192.168.10.210:9084/v1/embeddings` |
| `EMBED_MODEL` | `qwen-embed_8b` |

These are process-wide singletons, so endpoint calls do not recreate clients on every request.

---

## 3. API Surface

The service currently exposes 6 retrieval endpoints:

| Endpoint | Purpose |
|---|---|
| `POST /` | Raw chat-context messages from `ContextFetcher.fetch()` |
| `POST /chat_context_new` | Structured chat context for workers |
| `POST /big_data_documents_context` | Semantic big-data retrieval |
| `POST /big_data_documents_exact_context` | Exact / keyword big-data retrieval |
| `POST /uploaded_file_context` | Raw uploaded-file chunk retrieval |
| `POST /uploaded_file_context_new` | Structured uploaded-file context wrapper |

Error handling pattern:
- `ValueError` -> HTTP 400
- `RuntimeError` -> HTTP 500 with logged server-side detail
- other exceptions -> HTTP 500

---

## 4. Endpoint Contracts

### 4.1 `POST /`

Raw chat message retrieval.

Parameters:
- `user_id`
- `chat_id`
- `message_id`
- `query` optional
- `semantic_threshold` optional

Return shape:
- flat list of message payloads with `_scope`, `_sim_score`, `_turn_*` metadata

Important current behavior:
- if `query` is empty, the endpoint does not fail
- it automatically disables semantic retrieval and returns the recent tail messages only

This is the lower-level form of chat retrieval. Most current callers should use `/chat_context_new` instead.

### 4.2 `POST /chat_context_new`

Structured chat context endpoint used by workers.

Parameters:
- `user_id`
- `chat_id`
- `message_id`
- `query` optional
- `top_n`
- `semantic_threshold`
- `enable_semantic_search`
- `tail_max_turns`
- `tail_token_budget`
- `semantic_history_max_turns`
- `semantic_history_token_budget`
- `total_token_budget`
- `chars_per_token`
- `include_chronological_anchors`
- `chronological_anchor_turns`

Return shape:
- `conversation_history`
- `recent_conversations`
- `ordered_turns`
- `user_query`
- count fields for all three lists

Important current behavior:
- if `query` is empty, the endpoint does not fail
- semantic search is automatically disabled for that request
- the response is built from recent tail messages only
- this supports flows such as file upload turns where no user query text exists yet

### 4.3 `POST /big_data_documents_context`

Semantic report/document retrieval.

Accepted input:
- JSON body via `BigDataRequest`
- or query parameters

Important parameters:
- `query`
- `top_n`
- `min_score`
- `report_type`
- `branch`
- `doc_id`
- `parent_id`
- `lang`
- `is_attachment`
- `chunk_no`
- `document_date_gte`
- `document_date_lte`
- `ingestion_date_gte`
- `ingestion_date_lte`
- `collection_name`

Return shape:
- list of chunk payloads with content and retrieval metadata

### 4.4 `POST /big_data_documents_exact_context`

Exact / keyword retrieval over report indexes.

Required:
- `query`
- `keywords`

Optional:
- all big-data filters
- `elasticsearch_base_url`
- `elasticsearch_index`

Important current behavior:
- `elasticsearch_base_url` is still respected
- `elasticsearch_index` is accepted for backward compatibility but ignored for index selection
- the code always uses `report_mains_v1` and `report_attachments_v1`

### 4.5 `POST /uploaded_file_context`

Raw uploaded-file chunk retrieval.

Parameters:
- `user_id`
- `chat_id`
- `query` (not required for `full_file`)
- `top_n`
- `min_score`
- `retrieval_mode` = `semantic | full_file`
- `file_id`
- `resolve_latest_file`
- `latest_file_limit`
- `before_created_at`
- `collection_name`

Important current behavior:
- this path is Qdrant-only
- there is no Elasticsearch chat-file mapping lookup anymore

### 4.6 `POST /uploaded_file_context_new`

Structured uploaded-file retrieval wrapper.

Same input contract as `/uploaded_file_context`, but returns:
- `context`
- `file_context`
- `file_context_count`
- `source_documents`

This is the endpoint consumed by current workers.

---

## 5. Chat Retrieval Internals (`context_creator.py`)

`ContextFetcher` is the core chat retrieval engine.

### 5.1 Chat Collection and Defaults

Current defaults:

| Config field | Default |
|---|---|
| `chat_collection` | `chat_messages_production_1` |
| `semantic_threshold` | `0.4` |
| `semantic_trigger_count` | `10` |
| `recent_window_size` | `10` |
| `max_session_messages` | `1000` |
| `tail_max_turns` | `CHAT_CONTEXT_TAIL_MAX_TURNS` or `10` |
| `tail_token_budget` | `CHAT_CONTEXT_TAIL_TOKEN_BUDGET` or `8000` |
| `semantic_history_max_turns` | `CHAT_CONTEXT_SEMANTIC_HISTORY_MAX_TURNS` or `8` |
| `semantic_history_token_budget` | `CHAT_CONTEXT_SEMANTIC_HISTORY_TOKEN_BUDGET` or `4000` |
| `total_token_budget` | `CHAT_CONTEXT_TOTAL_TOKEN_BUDGET` or `12000` |
| `chars_per_token` | `CHAT_CONTEXT_CHARS_PER_TOKEN` or `4.0` |
| `include_chronological_anchors` | env or `true` |
| `chronological_anchor_turns` | env or `2` |

### 5.2 High-Level Fetch Algorithm

`fetch(...)` does this:
1. validate required identifiers
2. resolve the active context budget
3. normalize the query; if it is empty, semantic retrieval is disabled automatically
4. if semantic search is disabled:
   - fetch recent tail messages by `created_at DESC`
   - build turns
   - trim to recent turn and token budgets
   - tag as recent
5. if semantic search is enabled:
   - fetch all session messages up to `max_session_messages`
   - build normalized turns
   - always keep a recent tail under the recent budget
   - if total messages are at or below `semantic_trigger_count`, stop and return recent-only context
   - otherwise embed the query
   - optionally select earliest-turn anchors for "first/oldest" style queries
   - run semantic Qdrant search
   - build lexical candidate turns from the full session
   - combine semantic and lexical evidence while respecting turn and token budgets
   - merge recent, chronological-anchor, and semantic-history turns
   - flatten back to message-level output

### 5.3 Message and Turn Construction

Important helpers:
- `_fetch_all_session_messages(...)`
- `_build_turns(...)`
- `_flatten_turns(...)`
- `_order_messages_as_pairs(...)`

Behavior:
- user + assistant/system pairs become a turn
- unpaired messages are still retained as one-message turns
- ordering prefers `seq` when available, then timestamp
- very short / junk content is skipped

### 5.4 Recent Tail Selection

Recent context is chosen at turn level, not message level.

Selection helpers:
- `_take_recent_turns_by_budget(...)`
- `_take_recent_turns(...)`
- `_turn_token_count(...)`
- `_turns_token_count(...)`

The service uses both:
- a max turn count
- a token-estimate budget

### 5.5 Chronological Anchor Turns

This is the "earliest turn" feature.

It activates only when:
- `include_chronological_anchors` is enabled
- `chronological_anchor_turns > 0`
- the query matches earliest-turn intent such as first / earliest / oldest

These anchors are selected before semantic-history turns and count against the overall history budget.

### 5.6 Semantic Retrieval

Semantic candidate retrieval:
- uses `qdrant.query_points(...)`
- filters by `user_id`, `chat_id`, and excludes the current `message_id`
- applies `semantic_threshold`
- returns payloads tagged with `_scope="semantic"` and `_sim_score`

Important logging fields:
- raw hit count
- accepted hit count
- threshold
- raw top score

### 5.7 Lexical Fallback

The current chat service has a lexical fallback lane for referential chat queries.

Implemented signals:
- normalized rare query terms
- quoted phrases
- UUIDs
- reference-language detection such as pasted / mentioned / above / earlier / this / that

Lexical scoring is built in `_build_lexical_turn_hits(...)`.

This is useful for:
- "the thing I pasted above"
- UUID-heavy references
- quoted follow-ups
- entity recall when semantic scores are weak

### 5.8 Semantic + Lexical Turn Selection

`_select_semantic_turns(...)` combines both evidence streams.

Output scopes can be:
- `semantic`
- `lexical`
- `semantic+lexical`

Each selected turn can carry:
- `_sim_score`
- `_lexical_score`
- `_final_score`
- `_lexical_term_hits`
- `_lexical_phrase_hits`
- `_lexical_uuid_hits`

### 5.9 Returned Raw Message Metadata

When `ContextFetcher.fetch(...)` returns raw messages, downstream consumers may see:
- `_scope`
- `_sim_score`
- `_turn_id`
- `_turn_scope`
- `_turn_is_recent`
- `_final_score`
- lexical metadata fields

The structured chat endpoint converts these back into pair-based context.

---

## 6. Structured Chat Context (`chat_context_new_creator.py`)

`ChatContextNewFetcher` is a shaping layer over `ContextFetcher`.

Its job is to convert raw message lists into the worker-friendly structure:
- `conversation_history`
- `recent_conversations`
- `ordered_turns`

### 6.1 Pair Building

`_build_conversation_pairs(...)` reconstructs user -> assistant/system pairs from the raw message list.

Pair fields include:
- `turn_id`
- `scope`
- `is_recent`
- `created_at`
- `user_message`
- `assistant_reply`
- message ids
- seq values

### 6.2 Scope-Based Splitting

If scope tags are present, the splitter trusts them:
- recent scopes -> `recent_conversations`
- semantic/history scopes -> `conversation_history`

If scope tags are absent, it falls back to legacy max-recent and max-history pair splitting.

### 6.3 Ordered Turns

`ordered_turns` is the most important payload for downstream workers.

Each turn includes:
- `turn_index`
- `recency_rank`
- `is_recent`
- `scope`
- `created_at`
- `user_message`
- `assistant_reply`
- message ids
- seq values

Workers such as `qa_worker_v2` normalize and merge this structure locally.

---

## 7. Big-Data Semantic Retrieval (`big_data_documents_context_creator.py`)

This service handles semantic report/document chunk retrieval from Qdrant.

### 7.1 Current Defaults

Important defaults:

| Config field | Default |
|---|---|
| `big_data_collection` | `document_chunks_test_new_clean_v1` |
| `chunk_metadata_index` | `document_chunks_test_new_clean_metadata` |
| `report_main_index` | `report_mains_v1` |
| `report_attachment_index` | `report_attachments_v1` |
| `top_n_docs` | `8` |
| `semantic_threshold` | `0.22` |
| `semantic_min_score_floor` | `0.18` |
| `candidate_multiplier` | `1` |
| `max_qdrant_query_limit` | `800` |
| `qdrant_query_timeout_seconds` | `60` |
| `semantic_neighbor_enabled` | env or `true` |
| `semantic_neighbor_window` | env or `1` |
| `semantic_neighbor_max_contexts` | env or `200` |
| `semantic_neighbor_max_words` | env or `180` |

### 7.2 Semantic Fetch Behavior

`fetch(...)` does this:
1. validate query and `top_n`
2. apply minimum-score floor:
   - requested `min_score`
   - floored by `semantic_min_score_floor`
3. embed the query
4. build a Qdrant payload filter from all supplied retrieval filters
5. cap Qdrant query size to `max_qdrant_query_limit`
6. query Qdrant with `score_threshold`
7. keep only payloads with actual content
8. sort primarily by:
   - `_sim_score`
   - `quality_score`
   - `_final_score`
9. trim to `top_n`
10. optionally enrich with neighbor chunks from Elasticsearch metadata index

Important current behavior:
- `weight_recency` is `0.0`, so semantic ranking is similarity-first
- `quality_score` is used as a secondary tiebreaker if present in payload

### 7.3 Semantic Neighbor Enrichment

After top semantic chunks are selected, the fetcher can attach adjacent chunk text for reranker-friendly payloads.

Source of neighbors:
- Elasticsearch metadata index, not Qdrant

Current enrichment fields may include:
- `_neighbor_chunk_count`
- `_neighbor_offsets`
- `_rerank_content`

The main chunk content remains the base retrieval content; the neighbor bundle is an enrichment aid.

---

## 8. Big-Data Exact Retrieval (`big_data_documents_context_creator.py`)

This is the exact / keyword lane for reports.

### 8.1 Query Sources

Exact retrieval searches:
- `report_mains_v1`
- `report_attachments_v1`

It no longer uses the legacy single `document_chunks_v2` exact path for content lookup.

### 8.2 Keyword Normalization

Keywords are:
- whitespace-normalized
- deduplicated
- capped by `exact_max_keywords`

Current defaults:

| Config field | Default |
|---|---|
| `exact_top_n_docs` | `6` |
| `exact_max_keywords` | `6` |
| `exact_main_result_ratio` | `0.80` |
| `exact_attachment_result_ratio` | `0.20` |
| `exact_candidate_multiplier` | env or `3` |
| `exact_max_candidates_per_index` | env or `800` |
| `exact_elasticsearch_timeout_seconds` | env or `45` |

### 8.3 Exact Matching Logic

Matching signals:
- phrase match
- token-and match
- numeric signature match for long numbers / ids

Per-payload exact strength is built by `_score_exact_payload(...)`.

Scoring components used in final exact ranking:
- `_exact_match_score`
- normalized Elasticsearch `_score`
- recency score from document or ingestion timestamps

### 8.4 80:20 Main vs Attachment Allocation

Results are allocated across main and attachment indexes using quotas:
- main: 80%
- attachment: 20%

If one side under-fills, leftover capacity is backfilled from the other side.

### 8.5 Legacy Parameters

`fetch_exact(...)` still accepts:
- `elasticsearch_base_url`
- `elasticsearch_index`

Current behavior:
- `elasticsearch_base_url` can override the ES base URL
- `elasticsearch_index` is logged and ignored

---

## 9. Uploaded File Retrieval (`uploaded_file_context_creator.py`)

This path is now fully Qdrant-based.

There is no Elasticsearch chat-file mapping lookup in the active upload flow.

### 9.1 Current Defaults

| Config field | Default |
|---|---|
| `uploaded_file_collection` | `chat_attachment_chunks_production_1` |
| `top_n_docs` | `12` |
| `semantic_threshold` | `0.3` |
| `candidate_multiplier` | `6` |
| `weight_similarity` | `1.0` |
| `weight_file_recency` | `2.2` |
| `weight_chunk_recency` | `0.6` |
| `scoped_weight_file_recency` | env or `0.25` |
| `scoped_weight_chunk_recency` | env or `0.10` |
| `semantic_neighbor_radius` | env or `1` |
| `semantic_neighbor_seed_limit` | env or `8` |
| `full_file_token_budget` | env or `22000` |
| `full_file_chars_per_token` | env or `4.0` |
| `oversize_full_file_semantic_candidate_limit` | env or `64` |

### 9.2 Common Filters

All file retrieval filters on:
- `user_id`
- `chat_id`
- optional `file_id` list

`file_id` can be:
- one id
- comma / newline / semicolon-separated ids

### 9.3 Latest File Resolver

If `resolve_latest_file=True` and no explicit `file_id` is supplied:
- the fetcher scans the Qdrant file collection
- finds the most recent file ids visible in the chat
- can apply `before_created_at` so "latest" is evaluated relative to the current message timestamp

This is how singular follow-up file references are resolved without an external mapping index.

### 9.4 Semantic File Retrieval

Semantic retrieval algorithm:
1. normalize explicit file ids or resolve latest file ids
2. embed query text
3. query Qdrant with a chat/file filter
4. collect candidate chunks with content
5. compute per-file latest timestamps
6. compute final ranking score using similarity plus recency

Important current ranking behavior:
- explicit file-scoped semantic retrieval uses reduced recency weights:
  - `scoped_weight_file_recency`
  - `scoped_weight_chunk_recency`
- chat-wide semantic retrieval keeps the stronger recency weights

This is a major recent change. It prevents recency from overwhelming relevance once file scope is already explicit.

### 9.5 Neighbor Chunk Expansion

For explicit file-scoped semantic retrieval, the fetcher can expand top hits with adjacent chunks.

Mechanism:
- take top semantic seeds
- derive neighboring `chunk_no` values within configured radius
- fetch neighbor rows by scrolling the same Qdrant collection
- merge and dedupe neighbors with original hits

Neighbor scopes may appear as:
- `semantic`
- `semantic_neighbor`
- `semantic_fallback`

### 9.6 Full-File Retrieval

Full-file mode:
- does not embed the query for the initial full-file fetch
- scrolls Qdrant rows for the selected file scope
- preserves original document order, preferring `chunk_no` when present
- tags rows with `_scope="full_file"`

### 9.7 Oversize Full-File Fallback

If full-file retrieval exceeds the configured token budget:
1. estimate total tokens from chunk text
2. if over budget, try semantic fallback under the same file scope
3. if semantic fallback fails, use lexical top-chunk fallback

Fallback markers:
- `_full_file_oversize_fallback = true`
- `_scope = "semantic_fallback"` or `lexical_fallback`

This is how the upload path avoids dumping very large files directly into worker prompts.

---

## 10. Uploaded File Wrapper (`uploaded_file_context_new_creator.py`)

This is the structured response layer used by workers.

It wraps the raw uploaded-file fetcher and returns:
- `context`
- `file_context`
- `file_context_count`
- `source_documents`

### 10.1 File Context Fields

Current output fields include:
- `file_id`
- `chunk_id`
- `chunk_no`
- `file_name`
- `content`
- `score`
- `sim_score`
- `chunk_recency_score`
- `file_recency_score`
- `scope`
- `full_file_oversize_fallback`
- `created_at`

### 10.2 Content Trimming

Config:
- `UPLOADED_FILE_CONTEXT_SEMANTIC_MAX_CHARS`
- `UPLOADED_FILE_CONTEXT_FULL_FILE_MAX_CHARS`

Current defaults are `0`, so the wrapper does not truncate by default.

### 10.3 Source Documents

The wrapper still builds compact `source_documents` for the caller even though `qa_worker_v2` no longer persists them.

---

## 11. Qdrant Utilities (`qdrant_utils.py`)

`QdrantUtils` is an admin/helper module, not a retrieval endpoint.

Available helper methods:
- `delete_points_by_file_id(user_id, file_id)`
- `delete_points_by_chat_id(user_id, chat_id)`

Collections:
- attachment collection: `chat_attachment_chunks_production_1`
- chat collection: `chat_messages_production_1`

This file is operationally useful, but its standalone `main()` snippet is not part of the HTTP service contract.

---

## 12. Environment and Settings Reference

### 12.1 Router-Level Runtime

| Variable | Default | Used by |
|---|---|---|
| `QDRANT_URL` | `http://192.168.10.32:6333` | `router.py` |
| `QDRANT_TIMEOUT_SECONDS` | `60` | `router.py` |
| `EMBED_URL` | `http://192.168.10.210:9084/v1/embeddings` | all embedder instances |
| `EMBED_MODEL` | `qwen-embed_8b` | all embedder instances |
| `EMBED_QUERY_INSTRUCTION_ENABLED` | `true` | chat, big-data, upload embedders |
| `EMBED_QUERY_INSTRUCTION` | instruction text | query embedding prefix |

### 12.2 Chat Context Budget Controls

| Variable | Default |
|---|---|
| `CHAT_CONTEXT_TAIL_MAX_TURNS` | `10` |
| `CHAT_CONTEXT_TAIL_TOKEN_BUDGET` | `8000` |
| `CHAT_CONTEXT_SEMANTIC_HISTORY_MAX_TURNS` | `8` |
| `CHAT_CONTEXT_SEMANTIC_HISTORY_TOKEN_BUDGET` | `4000` |
| `CHAT_CONTEXT_TOTAL_TOKEN_BUDGET` | `12000` |
| `CHAT_CONTEXT_CHARS_PER_TOKEN` | `4.0` |
| `CHAT_CONTEXT_INCLUDE_CHRONOLOGICAL_ANCHORS` | `true` |
| `CHAT_CONTEXT_CHRONOLOGICAL_ANCHOR_TURNS` | `2` |

### 12.3 Big-Data Retrieval Controls

| Variable | Default |
|---|---|
| `BIGDATA_QDRANT_COLLECTION` | `document_chunks_test_new_clean_v1` |
| `BIGDATA_TOP_N_DOCS` | `8` |
| `BIGDATA_SEMANTIC_THRESHOLD` | `0.22` |
| `BIGDATA_MIN_SCORE_FLOOR` | `0.18` |
| `BIGDATA_CANDIDATE_MULTIPLIER` | `1` |
| `BIGDATA_QDRANT_MAX_QUERY_LIMIT` | `800` |
| `BIGDATA_QDRANT_QUERY_TIMEOUT_SECONDS` | `60` |
| `BIGDATA_SEMANTIC_NEIGHBOR_ENABLED` | `true` |
| `BIGDATA_SEMANTIC_NEIGHBOR_WINDOW` | `1` |
| `BIGDATA_SEMANTIC_NEIGHBOR_MAX_CONTEXTS` | `200` |
| `BIGDATA_SEMANTIC_NEIGHBOR_MAX_WORDS` | `180` |
| `BIGDATA_SEMANTIC_NEIGHBOR_TIMEOUT_SECONDS` | `20` |
| `BIGDATA_CHUNK_METADATA_INDEX` | `document_chunks_test_new_clean_metadata` |

### 12.4 Exact Retrieval / Elasticsearch Controls

| Variable | Default |
|---|---|
| `ELASTICSEARCH_BASE_URL` | `https://192.168.10.236:9200` |
| `ELASTICSEARCH_USER` | `elastic` |
| `ELASTICSEARCH_PASS` | `Elastic@123` |
| `ELASTICSEARCH_PASSWORD` | fallback source for password |
| `ELASTICSEARCH_DISABLE_SSL_VERIFY` | `true` |
| `REPORT_MAIN_INDEX` | `report_mains_v1` |
| `REPORT_ATTACHMENT_INDEX` | `report_attachments_v1` |

### 12.5 Uploaded File Retrieval Controls

| Variable | Default |
|---|---|
| `UPLOADED_FILE_SCOPED_WEIGHT_FILE_RECENCY` | `0.25` |
| `UPLOADED_FILE_SCOPED_WEIGHT_CHUNK_RECENCY` | `0.10` |
| `UPLOADED_FILE_SEMANTIC_NEIGHBOR_RADIUS` | `1` |
| `UPLOADED_FILE_SEMANTIC_NEIGHBOR_SEED_LIMIT` | `8` |
| `UPLOADED_FILE_FULL_FILE_TOKEN_BUDGET` | `22000` |
| `UPLOADED_FILE_FULL_FILE_CHARS_PER_TOKEN` | `4.0` |
| `UPLOADED_FILE_OVERSIZE_SEMANTIC_CANDIDATE_LIMIT` | `64` |
| `UPLOADED_FILE_CONTEXT_SEMANTIC_MAX_CHARS` | `0` |
| `UPLOADED_FILE_CONTEXT_FULL_FILE_MAX_CHARS` | `0` |

---

## 13. Known Behaviors and Caveats

- `/uploaded_file_context` and `/uploaded_file_context_new` are now Qdrant-only. They do not depend on `chat_file_mapping_v1`.
- `big_data_documents_exact_context` still accepts `elasticsearch_index` for backward compatibility, but the active code ignores it and uses report indexes directly.
- Chat retrieval is turn-aware, not raw-message-budget-only.
- Chat lexical fallback is conservative and exists only for the chat-history lane.
- Big-data semantic retrieval can enrich results with neighbor text from Elasticsearch metadata, but uploaded-file semantic neighbor expansion is Qdrant-based.
- Uploaded-file full-file retrieval is no longer guaranteed to return the entire file if the estimated token budget is exceeded.
- `context_fetcher_prod` does not own client response streams and does not delete them on cancellation. Cancellation stream cleanup is handled by the consuming workers, such as `rag_agent_service` and `qa_worker_v2`, because they own `task:{message_id}` and the Redis response stream name.
- `qdrant_utils.py` is a helper module, not a public API.

---

## 14. Practical Reading Order

If you need to debug the service quickly, read files in this order:
1. `router.py`
2. `context_creator.py`
3. `chat_context_new_creator.py`
4. `big_data_documents_context_creator.py`
5. `uploaded_file_context_creator.py`
6. `uploaded_file_context_new_creator.py`
7. `qdrant_utils.py`

That order matches the runtime critical path for almost all production retrieval issues.
