# RAGWorker — End-to-End Technical Documentation (v3)

> **Files covered:**
> - `rag_agent_service/rag_worker.py` (778 lines — fully refactored)
> - `rag_agent_service/retrieval_planner.py` (546 lines — new unified LLM planner)
> - `rag_agent_service/retrieval_utils.py` (888 lines — new utility class)
> - `rag_agent_service/time_filter_parser.py` (374 lines — new date extractor)
> - `rag_agent_service/settings.py` (201 lines)
>
> **Last documented:** 2026-03-11 (v3 update)

---

## What Changed From the Previous Version

The service was **completely refactored**. The key change is the replacement of three separate LLM calls (query rewrite, contextual planner, exact-term extractor) with a **single unified `RetrievalPlanner` LLM call** that produces a complete `RetrievalPlan` JSON object in one shot. The new architecture also adds:

- `RetrievalPlanner` — single LLM call → one JSON with standalone query, variants, exact terms, filters, AND a time range
- `RetrievalSupport` — all utility functions (ranking, merging, filter extraction, dedup) now in one class
- `time_filter_parser.py` — pure regex date parser for extracting date ranges from natural language

## What Changed in v3 (Latest Update)

Four significant additions to `retrieval_planner.py`, one targeted change in `rag_worker.py`, and a new guard function in `retrieval_utils.py`:

| File | Change | Impact |
|---|---|---|
| `retrieval_planner.py` | `_is_context_dependent_query()` — heuristic regex pre-classifier | Detects follow-ups without LLM |
| `retrieval_planner.py` | `_extract_topic_chain()` — chat history walker | Finds base topic + contextual fragment from prior turns |
| `retrieval_planner.py` | `_build_deterministic_fallback_payload()` — full no-LLM plan | Runs on EVERY request alongside the LLM call |
| `retrieval_planner.py` | `_repair_model_output()` — 2nd LLM JSON-repair call | Used when the primary planner returns malformed output |
| `retrieval_planner.py` | **Deterministic override logic** — merges/replaces LLM plan conditionally | Guards against LLM failing to resolve follow-up context |
| `rag_worker.py` | Raw `user_query` excluded from `retrieval_queries` when `context_dependent=True` | Prevents literal unresolved follow-up queries from hitting vector store |
| `retrieval_utils.py` | `_is_exact_phrase_candidate()` — guard for exact term quality | Filters pronoun-only / all-stopword candidates before they enter the exact lane |

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Entry Point: `process()`](#2-entry-point-process)
3. [Stage 1 — Building the Retrieval Plan](#3-stage-1--building-the-retrieval-plan)
   - 3.1 [Fetch Recent Chat Context (Recency-only)](#31-fetch-recent-chat-context-recency-only)
   - 3.2 [The Unified LLM Planner (`RetrievalPlanner.plan()`)](#32-the-unified-llm-planner)
   - 3.3 [Optional History Expansion Pass](#33-optional-history-expansion-pass)
   - 3.4 [The `RetrievalPlan` Dataclass](#34-the-retrievalplan-dataclass)
4. [Stage 2 — Filter Resolution (`_build_bigdata_filters()`)](#4-stage-2--filter-resolution)
   - 4.1 [Filter Priority Ladder](#41-filter-priority-ladder)
   - 4.2 [Time Filter Resolution](#42-time-filter-resolution)
   - 4.3 [Time Filter Parser (`time_filter_parser.py`)](#43-time-filter-parser)
5. [Stage 3 — Context Fetching (`_fetch_rag_context()`)](#5-stage-3--context-fetching)
   - 5.1 [Query & Exact-Term Finalization](#51-query--exact-term-finalization)
   - 5.2 [Attachment Path](#52-attachment-path)
   - 5.3 [Big-Data Path (`_fetch_bigdata_context()`)](#53-big-data-path)
   - 5.4 [Strict vs Relaxed Pass](#54-strict-vs-relaxed-pass)
   - 5.5 [Exact-Match Parallel Lane](#55-exact-match-parallel-lane)
6. [Stage 4 — Reranking (`rank_chunks_for_query()`)](#6-stage-4--reranking)
7. [Stage 5 — Contextual Retry](#7-stage-5--contextual-retry)
8. [Stage 6 — Exact-Match Slot Guarantee](#8-stage-6--exact-match-slot-guarantee)
9. [Stage 7 — LLM Answer Generation](#9-stage-7--llm-answer-generation)
10. [Stage 8 — Save & Stream](#10-stage-8--save--stream)
11. [`RetrievalSupport` Utility Class](#11-retrievalsupport-utility-class)
    - 11.1 [Filter Extraction From Query](#111-filter-extraction-from-query)
    - 11.2 [Exact Term Extraction (`fallback_exact_terms()`)](#112-exact-term-extraction)
    - 11.3 [Chunk Merging (`merge_chunks_by_identity()`)](#113-chunk-merging)
    - 11.4 [Scoring Formula](#114-scoring-formula)
12. [All Settings Reference](#12-all-settings-reference)
13. [Scenario Walkthroughs](#13-scenario-walkthroughs)
    - 13.1 [Standalone: "Tell me about Chabahar port"](#131-standalone-query)
    - 13.2 [Follow-up: "Other countries involved"](#132-follow-up-query)
    - 13.3 [Entity: "profile of narendra modi"](#133-entity-query)
    - 13.4 [Time-scoped: "EISI reports from last 3 months"](#134-time-scoped-query)
    - 13.5 [History-expansion: Rare follow-up needing old context](#135-history-expansion)
14. [Full Pipeline Diagram](#14-full-pipeline-diagram)
15. [Known Behaviours & Edge Cases](#15-known-behaviours--edge-cases)

---

## 1. Architecture Overview

```
Redis Stream (tasks.rag)
        │
        ▼
  RAGWorker.process()
        │
        ▼
  _fetch_rag_context()
        │
        ├─ _build_retrieval_plan()
        │       ├─ _fetch_chat_context(enable_semantic=False)   [HTTP → conv API]
        │       ├─ RetrievalPlanner.plan()                      [LLM call → JSON]
        │       └─ [if needs_history_expansion]:
        │              _fetch_chat_context(enable_semantic=True) [HTTP → conv API]
        │              RetrievalPlanner.plan()  [2nd LLM call]
        │
        ├─ _build_bigdata_filters()         [local: merge explicit + planner + regex]
        ├─ RetrievalSupport.fallback_exact_terms()  [local regex + plan.exact_terms]
        │
        ├─ [if attachments] → _fetch_attachment_context()
        │
        ├─ [else] → _fetch_bigdata_context()
        │       ├─ asyncio.create_task(_fetch_bigdata_exact_context_multi())  [HTTP → exact API, ASYNC]
        │       ├─ _run_bigdata_batch() strict [HTTP × N queries × M filter variants]
        │       └─ [if 0 strict results] → _run_bigdata_batch() relaxed
        │
        ├─ RetrievalSupport.rank_chunks_for_query()       [local reranking]
        ├─ [if contextual_retry needed] → _fetch_bigdata_context() topic-queries
        │                                  + re-merge + re-rank
        └─ RetrievalSupport.ensure_exact_match_presence() [slot guarantee]
        │
        ├─ build_rag_system_prompt() + build_rag_context_text()
        ├─ _llm_generate()                  [HTTP → vLLM, streaming]
        └─ save_assistant_response()        [HTTP → message service]
```

**External services:**

| Service | Setting | Default |
|---|---|---|
| Conv context API | `conv_context_url` | `http://192.168.10.52:5010/chat_context_new` |
| File context API | `file_context_url` | `http://192.168.10.52:5010/uploaded_file_context_new` |
| Big-data (semantic) | `big_data_context_url` | `http://192.168.10.52:5010/big_data_documents_context` |
| Big-data (exact) | `big_data_exact_context_url` | `http://192.168.10.52:5010/big_data_documents_exact_context` |
| vLLM | `vllm_base_url` | `http://192.168.10.210:8000/v1` |
| Save response | `assistant_resp_endpoint` | `http://192.168.10.52:5000/message/save_assistant_response` |

---

## 2. Entry Point: `process()`

**Lines 711–777**

```
1. Parse JSON wrapper from data["data"]
2. Extract payload (supports {payload:{...}} and bare {message_id:...})
3. Validate: message_id exists, task:{message_id} key in Redis
4. Check cancellation (cancelled:{message_id}) → silent drop
5. Set defaults: content, attachments, has_attachments, search_mode="assistant", seq=0
6. Generate system_response_id = uuid4() → store to Redis
7. Set status: "answering" → current_stage: "rag_agent"
8. Get stream_name from task:{message_id} hash
9. Call _rag(payload, stream_name)
10. save_assistant_response()
11. On any exception: status="failed", emit error message to stream
```

---

## 3. Stage 1 — Building the Retrieval Plan

**Method:** `_build_retrieval_plan()` (line 218)

This is the new unified planning stage. It replaces three separate LLM calls from the old version (rewrite, contextual planner, exact-term extractor) with **one or two** calls to `RetrievalPlanner.plan()`.

### 3.1 Fetch Recent Chat Context (Recency-only)

```python
recent_top_n = max(8, settings.rag_chat_context_top_n)  # max(8, 14) = 14
recent_chat_context = await _fetch_chat_context(
    data,
    query=query,
    enable_semantic_search=False,   # ← ALWAYS False for the first pass
    top_n=14,
)
```

**Why recency-only first?** The planner will tell us if it needs semantically-retrieved older turns (via `needs_history_expansion=true`). This avoids a costly semantic search when not needed.

**Note:** `semantic_threshold = 0.18` is used when semantic search IS enabled (`rag_chat_context_semantic_threshold`).

### 3.2 The Unified LLM Planner

**Class:** `RetrievalPlanner` (retrieval_planner.py)  
**Called from:** `_build_retrieval_plan()` (line 241)

**Constructor parameters (from settings):**

| Parameter | Setting | Value |
|---|---|---|
| `temperature` | `rag_query_planner_temperature` | `0.0` (deterministic) |
| `max_tokens` | `rag_query_planner_max_tokens` | `380` |
| `max_turns` | `rag_query_planner_max_turns` | `6` |
| `max_context_chars` | `rag_query_planner_context_chars` | `2600` |
| `max_query_variants` | `rag_retrieval_query_variants` | `4` |
| `max_exact_terms` | `rag_exact_match_max_terms` | `4` |
| `default_time_field` | `rag_default_time_filter_field` | `"ingestion_date"` |

**Chat context formatting for the planner prompt** (`_chat_context_block()`):
- Takes the last `max_turns=6` turns from the normalized chat context
- Each turn formatted as:
  ```
  [Turn 3 | recent | most recent prior turn]
  User: Tell me about Chabahar port
  Assistant: Chabahar is Iran's deep-water port...
  ```
- Total capped at `max_context_chars=2600` chars (tail-truncated if longer)

**System prompt instructs the LLM to return one JSON object with keys:**

```json
{
  "followup": true,
  "context_dependent": true,
  "needs_history_expansion": false,
  "standalone_query": "Countries involved in Chabahar port development",
  "query_variants": [
    "Chabahar port international stakeholders",
    "India Iran cooperation Chabahar"
  ],
  "focus_subject": "Chabahar port",
  "focus_hint": "Tell me about Chabahar port",
  "exact_terms": ["Chabahar"],
  "filters": {
    "report_types": [],
    "branch": null,
    "doc_id": null,
    "parent_id": null,
    "lang": null,
    "is_attachment": null,
    "chunk_no": null
  },
  "time_filter": {
    "field": "none",
    "label": "",
    "start_date": "",
    "end_date": ""
  }
}
```

**Key LLM rules baked into the system prompt:**
- Resolve references like "above", "it", "same topic" using the supplied chat context
- `standalone_query` should be very close to original wording if NOT context-dependent
- `query_variants` must be short retrieval queries, best first, 1–4 items
- `exact_terms` = **only** literal entities, names, phone numbers, identifiers, operation names — NOT generic words like "explain", "profile", "report"
- Filters only when **explicit or strongly implied** in query
- Known `report_types` and `branch` values are injected into the prompt
- Time: Today's UTC date is injected. Relative phrases ("last 3 months") → absolute `start_date`/`end_date`
- If `time_filter.field = "none"`: no date filtering

---

#### 3.2.1 NEW — JSON Repair Call (`_repair_model_output()`)

**Lines 495–521 in retrieval_planner.py**

After the primary LLM call, if `_extract_json_object(raw)` fails to find a valid JSON object in the response, a **second LLM call** is triggered to repair the malformed output:

```python
payload = _extract_json_object(raw)          # try to parse primary response
if not payload:
    payload = await self._repair_model_output(
        raw_output=raw,
        query=query,
        chat_block=chat_block,
    )
```

`_repair_model_output()` sends:
- **System prompt:** "You repair malformed retrieval-planner output into strict JSON. Return one valid JSON object only with keys: followup, context_dependent, needs_history_expansion, standalone_query, query_variants, focus_subject, focus_hint, exact_terms, filters, time_filter. Do not add explanations."
- **User prompt:** The original user query + chat context + the malformed output from the first call

If _repair_ also fails → `payload = {}` → falls through to the deterministic fallback.

**When does this fire?** Only when the primary planner produces output that cannot be parsed as JSON at all (truncated response, markdown-wrapped JSON that the extractor missed, non-JSON prose). This is a rare path.

---

#### 3.2.2 NEW — Deterministic Pre-Classification (`_is_context_dependent_query()`)

**Lines 364–371 in retrieval_planner.py**

Before the deterministic fallback is built, the system uses a pure-regex heuristic to detect whether the user query is context-dependent:

```python
def _is_context_dependent_query(self, query: str) -> bool:
    if _FOLLOW_UP_PATTERN.search(text):           # "tell me more", "elaborate", "what about"
        return True
    if _DISCOURSE_REFERENCE_PATTERN.search(text): # "above", "earlier", "same topic", "aforementioned"
        return True
    tokens = self._query_tokens(text)
    return len(tokens) <= 12 and bool(_PRONOUN_PATTERN.search(text))  # short query with pronoun
```

**Patterns matched:**

| Pattern constant | Examples |
|---|---|
| `_FOLLOW_UP_PATTERN` | "tell me more", "more details", "what about", "continue", "elaborate", "expand", "go deeper", "and what", "what else" |
| `_DISCOURSE_REFERENCE_PATTERN` | "above", "below", "earlier", "previous", "prior", "before that", "mentioned above", "same topic", "same subject", "same one", "that one", "aforementioned" |
| `_PRONOUN_PATTERN` (only if ≤12 tokens) | "he", "she", "him", "her", "it", "its", "they", "them", "this", "that", "those", "these", "his", "their" |

**Why <=12 tokens for pronoun check?** Long queries (>12 tokens) with pronouns are usually standalone analytical queries ("What do they think about the nuclear deal history..."). Short ones ("what did he say?") are almost certainly follow-ups.

---

#### 3.2.3 NEW — Topic Chain Extraction (`_extract_topic_chain()`)

**Lines 397–431 in retrieval_planner.py**

Traverses the normalized chat history in **reverse** (newest → oldest) to extract structured topic information:

```python
{
    "base_topic": "Chabahar port",              # first non-context-dependent prior turn
    "contextual_fragment": "India involvement", # most recent context-dependent fragment
    "resolved_topic": "Chabahar port",          # base_topic or fallback_topic
    "combined_topic": "Chabahar port India involvement"  # merged if both are present
}
```

**Algorithm:**
1. Iterate prior turns in reverse (excluding the current query turn)
2. For each turn's `user_message`:
   - Clean it using `_clean_topic_candidate()` (strips time phrases, request prefixes like "tell me about", pronouns, discourse references, filler words)
   - If it is **context-dependent** (per `_is_context_dependent_query`): record it as `latest_contextual_fragment` (only the first one found going backward) and **continue** — do not use it as a base topic
   - If it is **standalone** and has specific terms: record it as `base_topic` and **stop**
3. `resolved_topic = base_topic OR fallback_topic (first cleaned turn)`
4. `combined_topic = "{base_topic} {latest_contextual_fragment}"` if both are non-empty

**`_clean_topic_candidate(text)`** strips:
- Time filter phrases (e.g., "from last 3 months")
- Request prefixes: "tell me about", "show me", "give me", "describe", "explain", "summarize", "what is", etc.
- Leading "about", "on"
- Discourse references: "above", "earlier", "previously"
- Pronouns: "he", "she", "it", "this", "that", etc.
- Filler: "please", "briefly", "in brief"

**Example chain for a conversation:**
```
Turn 1: "Tell me about Chabahar port"       → base_topic = "Chabahar port"
Turn 2: "What countries are involved?"      → context_dependent, contextual_fragment = "countries involved"
Turn 3 (current): "Role of India in above" → being planned now

Result:
  base_topic          = "Chabahar port"
  contextual_fragment = "countries involved"
  combined_topic      = "Chabahar port countries involved"
```

---

#### 3.2.4 NEW — Deterministic Fallback Payload (`_build_deterministic_fallback_payload()`)

**Lines 433–493 in retrieval_planner.py**

Builds a **complete retrieval plan purely from deterministic logic** (no LLM). This runs on **every single request** alongside the primary LLM call, and the result is used to potentially override or repair the LLM plan.

```python
def _build_deterministic_fallback_payload(
    query, chat_context, allow_history_expansion, expanded_history_used
) -> Dict[str, Any]:
```

**Logic flow:**

1. `context_dependent = _is_context_dependent_query(query)` — regex classifier
2. `topic_info = _extract_topic_chain(chat_context, query)` — chat walker
3. `cleaned_query = _clean_topic_candidate(query)` — strip noise from current query
4. `generic_only = (specific_terms ≤ 2 AND matches generic pattern words like "mentioned", "details", "said")`

**Query construction rules:**

| Condition | `standalone_query` produced |
|---|---|
| `context_dependent=True`, `generic_only=True`, topic resolved | `resolved_topic` (replace vague query entirely) |
| `context_dependent=True`, `base_topic` found | `"{base_topic} {cleaned_query}"` |
| `context_dependent=True`, no `base_topic`, cleaned_query is sub-string of resolved | `resolved_topic` |
| `context_dependent=True`, no `base_topic`, novel cleaned_query | `"{resolved_topic} {cleaned_query}"` |
| `context_dependent=False` | `cleaned_query` (strip noise only) |
| No topic resolved at all | `cleaned_query` |

**Query variants** always include `standalone_query`, `resolved_topic`, and `combined_topic` (deduped).

**Exact terms:** If a `focus_subject` is found, it is added to `exact_terms`.

**`needs_history_expansion`** is set `True` only when:
- `context_dependent=True`
- AND no `resolved_topic` was found (can’t resolve from available history)
- AND `allow_history_expansion=True`
- AND `expanded_history_used=False`

---

#### 3.2.5 NEW — Deterministic Override Logic

**Lines 741–784 in retrieval_planner.py**

After both the LLM plan (`plan`) and the deterministic plan (`deterministic_plan`) are built, the system decides whether to **merge the deterministic plan into the LLM plan**:

```python
should_override = bool(
    deterministic_plan.context_dependent
    and (
        not plan.planner_used                # LLM failed entirely
        or not plan.context_dependent        # LLM missed that it's a follow-up
        or plan.standalone_query == query    # LLM returned raw query unchanged (failed to resolve)
        or (deterministic_plan.focus_hint and not plan.focus_hint)  # LLM has no focus but deterministic does
    )
)
```

**If override triggered:**
1. Start with `merged_payload = LLM payload dict`
2. **Overwrite** with all deterministic fields (standalone_query, query_variants, focus_subject, focus_hint, context_dependent, followup, exact_terms, needs_history_expansion)
3. **Preserve** from the LLM output (if present): `filters`, `time_filter`, `exact_terms`
   - This is the key merge strategy: deterministic controls **query construction**, LLM controls **filter/time/exact term extraction**
4. Rebuild `plan` from the merged payload
5. Log: `"RAG planner deterministic fallback applied"`

**Why is this important?** The LLM can sometimes:
- Fail to detect a follow-up (marks `context_dependent=False` when it should be `True`)
- Return `standalone_query` identical to the raw user input (no context resolution happened)
- Have no `focus_hint` even when prior chat context clearly establishes the topic

The deterministic logic catches all these cases using pure regex + chat history walking, and patches up the LLM’s output.

**Summary of the full `plan()` flow (updated):**

```
plan() called
  ├─ Build chat_block for prompt
  ├─ Call LLM (temp=0.0, max_tokens=380)
  │     └─ On LLM error → skip to ⓧ
  ├─ _extract_json_object(raw_response)
  │     └─ If fails → _repair_model_output() [2nd LLM call]
  │           └─ If repair fails → payload = {}
  ├─ _normalize_plan_payload(payload) → plan (LLM plan)
  └─ _build_deterministic_fallback_payload() [always runs, no LLM]
       ├─ _is_context_dependent_query()
       └─ _extract_topic_chain()
  └─ _normalize_plan_payload(deterministic_payload) → deterministic_plan
  └─ should_override_with_deterministic?
       ├─ YES: merge(LLM filters/time/exact_terms + deterministic queries/focus)
       └─ NO:  use LLM plan as-is
  └─ Log and return final plan
ⓧ Fallback: _normalize_plan_payload({}) = minimal plan from deterministic only
```


**Output normalization (`_normalize_plan_payload()`):**
- `standalone_query` → cleaned, max 220 chars, defaults to raw query if empty
- `query_variants` → deduped, cleaned, max 4 items. Always starts with `standalone_query`, ends with original raw query
- `focus_hint` → falls back to `focus_subject` if empty
- `time_filter` → validated ISO dates, swapped if end < start, field must be in `{none, ingestion_date, document_date, either}`
- If `field = "either"` → resolved to `rag_default_time_filter_field = "ingestion_date"`

### 3.3 Optional History Expansion Pass

**Lines 250–272**

```python
if plan.needs_history_expansion:
    hybrid_top_n = max(14, rag_query_planner_hybrid_chat_top_n=18)  # = 18
    hybrid_chat_context = await _fetch_chat_context(
        query=plan.standalone_query,   # ← searches with the rewritten query
        enable_semantic_search=True,   # ← NOW semantic search is on
        top_n=18,
    )
    if chat_turn_count(hybrid_chat_context) > chat_turn_count(recent_chat_context):
        # Worth re-planning with richer context
        expanded_plan = await planner.plan(
            ...,
            chat_context=hybrid_chat_context,
            expanded_history_used=True,   # prevents infinite recursion
        )
        return expanded_plan, hybrid_chat_context
```

**When does `needs_history_expansion` get set True by the LLM?**  
When the planner determines the recent chat snippet is insufficient to resolve the query AND semantically-related older turns would materially help.

**Guard:** `rag_query_planner_allow_history_expansion = True` must be set.

**Why only if turn count increases?** If the semantic search didn't find more turns than recency, there's no point re-planning.

### 3.4 The `RetrievalPlan` Dataclass

```python
@dataclass
class RetrievalPlan:
    standalone_query: str        # LLM-resolved standalone query
    query_variants: List[str]    # Up to 4 retrieval queries (deduplicated)
    focus_subject: str           # Topic entity (e.g., "Chabahar port")
    focus_hint: str              # Prior turn context (e.g., "Tell me about Chabahar port")
    exact_terms: List[str]       # Entities for exact-match lane
    filters: Dict[str, Any]      # Structured metadata filters from LLM
    time_filter: Dict[str, Any]  # Date range {field, start_date, end_date, start_ms, end_ms}
    followup: bool               # True if conversational follow-up
    context_dependent: bool      # True if query references prior context
    needs_history_expansion: bool # True (but already consumed by planner)
    planner_used: bool           # False if LLM failed (fallback plan)
    raw_plan: Dict[str, Any]     # Raw LLM JSON for debugging
```

---

## 4. Stage 2 — Filter Resolution

**Method:** `_build_bigdata_filters()` (line 274)

Combines three filter sources into a single effective filter dict.

### 4.1 Filter Priority Ladder

For each filter field, precedence is: **Payload (explicit) → Planner (LLM) → Regex (inferred from query text)**

```python
# report_type priority:
report_types = explicit_report_types \
            or planner_report_types \
            or inferred_report_types

# branch, doc_id, parent_id, lang: first non-null wins in priority order
branch = payload["branch"] or plan.filters["branch"] or regex_inferred["branch"]

# is_attachment, chunk_no: same pattern
```

**`extract_bigdata_filters_from_query(query)`** — regex-based extraction using `RetrievalSupport`:

| Filter | Regex Pattern | Example Query |
|---|---|---|
| `report_types` | `_REPORT_TYPE_FIELD_PATTERN` → label-aware, plus `_report_type_value_pattern` (exact known values) | "show me EISI or FR reports" |
| `branch` | `_branch_value_pattern` (exact known branch names) | "ACE branch data" |
| `doc_id` | `_DOC_ID_FIELD_PATTERN` | "doc id ABC123" |
| `parent_id` | `_PARENT_ID_FIELD_PATTERN` | "parent_id 0B10EA6B" |
| `chunk_no` | `_CHUNK_NO_FIELD_PATTERN` | "chunk 3" |
| `lang` | `_LANG_FIELD_PATTERN` | "in arabic" |
| `is_attachment` | `_INCLUDE_ATTACHMENT_PATTERN` / `_EXCLUDE_ATTACHMENT_PATTERN` | "with attachments" / "exclude attachments" |

**Known values:** Loaded from `rag_known_report_types_csv` and `rag_known_branches_csv`. A compiled regex pattern (`_compile_known_value_pattern()`) is built for each — sorted longest-first, with word-boundary anchors. Case-insensitive, allows `[\s_-]` between tokens.

### 4.2 Time Filter Resolution

**`resolve_time_filter(query, plan)`** (retrieval_utils.py line 805):

```python
if plan.time_filter:       # LLM set a time filter
    return plan.time_filter

parsed = extract_ingestion_time_filter(query)   # regex fallback
if parsed:
    return parsed.as_dict()
return {}
```

**Priority:** LLM planner's time filter **first**. If the LLM said `field=none` (no time filter), the regex fallback still runs on the raw query.

**Applied to filters (line 316):**
- Only if no explicit date fields in payload
- If `field="document_date"` → sets `document_date_gte/lte`
- Else → sets `ingestion_date_gte/lte`

**Single-day date shortcut:** If payload contains `document_date` or `ingestion_date` (exact epoch ms) but no range is specified, a ±1 day window is auto-applied:
```python
document_date_gte = doc_date - 86_400_000
document_date_lte = doc_date + 86_400_000
```

### 4.3 Time Filter Parser

**`time_filter_parser.py`** — Pure regex, no LLM. Handles natural language date phrases:

| Pattern | Example | Result |
|---|---|---|
| `between X and Y` / `from X to Y` | "from Jan 2024 to March 2024" | start=Jan 1, end=Mar 31 |
| `after X` / `since X` | "since 15 March 2024" | start=Mar 15, end=now |
| `before X` / `until X` | "before June 2025" | start=epoch, end=Jun 30 |
| `in X` / `during X` | "in 2024" / "during February 2025" | Full year / full month range |
| `last N unit` / `past N unit` | "last 3 months" / "past 2 weeks" | rolling window from now |
| `this week/month/year` | "this month" | start-of-period to now |
| `today` | "today" | midnight to now |
| `yesterday` | "yesterday" | full previous day |

**Supported date formats:** ISO (`2024-01-15`), Day Month Year (`15 January 2024`), Month Day Year (`January 15, 2024`), Month Year (`March 2025`), Year only (`2024`)

**Numbers:** Supports written numbers (`one`, `two`, ... `twelve`) and digits up to 12.

**Returns:** `RelativeTimeFilter(field="ingestion_date", start_ms, end_ms, label, matched_text)` or `None`.

---

## 5. Stage 3 — Context Fetching

**Method:** `_fetch_rag_context()` (line 477)

### 5.1 Query & Exact-Term Finalization

```python
retrieval_queries = support.dedupe_queries(
    [plan.standalone_query, *plan.query_variants, user_query],
    max_items=rag_retrieval_query_variants  # 4
)
```

Always starts with `plan.standalone_query` (LLM-rewritten). The raw `user_query` is appended as a guaranteed last resort.

**Exact terms: `fallback_exact_terms(query, plan, max_items=4)`** (retrieval_utils.py line 488):

Candidates assembled in priority order:
1. `plan.exact_terms` (from LLM planner) — highest priority
2. Quoted phrases: `"narendra modi"`, `'operation X'`
3. Long numbers (≥5 digits): phone numbers, IDs — `+971189292`, `9873456789`
4. ID-like tokens with digits: `ABC-123`, `DOC-456`
5. `derive_entity_like_phrase()` — extracts subject from structured queries:
   - `"profile of narendra modi"` → `"narendra modi"`
   - `"give me number details +971..."` → `"+971..."`
   - `"who is john doe"` → `"john doe"`
   - Fallback: `specific_query_terms()` if 1–4 meaningful non-stopword tokens
6. `plan.focus_subject` if query ≤12 tokens (follow-up context enrichment)

Then deduped via `clean_exact_candidate()` (strips trailing noise like "please", "in brief") and normalized.

### 5.2 Attachment Path

**Triggered when:** `attachment_ids` non-empty OR (`has_attachments=True` AND `"bigdata"` NOT in `search_mode`)

```python
file_id_value = ",".join(attachment_ids)  # comma-separated for multi-file
retrieval_mode = "full_file" if attachment_ids or has_attachments else "semantic"
top_n = rag_file_chunk_cap(320) if full_file else rag_top_n_docs(16)
min_score = 0.0 if full_file else rag_min_score(0.08)
```

If attachment chunks are returned → **big-data path is skipped entirely.**

### 5.3 Big-Data Path

**`_fetch_bigdata_context(data, retrieval_queries, exact_terms, filters)`** (line 414)

```
candidate_top_n = rag_top_n_docs × rag_candidate_multiplier = 16 × 4 = 64

filter_variants = _expand_bigdata_filter_variants(filters)
# If report_types=[A, B, C] → 3 filter variants, each with one report_type
# If report_types=[] → 1 filter variant

# Exact-match lane starts ASYNC (does not block semantic):
if exact_terms and rag_exact_match_enabled:
    exact_task = asyncio.create_task(
        _fetch_bigdata_exact_context_multi(query=retrieval_queries[0], ...)
    )
```

### 5.4 Strict vs Relaxed Pass

```
STRICT PASS:
  _run_bigdata_batch(queries, filter_variants, top_n=64, min_score=0.08)
  → N_queries × M_filter_variants parallel HTTP calls to big_data_context_url
  Merge all results by identity

If merged is non-empty:
  → await exact_task, merge exact results, return

RELAXED PASS (only when strict returns 0):
  filters_relaxed = relax(filters)  # remove doc_id, parent_id, chunk_no
  _run_bigdata_batch(queries, filter_variants_relaxed, top_n=64, min_score=0.0)
  → await exact_task, merge exact results, return
```

**`_expand_bigdata_filter_variants(filters)`** (line 397):
- Pops `report_types` list
- Creates one filter dict per report type with `report_type` set to that value
- If `report_types` is empty → returns `[base_filters]` (single variant)

### 5.5 Exact-Match Parallel Lane

**`_fetch_bigdata_exact_context_multi()`** (line 178):
- Expands filter variants same way as semantic search
- Runs `_fetch_bigdata_exact_context()` per variant in parallel
- Merges results with `merge_chunks_by_identity()`
- Strips `report_types` from exact filters (keeps branch, doc_id, etc.)
- Also passes `elasticsearch_base_url` and `elasticsearch_index` from payload if present

**POST to `big_data_exact_context_url`:**
```json
{
  "query": "primary_retrieval_query",
  "keywords": ["narendra modi", "+971189292"],
  "top_n": 6,
  "report_type": "EISI",   // from filter variant
  "branch": "ACE",
  "elasticsearch_base_url": "...",  // optional, from payload
  "elasticsearch_index": "..."      // optional, from payload
}
```

---

## 6. Stage 4 — Reranking

**`RetrievalSupport.rank_chunks_for_query(chunks, primary_query, secondary_queries, topic_hint)`** (retrieval_utils.py line 615)

### Scoring Formula

**When `topic_hint` is non-empty (follow-up / context-dependent):**

```
score =   0.46 × semantic_similarity      (from _sim_score / _final_score / score field)
        + 0.22 × primary_keyword_overlap  (primary query terms in content)
        + 0.06 × secondary_keyword_overlap (best overlap across all secondary queries)
        + 0.22 × topic_term_overlap        (topic_hint specific terms, fuzzy-matched)
        + phrase_bonus                     (0.12 if full primary query in content,
                                            +0.04 if any secondary query ≥10 chars in content)
        + query_match_bonus                (0.04 per extra query match, max 0.12)
        + exact_bonus                      (0.34 × _exact_match_score, if > 0)
        + exact_keyword_count_bonus        (0.03 per extra matched keyword, max 0.10)
        - 0.06 penalty if topic_overlap ≤ 0.01
```

**When no `topic_hint` (standalone query):**

```
score =   0.62 × semantic_similarity
        + 0.26 × primary_keyword_overlap
        + 0.08 × secondary_keyword_overlap
        + phrase_bonus + query_match_bonus + exact_bonus + exact_keyword_count_bonus
```

**`term_match_score(term, content)`** (fuzzy):
- Exact substring → `1.0`
- Fuzzy ratio ≥ 0.92 → `0.90`
- Fuzzy ratio ≥ 0.86 → `0.65`
- Otherwise → `0.0`

**After scoring:**
- Sort descending
- Deduplicate by `chunk_dedupe_key` = `"{doc_id}|{file_id}|{chunk_no}"` (SHA1 of content if IDs absent)

**`secondary_queries`** passed in are:
```python
secondary_queries = dedupe_queries([
    *retrieval_queries[1:],  # all variants except primary
    user_query,
    plan.focus_hint,
    plan.focus_subject,
], max_items=max(4, len(retrieval_queries) + 2))
```

---

## 7. Stage 5 — Contextual Retry

**Lines 520–547** — Triggered when **ALL** of:
- `rag_contextual_retry_enabled = True`
- `plan.context_dependent = True`
- `plan.focus_subject` or `plan.focus_hint` is non-empty
- No attachments

**Process:**
1. `topic_hint = plan.focus_subject or plan.focus_hint`
2. `alignment = topic_alignment_score(top_6_ranked_chunks, topic_hint)`
   - Extracts specific (non-stopword, ≥4 char) terms from `topic_hint`
   - For each of top 6 chunks: fuzzy-scores their term overlap
   - Final = `0.65 × best + 0.35 × average`
3. If `alignment < rag_contextual_retry_topic_overlap_threshold (0.18)`:
   ```python
   topic_queries = dedupe_queries([
       f"{retrieval_queries[0]} {topic_hint}",
       f"{topic_hint} {retrieval_queries[0]}",
       topic_hint,
       user_query,
   ], max_items=4)
   retry_chunks = await _fetch_bigdata_context(topic_queries, exact_terms, filters)
   ranked = rank_chunks_for_query(
       merge(initial_chunks + retry_chunks),
       primary_query,
       secondary_queries=[*secondary_queries, *topic_queries],
       topic_hint=topic_hint
   )
   ```

**No LLM call in retry.** Just new retrieval queries composed from the focus subject.

---

## 8. Stage 6 — Exact-Match Slot Guarantee

**`ensure_exact_match_presence(ranked, top_k=20, guaranteed=4)`** (retrieval_utils.py line 705):

```
1. Filter ranked for chunks with _exact_match_score > 0.0
2. Take first min(len(exact_chunks), guaranteed=4) → force-include
3. Fill remaining slots (up to top_k=20) from full ranked list (no duplicates)
4. Return capped at top_k
```

This guarantees that up to `rag_exact_match_guaranteed_chunks=4` exact-match chunks are always in the final prompt, even if their overall ranking score was displaced by high-scoring semantic chunks.

---

## 9. Stage 7 — LLM Answer Generation

**`_rag()` (line 681) → `_llm_generate()` (line 635)**

```python
system_prompt = build_rag_system_prompt()
user_prompt = build_rag_context_text(
    query=user_query,
    kb_chunks=final_chunks,                   # after ensure_exact_match_presence
    chat_context=chat_ctx,
    max_chunk_chars=rag_max_chunk_chars,       # 2600 per chunk
    max_chat_messages=rag_max_chat_messages,   # 8 messages
    resolved_query=retrieval_plan["primary_query"],
    retrieval_focus=plan.focus_subject or plan.focus_hint,
    applied_time_filter=retrieval_plan["time_filter"]["label"],  # e.g. "last 3 months"
)
```

If `kb_chunks` is empty → `build_no_context_message()` is emitted:
- If time filter was applied: `"I could not find documents matching the requested filters for \`{label}\`. Try a broader time range..."`
- Otherwise: `"I could not find relevant knowledge-base context... Please refine the query..."`

**LLM call:**
```
model = openai/gpt-oss-120b
temperature = rag_temperature = 0.15
max_tokens = 2048
stream = True
```

**LLM Trace** (`llm_trace_enabled=True`): Final answer prompts are written to `logs/rag_llm_trace.log` via `asyncio.to_thread` (non-blocking I/O). Protected by `asyncio.Lock`.

**Planner Trace** (`rag_planner_trace_enabled=True`): Retrieval-planner prompts, raw planner outputs, repair-call outputs, and the final normalized retrieval plan are written to `logs/rag_planner_trace.log`.

---

## 10. Stage 8 — Save & Stream

**Token streaming:** `redis.xadd(stream, {"data": token}, maxlen=10000)`  
**End signal:** `redis.xadd(stream, {"end": "1"}, maxlen=1000)`

**Status transitions:**
```
"answering" → first token arrives → "completed" → after {end:1} → "finished"
```

**`save_assistant_response()`** (line 653):
```
POST http://192.168.10.52:5000/message/save_assistant_response
{
    "user_id": ...,
    "chat_id": ...,
    "message_id": system_response_id,    ← fresh UUID
    "role": "system",
    "content": answer,
    "seq": user_seq + 1,
    "search_mode": search_mode,
    "source_documents": []               ← always empty (API contract)
}
aiohttp timeout: 10s
```

---

## 11. `RetrievalSupport` Utility Class

All utility functions now consolidated in `retrieval_utils.py`.

### 11.1 Filter Extraction From Query

- `extract_bigdata_filters_from_query(query)` — regex extraction for report_type, branch, doc_id, parent_id, chunk_no, lang, is_attachment
- `normalize_report_type_values(raw)` — canonicalize to known report types, split on `,/;|or and`
- `canonicalize_report_type(value)` — lookup → known value; if not found, pass-through as-is (case-preserved)
- `match_known_value(text, value_pattern, lookup)` — single-match
- `extract_known_values(text, value_pattern, lookup, max_items)` — all matches

### 11.2 Exact Term Extraction

`fallback_exact_terms(query, plan, max_items)` — see [Section 5.1](#51-query--exact-term-finalization)

`derive_entity_like_phrase(text)`:
1. Strip time filter phrase from text first (`strip_time_filter_phrase()`)
2. Try `_EXACT_ENTITY_CUE_PATTERNS`:
   - `"profile of X"` / `"details of X"` / `"bio of X"` → extracts `X`
   - `"who is X"` / `"what is X"` / `"tell me about X"` → extracts `X`
   - `"give me number details X"` → extracts `X`
3. Fallback: extract 1–4 specific terms (non-stopword, ≥4 chars)

### 11.3 Chunk Merging

**`merge_chunks_by_identity(query_to_chunks: List[(query, chunks)])`** (line 752):

For each chunk:
- Key = `chunk_dedupe_key(chunk)` = `"{doc_id}|{file_id}|{chunk_no}"`
- If new key: store chunk as-is, set `_retrieval_queries=[query]`, `_query_match_count=1`
- If existing key:
  - Keep **highest** `_sim_score`
  - Keep **highest** `_exact_match_score`
  - Keep **highest** `_final_score`
  - **Append** query to `_retrieval_queries`
  - Set `_query_match_count = len(_retrieval_queries)`
  - **Merge** `_matched_keywords` (deduplicated)

The `_query_match_count` is used in the scoring formula as `query_match_bonus`.

### 11.4 Scoring Formula

See [Section 6](#6-stage-4--reranking) for the complete formula. New vs old version:

| Component | Old Version | New Version |
|---|---|---|
| `exact_bonus` | Not present | `0.34 × _exact_match_score` |
| `exact_keyword_count_bonus` | Not present | `0.03 × (matched_keywords - 1)`, max `0.10` |
| Max `phrase_bonus` | `0.12` | `0.12 + 0.04 = 0.16` (secondary phrase adds 0.04) |
| `secondary_weight` (topic mode) | `0.06` | `0.06` (same) |

---

## 12. All Settings Reference

| Setting | Default | Role |
|---|---|---|
| **Infrastructure** | | |
| `redis_url` | `redis://:Redis@123@192.168.10.35:6379/0` | Redis |
| `stream_name` | `tasks.rag` | Input job stream |
| `group_name` | `rag` | Consumer group |
| `worker_name` | `rag_01` | Worker identity |
| **LLM** | | |
| `vllm_base_url` | `http://192.168.10.210:8000/v1` | vLLM endpoint |
| `model_name` | `openai/gpt-oss-120b` | Used for ALL LLM calls |
| `rag_temperature` | `0.15` | Answer generation |
| **Retrieval Core** | | |
| `rag_top_n_docs` | `16` | Candidates per vector search call |
| `rag_min_score` | `0.08` | Min cosine sim (strict pass) |
| `rag_candidate_multiplier` | `4` | `top_n = 16 × 4 = 64` per search |
| `rag_file_chunk_cap` | `320` | Full-file attachment chunk cap |
| `rag_max_prompt_chunks` | `20` | Max chunks in LLM prompt |
| `rag_max_chunk_chars` | `2600` | Per-chunk char cap in prompt |
| `rag_emit_source_documents_limit` | `12` | Max source docs in response |
| **Chat Context** | | |
| `rag_max_chat_messages` | `8` | Messages in LLM answer prompt |
| `rag_chat_context_top_n` | `14` | Messages fetched for planning |
| `rag_chat_context_semantic_threshold` | `0.18` | Min score when semantic search enabled |
| **Unified Planner (NEW)** | | |
| `rag_query_planner_temperature` | `0.0` | Deterministic planner output |
| `rag_query_planner_max_tokens` | `380` | Max JSON output tokens |
| `rag_query_planner_max_turns` | `6` | Chat turns fed to planner |
| `rag_query_planner_context_chars` | `2600` | Max chat chars in planner prompt |
| `rag_query_planner_allow_history_expansion` | `True` | Enable 2nd semantic pass |
| `rag_query_planner_hybrid_chat_top_n` | `18` | Messages for expanded history pass |
| `rag_default_time_filter_field` | `"ingestion_date"` | Default when `field="either"` |
| `rag_retrieval_query_variants` | `4` | Max search query variants |
| **Legacy Settings (kept for compatibility)** | | |
| `rag_enable_query_rewrite` | `True` | Superseded by planner |
| `rag_rewrite_only_on_followups` | `True` | Superseded by planner |
| `rag_query_rewrite_max_turns` | `4` | Superseded by planner |
| `rag_enable_contextual_query_planner` | `True` | Superseded by unified planner |
| `rag_contextual_query_planner_only_on_contextual` | `True` | Superseded by unified planner |
| **Contextual Retry** | | |
| `rag_contextual_retry_enabled` | `True` | Topic-misalignment retry |
| `rag_contextual_retry_topic_overlap_threshold` | `0.18` | Min score before retry |
| **Exact Match** | | |
| `rag_exact_match_enabled` | `True` | Master switch |
| `rag_exact_match_top_n` | `6` | Max exact chunks per request |
| `rag_exact_match_guaranteed_chunks` | `4` | Forced slots in final context |
| `rag_exact_match_max_terms` | `4` | Max extracted exact terms |
| **Known Values** | | |
| `rag_known_branches_csv` | `ACE,DELHI,MUMBAI,BENGALURU,CHENNAI` | Filter inference |
| `rag_known_report_types_csv` | `Collation,FR,MR,SPR,WR,Notes,UO,SR,...` | Filter inference |
| **Tracing** | | |
| `llm_trace_enabled` | `True` | Log prompts to file |
| `llm_trace_file` | `logs/rag_llm_trace.log` | Trace log path |
| `rag_planner_trace_enabled` | `True` | Log retrieval-planner prompts/results to file |
| `rag_planner_trace_file` | `logs/rag_planner_trace.log` | Retrieval-planner trace log path |

---

## 13. Scenario Walkthroughs

### 13.1 Standalone Query

**User:** `"Tell me about Chabahar port"`

```
1. _build_retrieval_plan():
   a. Recent chat: empty (first message), top_n=14, semantic=False
   b. Planner LLM call with no prior chat context:
      → followup=False, context_dependent=False
      → standalone_query = "Chabahar port overview"
      → query_variants = ["Chabahar port development", "Iran deep water port"]
      → focus_subject = "", focus_hint = ""
      → exact_terms = ["Chabahar"]
      → time_filter = {field: "none"}
   c. needs_history_expansion=False → no 2nd pass

2. _build_bigdata_filters():
   → no explicit filters, no planner filters, no regex match
   → filters = all None

3. retrieval_queries = ["Chabahar port overview", "...", "Tell me about Chabahar port"]

4. exact_terms = dedupe(plan.exact_terms=["Chabahar"] + fallback)
   → derive_entity_like_phrase("Tell me about Chabahar port") → "Chabahar port"
   → exact_terms = ["Chabahar", "Chabahar port"] (deduped to 2)

5. _fetch_bigdata_context():
   - exact_task: POST /big_data_documents_exact_context {keywords: ["Chabahar", "Chabahar port"]}
   - strict: 3 queries × 1 filter = 3 parallel semantic searches
   - Merge → say 48 candidates

6. rank_chunks_for_query(chunks, "Chabahar port overview", secondary_queries=[...], topic_hint="")
   - No topic_hint → weights: 0.62 semantic, 0.26 primary overlap
   - exact_bonus applied to chunks from exact lane

7. topic_alignment: context_dependent=False → retry SKIPPED

8. ensure_exact_match_presence(ranked, top_k=20, guaranteed=4)
   → Guaranteed 4 exact-match chunks in final context

9. LLM generates streaming answer
```

### 13.2 Follow-up Query

**User:** `"Other countries involved"` (after Chabahar answer)

```
1. _build_retrieval_plan():
   a. Recent chat: [{user: "Tell me about Chabahar port", assistant: "..."}]
   b. Planner LLM (with chat context):
      → followup=True, context_dependent=True
      → standalone_query = "Countries involved in Chabahar port development"
      → query_variants = ["Chabahar port international stakeholders",
                          "India Iran cooperation Chabahar"]
      → focus_subject = "Chabahar port"
      → focus_hint = "Tell me about Chabahar port"
      → exact_terms = ["Chabahar"]
      → time_filter = {field: "none"}
   c. needs_history_expansion=False

2. retrieval_queries = ["Countries involved in Chabahar port development",
                         "Chabahar port international stakeholders",
                         "India Iran cooperation Chabahar",
                         "Other countries involved"]

3. _fetch_bigdata_context():
   - 4 queries × 1 filter = 4 parallel searches
   - Exact lane: keywords = ["Chabahar"]

4. rank_chunks_for_query(..., topic_hint="Chabahar port")
   - topic_hint present → weights: 0.46 semantic, 0.22 primary, 0.22 topic
   - Chunks not mentioning "Chabahar" → penalized -0.06

5. topic_alignment_score(top_6, "Chabahar port"):
   - All chunks from vector search for "Chabahar port development" likely have
     high Chabahar overlap → alignment > 0.18 → retry SKIPPED

6. LLM prompt includes:
   - resolved_query = "Countries involved in Chabahar port development"
   - retrieval_focus = "Chabahar port"
   - 8 chat messages (full conversation history)
```

**Key difference from old version:** Whereas the old version would NOT have enriched "Other countries involved" (3 specific terms ≥ threshold), the new unified planner **ALWAYS** runs and resolves the query in context. There's no threshold logic — the LLM decides.

### 13.3 Entity Query

**User:** `"give me profile of narendra modi"`

```
1. Planner:
   → standalone_query = "Narendra Modi profile background"
   → exact_terms = ["Narendra Modi"]  ← LLM extracts the name
   → focus_subject = "Narendra Modi"

2. fallback_exact_terms("give me profile of narendra modi", plan):
   - plan.exact_terms = ["Narendra Modi"] → comes first
   - derive_entity_like_phrase: _EXACT_ENTITY_CUE_PATTERNS match "profile of narendra modi"
     → extracted = "narendra modi"
   - deduped: ["Narendra Modi", "narendra modi"] → normalized → ["Narendra Modi"]

3. exact_task fires: POST /exact_context {keywords: ["Narendra Modi"]}
   - Elasticsearch does phrase-match on content for "narendra modi"
   - Returns chunks with _exact_match_score values

4. In ranking: exact_bonus = 0.34 × _exact_match_score pushes exact-match chunks up
5. ensure_exact_match_presence: up to 4 exact-match slots guaranteed
```

### 13.4 Time-Scoped Query

**User:** `"show me Collation reports from last 3 months"`

```
1. Planner:
   → standalone_query = "Collation reports recent"
   → filters.report_types = ["Collation"]
   → time_filter = {field: "ingestion_date", start_date: "2025-12-11", end_date: "2026-03-11"}
     (LLM computes: today=2026-03-11, 3 months back = 2025-12-11)

2. _build_bigdata_filters():
   → report_types = ["Collation"] (from planner)
   → time_filter resolved → ingestion_date_gte = 1765411200000, ingestion_date_lte = 1741651200000 (approx)

3. filter_variants = [{ report_type: "Collation", ingestion_date_gte: ..., ingestion_date_lte: ... }]

4. Retrieval: vector search constrained to Collation docs in last 3 months

5. If 0 results:
   - Relaxed pass: remove doc_id/parent_id/chunk_no (report_type and dates kept)
   - min_score = 0.0

6. No context found → build_no_context_message():
   "I could not find documents matching the requested filters for `last 3 months`.
    Try a broader time range or fewer filters."
```

Note that `time_filter_parser.py` also runs as fallback if LLM sets `field="none"` despite "last 3 months" being in the query. The regex parser would independently extract the same range.

### 13.5 History Expansion

**User:** `"What was the earlier suggestion given about the port?"` (chat is old, not recently discussed)

```
1. Recent chat (top_n=14, semantic=False):
   → Only today's messages, no port discussion

2. Planner (first call):
   → context_dependent=True, followup=False
   → standalone_query = "earlier suggestion about the port"
   → needs_history_expansion = True  ← LLM says: recent turns insufficient, need older context

3. rag_query_planner_allow_history_expansion = True → trigger expansion:
   a. hybrid_chat_context = _fetch_chat_context(
          query="earlier suggestion about the port",
          enable_semantic_search=True,   ← NOW semantic is on
          top_n=18,
      )
   b. If hybrid has more turns than recent → re-plan:
      expanded_plan = await planner.plan(
          chat_context=hybrid_chat_context, expanded_history_used=True
      )
   c. Return expanded_plan with richer context

4. With the older conversation turns now available, the planner can:
   → standalone_query = "Chabahar port development suggestions from prior discussion"
   → focus_subject = "Chabahar port"
   → query_variants = ["port development recommendations", "...]
```

---

## 14. Full Pipeline Diagram

```
Job arrives from tasks.rag stream
        │
        ▼
process(job_id, data)
  ├── Validate: message_id, task key, not cancelled
  ├── Set Redis: status=answering
  │
  ▼
_rag(payload, stream_name)
  │
  ▼
_fetch_rag_context()
  │
  ├─ _build_retrieval_plan()
  │     │
  │     ├─ _fetch_chat_context(semantic=False, top_n=14)
  │     │       [HTTP → /chat_context_new]
  │     │
  │     ├─ RetrievalPlanner.plan(query, recent_chat, explicit_filters)
  │     │       [LLM call: temp=0.0, max_tokens=380, returns JSON]
  │     │       → RetrievalPlan {
  │     │           standalone_query, query_variants (up to 4),
  │     │           focus_subject, focus_hint,
  │     │           exact_terms (up to 4),
  │     │           filters, time_filter,
  │     │           followup, context_dependent,
  │     │           needs_history_expansion
  │     │         }
  │     │
  │     └─ [if needs_history_expansion AND allow_expansion]:
  │           _fetch_chat_context(semantic=True, top_n=18)
  │           RetrievalPlanner.plan(query, hybrid_chat, expanded_history_used=True)
  │           [2nd LLM call, only if turn count increased]
  │
  ├─ _build_bigdata_filters(data, plan=plan)
  │     Merge (priority: payload → planner → regex inferred)
  │     + resolve_time_filter(query, plan)
  │     → {report_type, branch, doc_id, lang, ingestion_date_gte/lte, ...}
  │
  ├─ fallback_exact_terms(query, plan, max_items=4)
  │     Priority: [planner terms] → [quoted phrases] → [long numbers] →
  │               [ID-like with digits] → [entity cue extraction] → [specific terms]
  │
  ├─ Has attachments?
  │     YES → _fetch_attachment_context(retrieval_queries[0])  [HTTP → /uploaded_file_context_new]
  │           If chunks → skip big-data
  │
  └─ (else) _fetch_bigdata_context(retrieval_queries, exact_terms, filters)
        │
        ├─ _expand_bigdata_filter_variants(filters)
        │     → [variant_per_report_type] or [base_filters]
        │
        ├─ asyncio.create_task(exact lane)     [HTTP → /big_data_documents_exact_context]
        │     Parallel per filter variant, merge results
        │
        ├─ STRICT: N queries × M variants = N×M parallel searches
        │     [HTTP → /big_data_documents_context, top_n=64, min_score=0.08]
        │     merge_chunks_by_identity()
        │
        ├─ If strict > 0: await exact_task, merge, return
        │
        └─ RELAXED: N queries × M (relaxed) variants
              [min_score=0.0, no doc_id/parent_id/chunk_no]
              await exact_task, merge, return

  ├─ rank_chunks_for_query(merged, primary_query, secondary_queries, topic_hint)
  │     Scoring: semantic + keyword overlap + topic overlap + phrase bonus +
  │              query_match_bonus + exact_bonus + exact_keyword_count_bonus
  │
  ├─ [if context_dependent AND not attachments]:
  │     topic_alignment_score(top_6, focus_subject)
  │     if alignment < 0.18 → topic-focused retry:
  │         _fetch_bigdata_context(topic_queries)
  │         merge(initial + retry)
  │         re-rank with topic_hint
  │
  ├─ ensure_exact_match_presence(ranked, top_k=20, guaranteed=4)
  │     → Force 4 exact-match slots
  │
  ▼
_rag() cont.
  ├─ No chunks? → emit no-context message (with time filter label if applicable)
  │
  ├─ build_rag_system_prompt()
  ├─ build_rag_context_text(query, chunks, chat_ctx, resolved_query, focus, time_label)
  ├─ _llm_generate(stream=True, temp=0.15, max_tokens=2048)
  │     → stream tokens → Redis xadd
  │     → {end:1} → status=finished
  │
  └─ save_assistant_response()  [HTTP POST, aiohttp, 10s timeout]
```

---

## 15. Known Behaviours & Edge Cases

### The Planner Always Runs (Unlike the Old Version)
The old system had threshold-based logic: "only rewrite if context_dependent=True". The new planner **always runs for every request**. The only exception is a planner LLM failure, in which case `planner_used=False` and the raw query is used directly.

### Two LLM Calls for History Expansion
If `needs_history_expansion=True`, the system makes **two planner calls** per request (each up to 380 tokens). This is the only case where the planning stage does an extra HTTP call AND an extra LLM call. The total latency budget for planning is therefore up to `2 × planner_call_time`.

### Time Filter: Regex Fallback When Planner Says "none"
The planner may say `time_filter.field="none"` even when the query has "last 3 months". The `resolve_time_filter()` method always tries the regex parser as a fallback. If BOTH return nothing, no date filter is applied.

### report_types vs report_type in Filter Dict
The filter dict has **both** `report_types: List[str]` AND `report_type: str` (the first element of the list or None). `_expand_bigdata_filter_variants()` uses `report_types` to fan out into one variant per type. The downstream API receives `report_type` (singular) per variant call.

### Exact Match Timing
The exact lane (`exact_task`) is started **before** the strict semantic pass. It runs concurrently. It's only awaited AFTER the strict results arrive. If strict returns 0 results, the exact lane is still awaited once in the relaxed path. This means the exact-match API call happens regardless of the semantic outcome.

### Zero-Results Fallback Message Has Time Context
If no chunks were found AND a time filter was applied, the error message says `"matching the requested filters for \`{label}\`. Try a broader time range..."`. If no time filter, it gives the generic metadata prompt message.

### Cancellation Not Checked Mid-Retrieval
Unlike the QAWorker summarization batch loop, the RAGWorker does NOT check cancellation between retrieval stages. Cancellation is only checked **at the start** of `process()`. A running RAG job will complete even if cancelled mid-flight.

---

## 16. Why 12 Source Documents Are Always Emitted to the UI

### 16.1 The Problem

Every RAG response sends exactly up to **12 source documents** to the UI, regardless of whether:
- The LLM actually used or referenced those documents in its answer
- The documents are topically relevant to the final answer
- The documents have meaningful content related to the user's query

This happens because the source-document list is built **before** the LLM generates its answer, and no post-generation filtering is performed.

### 16.2 Root Cause — Code Flow Traced

The entire chain lives in `_rag()` (rag_worker.py lines 1085–1125):

```python
# Step 1: Fetch & rank chunks (retrieval pipeline)
kb_chunks, chat_ctx, retrieval_plan = await self._fetch_rag_context(payload)

# Step 2: Build source docs FROM THE SAME kb_chunks — BEFORE asking the LLM
source_documents = build_source_documents(
    kb_chunks,
    limit=int(settings.rag_emit_source_documents_limit)   # default = 12
)

# Step 3: Resolve branch-based access control
access_by_branch = await self._resolve_source_document_access(...)
if access_by_branch:
    source_documents = build_source_documents(
        kb_chunks,
        limit=int(settings.rag_emit_source_documents_limit),   # still 12
        access_by_branch=access_by_branch,
    )

# Step 4: Call LLM — source_documents are passed INTO _llm_generate
#          and emitted to Redis BEFORE the first LLM token arrives
answer = await self._llm_generate(
    message_id=..., stream=...,
    system_prompt=system_prompt,
    user_query=user_prompt,
    source_documents=source_documents,   # ← emitted immediately
)
```

Inside `_llm_generate()` (lines 916–947):
```python
if source_documents:
    await self.redis.xadd(
        stream,
        {"sources": json.dumps(source_documents, ensure_ascii=False)},
        maxlen=10000, approximate=True,
    )
# THEN the LLM streaming begins...
async for token in self._call_llm_stream(...):
    await self.redis.xadd(stream, {"data": token}, ...)
```

**Key insight:** The `sources` message is pushed to the Redis stream **before the LLM has produced any output**. The UI receives sources first, then the streaming answer tokens. By the time the LLM generates its answer and cites (or doesn't cite) specific chunks, the source documents are already sent.

### 16.3 How `build_source_documents()` Selects The 12 Documents

The function is in `prompt.py` (lines 257–304):

```python
def build_source_documents(
    kb_chunks: List[Dict[str, Any]],
    *,
    limit: int = 12,             # ← hardcoded default, overridden by settings
    excerpt_chars: int = 220,
    access_by_branch: Dict[str, bool] | None = None,
) -> List[Dict[str, Any]]:
```

It iterates through `kb_chunks` **in their ranked order** and:
1. Deduplicates by `(url, doc_id, parent_id)` tuple
2. Skips chunks with no content
3. Takes the first `limit` (12) unique results
4. For each, includes: `doc_id`, `parent_id`, `chunk_no`, `report_type`, `branch`, `score`, `url`, `document_date`, `excerpt` (truncated to 220 chars), `access` (branch ACL)

**There is NO relevance threshold.** If `kb_chunks` contains 20 ranked chunks, the top 12 unique documents are always emitted regardless of their scores or relevance to the actual answer.

### 16.4 Why Documents Can Appear Unrelated

The `kb_chunks` list arriving at `build_source_documents()` is the output of `ensure_exact_match_presence()`, which caps at `rag_max_prompt_chunks = 20`. These 20 chunks come from a multi-stage pipeline:

1. **64 candidates** fetched per search call (16 × 4 multiplier)
2. Multiple query variants and filter variants → potentially **hundreds** of raw candidates
3. Merged, scored, ranked by a heuristic formula (semantic + keyword + exact + topic overlap)
4. Optionally reranked by an external reranker
5. Guaranteed 4 exact-match slots inserted even if their overall score is low
6. Capped to top 20

Among these 20 chunks, the **bottom half** can have low relevance scores. When deduplicated to 12 unique source documents, several may be topically unrelated to the answer.

Additionally, the LLM's system prompt says to cite `[doc_id:chunk_no]` or `[source:<index>]` — but the LLM may choose to only use 3–4 of the 20 provided chunks while ignoring the rest. All 12 are still sent to the UI.

### 16.5 The Numbers Gap

| Constant | Default | Purpose |
|---|---|---|
| `rag_max_prompt_chunks` | **20** | Chunks injected into LLM prompt |
| `rag_emit_source_documents_limit` | **12** | Source docs emitted to UI |
| LLM actually cites | **2–6 typically** | Chunks the LLM references in its answer |

So the pipeline gives 20 chunks to the LLM, shows 12 to the user, but the LLM typically only cites 2–6 of them.

### 16.6 Can You Send Only LLM-Referenced Documents?

**Yes, but it requires a design change. The current architecture emits sources BEFORE the LLM responds, so you cannot know which ones the LLM will cite.**

Here are the approaches, ordered from simplest to most robust:

---

#### Approach A: Post-Generation Citation Extraction (Recommended)

**Concept:** Let the LLM finish generating its answer, then scan the answer for citation patterns, and emit only the cited source documents.

**How it works:**

1. **Buffer the full LLM answer** (already done — `collected` variable in `_llm_generate()` accumulates all tokens)
2. **Extract citations** from the answer text using regex:
   ```python
   # The system prompt tells the LLM to cite as:
   #   [doc_id:chunk_no]  or  [parent_id:chunk_no]  or  [source:<index>]
   import re
   
   def extract_cited_indices(answer: str, source_documents: list) -> set:
       cited = set()
       # Match [source:N] patterns
       for m in re.finditer(r'\[source:(\d+)\]', answer):
           cited.add(int(m.group(1)) - 1)  # 0-indexed
       # Match [doc_id:chunk_no] or [parent_id:chunk_no]
       for m in re.finditer(r'\[([^\]]+?):(\d+)\]', answer):
           ref_id = m.group(1).strip()
           ref_chunk = m.group(2).strip()
           for idx, doc in enumerate(source_documents):
               if (doc.get("doc_id") == ref_id or doc.get("parent_id") == ref_id):
                   if str(doc.get("chunk_no")) == ref_chunk:
                       cited.add(idx)
       return cited
   ```
3. **Filter source_documents** to only those cited
4. **Emit the filtered list AFTER the answer stream completes** (move the `sources` xadd after the streaming loop)

**Required changes to `_llm_generate()`:**

```python
async def _llm_generate(self, *, message_id, stream, system_prompt, user_query,
                        source_documents=None) -> str:
    await self._append_llm_trace(...)
    
    # DO NOT emit sources here anymore
    
    collected = ""
    stream_started = False
    async for token in self._call_llm_stream(system_prompt=system_prompt, user_query=user_query):
        if not token:
            continue
        if not stream_started:
            stream_started = True
            await self.redis.hset(f"task:{message_id}", "status", "completed")
        collected += token
        await self.redis.xadd(stream, {"data": token}, maxlen=10000, approximate=True)
    
    # NOW filter and emit sources after LLM is done
    if source_documents:
        cited_indices = extract_cited_indices(collected, source_documents)
        if cited_indices:
            filtered = [source_documents[i] for i in sorted(cited_indices) if i < len(source_documents)]
        else:
            filtered = source_documents  # fallback: send all if no citations detected
        await self.redis.xadd(
            stream,
            {"sources": json.dumps(filtered, ensure_ascii=False)},
            maxlen=10000, approximate=True,
        )
    
    if not stream_started:
        await self.redis.hset(f"task:{message_id}", "status", "completed")
    await self.redis.xadd(stream, {"end": "1"}, maxlen=1000, approximate=True)
    await self.redis.hset(f"task:{message_id}", "status", "finished")
    return collected
```

**Trade-off:** Sources appear in the UI **after** the answer (instead of before). The UI would need to handle this — either show sources at the bottom, or wait for the full response before rendering the source panel.

---

#### Approach B: Score Threshold Filtering (Quick Fix, Partial Solution)

**Concept:** Before emitting, filter out source documents below a minimum relevance score.

```python
# In _rag(), after building source_documents:
MIN_SOURCE_SCORE = 0.15   # configurable threshold
source_documents = [doc for doc in source_documents if doc.get("score", 0.0) >= MIN_SOURCE_SCORE]
```

**Limitation:** This reduces noise but still doesn't know which docs the LLM actually used. A high-scoring document might still not be cited.

---

#### Approach C: Hybrid — Score Filter + Post-Generation Citation Match

Combine Approach A and B:
1. Apply a score threshold to pre-filter obviously irrelevant docs
2. After LLM completes, further filter to only cited docs
3. If the LLM cited nothing (no citations in output), fall back to the score-filtered list

This is the most robust approach but requires both the streaming change and a score threshold.

---

#### Approach D: Two-Phase Source Emission

**Concept:** Emit a preliminary source list before the LLM streams (for immediate UI rendering), then emit a refined list after the LLM finishes.

```python
# Phase 1: emit all sources (existing behavior)
await self.redis.xadd(stream, {"sources": json.dumps(source_documents)}, ...)

# LLM streams answer...
collected = await self._stream_llm(...)

# Phase 2: emit refined sources
cited = extract_cited_indices(collected, source_documents)
if cited:
    await self.redis.xadd(stream, {"refined_sources": json.dumps([source_documents[i] for i in cited])}, ...)
```

**Trade-off:** Requires UI to handle a `refined_sources` event and replace the initial list.

---

### 16.7 Summary

| Aspect | Current State |
|---|---|
| **Why always 12?** | `rag_emit_source_documents_limit = 12` in settings; `build_source_documents()` takes the top 12 unique chunks by rank |
| **When emitted?** | **Before** the LLM generates its answer (inside `_llm_generate()`, before streaming starts) |
| **Filtering by relevance?** | **None** — no score threshold, no citation matching |
| **Does LLM use all 12?** | **No** — LLM typically cites 2–6 of the 20 chunks in its prompt; the rest are ignored |
| **Best fix?** | **Approach A** (post-generation citation extraction) — move source emission after streaming, filter to cited-only docs |
| **Quickest fix?** | **Approach B** (score threshold) — add `MIN_SOURCE_SCORE` filter, removes obvious noise but not perfect |
```
