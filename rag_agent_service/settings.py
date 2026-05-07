from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    redis_url: str = Field(
        default="redis://:Redis@123@192.168.10.35:6379/0",
        description="Redis connection URL.",
    )
    stream_name: str = Field(default="tasks.rag_production", description="RAG worker stream.")
    stream_llm_response: str = Field(
        default="tasks.llm_response_production",
        description="Redis stream for logging final user-facing LLM responses.",
    )
    dead_letter_queue: str = Field(default="tasks.dlq_production", description="DLQ stream name.")
    group_name: str = Field(default="rag", description="Redis consumer group name.")
    worker_name: str = Field(default="rag_01", description="Worker name.")
    rag_worker_max_concurrency: int = Field(
        default=10,
        description="Maximum number of RAG jobs processed concurrently by one worker process.",
    )
    rag_worker_read_count: int = Field(
        default=10,
        description="Maximum RAG stream messages read per polling cycle.",
    )
    rag_worker_read_block_ms: int = Field(
        default=2000,
        description="Blocking read timeout in milliseconds for the RAG Redis stream.",
    )
    rag_worker_idle_sleep_s: float = Field(
        default=0.05,
        description="Small sleep between RAG polling cycles.",
    )
    rag_worker_job_timeout_s: float = Field(
        default=900.0,
        description="Hard timeout in seconds for one RAG stream job. <=0 disables the guard.",
    )
    rag_task_ttl_s: int = Field(
        default=1800,
        description="TTL refreshed on task:{message_id} while a RAG job is actively processing.",
    )
    rag_response_stream_ttl_s: int = Field(
        default=3600,
        description="TTL applied to per-message RAG response streams after finish, failure, or timeout.",
    )
    llm_shared_prompt_dir: str = Field(
        default="/home/user1/llm_shared/prompts/prod",
        description="Shared directory containing live prompt text files for QA, summary, translation, and RAG.",
    )

    conv_context_url: str = Field(
        default="http://192.168.10.67:5010/chat_context_new",
        description="Conversation context endpoint URL.",
    )
    file_context_url: str = Field(
        default="http://192.168.10.67:5010/uploaded_file_context_new",
        description="Uploaded file context endpoint URL.",
    )
    big_data_context_url: str = Field(
        default="http://192.168.10.67:5010/big_data_documents_context",
        description="Big-data context endpoint URL.",
    )
    big_data_exact_context_url: str = Field(
        default="http://192.168.10.67:5010/big_data_documents_exact_context",
        description="Exact-match big-data context endpoint URL.",
    )
    context_fetch_timeout_s: float = Field(default=60.0, description="Timeout for context fetch API calls.")
    context_fetch_retries: int = Field(default=3, description="Retries for context fetch API calls.")

    assistant_resp_endpoint: str = Field(
        default="http://192.168.10.182:5000/message/save_assistant_response",
        description="Assistant response save endpoint.",
    )
    rag_save_assistant_response_timeout_s: float = Field(
        default=10.0,
        description="Timeout for saving the final RAG assistant response.",
    )
    rag_save_assistant_response_retries: int = Field(
        default=2,
        description="Retry count for transient final RAG assistant-response save failures.",
    )

    # changed: RAG source access is now resolved from the user's access-code list in one API call per answer.
    rag_user_access_branches_endpoint: str = Field(
        default="http://192.168.10.182:5000/admin/user_access_branches",
        description="Endpoint that returns the user's access code list for chunk ACL evaluation.",
    )
    rag_file_access_endpoint: str = Field(
        default="http://192.168.10.182:5000/admin/file_access",
        description="Endpoint that returns whether a user can access a branch-restricted source document.",
    )
    rag_file_access_timeout_s: float = Field(
        default=6.0,
        description="Timeout for per-branch source-document access checks.",
    )
    rag_file_access_retries: int = Field(
        default=1,
        description="Retry count for per-branch source-document access checks.",
    )

    vllm_base_url: str = Field(
        default="http://192.168.10.210:8000/v1",
        description="OpenAI-compatible vLLM base URL.",
    )
    vllm_api_key: str = Field(default="EMPTY", description="vLLM API key.")
    model_name: str = Field(default="openai/gpt-oss-120b", description="Model name.")

    rag_temperature: float = Field(default=0.15, description="Sampling temperature for RAG answers.")
    rag_answer_max_tokens: int = Field(
        default=2048,
        description="Max output tokens for the main DB-grounded RAG answer call.",
    )
    rag_answer_stream_start_timeout_s: float = Field(
        default=120.0,
        description="Maximum seconds to wait for the answer stream request or first streamed event before failing the RAG job.",
    )
    rag_answer_stream_idle_timeout_s: float = Field(
        default=90.0,
        description="Maximum seconds to wait between streamed answer events before failing the RAG job.",
    )
    rag_top_n_docs: int = Field(
        default=16,
        description="Top chunks fetched for attachment-scoped semantic retrieval paths.",
    )
    rag_bigdata_semantic_top_n_per_query: int = Field(
        default=200,
        description="Explicit top_n fetched per semantic big-data query variant during the initial strict retrieval pass.",
    )
    rag_min_score: float = Field(default=0.3, description="Minimum semantic score for big-data retrieval.")
    rag_bigdata_min_score_floor: float = Field(
        default=0.18,
        description="Minimum semantic score floor for big-data retrieval, even when fallback retrieval asks for a weaker min_score.",
    )
    rag_relaxed_min_score: float = Field(
        default=0.18,
        description="Minimum semantic score used by the targeted relaxed semantic recovery pass.",
    )
    rag_reranker_enabled: bool = Field(
        default=True,
        description="If true, rerank merged big-data candidates with the external reranker before prompt selection.",
    )
    rag_reranker_base_url: str = Field(
        default="http://192.168.10.210:9082",
        description="Base URL for the external reranker service.",
    )
    rag_reranker_model: str = Field(
        default="",
        description="Optional model name included in vLLM /v1/rerank requests. Set this to the served-model-name for vLLM.",
    )
    rag_reranker_timeout_s: float = Field(
        default=120.0,
        description="Timeout for one reranker request.",
    )
    rag_reranker_candidate_pool_size: int = Field(
        default=1024,
        description="Maximum number of merged big-data candidates sent to the reranker.",
    )
    rag_reranker_semantic_pool_size: int = Field(
        default=800,
        description="Target semantic candidate count inside the reranker pool before exact/backfill rows.",
    )
    rag_semantic_query_variants: int = Field(
        default=2,
        description="Maximum top-ranked query variants that are allowed to hit Qdrant semantic retrieval.",
    )
    rag_reranker_exact_pool_size: int = Field(
        default=224,
        description="Target exact big-data candidate count inside the reranker pool before backfill rows.",
    )
    rag_reranker_min_candidates: int = Field(
        default=8,
        description="Minimum merged big-data candidates required before reranking is attempted.",
    )
    rag_reranker_doc_max_chars: int = Field(
        default=5000,
        description="Maximum per-document text length sent to the reranker after neighbor enrichment and template wrapping.",
    )
    rag_reranker_use_instruction: bool = Field(
        default=True,
        description="If true, prepend a Qwen-style task instruction to the reranker query before calling the rerank API.",
    )
    rag_reranker_qwen3_template_enabled: bool = Field(
        default=True,
        description="If true, wrap reranker query/documents with the Qwen3-Reranker chat-style query/document templates.",
    )
    rag_reranker_instruction: str = Field(
        default=(
            "Given a user question for an internal report/document RAG system, "
            "retrieve passages that directly answer the query. Prefer chunks with "
            "specific factual evidence, named entities, dates, locations, organizations, "
            "and event details over broad or generic mentions."
        ),
        description="Instruction text injected into the reranker query for instruction-aware reranker models such as Qwen3-Reranker.",
    )
    rag_file_chunk_cap: int = Field(default=320, description="Chunk cap for attachment-scoped full-file retrieval.")
    rag_max_chat_messages: int = Field(default=8, description="Maximum recent chat messages included in prompt.")
    rag_chat_context_top_n: int = Field(
        default=14,
        description="How many chat messages to request from chat-context service for retrieval planning.",
    )
    rag_chat_context_semantic_threshold: float = Field(
        default=0.18,
        description="Minimum semantic similarity for retrieving older chat turns when hybrid chat-context search is enabled.",
    )
    rag_chat_context_enable_semantic_for_standalone: bool = Field(
        default=False,
        description="If true, always run semantic chat-history retrieval even for standalone queries. Disabled by default to save latency and avoid cross-topic bleed.",
    )
    rag_max_prompt_chunks: int = Field(
        default=0,
        description="Hard cap for retrieved chunks injected into the final LLM answer prompt. <=0 uses token-budget-only dynamic selection.",
    )
    rag_max_chunk_chars: int = Field(default=0, description="Per-chunk content chars in RAG context. <=0 disables clipping.")
    rag_prompt_token_budget: int = Field(
        default=30000,
        description="Approximate token budget for the final RAG user prompt context.",
    )
    rag_prompt_chars_per_token: float = Field(
        default=4.0,
        description="Approximate chars-per-token used for dynamic prompt budgeting.",
    )
    rag_prompt_static_overhead_tokens: int = Field(
        default=1400,
        description="Reserved token budget for query, chat context, metadata, and instructions.",
    )
    rag_prompt_chunk_overhead_tokens: int = Field(
        default=80,
        description="Estimated prompt metadata overhead per selected context chunk.",
    )
    rag_prompt_min_chunk_tokens: int = Field(
        default=180,
        description="Minimum remaining token budget required to include a truncated chunk excerpt.",
    )
    rag_evidence_selector_enabled: bool = Field(
        default=True,
        description="If true, run a post-rerank evidence-selection stage before the coverage judge and final answer call.",
    )
    rag_evidence_selector_batch_token_budget: int = Field(
        default=30000,
        description="Approximate token budget for each selector LLM batch over the reranked chunk list.",
    )
    rag_evidence_selector_max_batches: int = Field(
        default=2,
        description="Maximum number of selector LLM batches taken sequentially from the reranked list.",
    )
    rag_evidence_selector_static_overhead_tokens: int = Field(
        default=1000,
        description="Reserved token budget per selector batch for query, metadata, and instructions.",
    )
    rag_evidence_selector_chunk_overhead_tokens: int = Field(
        default=80,
        description="Estimated selector prompt metadata overhead per chunk row.",
    )
    rag_evidence_selector_min_chunk_tokens: int = Field(
        default=160,
        description="Minimum remaining token budget required to include a truncated chunk excerpt in a selector batch.",
    )
    rag_evidence_selector_max_chars_per_chunk: int = Field(
        default=2200,
        description="Maximum chars from one reranked chunk shown to the selector LLM before batch-level truncation.",
    )
    rag_evidence_selector_model_name: str = Field(
        default="",
        description="Optional model override for evidence-selector LLM calls. Empty uses the main answer model.",
    )
    rag_evidence_selector_temperature: float = Field(
        default=0.0,
        description="Temperature for evidence-selector LLM calls.",
    )
    rag_evidence_selector_max_tokens: int = Field(
        default=2048,
        description="Maximum output tokens for one evidence-selector JSON response.",
    )
    rag_evidence_selector_max_evidence_ids_per_batch: int = Field(
        default=48,
        description="Maximum usable direct/indirect evidence source IDs accepted from one selector batch response.",
    )
    rag_evidence_selector_fallback_prompt_token_budget: int = Field(
        default=20000,
        description="Prompt token budget used for top-reranked fallback chunks when selector returns no valid evidence IDs.",
    )
    rag_coverage_judge_enabled: bool = Field(
        default=True,
        description="If true, run an intermediate answerability / coverage judge before final answer generation.",
    )
    rag_coverage_judge_model_name: str = Field(
        default="",
        description="Optional model override for the coverage judge. Empty uses the main answer model.",
    )
    rag_coverage_judge_temperature: float = Field(
        default=0.0,
        description="Temperature for the intermediate coverage judge call.",
    )
    rag_coverage_judge_max_tokens: int = Field(
        default=1024,
        description="Maximum output tokens for the intermediate coverage judge JSON response.",
    )
    rag_coverage_judge_max_chunks: int = Field(
        default=0,
        description="Maximum selected prompt chunks shown to the coverage judge. <=0 uses all chunks selected by the prompt token budget.",
    )
    rag_coverage_judge_max_chars_per_chunk: int = Field(
        default=0,
        description="Maximum chars per chunk excerpt shown to the coverage judge. <=0 uses each chunk's prompt-budgeted content.",
    )
    rag_recovery_enabled: bool = Field(
        default=True,
        description="If true, allow one targeted recovery pass after the coverage judge reports partial or insufficient evidence.",
    )
    rag_recovery_max_attempts: int = Field(
        default=1,
        description="Maximum number of targeted retrieval recovery passes after the coverage judge. Phase 1 should remain 1.",
    )
    rag_recovery_semantic_rescue_chunks: int = Field(
        default=8,
        description="Maximum count of pre-rerank strict semantic rescue chunks injected during the recovery pass.",
    )
    rag_recovery_next_reranked_window_chunks: int = Field(
        default=8,
        description="Maximum count of additional post-rerank chunks injected during the recovery pass when prompt budget clipping is suspected.",
    )
    rag_recovery_relaxed_semantic_enabled: bool = Field(
        default=True,
        description="If true, recovery may run one small relaxed semantic retrieval pass instead of a second broad pre-rerank sweep.",
    )
    rag_recovery_relaxed_semantic_top_n_per_query: int = Field(
        default=20,
        description="Per-query top_n for the targeted relaxed semantic recovery pass.",
    )
    rag_recovery_relaxed_semantic_total_chunks: int = Field(
        default=30,
        description="Maximum merged relaxed semantic rescue chunks kept from the targeted recovery pass.",
    )
    rag_recovery_exact_match_top_n: int = Field(
        default=96,
        description="Exact-match candidate cap for targeted recovery retries.",
    )
    rag_recovery_max_suggested_queries: int = Field(
        default=3,
        description="Maximum judge-suggested alternative queries considered during the targeted recovery pass.",
    )
    rag_recovery_max_suggested_exact_terms: int = Field(
        default=4,
        description="Maximum judge-suggested exact terms considered during the targeted recovery pass.",
    )
    rag_utility_llm_reasoning_effort: str = Field(
        default="low",
        description="Optional reasoning effort hint for non-streaming JSON utility calls such as selector and coverage judge. Empty disables the hint.",
    )
    # rag_emit_source_documents_limit: int = Field(default=12, description="Limit for response source document payload.")
    # rag_retrieval_query_variants: int = Field(
    #     default=4,
    #     description="Maximum retrieval query variants per request after planner normalization.",
    # )

    rag_emit_source_documents_limit: int = Field(default=40, description="Limit for response source document payload.")
    rag_general_background_enabled: bool = Field(
        default=True,
        description="If true, optionally append a separate general model-knowledge background section after DB-grounded RAG answers.",
    )
    rag_general_background_trigger_mode: str = Field(
        default="default_on",
        description="Controls addendum triggering: 'default_on' appends for most non-internal, non-current informational answers; 'explicit_or_insufficient' is conservative.",
    )
    rag_general_background_on_no_context: bool = Field(
        default=True,
        description="If true, allow the general-background section when no internal KB context was found for a general-public query.",
    )
    rag_general_background_on_insufficient_context: bool = Field(
        default=True,
        description="If true, allow the general-background section when the DB-grounded answer says evidence is insufficient for a general-public query.",
    )
    rag_general_background_max_tokens: int = Field(
        default=700,
        description="Max output tokens for the optional general-background LLM call.",
    )
    rag_general_background_temperature: float = Field(
        default=0.2,
        description="Temperature for the optional general-background LLM call.",
    )
    rag_retrieval_query_variants: int = Field(
        default=4,
        description="Maximum retrieval query variants per request after planner normalization.",
    )

    rag_query_planner_temperature: float = Field(
        default=0.0,
        description="Temperature for the unified retrieval planner.",
    )
    rag_query_planner_max_tokens: int = Field(
        default=1024,
        description="Max output tokens for the unified retrieval planner JSON output.",
    )
    rag_query_planner_max_turns: int = Field(
        default=6,
        description="Max chat turns passed to the retrieval planner.",
    )
    rag_query_planner_context_chars: int = Field(
        default=10000,
        description="Max chat-context chars passed to the retrieval planner.",
    )
    rag_query_planner_allow_history_expansion: bool = Field(
        default=True,
        description="If true, planner may trigger one semantic chat-history expansion pass when recent turns are insufficient.",
    )
    rag_query_planner_hybrid_chat_top_n: int = Field(
        default=18,
        description="Chat-context retrieval budget for planner-triggered semantic history expansion.",
    )
    rag_default_time_filter_field: str = Field(
        default="ingestion_date",
        description="Default field used when planner resolves a time range without a strong field preference.",
    )
    rag_enable_query_rewrite: bool = Field(
        default=True,
        description="Legacy setting retained for compatibility; the unified retrieval planner supersedes standalone rewrite.",
    )
    rag_rewrite_only_on_followups: bool = Field(
        default=True,
        description="If true, run query rewrite only when query appears to be conversational follow-up.",
    )
    rag_query_rewrite_max_turns: int = Field(
        default=4,
        description="Max recent turns included in query rewrite prompt.",
    )
    rag_query_rewrite_context_chars: int = Field(
        default=1800,
        description="Max conversation chars included in rewrite prompt.",
    )
    rag_query_rewrite_max_tokens: int = Field(
        default=180,
        description="Max model output tokens for standalone query rewrite step.",
    )
    rag_query_rewrite_temperature: float = Field(
        default=0.0,
        description="Temperature for standalone query rewrite step.",
    )
    rag_enable_contextual_query_planner: bool = Field(
        default=True,
        description="If true, use an LLM retrieval-planning step for contextual or ambiguous follow-up queries.",
    )
    rag_contextual_query_planner_only_on_contextual: bool = Field(
        default=True,
        description="If true, run the retrieval planner only for context-dependent or discourse-referential queries.",
    )
    rag_contextual_query_planner_max_tokens: int = Field(
        default=1024,
        description="Max output tokens for contextual retrieval planner JSON output.",
    )
    rag_contextual_query_planner_temperature: float = Field(
        default=0.0,
        description="Temperature for contextual retrieval planner.",
    )
    rag_contextual_retry_enabled: bool = Field(
        default=True,
        description="If true, run one focused retrieval retry when follow-up retrieval appears topic-misaligned.",
    )
    rag_contextual_retry_topic_overlap_threshold: float = Field(
        default=0.18,
        description="Minimum topic-term alignment score expected in top ranked chunks before skipping topic-focused retry.",
    )
    rag_exact_match_enabled: bool = Field(
        default=True,
        description="If true, run an exact-match retrieval lane for entity/number-centric RAG queries.",
    )
    rag_exact_match_top_n: int = Field(
        default=224,
        description="Maximum exact-match chunks requested from the context service.",
    )
    rag_exact_match_guaranteed_chunks: int = Field(
        default=2,
        description="Minimum number of exact-match chunks to keep in final ranked context when available.",
    )
    rag_exact_match_max_terms: int = Field(
        default=4,
        description="Maximum exact-match terms/phrases extracted per request.",
    )
    rag_exact_match_query_token_limit: int = Field(
        default=12,
        description="Skip exact-match extraction for very long broad queries unless strong exact signals exist.",
    )
    rag_enable_exact_term_extractor: bool = Field(
        default=True,
        description="If true, use an LLM to extract literal entity/number phrases for the exact-match retrieval lane.",
    )
    rag_exact_term_extractor_temperature: float = Field(
        default=0.0,
        description="Temperature for the exact-term extraction LLM call.",
    )
    rag_exact_term_extractor_max_tokens: int = Field(
        default=120,
        description="Max output tokens for exact-term extraction JSON output.",
    )
    rag_known_branches_csv: str = Field(
        default="ACE,DELHI,MUMBAI,BENGALURU,CHENNAI",
        description="Comma-separated branch labels used for query-to-filter inference when payload metadata is absent.",
    )
    rag_known_report_types_csv: str = Field(
        default="Collation,FR,MR,SPR,WR,UO,SR,MISC,EIS",
        description="Comma-separated report types used for query-to-filter inference when payload metadata is absent.",
    )

    llm_trace_enabled: bool = Field(default=True, description="If true, append system/user prompts to trace file.")
    llm_trace_file: str = Field(default="logs/rag_llm_trace.log", description="Trace log path.")
    rag_planner_trace_enabled: bool = Field(
        default=True,
        description="If true, append retrieval-planner prompts, raw outputs, and normalized plans to a separate trace file.",
    )
    rag_planner_trace_file: str = Field(
        default="logs/rag_planner_trace.log",
        description="Trace log path for retrieval-planner LLM calls.",
    )
    rag_context_trace_enabled: bool = Field(
        default=True,
        description="If true, append compact retrieval context traces with selected chunk metadata, ACL fields, and selector/coverage diagnostics.",
    )
    rag_context_trace_file: str = Field(
        default="logs/rag_context_trace.log",
        description="Deprecated context trace file path.",
    )
    rag_judge_trace_enabled: bool = Field(
        default=True,
        description="If true, append every coverage-judge prompt and response to a dedicated judge trace log.",
    )
    rag_judge_trace_file: str = Field(
        default="rag_judge_call.log",
        description="Trace log path for coverage judge prompts and responses.",
    )
    rag_selector_trace_enabled: bool = Field(
        default=True,
        description="If true, append every evidence-selector prompt and response to a dedicated selector trace log.",
    )
    rag_selector_trace_file: str = Field(
        default="rag_selector_call.log",
        description="Trace log path for evidence-selector prompts and responses.",
    )


settings = Settings()
