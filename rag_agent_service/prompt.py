from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from prompt_store import read_active_prompt

logger = logging.getLogger(__name__)
_SPECIAL_CHAT_TOKEN_PATTERN = re.compile(r"<\|[^>]+?\|>")
USED_SOURCES_JSON_START = "<<USED_SOURCES_JSON>>"
USED_SOURCES_JSON_END = "<<END_USED_SOURCES_JSON>>"


def _backend_source_output_contract() -> str:
    return (
        "Backend output contract:\n"
        "1. In the visible answer, do not include any citation markers, source tags, doc_id references, parent_id references, or [source:N] tokens.\n"
        "2. After the visible answer, append exactly one hidden machine-readable block in this exact format:\n"
        f"{USED_SOURCES_JSON_START}\n"
        "{\"used_sources\":[]}\n"
        f"{USED_SOURCES_JSON_END}\n"
        "3. `used_sources` must contain only the actual source numbers used in the answer, for example [3,7]. Use an empty list if no source was used.\n"
        "4. Do not mention the machine-readable block in the visible answer.\n"
        "5. The hidden block is for backend parsing only and must not be described or referenced in the visible answer."
    )


def _sanitize_text(value: Any) -> str:
    text = _SPECIAL_CHAT_TOKEN_PATTERN.sub(" ", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def _truncate(text: str, max_chars: int) -> str:
    value = _sanitize_text(text)
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "..."


def _display_score(chunk: Dict[str, Any]) -> float:
    raw = chunk.get("_rerank_score", chunk.get("_final_score", chunk.get("_heuristic_score", chunk.get("_sim_score", chunk.get("score", 0.0)))))
    try:
        return float(raw)
    except Exception:
        return 0.0


def _recency_label(rank: Any) -> str:
    try:
        value = int(rank)
    except Exception:
        return ""
    if value <= 0:
        return ""
    if value == 1:
        return "most recent prior turn"
    if value == 2:
        return "2nd most recent prior turn"
    if value == 3:
        return "3rd most recent prior turn"
    return f"{value} turns back"


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _normalize_branch_key(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_access_code(value: Any) -> str:
    # changed: ACL code matching is exact-token after trimming and lowercasing.
    return str(value or "").strip().lower()


def _split_acl_codes(value: Any) -> set[str]:
    # changed: semantic and exact payloads may carry ACLs as comma-separated strings or simple lists.
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = str(value).split(",")
    codes: set[str] = set()
    for item in items:
        code = _normalize_access_code(item)
        if code:
            codes.add(code)
    return codes


def _normalize_chat_turns(chat_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    ordered_turns = list(chat_context.get("ordered_turns") or [])
    turns: List[Dict[str, Any]] = []
    if ordered_turns:
        for idx, turn in enumerate(ordered_turns, start=1):
            user_message = _sanitize_text(turn.get("user_message"))
            assistant_reply = _sanitize_text(turn.get("assistant_reply"))
            if not user_message and not assistant_reply:
                continue
            turns.append(
                {
                    "turn_index": int(turn.get("turn_index") or idx),
                    "recency_rank": turn.get("recency_rank"),
                    "is_recent": bool(turn.get("is_recent")),
                    "scope": str(turn.get("scope") or "").strip(),
                    "user_message": user_message,
                    "assistant_reply": assistant_reply,
                }
            )
        return turns

    history = list(chat_context.get("conversation_history") or [])
    recent_pairs = list(chat_context.get("recent_conversations") or [])

    idx = 0
    while idx < len(history):
        current = history[idx]
        role = str(current.get("role") or "").strip().lower()
        text = _sanitize_text(current.get("text") or current.get("content"))
        if role in {"user", "human"} and text and idx + 1 < len(history):
            nxt = history[idx + 1]
            next_role = str(nxt.get("role") or "").strip().lower()
            next_text = _sanitize_text(nxt.get("text") or nxt.get("content"))
            if next_role in {"assistant", "system"} and next_text:
                turns.append(
                    {
                        "turn_index": len(turns) + 1,
                        "recency_rank": None,
                        "is_recent": False,
                        "scope": "history",
                        "user_message": text,
                        "assistant_reply": next_text,
                    }
                )
                idx += 2
                continue
        idx += 1

    for pair in recent_pairs:
        user_message = _sanitize_text(pair.get("user_message"))
        assistant_reply = _sanitize_text(pair.get("assistant_reply"))
        if not user_message and not assistant_reply:
            continue
        turns.append(
            {
                "turn_index": len(turns) + 1,
                "recency_rank": None,
                "is_recent": True,
                "scope": str(pair.get("scope") or "recent").strip(),
                "user_message": user_message,
                "assistant_reply": assistant_reply,
            }
        )

    total_turns = len(turns)
    for idx, turn in enumerate(turns, start=1):
        if turn.get("recency_rank") is None:
            turn["recency_rank"] = (total_turns - idx) + 1
    return turns


def _select_turns_for_prompt(turns: List[Dict[str, Any]], *, max_messages: int) -> List[Dict[str, Any]]:
    if not turns:
        return []
    max_turns = max(1, int(max_messages))
    if len(turns) <= max_turns:
        return turns

    recent_turns = [turn for turn in turns if bool(turn.get("is_recent"))]
    older_turns = [turn for turn in turns if not bool(turn.get("is_recent"))]
    if recent_turns and older_turns and max_turns >= 2:
        older_budget = min(len(older_turns), 1 if max_turns <= 3 else 2)
        recent_budget = max_turns - older_budget
        if recent_budget > 0:
            return [*older_turns[-older_budget:], *recent_turns[-recent_budget:]]
    return turns[-max_turns:]


def build_rag_system_prompt() -> str:
    base_prompt = read_active_prompt().rstrip("\n")
    contract = _backend_source_output_contract()
    if not base_prompt:
        return contract
    return f"{base_prompt}\n\n{contract}".strip()


def build_chat_context_system_prompt() -> str:
    return (
        "You answer chat-context-only follow-up and transformation requests.\n"
        "Use only the prior chat context provided in the user prompt and the current user request.\n"
        "This path intentionally has no knowledge-base/vector retrieval context.\n"
        "Do not mention missing KB context, missing retrieved documents, vector search, or failed retrieval.\n"
        "If the referenced prior answer cannot be identified from chat context, ask one short clarification question.\n\n"
        f"{_backend_source_output_contract()}"
    )


def build_general_background_system_prompt() -> str:
    return (
        "You write a clearly separated general-background addendum.\n"
        "Use broad public/general model knowledge only; do not claim access to training data, private data, internal documents, or live web.\n"
        "Do not cite source numbers, doc ids, parent ids, or retrieved documents.\n"
        "Do not include machine-readable JSON or backend markers.\n"
        "If the request needs current, live, legal, medical, financial, or citation-grade verification, say external verified retrieval is needed.\n"
        "Treat the DB-grounded answer as already shown to the user, then add only useful public/general context beyond it.\n"
        "Keep the addendum concise and avoid repeating the DB-grounded answer except for a short bridge when needed."
    )


def build_general_background_prompt(
    *,
    query: str,
    db_answer: str,
    reason: str = "",
    resolved_query: str = "",
    retrieval_focus: str = "",
    max_answer_chars: int = 4000,
) -> str:
    lines: List[str] = []
    lines.append("## User Query")
    lines.append(_sanitize_text(query))
    lines.append("")
    resolved = _truncate(_sanitize_text(resolved_query), 1000)
    if resolved and resolved.lower() != _sanitize_text(query).lower():
        lines.append("## Resolved Retrieval Query")
        lines.append(resolved)
        lines.append("")
    focus = _truncate(_sanitize_text(retrieval_focus), 700)
    if focus:
        lines.append("## Retrieval Focus")
        lines.append(focus)
        lines.append("")
    answer = _truncate(db_answer, max_answer_chars)
    if answer:
        lines.append("## DB-Grounded Answer Already Sent To User")
        lines.append(answer)
        lines.append("")
    if reason:
        lines.append("## Why This Addendum Is Being Requested")
        lines.append(_sanitize_text(reason))
        lines.append("")
    lines.append("## Task")
    lines.append("Write only the general-background addendum body that directly helps answer the user query.")
    lines.append("Use the resolved retrieval query and retrieval focus only to disambiguate what the user asked.")
    lines.append("Read the DB-grounded answer first so you know exactly what has already been answered.")
    lines.append("Add what is publicly or generally known beyond that answer: relevant background, definitions, common context, caveats, and broader interpretation when useful.")
    lines.append("Do not treat the DB-grounded answer as a public source, and do not repeat it line-by-line.")
    lines.append("If the topic appears private/internal and there is no reliable public background to add, say that public background is limited and give only safe general context.")
    lines.append("Do not add generic encyclopedia text that is unrelated to the actual question or the DB-grounded answer.")
    lines.append("Do not add a heading; the backend will add the heading.")
    lines.append("Do not mention source ids, retrieved documents, internal DB, or backend markers.")
    lines.append("Do not claim this comes from training data; call it general background if needed.")
    return "\n".join(lines).strip()


def build_rag_context_text(
    *,
    query: str,
    kb_chunks: List[Dict[str, Any]],
    chat_context: Dict[str, Any],
    max_chunk_chars: int,
    max_chat_messages: int,
    resolved_query: str = "",
    retrieval_focus: str = "",
    applied_time_filter: str = "",
    coverage_guidance: str = "",
) -> str:
    def append_chunk_section(section_title: str, chunks: List[Dict[str, Any]], start_index: int) -> int:
        lines.append(section_title)
        if not chunks:
            lines.append("No chunks retrieved.")
            lines.append("")
            return start_index

        source_index = start_index
        for chunk in chunks:
            doc_id = str(chunk.get("doc_id") or "").strip()
            parent_id = str(chunk.get("parent_id") or "").strip()
            file_id = str(chunk.get("file_id") or "").strip()
            title = str(chunk.get("title") or chunk.get("file_name") or "").strip()
            report_type = str(chunk.get("report_type") or "").strip()
            branch = str(chunk.get("branch") or "").strip()
            chunk_no = str(chunk.get("chunk_no") or "").strip()
            lang = str(chunk.get("lang") or "").strip()
            document_date = str(chunk.get("document_date") or "").strip()
            ingestion_date = str(chunk.get("ingestion_date") or chunk.get("created_at") or "").strip()
            score = _display_score(chunk)

            content = _truncate(
                chunk.get("_prompt_content")
                or chunk.get("content")
                or chunk.get("text")
                or chunk.get("chunk_text")
                or chunk.get("page_content")
                or "",
                max_chunk_chars,
            )
            if not content:
                continue

            id_label = doc_id or parent_id or file_id or f"source:{source_index}"
            if chunk_no:
                id_label = f"{id_label}:{chunk_no}"
            source_kind = str(chunk.get("_context_source") or chunk.get("_scope") or "retrieved").strip() or "retrieved"
            lines.append(
                f"[source:{source_index}] source_type={source_kind} id={id_label} score={score:.4f} "
                f"title={title or '-'} report_type={report_type or '-'} branch={branch or '-'} "
                f"lang={lang or '-'} document_date={document_date or '-'} ingestion_date={ingestion_date or '-'}"
            )
            lines.append(content)
            lines.append("")
            source_index += 1
        return source_index

    lines: List[str] = []
    lines.append("## User Query")
    lines.append(_sanitize_text(query))
    lines.append("")

    resolved = _truncate(_sanitize_text(resolved_query), 1000)
    if resolved and resolved.lower() != _sanitize_text(query).lower():
        lines.append("## Resolved Retrieval Query")
        lines.append(resolved)
        lines.append("")

    focus = _truncate(_sanitize_text(retrieval_focus), 700)
    if focus:
        lines.append("## Retrieval Focus")
        lines.append(focus)
        lines.append("")

    time_filter = _sanitize_text(applied_time_filter)
    if time_filter:
        lines.append("## Applied Time Filter")
        lines.append(f"Retrieved documents were filtered by `ingestion_date` for: {time_filter}")
        lines.append("")

    uploaded_chunks = [
        chunk for chunk in list(kb_chunks or [])
        if str(chunk.get("_context_source") or "").strip().lower() == "uploaded_file"
    ]
    bigdata_chunks = [
        chunk for chunk in list(kb_chunks or [])
        if str(chunk.get("_context_source") or "").strip().lower() != "uploaded_file"
    ]
    if uploaded_chunks:
        next_source = append_chunk_section("## Uploaded File Context", uploaded_chunks, 1)
        append_chunk_section("## BigData Retrieved Context", bigdata_chunks, next_source)
    else:
        append_chunk_section("## Knowledge Base Context", bigdata_chunks, 1)

    lines.append("## Chat Context (Oldest To Newest)")
    turns = _select_turns_for_prompt(
        _normalize_chat_turns(chat_context),
        max_messages=max_chat_messages,
    )
    if turns:
        for turn in turns:
            scope = "recent" if bool(turn.get("is_recent")) else "older relevant"
            recency = _recency_label(turn.get("recency_rank"))
            turn_header = f"[Turn {turn.get('turn_index')} | {scope}"
            if recency:
                turn_header += f" | {recency}"
            turn_header += "]"
            lines.append(turn_header)

            user_message = str(turn.get("user_message") or "").strip()
            assistant_reply = str(turn.get("assistant_reply") or "").strip()
            if user_message:
                lines.append(f"User: {_truncate(user_message, 420)}")
            if assistant_reply:
                lines.append(f"Assistant: {_truncate(assistant_reply, 420)}")
    else:
        lines.append("No prior chat context.")

    lines.append("")
    guidance = _sanitize_text(coverage_guidance)
    if guidance:
        lines.append("## Retrieval Coverage Guidance")
        lines.append(guidance)
        lines.append("")

    lines.append("## Instructions")
    lines.append("- Answer using only evidence from the sections above.")
    lines.append("- If an Uploaded File Context section is present, treat it as the user's reference document and use BigData Retrieved Context only to enrich, compare, or find related information requested by the user.")
    lines.append("- Be exhaustive for the user's requested entity or topic: include every concrete supported detail that helps answer the query.")
    lines.append("- For person-detail queries, include all supported names, aliases, relationships, family details, associates, roles, identifiers, locations, dates, events, travel, contact details, and caveats that appear in the sources.")
    lines.append("- If multiple plausible people/entities match the requested name, do not collapse them into one identity. Provide separate details for each plausible match and state the ambiguity.")
    lines.append("- Do not omit a supported detail merely because it is minor; omit only duplicate, irrelevant, or unsupported material.")
    lines.append("- Do not include citation markers, source tags, or [source:N] tokens in the visible answer.")
    lines.append("- If the user asks for full content, provide full available content from retrieved chunks.")
    lines.append("- If an applied time filter is shown and it matters to the answer, mention that the findings come from documents in that ingestion-date window.")
    lines.append("- If evidence is missing, state that explicitly and ask one concise follow-up.")
    lines.append(
        f"- After the visible answer, append exactly this hidden block: "
        f"{USED_SOURCES_JSON_START} "
        '{"used_sources":[]} '
        f"{USED_SOURCES_JSON_END}; replace [] with only the actual source numbers used."
    )
    return "\n".join(lines).strip()


def build_chat_context_answer_text(
    *,
    query: str,
    chat_context: Dict[str, Any],
    max_chat_messages: int,
    resolved_query: str = "",
    retrieval_focus: str = "",
    max_user_message_chars: int = 1200,
    max_assistant_reply_chars: int = 8000,
) -> str:
    lines: List[str] = []
    lines.append("## User Query")
    lines.append(_sanitize_text(query))
    lines.append("")

    resolved = _sanitize_text(resolved_query)
    if resolved and resolved.lower() != _sanitize_text(query).lower():
        lines.append("## Resolved Chat-Only Request")
        lines.append(resolved)
        lines.append("")

    focus = _sanitize_text(retrieval_focus)
    if focus:
        lines.append("## Conversation Focus")
        lines.append(focus)
        lines.append("")

    lines.append("## Chat Context To Use (Oldest To Newest)")
    turns = _select_turns_for_prompt(
        _normalize_chat_turns(chat_context),
        max_messages=max_chat_messages,
    )
    if turns:
        for turn in turns:
            scope = "recent" if bool(turn.get("is_recent")) else "older relevant"
            recency = _recency_label(turn.get("recency_rank"))
            turn_header = f"[Turn {turn.get('turn_index')} | {scope}"
            if recency:
                turn_header += f" | {recency}"
            turn_header += "]"
            lines.append(turn_header)

            user_message = str(turn.get("user_message") or "").strip()
            assistant_reply = str(turn.get("assistant_reply") or "").strip()
            if user_message:
                lines.append(f"User: {_truncate(user_message, max_user_message_chars)}")
            if assistant_reply:
                lines.append(f"Assistant: {_truncate(assistant_reply, max_assistant_reply_chars)}")
            lines.append("")
    else:
        lines.append("No prior chat context was available.")
        lines.append("")

    lines.append("## Instructions")
    lines.append("- This is intentionally a chat-context-only answer. Knowledge-base/vector retrieval was intentionally skipped.")
    lines.append("- Answer only from the prior chat context above and the user's current transform/follow-up request.")
    lines.append("- Preserve, summarize, shorten, reformat, or continue the previous answer according to the user's request.")
    lines.append("- If the user says 'above', 'previous answer', 'that', or similar, resolve it to the relevant prior assistant answer.")
    lines.append("- If the reference cannot be resolved from the chat context, ask one short clarification question.")
    lines.append("- Do not mention missing KB context, missing retrieved documents, vector search, or failed retrieval.")
    lines.append("- Do not include citation markers, source tags, doc IDs, parent IDs, or [source:N] tokens in the visible answer.")
    lines.append(
        f"- After the visible answer, append exactly this hidden block: "
        f"{USED_SOURCES_JSON_START} "
        '{"used_sources":[]} '
        f"{USED_SOURCES_JSON_END}; keep used_sources empty because no KB source chunks are used."
    )
    return "\n".join(lines).strip()


def build_source_documents(
    kb_chunks: List[Dict[str, Any]],
    *,
    limit: int = 12,
    excerpt_chars: int = 220,
    user_access_codes: List[str] | set[str] | None = None,
    access_override: bool = False,
) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    # changed: source-document access is now derived strictly from chunk ACL fields.
    normalized_user_access = {_normalize_access_code(value) for value in list(user_access_codes or []) if _normalize_access_code(value)}
    for chunk in kb_chunks:
        doc_id = str(chunk.get("doc_id") or "").strip()
        parent_id = str(chunk.get("parent_id") or "").strip()
        url = str(chunk.get("system_path") or "").strip()
        branch = chunk.get("branch")
        dedupe_key = (url.lower(), doc_id.lower(), parent_id.lower())
        if any(dedupe_key) and dedupe_key in seen:
            continue

        content = _truncate(
            chunk.get("content")
            or chunk.get("text")
            or chunk.get("chunk_text")
            or chunk.get("page_content")
            or "",
            excerpt_chars,
        )
        if not content:
            continue
        if any(dedupe_key):
            seen.add(dedupe_key)
        acl_branch_codes = _split_acl_codes(chunk.get("access_branches"))
        acl_group_codes = _split_acl_codes(chunk.get("access_groups"))
        acl_codes = acl_branch_codes | acl_group_codes
        has_access = bool(access_override or (acl_codes and normalized_user_access.intersection(acl_codes)))
        logger.info(
            "RAG source document access doc_id=%s parent_id=%s chunk_no=%s access_branches=%s access_groups=%s override=%s resolved_access=%s",
            doc_id or "",
            parent_id or "",
            chunk.get("chunk_no"),
            sorted(acl_branch_codes),
            sorted(acl_group_codes),
            access_override,
            has_access,
        )
        docs.append(
            {
                "doc_id": doc_id or '',
                "parent_id": parent_id or '',
                "chunk_no": chunk.get("chunk_no"),
                "report_type": str(chunk.get("report_type") or ""),
                "branch": str(branch or ""),
                "score": _display_score(chunk),
                "url": url or '',
                "document_date": _safe_int(chunk.get("document_date")) or 0,
                "excerpt": content,
                "access": has_access,
            }
        )
        if len(docs) >= max(1, int(limit)):
            break
    return docs
