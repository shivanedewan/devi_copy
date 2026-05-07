from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

from logger import logger
from time_filter_parser import extract_ingestion_time_filter, should_apply_retrieval_time_filter


_TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
_SPACE_PATTERN = re.compile(r"\s+")
_ALLOWED_TIME_FIELDS = {"none", "ingestion_date", "document_date", "either"}
_ALLOWED_ANSWER_INTENTS = {"inform", "format", "summarize", "continue"}
_ALLOWED_RETRIEVAL_ACTIONS = {"fresh_retrieval", "reuse_previous_topic"}
_SPECIAL_CHAT_TOKEN_PATTERN = re.compile(r"<\|[^>]+?\|>")
_FOLLOW_UP_PATTERN = re.compile(
    r"\b(tell me more|more details|what about|continue|elaborate|expand|go deeper|and what|what else)\b",
    flags=re.IGNORECASE,
)
_DISCOURSE_REFERENCE_PATTERN = re.compile(
    r"\b(?:above|below|earlier|previous|previously|prior|before that|mentioned above|mentioned earlier|same topic|same subject|same one|that one|aforementioned)\b",
    flags=re.IGNORECASE,
)
_PRONOUN_PATTERN = re.compile(r"\b(he|she|him|her|it|its|they|them|this|that|those|these|his|their)\b", flags=re.IGNORECASE)
_SOURCE_REFERENCE_PATTERN = re.compile(
    r"\b(?:source|sources)\s*(?::|#|no\.?|num(?:ber)?)?\s*\d{1,4}\b|"
    r"\b(?:source|sources)\s+(?:number|numbers)\s+\d{1,4}\b",
    flags=re.IGNORECASE,
)
_REQUEST_PREFIX_PATTERN = re.compile(
    r"^\s*(?:tell me|tell me about|what is mentioned about|what is said about|what is written about|what about|give me|show me|describe|explain|summari[sz]e|brief me on|what can you tell me about|what do you know about)\s+",
    flags=re.IGNORECASE,
)
_GENERIC_QUERY_PATTERN = re.compile(
    r"\b(?:mentioned|said|written|details?|information|summary|brief|overview|about)\b",
    flags=re.IGNORECASE,
)
_CONTEXT_CARRY_PATTERN = re.compile(
    r"\b(?:first question|last question|previous question|my first question|my previous question|"
    r"previous response|previous answer|that response|that answer)\b",
    flags=re.IGNORECASE,
)
_FORMAT_INTENT_PATTERN = re.compile(
    r"\b(?:respond|answer|write|rewrite|rephrase|reformat|format|present|provide|give|prepare|make|turn|convert|draft|"
    r"summari[sz]e|continue)\b",
    flags=re.IGNORECASE,
)
_FORMAT_TARGET_PATTERN = re.compile(
    r"\b(?:report|summary|brief|briefing|note|memo|email|mail|table|bullet(?:s)?|paragraph|page|pages|points?)\b",
    flags=re.IGNORECASE,
)
_RETRIEVAL_SCOPE_PHRASE_PATTERN = re.compile(
    r"\b(?:from|in|inside|within|using|search(?:ing)?|look(?:ing)?\s+in|based\s+on|according\s+to)\s+"
    r"(?:the\s+)?(?:big\s*data|bigdata|internal\s+(?:documents?|docs?|records?|database|db)|"
    r"(?:our|the|my)\s+(?:documents?|docs?|records?|database|db)|knowledge\s*base|kb|database|db|records?)\b|"
    r"\b(?:big\s*data|bigdata|internal\s+(?:documents?|docs?|records?|database|db)|knowledge\s*base|kb|database|db|records?)\s+"
    r"(?:results?|search|records?|documents?|docs?|data)\b",
    flags=re.IGNORECASE,
)
_SUMMARIZE_INTENT_PATTERN = re.compile(
    r"\b(?:summari[sz]e|summary|brief|briefing|overview|in brief|short summary)\b",
    flags=re.IGNORECASE,
)
_LOW_SIGNAL_TOKENS = {
    "a", "an", "about", "all", "and", "any", "are", "be", "details", "for", "give", "he", "her", "him",
    "his", "i", "in", "information", "is", "it", "its", "me", "mentioned", "of", "on", "relation",
    "relations", "relationship", "relationships", "said", "show", "summary", "tell", "that", "the",
    "their", "them", "there", "these", "they", "this", "those", "what", "which", "who", "written",
    "big", "bigdata", "data", "database", "db", "document", "documents", "internal", "kb", "knowledge",
    "record", "records", "result", "results",
}
_FORMAT_LOW_SIGNAL_TOKENS = {
    "answer",
    "answers",
    "brief",
    "briefing",
    "bullet",
    "bullets",
    "consider",
    "continue",
    "convert",
    "draft",
    "email",
    "first",
    "format",
    "formatted",
    "give",
    "mail",
    "make",
    "memo",
    "note",
    "page",
    "pages",
    "paragraph",
    "points",
    "prepare",
    "present",
    "provide",
    "question",
    "questions",
    "reformat",
    "report",
    "reports",
    "rephrase",
    "respond",
    "response",
    "rewrite",
    "summarize",
    "summary",
    "table",
    "turn",
    "write",
}


def _normalize_space(value: Any) -> str:
    text = _SPECIAL_CHAT_TOKEN_PATTERN.sub(" ", str(value or ""))
    return _SPACE_PATTERN.sub(" ", text).strip()


def _truncate(value: Any, max_chars: int) -> str:
    text = _normalize_space(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _normalize_query_key(value: Any) -> str:
    return " ".join(_normalize_space(value).lower().split())


def _clean_text(value: Any, *, max_chars: int = 240) -> str:
    text = _normalize_space(value).strip(" \t\r\n,.;:!?-")
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars].rstrip()
    return text


def _extract_json_object(raw: Any) -> Dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}

    candidates: List[str] = []
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced_match:
        candidates.append(str(fenced_match.group(1) or "").strip())

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1].strip())
    candidates.append(text)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return None


def _coerce_query_list(raw: Any, *, max_items: int) -> List[str]:
    items = list(raw) if isinstance(raw, (list, tuple, set)) else [raw]
    values: List[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        for part in re.split(r"[\r\n]+", text):
            candidate = _clean_text(part, max_chars=220)
            if not candidate:
                continue
            values.append(candidate)
            if len(values) >= max(1, int(max_items)):
                return values
    return values


def _coerce_string_list(raw: Any, *, max_items: int, max_chars: int = 120) -> List[str]:
    items = list(raw) if isinstance(raw, (list, tuple, set)) else [raw]
    values: List[str] = []
    for item in items:
        if isinstance(item, str):
            parts = re.split(r"[\r\n,;|]+", item)
        else:
            parts = [item]
        for part in parts:
            candidate = _clean_text(part, max_chars=max_chars)
            if not candidate:
                continue
            values.append(candidate)
            if len(values) >= max(1, int(max_items)):
                return values
    return values


def _dedupe_strings(values: List[str], *, max_items: int) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for raw in values:
        value = _clean_text(raw)
        if not value:
            continue
        key = _normalize_query_key(value)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(value)
        if len(out) >= max(1, int(max_items)):
            break
    return out


def _is_exact_term_candidate(value: Any) -> bool:
    text = _clean_text(value)
    if not text:
        return False
    if _SOURCE_REFERENCE_PATTERN.search(text):
        return False
    tokens = [token for token in _TOKEN_PATTERN.findall(text.lower()) if token]
    if not tokens:
        return False
    if all(token.isdigit() for token in tokens):
        digits = "".join(re.findall(r"\d+", text))
        return len(digits) >= 5
    signal_tokens = [token for token in tokens if token not in _LOW_SIGNAL_TOKENS]
    return bool(signal_tokens)


def _parse_iso_date(value: Any) -> Optional[str]:
    text = _clean_text(value, max_chars=16)
    if not text:
        return None
    try:
        parsed = date.fromisoformat(text)
    except Exception:
        return None
    return parsed.isoformat()


def _to_epoch_ms(day_text: str, *, end_of_day: bool) -> int:
    parsed = date.fromisoformat(day_text)
    clock = time(23, 59, 59, 999000, tzinfo=timezone.utc) if end_of_day else time(0, 0, 0, 0, tzinfo=timezone.utc)
    dt = datetime.combine(parsed, clock)
    return int(dt.timestamp() * 1000)


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


def _normalize_chat_turns(chat_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    ordered_turns = list(chat_context.get("ordered_turns") or [])
    turns: List[Dict[str, Any]] = []
    if ordered_turns:
        for idx, turn in enumerate(ordered_turns, start=1):
            user_message = _clean_text(turn.get("user_message"), max_chars=600)
            assistant_reply = _clean_text(turn.get("assistant_reply"), max_chars=600)
            if not user_message and not assistant_reply:
                continue
            turns.append(
                {
                    "turn_index": int(turn.get("turn_index") or idx),
                    "recency_rank": turn.get("recency_rank"),
                    "is_recent": bool(turn.get("is_recent")),
                    "scope": _clean_text(turn.get("scope"), max_chars=32),
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
        text = _clean_text(current.get("text") or current.get("content"), max_chars=600)
        if role in {"user", "human"} and text and idx + 1 < len(history):
            nxt = history[idx + 1]
            next_role = str(nxt.get("role") or "").strip().lower()
            next_text = _clean_text(nxt.get("text") or nxt.get("content"), max_chars=600)
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
        user_message = _clean_text(pair.get("user_message"), max_chars=600)
        assistant_reply = _clean_text(pair.get("assistant_reply"), max_chars=600)
        if not user_message and not assistant_reply:
            continue
        turns.append(
            {
                "turn_index": len(turns) + 1,
                "recency_rank": None,
                "is_recent": True,
                "scope": _clean_text(pair.get("scope") or "recent", max_chars=32),
                "user_message": user_message,
                "assistant_reply": assistant_reply,
            }
        )

    total_turns = len(turns)
    for idx, turn in enumerate(turns, start=1):
        if turn.get("recency_rank") is None:
            turn["recency_rank"] = (total_turns - idx) + 1
    return turns


@dataclass
class RetrievalPlan:
    standalone_query: str
    query_variants: List[str]
    focus_subject: str
    focus_hint: str
    answer_intent: str
    retrieval_action: str
    exact_terms: List[str]
    filters: Dict[str, Any]
    time_filter: Dict[str, Any]
    followup: bool
    context_dependent: bool
    needs_history_expansion: bool
    planner_used: bool
    raw_plan: Dict[str, Any]


class RetrievalPlanner:
    def __init__(
        self,
        *,
        client: Any,
        model_name: str,
        known_branches: List[str],
        known_report_types: List[str],
        temperature: float,
        max_tokens: int,
        max_query_variants: int,
        max_exact_terms: int,
        max_turns: int,
        max_context_chars: int,
        default_time_field: str,
        trace_hook: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    ) -> None:
        self._client = client
        self._model_name = model_name
        self._known_branches = list(known_branches or [])
        self._known_report_types = list(known_report_types or [])
        self._temperature = float(temperature)
        self._max_tokens = max(128, int(max_tokens))
        self._max_query_variants = max(1, int(max_query_variants))
        self._max_exact_terms = max(1, int(max_exact_terms))
        self._max_turns = max(1, int(max_turns))
        self._max_context_chars = max(600, int(max_context_chars))
        self._default_time_field = (
            default_time_field
            if str(default_time_field or "").strip() in {"ingestion_date", "document_date"}
            else "ingestion_date"
        )
        self._trace_hook = trace_hook

    async def _emit_trace(self, payload: Dict[str, Any]) -> None:
        if not self._trace_hook:
            return
        try:
            await self._trace_hook(payload)
        except Exception as exc:
            logger.warning(
                "RAG planner trace hook failed chat_id=%s message_id=%s stage=%s error=%s",
                payload.get("chat_id"),
                payload.get("message_id"),
                payload.get("stage"),
                exc,
            )

    @staticmethod
    def _chat_context_block(chat_context: Dict[str, Any], *, max_turns: int, max_chars: int) -> str:
        lines: List[str] = []
        turns = _normalize_chat_turns(chat_context)
        if max_turns > 0 and len(turns) > max_turns:
            turns = turns[-max_turns:]
        for turn in turns:
            scope = "recent" if bool(turn.get("is_recent")) else (_clean_text(turn.get("scope"), max_chars=32) or "older relevant")
            recency = _recency_label(turn.get("recency_rank"))
            header = f"[Turn {turn.get('turn_index')} | {scope}"
            if recency:
                header += f" | {recency}"
            header += "]"
            lines.append(header)
            if turn.get("user_message"):
                lines.append(f"User: {_truncate(turn.get('user_message'), 240)}")
            if turn.get("assistant_reply"):
                lines.append(f"Assistant: {_truncate(turn.get('assistant_reply'), 240)}")

        text = "\n".join(lines).strip()
        if len(text) <= max_chars:
            return text
        return text[-max_chars:]

    async def _call_model(self, *, system_prompt: str, user_prompt: str) -> str:
        request_payload = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": False,
        }
        try:
            response = await self._client.chat.completions.create(
                **request_payload,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            error_text = str(exc).lower()
            if any(token in error_text for token in ("response_format", "json_object", "json schema", "extra_forbidden")):
                logger.info("RAG planner JSON mode unsupported by backend; retrying without response_format.")
                response = await self._client.chat.completions.create(**request_payload)
            else:
                raise
        if not response.choices:
            return ""
        content = response.choices[0].message.content
        return str(content or "").strip()

    @staticmethod
    def _query_tokens(text: Any) -> List[str]:
        return [token for token in _TOKEN_PATTERN.findall(str(text or "").lower()) if token]

    def _specific_terms(self, text: Any) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        for token in self._query_tokens(text):
            if len(token) < 3 or token in _LOW_SIGNAL_TOKENS:
                continue
            if token in seen:
                continue
            seen.add(token)
            out.append(token)
        return out

    def _looks_like_format_only_fragment(self, text: str) -> bool:
        tokens = [token for token in self._query_tokens(text) if token]
        if not tokens:
            return False
        significant = [
            token
            for token in tokens
            if not token.isdigit()
            and token not in _LOW_SIGNAL_TOKENS
            and token not in _FORMAT_LOW_SIGNAL_TOKENS
        ]
        return len(significant) == 0

    def _is_format_transform_query(self, query: str) -> bool:
        text = _normalize_space(query).lower()
        if not text:
            return False
        has_format_signal = bool(_FORMAT_INTENT_PATTERN.search(text) or _FORMAT_TARGET_PATTERN.search(text))
        if not has_format_signal:
            return False
        if _DISCOURSE_REFERENCE_PATTERN.search(text) or _CONTEXT_CARRY_PATTERN.search(text):
            return True
        cleaned = self._clean_topic_candidate(text)
        if not cleaned:
            return True
        if self._looks_like_format_only_fragment(cleaned):
            return True
        return not bool(self._specific_terms(cleaned))

    def _classify_answer_intent(self, query: str, *, context_dependent: bool) -> str:
        text = _normalize_space(query).lower()
        if not text:
            return "inform"
        if self._is_format_transform_query(text):
            if _SUMMARIZE_INTENT_PATTERN.search(text) and not _FORMAT_TARGET_PATTERN.search(text):
                return "summarize"
            if re.search(r"\b(?:continue|respond|answer)\b", text) and not _FORMAT_TARGET_PATTERN.search(text):
                return "continue"
            return "format"
        if _SUMMARIZE_INTENT_PATTERN.search(text):
            return "summarize"
        if context_dependent and re.search(r"\b(?:continue|respond|answer)\b", text):
            return "continue"
        return "inform"

    @staticmethod
    def _infer_retrieval_action(*, context_dependent: bool, focus_subject: str, focus_hint: str) -> str:
        if context_dependent and str(focus_subject or focus_hint or "").strip():
            return "reuse_previous_topic"
        return "fresh_retrieval"

    def _is_context_dependent_query(self, query: str) -> bool:
        text = _normalize_space(query).lower()
        if not text:
            return False
        if self._is_format_transform_query(text):
            return True
        if _CONTEXT_CARRY_PATTERN.search(text):
            return True
        if _FOLLOW_UP_PATTERN.search(text) or _DISCOURSE_REFERENCE_PATTERN.search(text):
            return True
        tokens = self._query_tokens(text)
        return len(tokens) <= 12 and bool(_PRONOUN_PATTERN.search(text))

    def _has_strong_context_dependency_signal(self, query: str) -> bool:
        text = _normalize_space(query).lower()
        if not text:
            return False
        return bool(
            self._is_format_transform_query(text)
            or _CONTEXT_CARRY_PATTERN.search(text)
            or _FOLLOW_UP_PATTERN.search(text)
            or _DISCOURSE_REFERENCE_PATTERN.search(text)
        )

    @staticmethod
    def _strip_time_phrase(text: str) -> str:
        value = _normalize_space(text)
        if not value:
            return ""
        time_filter = extract_ingestion_time_filter(value)
        if not time_filter or not str(time_filter.matched_text or "").strip():
            return value
        stripped = re.sub(re.escape(str(time_filter.matched_text or "").strip()), " ", value, flags=re.IGNORECASE)
        return _normalize_space(stripped).strip(" \t\r\n,.;:-")

    def _clean_topic_candidate(self, text: Any) -> str:
        value = _normalize_space(text)
        if not value:
            return ""
        value = self._strip_time_phrase(value)
        value = _RETRIEVAL_SCOPE_PHRASE_PATTERN.sub(" ", value)
        value = _REQUEST_PREFIX_PATTERN.sub("", value)
        value = re.sub(r"^\s*(?:about|on)\s+", "", value, flags=re.IGNORECASE)
        value = _DISCOURSE_REFERENCE_PATTERN.sub(" ", value)
        value = _PRONOUN_PATTERN.sub(" ", value)
        value = re.sub(r"\b(?:please|briefly|brief|details only|in brief|in short)\b", " ", value, flags=re.IGNORECASE)
        value = _CONTEXT_CARRY_PATTERN.sub(" ", value)
        value = _FORMAT_INTENT_PATTERN.sub(" ", value)
        value = _FORMAT_TARGET_PATTERN.sub(" ", value)
        value = _normalize_space(value).strip(" \t\r\n,.;:-")
        if self._looks_like_format_only_fragment(value):
            return ""
        return value

    def _extract_topic_chain(self, chat_context: Dict[str, Any], current_query: str) -> Dict[str, str]:
        current_key = _normalize_query_key(current_query)
        latest_contextual_fragment = ""
        base_topic = ""
        fallback_topic = ""

        for turn in reversed(_normalize_chat_turns(chat_context)):
            user_message = _clean_text(turn.get("user_message"), max_chars=320)
            if not user_message:
                continue
            if _normalize_query_key(user_message) == current_key:
                continue
            cleaned = self._clean_topic_candidate(user_message)
            if cleaned and not fallback_topic:
                fallback_topic = cleaned
            if self._is_context_dependent_query(user_message):
                if cleaned and self._specific_terms(cleaned) and not latest_contextual_fragment:
                    latest_contextual_fragment = cleaned
                continue
            if cleaned and self._specific_terms(cleaned):
                base_topic = cleaned
                break

        resolved_topic = base_topic or fallback_topic
        combined_topic = ""
        if base_topic and latest_contextual_fragment:
            combined_topic = _normalize_space(f"{base_topic} {latest_contextual_fragment}")
        elif resolved_topic:
            combined_topic = resolved_topic
        return {
            "base_topic": base_topic,
            "contextual_fragment": latest_contextual_fragment,
            "resolved_topic": resolved_topic,
            "combined_topic": combined_topic,
        }

    def _should_carry_prior_resolved_topic(
        self,
        *,
        base_topic: str,
        resolved_topic: str,
        current_specific_terms: List[str],
    ) -> bool:
        if not resolved_topic:
            return False
        if not current_specific_terms:
            return True
        base_terms = set(self._specific_terms(base_topic))
        resolved_terms = [term for term in self._specific_terms(resolved_topic) if term not in base_terms]
        if not resolved_terms:
            return False
        return bool(set(current_specific_terms) & set(resolved_terms))

    def _build_deterministic_fallback_payload(
        self,
        *,
        query: str,
        chat_context: Dict[str, Any],
        allow_history_expansion: bool,
        expanded_history_used: bool,
    ) -> Dict[str, Any]:
        context_dependent = self._is_context_dependent_query(query)
        topic_info = self._extract_topic_chain(chat_context, query)
        combined_topic = topic_info.get("combined_topic") or ""
        resolved_topic = combined_topic or topic_info.get("resolved_topic") or ""
        base_topic = topic_info.get("base_topic") or ""
        cleaned_query = self._clean_topic_candidate(query)
        specific_terms = self._specific_terms(cleaned_query)
        answer_intent = self._classify_answer_intent(query, context_dependent=context_dependent)
        generic_only = not specific_terms or (
            len(specific_terms) <= 2 and bool(_GENERIC_QUERY_PATTERN.search(cleaned_query.lower()))
        )
        if answer_intent in {"format", "continue"}:
            generic_only = True

        standalone_query = query
        query_variants: List[str] = []
        focus_subject = base_topic or topic_info.get("resolved_topic") or ""
        focus_hint = resolved_topic or focus_subject
        exact_terms: List[str] = []

        if resolved_topic:
            if context_dependent and generic_only:
                standalone_query = resolved_topic
                query_variants = [resolved_topic]
                focus_hint = resolved_topic
            elif context_dependent:
                if base_topic:
                    standalone_query = _normalize_space(f"{base_topic} {cleaned_query}")
                elif _normalize_query_key(cleaned_query) in _normalize_query_key(resolved_topic):
                    standalone_query = resolved_topic
                else:
                    standalone_query = _normalize_space(f"{resolved_topic} {cleaned_query}")
                query_variants = [standalone_query]
                compact_query = ""
                if specific_terms and base_topic:
                    compact_query = _normalize_space(f"{base_topic} {' '.join(specific_terms)}")
                elif specific_terms:
                    compact_query = " ".join(specific_terms)
                if compact_query and _normalize_query_key(compact_query) != _normalize_query_key(standalone_query):
                    query_variants.append(compact_query)
                if self._should_carry_prior_resolved_topic(
                    base_topic=base_topic,
                    resolved_topic=resolved_topic,
                    current_specific_terms=specific_terms,
                ):
                    query_variants.append(resolved_topic)
                elif base_topic and _normalize_query_key(base_topic) != _normalize_query_key(standalone_query):
                    query_variants.append(base_topic)
                focus_hint = compact_query or cleaned_query or resolved_topic or focus_subject
            else:
                standalone_query = cleaned_query or query
                query_variants = [standalone_query]
                focus_hint = cleaned_query or resolved_topic or focus_subject
            if focus_subject:
                exact_terms.append(focus_subject)
        else:
            standalone_query = cleaned_query or query
            query_variants = [standalone_query]
            focus_hint = cleaned_query or focus_subject

        needs_history_expansion = bool(
            context_dependent and not resolved_topic and allow_history_expansion and not expanded_history_used
        )
        retrieval_action = self._infer_retrieval_action(
            context_dependent=context_dependent,
            focus_subject=focus_subject,
            focus_hint=focus_hint,
        )
        return {
            "followup": context_dependent,
            "context_dependent": context_dependent,
            "needs_history_expansion": needs_history_expansion,
            "standalone_query": standalone_query,
            "query_variants": query_variants,
            "focus_subject": focus_subject,
            "focus_hint": focus_hint,
            "answer_intent": answer_intent,
            "retrieval_action": retrieval_action,
            "exact_terms": exact_terms,
        }

    async def _repair_model_output(
        self,
        *,
        raw_output: str,
        query: str,
        chat_block: str,
        chat_id: str,
        message_id: str,
        expanded_history_used: bool,
    ) -> Dict[str, Any]:
        text = str(raw_output or "").strip()
        if not text:
            return {}
        system_prompt = (
            "You repair malformed retrieval-planner output into strict JSON.\n"
            "Return one valid JSON object only with keys: followup, context_dependent, needs_history_expansion, "
            "standalone_query, query_variants, focus_subject, focus_hint, answer_intent, retrieval_action, exact_terms, filters, time_filter.\n"
            "Do not add explanations."
        )
        user_prompt = (
            f"User query:\n{query}\n\n"
            f"Chat context:\n{chat_block or 'No prior chat context.'}\n\n"
            f"Malformed planner output:\n{text}\n\n"
            "Return repaired JSON only."
        )
        try:
            repaired = await self._call_model(system_prompt=system_prompt, user_prompt=user_prompt)
        except Exception as exc:
            await self._emit_trace(
                {
                    "stage": "planner_repair_error",
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "expanded_history_used": bool(expanded_history_used),
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "error": str(exc),
                }
            )
            return {}
        await self._emit_trace(
            {
                "stage": "planner_repair",
                "chat_id": chat_id,
                "message_id": message_id,
                "expanded_history_used": bool(expanded_history_used),
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "raw_output": repaired,
            }
        )
        return _extract_json_object(repaired)

    def _normalize_filters(self, raw: Any) -> Dict[str, Any]:
        payload = raw if isinstance(raw, dict) else {}
        report_types = _dedupe_strings(
            _coerce_string_list(payload.get("report_types"), max_items=8, max_chars=48),
            max_items=8,
        )
        return {
            "report_types": report_types,
            "branch": _clean_text(payload.get("branch"), max_chars=64) or None,
            "doc_id": _clean_text(payload.get("doc_id"), max_chars=96) or None,
            "parent_id": _clean_text(payload.get("parent_id"), max_chars=96) or None,
            "lang": _clean_text(payload.get("lang"), max_chars=24).lower() or None,
            "is_attachment": _coerce_bool(payload.get("is_attachment")),
            "chunk_no": self._safe_int(payload.get("chunk_no")),
        }

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        try:
            if value is None:
                return None
            return int(value)
        except Exception:
            return None

    def _normalize_time_filter(self, raw: Any) -> Dict[str, Any]:
        payload = raw if isinstance(raw, dict) else {}
        requested_field = _clean_text(payload.get("field"), max_chars=24).lower() or "none"
        if requested_field not in _ALLOWED_TIME_FIELDS:
            requested_field = "none"
        start_date = _parse_iso_date(payload.get("start_date"))
        end_date = _parse_iso_date(payload.get("end_date"))
        label = _clean_text(payload.get("label"), max_chars=96)

        if requested_field == "none" or not start_date or not end_date:
            return {}
        if end_date < start_date:
            start_date, end_date = end_date, start_date

        effective_field = requested_field
        if effective_field == "either":
            effective_field = self._default_time_field

        return {
            "field": effective_field,
            "requested_field": requested_field,
            "label": label or f"{start_date} to {end_date}",
            "matched_text": label or f"{start_date} to {end_date}",
            "start_date": start_date,
            "end_date": end_date,
            "start_ms": _to_epoch_ms(start_date, end_of_day=False),
            "end_ms": _to_epoch_ms(end_date, end_of_day=True),
        }

    def _normalize_plan_payload(
        self,
        *,
        query: str,
        raw_payload: Dict[str, Any],
        allow_history_expansion: bool,
        expanded_history_used: bool,
    ) -> RetrievalPlan:
        followup = bool(_coerce_bool(raw_payload.get("followup")))
        context_dependent = bool(_coerce_bool(raw_payload.get("context_dependent")))
        if followup:
            context_dependent = True

        focus_subject = _clean_text(raw_payload.get("focus_subject"), max_chars=120)
        focus_hint = _clean_text(raw_payload.get("focus_hint"), max_chars=120) or focus_subject
        raw_answer_intent = _clean_text(raw_payload.get("answer_intent"), max_chars=24).lower()
        answer_intent = (
            raw_answer_intent
            if raw_answer_intent in _ALLOWED_ANSWER_INTENTS
            else self._classify_answer_intent(query, context_dependent=context_dependent)
        )
        raw_retrieval_action = _clean_text(raw_payload.get("retrieval_action"), max_chars=32).lower()
        retrieval_action = (
            raw_retrieval_action
            if raw_retrieval_action in _ALLOWED_RETRIEVAL_ACTIONS
            else self._infer_retrieval_action(
                context_dependent=context_dependent,
                focus_subject=focus_subject,
                focus_hint=focus_hint,
            )
        )
        exact_terms = _dedupe_strings(
            [
                value
                for value in _coerce_string_list(
                    raw_payload.get("exact_terms"),
                    max_items=self._max_exact_terms,
                    max_chars=120,
                )
                if _is_exact_term_candidate(value)
            ],
            max_items=self._max_exact_terms,
        )
        standalone_query = _clean_text(raw_payload.get("standalone_query"), max_chars=220) or query
        if retrieval_action == "reuse_previous_topic":
            topic_query = focus_subject or focus_hint or (exact_terms[0] if exact_terms else "")
            cleaned_standalone = self._clean_topic_candidate(standalone_query)
            if topic_query and (
                not self._specific_terms(cleaned_standalone)
                or self._is_format_transform_query(standalone_query)
                or _normalize_query_key(standalone_query) == _normalize_query_key(query)
            ):
                standalone_query = topic_query

        query_candidates: List[str] = [
            standalone_query,
            *_coerce_query_list(raw_payload.get("query_variants"), max_items=self._max_query_variants + 2),
        ]
        if retrieval_action == "reuse_previous_topic":
            query_candidates = [
                candidate
                for candidate in query_candidates
                if candidate and not self._is_format_transform_query(candidate)
            ] or [standalone_query]
        if retrieval_action == "reuse_previous_topic":
            if focus_subject:
                query_candidates.append(focus_subject)
            if focus_hint:
                query_candidates.append(focus_hint)
        elif not context_dependent or _normalize_query_key(standalone_query) == _normalize_query_key(query):
            query_candidates.append(query)
        query_variants = _dedupe_strings(
            query_candidates,
            max_items=self._max_query_variants,
        )
        if not query_variants:
            query_variants = [standalone_query or query]

        filters = self._normalize_filters(raw_payload.get("filters"))
        time_filter = self._normalize_time_filter(raw_payload.get("time_filter"))
        if time_filter and not should_apply_retrieval_time_filter(query, time_filter):
            logger.info(
                "RAG planner suppressed weak time filter query=%s time_filter=%s",
                _truncate(query, 160),
                time_filter,
            )
            time_filter = {}
        needs_history_expansion = bool(_coerce_bool(raw_payload.get("needs_history_expansion")))
        if expanded_history_used:
            needs_history_expansion = False
        elif not allow_history_expansion:
            needs_history_expansion = False

        return RetrievalPlan(
            standalone_query=standalone_query,
            query_variants=query_variants,
            focus_subject=focus_subject,
            focus_hint=focus_hint,
            answer_intent=answer_intent,
            retrieval_action=retrieval_action,
            exact_terms=exact_terms,
            filters=filters,
            time_filter=time_filter,
            followup=followup,
            context_dependent=context_dependent,
            needs_history_expansion=needs_history_expansion,
            planner_used=bool(raw_payload),
            raw_plan=raw_payload,
        )

    async def plan(
        self,
        *,
        query: str,
        chat_context: Dict[str, Any],
        explicit_filters: Dict[str, Any],
        allow_history_expansion: bool,
        expanded_history_used: bool,
        chat_id: str,
        message_id: str,
    ) -> RetrievalPlan:
        today_utc = datetime.now(timezone.utc).date().isoformat()
        chat_block = self._chat_context_block(
            chat_context,
            max_turns=self._max_turns,      # for retrieval planner it is fixed to 6
            max_chars=self._max_context_chars,    # max context chars here are 2600
        )
        system_prompt = (
            "You are a retrieval planner for an enterprise RAG system.\n"
            "You do not answer the user. You only produce a retrieval plan as strict JSON.\n"
            "Return exactly one JSON object with keys:\n"
            "{\n"
            '  "followup": boolean,\n'
            '  "context_dependent": boolean,\n'
            '  "needs_history_expansion": boolean,\n'
            '  "standalone_query": string,\n'
            '  "query_variants": [string],\n'
            '  "focus_subject": string,\n'
            '  "focus_hint": string,\n'
            '  "answer_intent": "inform" | "format" | "summarize" | "continue",\n'
            '  "retrieval_action": "fresh_retrieval" | "reuse_previous_topic",\n'
            '  "exact_terms": [string],\n'
            '  "filters": {\n'
            '    "report_types": [string],\n'
            '    "branch": string,\n'
            '    "doc_id": string,\n'
            '    "parent_id": string,\n'
            '    "lang": string,\n'
            '    "is_attachment": boolean,\n'
            '    "chunk_no": integer\n'
            "  },\n"
            '  "time_filter": {\n'
            '    "field": "none" | "ingestion_date" | "document_date" | "either",\n'
            '    "label": string,\n'
            '    "start_date": "YYYY-MM-DD",\n'
            '    "end_date": "YYYY-MM-DD"\n'
            "  }\n"
            "}\n"
            "Rules:\n"
            "- Resolve references like above, earlier, that, it, same topic using only the supplied chat context.\n"
            "- If the query is standalone, keep standalone_query very close to the original wording.\n"
            "- query_variants must be short retrieval queries, best first, 1 to 4 items.\n"
            "- Treat phrases like 'from BigData', 'from big data', 'from internal documents', 'from our records', 'from my records', 'in the database', 'from KB', or 'from knowledge base' as retrieval-scope instructions, not topic keywords. Remove those phrases from standalone_query, query_variants, focus_subject, focus_hint, and exact_terms unless the user is literally asking about a system named BigData/database/records.\n"
            "- Example: 'more results of Narendra Modi from big data' should search for 'Narendra Modi', not 'Narendra Modi big data'.\n"
            "- exact_terms must contain only literal entities, names, phone numbers, identifiers, operation names, or quoted phrases likely to appear verbatim.\n"
            "- Do not put generic words in exact_terms: explain, details, profile, report, document, issue, summary.\n"
            "- Do not put UI/source-reference phrases in exact_terms, such as 'source 2', 'source number 2', or 'inference made from source'; those refer to prior answer source numbering, not document text.\n"
            "- Extract filters only when explicit or strongly implied.\n"
            "- Important distinction: report type words can mean either source filters or requested output format.\n"
            "- If UO, EIS, or SR appears with output-generation words such as draft, generate, prepare, write, revise, enrich, modify, update, or provide a revised draft, treat it as the requested output/report format only; do not put that value in filters.report_types.\n"
            "- Use filters.report_types for UO, EIS, SR, or other report types only when the user clearly asks to search existing source reports/documents, for example 'find UO reports about Tripti', 'from EIS documents', 'UO reports in last year', or 'search SR documents'.\n"
            "- Be conservative with time_filter. Do not create a metadata time_filter just because the user mentions an event date such as 'movement of X on 01 Dec 2019' or 'what happened on 12 Sept 2012'. Treat those dates as query text unless the user explicitly asks to restrict reports/documents/database by date.\n"
            "- Use time_filter for clear retrieval-scope date constraints like 'reports from the database in 2024', 'documents dated 12 Sept 2012', 'what is mentioned in reports from last year', 'last six months', or explicit between/from-to ranges.\n"
            "- Use only these report types when relevant: "
            f"{', '.join(self._known_report_types) or 'none provided'}.\n"
            "- Use only these branch values when relevant: "
            f"{', '.join(self._known_branches) or 'none provided'}.\n"
            "- answer_intent='format' when the user mainly changes output style or format of the same topic.\n"
            "- answer_intent='summarize' when the user asks for a shorter or summarized version.\n"
            "- answer_intent='continue' when the user asks to continue or respond on the same prior topic.\n"
            "- retrieval_action='reuse_previous_topic' when the query depends on prior chat context and mostly changes format, style, or continuation.\n"
            "- For formatting follow-ups like 'respond in 2 page report' or 'consider my first question and respond', keep standalone_query focused on the underlying topic, not the formatting instruction.\n"
            "- Use retrieval_action='fresh_retrieval' for new information-seeking turns.\n"
            f"- Today UTC is {today_utc}. Convert relative time requests into absolute start_date/end_date.\n"
            "- If there is no time constraint, return time_filter.field='none' and empty dates.\n"
            "- Set needs_history_expansion=true only when the current chat snippet is insufficient to resolve the query and older semantically-related chat turns would materially help.\n"
            "- Do not invent missing entity names, document ids, or date ranges.\n"
            "- Return JSON only. No markdown."
        )
        user_prompt = (
            f"User query:\n{query}\n\n"
            f"Explicit payload filters already present:\n{json.dumps(explicit_filters, ensure_ascii=False)}\n\n"
            "Chat context available to you (oldest to newest):\n"
            f"{chat_block or 'No prior chat context.'}\n\n"
            "Return JSON only."
        )

        try:
            raw = await self._call_model(system_prompt=system_prompt, user_prompt=user_prompt)
        except Exception as exc:
            await self._emit_trace(
                {
                    "stage": "planner_primary_error",
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "expanded_history_used": bool(expanded_history_used),
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "error": str(exc),
                }
            )
            logger.warning(
                "RAG planner call failed chat_id=%s message_id=%s error=%s",
                chat_id,
                message_id,
                exc,
            )
            return self._normalize_plan_payload(
                query=query,
                raw_payload={},
                allow_history_expansion=allow_history_expansion,
                expanded_history_used=expanded_history_used,
            )
        await self._emit_trace(
            {
                "stage": "planner_primary",
                "chat_id": chat_id,
                "message_id": message_id,
                "expanded_history_used": bool(expanded_history_used),
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "raw_output": raw,
            }
        )

        payload = _extract_json_object(raw)
        if not payload:
            payload = await self._repair_model_output(
                raw_output=raw,
                query=query,
                chat_block=chat_block,
                chat_id=chat_id,
                message_id=message_id,
                expanded_history_used=expanded_history_used,
            )
        if not payload:
            logger.warning(
                "RAG planner returned non-json chat_id=%s message_id=%s raw=%s",
                chat_id,
                message_id,
                _truncate(raw, 240),
            )

        plan = self._normalize_plan_payload(
            query=query,
            raw_payload=payload,
            allow_history_expansion=allow_history_expansion,
            expanded_history_used=expanded_history_used,
        )
        deterministic_payload = self._build_deterministic_fallback_payload(
            query=query,
            chat_context=chat_context,
            allow_history_expansion=allow_history_expansion,
            expanded_history_used=expanded_history_used,
        )
        deterministic_plan = self._normalize_plan_payload(
            query=query,
            raw_payload=deterministic_payload,
            allow_history_expansion=allow_history_expansion,
            expanded_history_used=expanded_history_used,
        )
        strong_context_signal = self._has_strong_context_dependency_signal(query)
        should_override_with_deterministic = bool(
            deterministic_plan.context_dependent
            and (
                not plan.planner_used
                or (strong_context_signal and not plan.context_dependent)
                or (
                    strong_context_signal
                    and _normalize_query_key(plan.standalone_query) == _normalize_query_key(query)
                    and _normalize_query_key(deterministic_plan.standalone_query) != _normalize_query_key(query)
                )
                or (strong_context_signal and deterministic_plan.focus_hint and not plan.focus_hint)
                or (
                    deterministic_plan.answer_intent != "inform"
                    and plan.answer_intent == "inform"
                )
            )
        )
        if should_override_with_deterministic:
            merged_payload = dict(payload or {})
            merged_payload.update(deterministic_payload)
            if payload:
                if payload.get("filters"):
                    merged_payload["filters"] = payload.get("filters")
                if payload.get("time_filter"):
                    merged_payload["time_filter"] = payload.get("time_filter")
                if payload.get("exact_terms"):
                    merged_payload["exact_terms"] = payload.get("exact_terms")
            plan = self._normalize_plan_payload(
                query=query,
                raw_payload=merged_payload,
                allow_history_expansion=allow_history_expansion,
                expanded_history_used=expanded_history_used,
            )
            logger.info(
                "RAG planner deterministic fallback applied chat_id=%s message_id=%s standalone=%s variants=%s",
                chat_id,
                message_id,
                _truncate(plan.standalone_query, 120),
                [_truncate(item, 120) for item in plan.query_variants],
            )
        await self._emit_trace(
            {
                "stage": "planner_result",
                "chat_id": chat_id,
                "message_id": message_id,
                "expanded_history_used": bool(expanded_history_used),
                "raw_output": raw,
                "parsed_payload": payload,
                "deterministic_payload": deterministic_payload,
                "deterministic_override_applied": bool(should_override_with_deterministic),
                "final_plan": {
                    "followup": bool(plan.followup),
                    "context_dependent": bool(plan.context_dependent),
                    "needs_history_expansion": bool(plan.needs_history_expansion),
                    "planner_used": bool(plan.planner_used),
                    "standalone_query": plan.standalone_query,
                    "query_variants": list(plan.query_variants or []),
                    "focus_subject": plan.focus_subject,
                    "focus_hint": plan.focus_hint,
                    "answer_intent": plan.answer_intent,
                    "retrieval_action": plan.retrieval_action,
                    "exact_terms": list(plan.exact_terms or []),
                    "filters": dict(plan.filters or {}),
                    "time_filter": dict(plan.time_filter or {}),
                },
            }
        )
        logger.info(
            "RAG planner chat_id=%s message_id=%s followup=%s context_dependent=%s answer_intent=%s retrieval_action=%s expand_history=%s standalone=%s variants=%s exact_terms=%s",
            chat_id,
            message_id,
            plan.followup,
            plan.context_dependent,
            plan.answer_intent,
            plan.retrieval_action,
            plan.needs_history_expansion,
            _truncate(plan.standalone_query, 120),
            [_truncate(item, 120) for item in plan.query_variants],
            plan.exact_terms,
        )
        return plan
