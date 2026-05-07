from __future__ import annotations

import asyncio
import json
import math
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from openai import AsyncOpenAI

from context_fetcher import ContextFetcher
from logger import logger
from prompt import (
    USED_SOURCES_JSON_END,
    USED_SOURCES_JSON_START,
    build_chat_context_answer_text,
    build_chat_context_system_prompt,
    build_general_background_prompt,
    build_general_background_system_prompt,
    build_rag_context_text,
    build_rag_system_prompt,
    build_source_documents,
)
from reranker_client import NativeRerankerClient
from retrieval_planner import RetrievalPlan, RetrievalPlanner
from retrieval_utils import RetrievalSupport
from settings import settings


class RAGWorker:
    _GENERAL_BACKGROUND_EXPLICIT_PATTERN = re.compile(
        r"\b("
        r"general\s+(?:model\s+)?knowledge|"
        r"background\s+(?:knowledge|context|information)|"
        r"public(?:ly)?\s+(?:known|available|knowledge|information)|"
        r"outside\s+(?:the\s+)?(?:documents?|db|database|knowledge\s*base|kb)|"
        r"beyond\s+(?:the\s+)?(?:documents?|db|database|knowledge\s*base|kb)|"
        r"not\s+(?:only\s+)?(?:from|in)\s+(?:the\s+)?(?:documents?|db|database|knowledge\s*base|kb)|"
        r"open\s+source|"
        r"from\s+(?:your|the\s+model'?s?)\s+(?:knowledge|training)|"
        r"what\s+do\s+you\s+know\s+about"
        r")\b",
        flags=re.IGNORECASE,
    )
    _NO_GENERAL_BACKGROUND_PATTERN = re.compile(
        r"\b(?:no|without)\s+(?:open[-\s]*source|outside|external|public|general\s+background)\b|"
        r"\b(?:do\s+not|don't|dont|avoid|skip|exclude)\s+"
        r"(?:use|include|add|give|append|provide|answer\s+from|respond\s+from|rely\s+on)?\s*"
        r"(?:the\s+)?(?:open[-\s]*source|outside|external|public|general\s+background)\b|"
        r"\b(?:only|strictly|solely)\s+(?:from|using|based\s+on)\s+"
        r"(?:the\s+)?(?:documents?|reports?|db|database|internal\s+(?:documents?|data)|"
        r"retrieved\s+context|knowledge\s*base|kb)\b|"
        r"\b(?:use|answer\s+from)\s+(?:only\s+)?(?:the\s+)?"
        r"(?:documents?|reports?|internal\s+data|db|database)\s+only\b",
        flags=re.IGNORECASE,
    )
    _INTERNAL_ONLY_QUERY_PATTERN = re.compile(
        r"\b("
        r"according\s+to\s+(?:the\s+)?(?:documents?|reports?|db|database)|"
        r"from\s+(?:our|the|these)\s+(?:documents?|reports?|db|database|internal\s+data)|"
        r"internal\s+(?:documents?|reports?|data|db|database)|"
        r"uploaded\s+files?|"
        r"audit\s+logs?|"
        r"doc(?:ument)?[\s_-]*id|"
        r"parent[\s_-]*id|"
        r"chunk(?:[\s_-]*(?:no|number|id))?|"
        r"system\s*path|"
        r"branch|"
        r"report\s+type"
        r")\b",
        flags=re.IGNORECASE,
    )
    _REPORT_OUTPUT_PATTERN = re.compile(
        r"\b(?:make|create|draft|write|prepare|generate|compose)\s+"
        r"(?:a\s+|an\s+|the\s+)?(?:report|briefing|memo|note|email|letter)\b|"
        r"\b(?:make|convert|format|turn)\s+(?:it|this|the\s+answer)?\s*"
        r"(?:into|as)\s+(?:a\s+|an\s+|the\s+)?"
        r"(?:report|briefing|memo|table|bullet\s+points?)\b|"
        r"\b(?:report|briefing|memo)\s+(?:format|style|template)\b",
        flags=re.IGNORECASE,
    )
    _CURRENT_OR_VERIFIED_QUERY_PATTERN = re.compile(
        r"\b("
        r"latest|current|today|yesterday|tomorrow|now|live|real[-\s]*time|"
        r"news|price|stock|share\s+price|weather|score|exchange\s+rate|"
        r"law|legal|medical|doctor|diagnosis|financial\s+advice|investment\s+advice"
        r")\b",
        flags=re.IGNORECASE,
    )
    _INSUFFICIENT_ANSWER_PATTERN = re.compile(
        r"\b("
        r"could\s+not\s+find|couldn't\s+find|not\s+find|"
        r"insufficient|not\s+enough|not\s+available|missing|"
        r"not\s+present|not\s+mentioned|not\s+provided|"
        r"(?:do|does|did)\s+not\s+contain|(?:don't|doesn't|didn't)\s+contain|"
        r"no\s+(?:relevant\s+)?(?:context|evidence|documents?)"
        r")\b",
        flags=re.IGNORECASE,
    )
    _CHAT_ONLY_TRANSFORM_PATTERN = re.compile(
        r"\b("
        r"from\s+above|above\s+answer|previous\s+answer|earlier\s+answer|last\s+answer|"
        r"same\s+answer|that\s+answer|summari[sz]e|shorten|condense|compress|"
        r"make\s+it|convert\s+it|rewrite\s+it|rephrase\s+it|format\s+it|"
        r"(?:one|two|three|\d+)\s+lines?|bullet\s+points?|bullets?|"
        r"in\s+(?:one|two|three|\d+)\s+lines?|in\s+brief|briefly"
        r")\b",
        flags=re.IGNORECASE,
    )
    _SOCIAL_GREETING_ONLY_PATTERN = re.compile(
        r"^\s*(?:hi|hello|hey|hii+|greetings|yo|good\s+(?:morning|afternoon|evening))"
        r"(?:\s+(?:there|assistant|team|buddy|sir))?\s*[!.?]*\s*$",
        flags=re.IGNORECASE,
    )
    _SOCIAL_ACK_ONLY_PATTERN = re.compile(
        r"^\s*(?:thanks|thank\s+you|thx|ok(?:ay)?|got\s+it|cool|nice|alright|understood)"
        r"(?:\s+(?:so\s+much|thanks))?\s*[!.?]*\s*$",
        flags=re.IGNORECASE,
    )
    _SOCIAL_FAREWELL_ONLY_PATTERN = re.compile(
        r"^\s*(?:bye|goodbye|see\s+you|take\s+care|catch\s+you\s+later)\s*[!.?]*\s*$",
        flags=re.IGNORECASE,
    )
    _SOCIAL_META_ONLY_PATTERN = re.compile(
        r"^\s*(?:"
        r"who\s+are\s+you|"
        r"what\s+are\s+you|"
        r"what\s+can\s+you\s+do|"
        r"what\s+do\s+you\s+do|"
        r"how\s+can\s+you\s+help(?:\s+me)?|"
        r"help|"
        r"can\s+you\s+help(?:\s+me)?"
        r")\s*[?.!]*\s*$",
        flags=re.IGNORECASE,
    )
    _RERANK_SPECIAL_TOKEN_PATTERN = re.compile(r"<\|[^>]+?\|>")
    _RERANK_THINK_TAG_PATTERN = re.compile(r"</?think>")

    def __init__(self, context_fetcher: ContextFetcher):
        self.cf = context_fetcher
        self.redis = None
        self.client = AsyncOpenAI(base_url=settings.vllm_base_url, api_key=settings.vllm_api_key)
        self._trace_lock = asyncio.Lock()
        self._trace_path = Path(str(getattr(settings, "llm_trace_file", "logs/rag_llm_trace.log")))
        self._llm_log_lock = asyncio.Lock()
        self._llm_log_contexts: Dict[str, List[Dict[str, Any]]] = {}
        self._context_trace_lock = asyncio.Lock()
        self._context_trace_path = self._resolve_service_log_path(
            str(getattr(settings, "rag_context_trace_file", "logs/rag_context_trace.log"))
        )
        self._planner_trace_lock = asyncio.Lock()
        self._planner_trace_path = Path(
            str(getattr(settings, "rag_planner_trace_file", "logs/rag_planner_trace.log"))
        )
        self._selector_trace_lock = asyncio.Lock()
        self._selector_trace_path = self._resolve_service_log_path(
            str(getattr(settings, "rag_selector_trace_file", "rag_selector_call.log"))
        )
        self._judge_trace_lock = asyncio.Lock()
        self._judge_trace_path = self._resolve_service_log_path(
            str(getattr(settings, "rag_judge_trace_file", "rag_judge_call.log"))
        )
        self._reranker = None
        if bool(getattr(settings, "rag_reranker_enabled", False)):
            self._reranker = NativeRerankerClient(
                str(getattr(settings, "rag_reranker_base_url", "")).strip(),
                timeout_seconds=float(getattr(settings, "rag_reranker_timeout_s", 6.0)),
                model_name=str(getattr(settings, "rag_reranker_model", "") or "").strip(),
            )
        self.support = RetrievalSupport(
            known_branches=getattr(settings, "rag_known_branches_csv", ""),
            known_report_types=getattr(settings, "rag_known_report_types_csv", ""),
        )
        self.planner = RetrievalPlanner(
            client=self.client,
            model_name=settings.model_name,
            known_branches=self.support._known_branches,
            known_report_types=self.support._known_report_types,
            temperature=float(getattr(settings, "rag_query_planner_temperature", 0.0)),
            max_tokens=int(getattr(settings, "rag_query_planner_max_tokens", 380)),
            max_query_variants=int(getattr(settings, "rag_retrieval_query_variants", 4)),
            max_exact_terms=int(getattr(settings, "rag_exact_match_max_terms", 4)),
            max_turns=int(getattr(settings, "rag_query_planner_max_turns", 6)),
            max_context_chars=int(getattr(settings, "rag_query_planner_context_chars", 2600)),
            default_time_field=str(getattr(settings, "rag_default_time_filter_field", "ingestion_date")),
            trace_hook=self._append_planner_trace,
        )

    @staticmethod
    def _resolve_service_log_path(raw_path: str) -> Path:
        path = Path(str(raw_path or "logs/rag_context_trace.log")).expanduser()
        if path.is_absolute():
            return path
        return Path(__file__).resolve().parent / path

    def setup_redis(self, redis_client) -> None:
        self.redis = redis_client

    @staticmethod
    def _task_timestamp() -> str:
        return str(datetime.now(timezone.utc))

    @staticmethod
    def _task_ttl_seconds() -> int:
        return max(60, int(getattr(settings, "rag_task_ttl_s", 1800)))

    @staticmethod
    def _response_stream_ttl_seconds() -> int:
        return max(60, int(getattr(settings, "rag_response_stream_ttl_s", 3600)))

    async def _refresh_task_ttl(self, message_id: str) -> None:
        normalized_message_id = str(message_id or "").strip()
        if not normalized_message_id or self.redis is None:
            return
        try:
            await self.redis.expire(f"task:{normalized_message_id}", self._task_ttl_seconds())
        except Exception as exc:
            logger.warning("RAG task TTL refresh failed message_id=%s error=%s", normalized_message_id, exc)

    async def _hset_task(self, message_id: str, mapping: Dict[str, Any]) -> None:
        normalized_message_id = str(message_id or "").strip()
        if not normalized_message_id or self.redis is None:
            return
        payload = {str(key): "" if value is None else str(value) for key, value in dict(mapping or {}).items()}
        if not payload:
            return
        try:
            await self.redis.hset(f"task:{normalized_message_id}", mapping=payload)
        except TypeError:
            for key, value in payload.items():
                await self.redis.hset(f"task:{normalized_message_id}", key, value)
        await self._refresh_task_ttl(normalized_message_id)

    async def _ensure_task_record(self, *, message_id: str, payload: Dict[str, Any]) -> str:
        normalized_message_id = str(message_id or "").strip()
        if not normalized_message_id:
            return ""
        default_stream = f"streaming:resp:{normalized_message_id}"
        record: Dict[str, Any] = {}
        if self.redis is not None:
            try:
                record = await self.redis.hgetall(f"task:{normalized_message_id}")
            except Exception as exc:
                logger.warning("RAG task lookup failed message_id=%s error=%s", normalized_message_id, exc)
        stream_name = str((record or {}).get("stream") or "").strip() or default_stream
        if not record:
            logger.warning(
                "RAG task record missing; reconstructing response stream message_id=%s stream=%s",
                normalized_message_id,
                stream_name,
            )
        await self._hset_task(
            normalized_message_id,
            {
                "message_id": normalized_message_id,
                "user_id": str((payload or {}).get("user_id") or ""),
                "chat_id": str((payload or {}).get("chat_id") or ""),
                "status": str((record or {}).get("status") or "queued"),
                "current_stage": str((record or {}).get("current_stage") or "rag_agent"),
                "ui": str((record or {}).get("ui") or "planning"),
                "ui_detail": str((record or {}).get("ui_detail") or (record or {}).get("ui_detailed") or "building retrieval plan"),
                "ui_detailed": str((record or {}).get("ui_detailed") or (record or {}).get("ui_detail") or "building retrieval plan"),
                "stream": stream_name,
                "search_mode": str((payload or {}).get("search_mode") or (record or {}).get("search_mode") or ""),
                "system_resp_id": str((record or {}).get("system_resp_id") or ""),
                "updated_at": self._task_timestamp(),
            },
        )
        return stream_name

    async def _is_task_cancelled(self, message_id: str) -> bool:
        normalized_message_id = str(message_id or "").strip()
        if not normalized_message_id or self.redis is None:
            return False
        try:
            if await self.redis.get(f"cancelled:{normalized_message_id}"):
                return True
            status = str(await self.redis.hget(f"task:{normalized_message_id}", "status") or "").strip().lower()
            return status == "cancelled"
        except Exception as exc:
            logger.warning("RAG cancellation check failed message_id=%s error=%s", normalized_message_id, exc)
            return False

    async def _delete_cancelled_response_stream(self, *, message_id: str, stream: str = "") -> None:
        normalized_message_id = str(message_id or "").strip()
        if not normalized_message_id or self.redis is None:
            return
        stream_name = str(stream or "").strip()
        try:
            if not stream_name:
                rec = await self.redis.hgetall(f"task:{normalized_message_id}")
                stream_name = str((rec or {}).get("stream") or "").strip()
            if stream_name:
                await self.redis.delete(stream_name)
                logger.info(
                    "RAG cancelled response stream deleted message_id=%s stream=%s",
                    normalized_message_id,
                    stream_name,
                )
            await self._hset_task(
                normalized_message_id,
                {
                    "status": "cancelled",
                    "ui": "cancelled",
                    "ui_detail": "request cancelled",
                    "ui_detailed": "request cancelled",
                    "updated_at": self._task_timestamp(),
                },
            )
        except Exception as exc:
            logger.warning(
                "RAG cancelled stream cleanup failed message_id=%s stream=%s error=%s",
                normalized_message_id,
                stream_name,
                exc,
            )

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        try:
            if value is None:
                return None
            return int(value)
        except Exception:
            return None

    async def _set_ui_state(self, *, message_id: str, ui: str, ui_detail: str = "") -> None:
        normalized_message_id = str(message_id or "").strip()
        if not normalized_message_id or self.redis is None:
            return
        try:
            await self._hset_task(
                normalized_message_id,
                {
                    "ui": str(ui or "").strip(),
                    "ui_detail": str(ui_detail or "").strip(),
                    "ui_detailed": str(ui_detail or "").strip(),
                    "updated_at": self._task_timestamp(),
                },
            )
        except Exception as exc:
            logger.warning(
                "RAG ui-state update failed message_id=%s ui=%s error=%s",
                normalized_message_id,
                ui,
                exc,
            )

    async def aclose(self) -> None:
        if self._reranker is not None:
            await self._reranker.aclose()

    @staticmethod
    def _normalize_query(data: Dict[str, Any]) -> str:
        query = str(data.get("content") or "").strip()
        return query or "Please help using available context."

    @classmethod
    def _classify_social_meta_query(cls, query: str) -> str:
        text = str(query or "").strip()
        if not text:
            return ""
        if cls._SOCIAL_GREETING_ONLY_PATTERN.fullmatch(text):
            return "greeting"
        if cls._SOCIAL_ACK_ONLY_PATTERN.fullmatch(text):
            return "acknowledgement"
        if cls._SOCIAL_FAREWELL_ONLY_PATTERN.fullmatch(text):
            return "farewell"
        if cls._SOCIAL_META_ONLY_PATTERN.fullmatch(text):
            return "meta"
        return ""

    @staticmethod
    def _social_meta_response(kind: str) -> str:
        if kind == "greeting":
            return "Hello. I can help with retrieval questions over the document database."
        if kind == "acknowledgement":
            return "Noted. Ask your next document question whenever you are ready."
        if kind == "farewell":
            return "Bye. Come back when you need help from the document database."
        return (
            "I am the retrieval assistant for the document database. "
            "You can ask about reports, entities, dates, branches, document IDs, or uploaded files."
        )

    @staticmethod
    def _truncate_text(text: Any, max_chars: int) -> str:
        value = str(text or "").strip()
        if max_chars <= 0 or len(value) <= max_chars:
            return value
        return value[:max_chars].rstrip() + "..."

    @staticmethod
    def _normalize_branch_key(value: Any) -> str:
        return str(value or "").strip().lower()

    @staticmethod
    def _safe_chunk_no(value: Any) -> Optional[int]:
        if value in (None, ""):
            return None
        try:
            return int(value)
        except Exception:
            return None

    @classmethod
    def _order_file_chunks(
        cls,
        chunks: List[Dict[str, Any]],
        attachment_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        rows = list(chunks or [])
        if not rows:
            return []
        ordered_ids = [str(file_id).strip() for file_id in list(attachment_ids or []) if str(file_id).strip()]
        file_order = {file_id: index for index, file_id in enumerate(ordered_ids)}

        def sort_key(chunk: Dict[str, Any]) -> tuple:
            file_id = str(chunk.get("file_id") or "").strip()
            chunk_no = cls._safe_chunk_no(chunk.get("chunk_no"))
            return (
                file_order.get(file_id, len(file_order)),
                file_id,
                0 if chunk_no is not None else 1,
                chunk_no if chunk_no is not None else 10**12,
                str(chunk.get("created_at") or ""),
                str(chunk.get("chunk_id") or chunk.get("message_id") or ""),
            )

        return sorted(rows, key=sort_key)

    def _mark_uploaded_file_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        marked: List[Dict[str, Any]] = []
        for chunk in list(chunks or []):
            payload = dict(chunk)
            payload["_context_source"] = "uploaded_file"
            payload.setdefault("_scope", "uploaded_file_full_file")
            payload.setdefault("_final_score", payload.get("_final_score", payload.get("score", 1.0) or 1.0))
            marked.append(payload)
        return marked

    def _mark_bigdata_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        marked: List[Dict[str, Any]] = []
        for chunk in list(chunks or []):
            payload = dict(chunk)
            payload["_context_source"] = "bigdata"
            marked.append(payload)
        return marked

    @staticmethod
    def _is_uploaded_file_context_chunk(chunk: Dict[str, Any]) -> bool:
        context_source = str((chunk or {}).get("_context_source") or "").strip().lower()
        if context_source == "uploaded_file":
            return True
        scope = str((chunk or {}).get("_scope") or "").strip().lower()
        return scope.startswith("uploaded_file")

    @classmethod
    def _filter_bigdata_source_chunks(cls, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [chunk for chunk in list(chunks or []) if not cls._is_uploaded_file_context_chunk(chunk)]

    @classmethod
    def _compact_file_anchor_terms(cls, text: str, *, max_terms: int) -> List[str]:
        value = str(text or "")
        if not value:
            return []
        candidates: List[str] = []
        candidates.extend(match.group(0).strip() for match in re.finditer(r"\b\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\b", value))
        candidates.extend(match.group(0).strip() for match in re.finditer(r"\b[A-Z][A-Za-z]+(?:[-\s][A-Z][A-Za-z]+){1,5}\b", value))
        candidates.extend(match.group(0).strip() for match in re.finditer(r"\b[A-Z0-9][A-Z0-9._/-]{3,}\b", value))

        out: List[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            cleaned = re.sub(r"\s+", " ", candidate).strip(" \t\r\n,.;:()[]{}")
            if len(cleaned) < 4:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(cleaned)
            if len(out) >= max(1, int(max_terms)):
                break
        return out

    def _build_uploaded_file_anchor(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        rows = list(chunks or [])
        if not rows:
            return {"text": "", "titles": [], "terms": [], "chunk_count": 0}

        titles: List[str] = []
        seen_titles: set[str] = set()
        excerpts: List[str] = []
        max_anchor_chunks = max(1, int(getattr(settings, "rag_file_anchor_max_chunks", 8)))
        max_excerpt_chars = max(200, int(getattr(settings, "rag_file_anchor_chunk_chars", 700)))
        for chunk in rows:
            title = str(chunk.get("title") or chunk.get("file_name") or chunk.get("bucket_name") or "").strip()
            if title and title.lower() not in seen_titles:
                seen_titles.add(title.lower())
                titles.append(title)
            if len(excerpts) >= max_anchor_chunks:
                continue
            content = self._chunk_raw_content(chunk)
            if content:
                chunk_no = str(chunk.get("chunk_no") or "").strip()
                prefix = f"chunk {chunk_no}: " if chunk_no else ""
                excerpts.append(prefix + self._truncate_text(re.sub(r"\s+", " ", content), max_excerpt_chars))

        combined = "\n".join(excerpts)
        terms = self._compact_file_anchor_terms(" ".join([*titles, combined]), max_terms=18)
        lines: List[str] = []
        if titles:
            lines.append("Uploaded file titles: " + "; ".join(titles[:6]))
        if terms:
            lines.append("High-signal terms from uploaded file: " + "; ".join(terms))
        if excerpts:
            lines.append("Representative uploaded file excerpts:")
            lines.extend(f"- {excerpt}" for excerpt in excerpts)
        anchor_text = self._truncate_text(
            "\n".join(lines),
            max(1000, int(getattr(settings, "rag_file_anchor_max_chars", 6000))),
        )
        return {
            "text": anchor_text,
            "titles": titles,
            "terms": terms,
            "chunk_count": len(rows),
        }

    def _build_file_anchored_planner_payload(
        self,
        data: Dict[str, Any],
        *,
        user_query: str,
        file_anchor: Dict[str, Any],
    ) -> Dict[str, Any]:
        anchor_text = str((file_anchor or {}).get("text") or "").strip()
        if not anchor_text:
            return dict(data)
        anchored = dict(data)
        anchored["content"] = "\n\n".join(
            [
                str(user_query or "").strip() or "Find BigData information related to the uploaded file.",
                "Uploaded file anchor for retrieval planning:",
                anchor_text,
                (
                    "Planner instruction: use the uploaded file anchor only to identify the subject, entities, "
                    "dates, and themes for BigData retrieval. Do not treat uploaded-file metadata as BigData filters "
                    "unless the user explicitly asks for those filters."
                ),
            ]
        ).strip()
        anchored["_original_content"] = data.get("content")
        anchored["_uploaded_file_anchor"] = file_anchor
        return anchored

    def _compact_file_anchored_retrieval_query(self, user_query: str, file_anchor: Dict[str, Any]) -> str:
        terms = [str(term).strip() for term in list((file_anchor or {}).get("terms") or []) if str(term).strip()]
        titles = [str(title).strip() for title in list((file_anchor or {}).get("titles") or []) if str(title).strip()]
        parts = [str(user_query or "").strip() or "Find related BigData information for the uploaded file."]
        if terms:
            parts.append("Uploaded-file terms: " + "; ".join(terms[:12]))
        elif titles:
            parts.append("Uploaded-file titles: " + "; ".join(titles[:4]))
        return self._truncate_text(" ".join(parts), 900)

    def _sanitize_file_anchored_plan(
        self,
        plan: RetrievalPlan,
        *,
        user_query: str,
        file_anchor: Dict[str, Any],
    ) -> RetrievalPlan:
        if not str((file_anchor or {}).get("text") or "").strip():
            return plan
        compact_query = self._compact_file_anchored_retrieval_query(user_query, file_anchor)
        marker = "uploaded file anchor for retrieval planning"

        standalone = str(plan.standalone_query or "").strip()
        if not standalone or len(standalone) > 900 or marker in standalone.lower():
            plan.standalone_query = compact_query

        clean_variants: List[str] = []
        for variant in list(plan.query_variants or []):
            value = str(variant or "").strip()
            if not value or len(value) > 900 or marker in value.lower():
                continue
            clean_variants.append(value)
        plan.query_variants = self.support.dedupe_queries(
            [compact_query, *clean_variants],
            max_items=max(1, int(settings.rag_retrieval_query_variants)),
        )
        return plan

    @staticmethod
    def _chunk_score(chunk: Dict[str, Any]) -> float:
        raw = chunk.get("_rerank_score", chunk.get("_final_score", chunk.get("_heuristic_score", chunk.get("_sim_score", chunk.get("score", 0.0)))))
        try:
            return float(raw)
        except Exception:
            return 0.0

    @staticmethod
    def _chunk_exact_score(chunk: Dict[str, Any]) -> float:
        try:
            return float(chunk.get("_exact_match_score", 0.0) or 0.0)
        except Exception:
            return 0.0

    @classmethod
    def _sanitize_rerank_text(cls, value: Any) -> str:
        text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
        text = cls._RERANK_SPECIAL_TOKEN_PATTERN.sub(" ", text)
        text = cls._RERANK_THINK_TAG_PATTERN.sub(" ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @classmethod
    def _chunk_body_text_for_rerank(cls, chunk: Dict[str, Any]) -> str:
        content = cls._sanitize_rerank_text(
            chunk.get("_rerank_content")
            or chunk.get("content")
            or chunk.get("text")
            or chunk.get("chunk_text")
            or chunk.get("page_content")
            or ""
        )
        if not content:
            return ""
        metadata_lines: List[str] = []
        for label, key in (
            ("DocId", "doc_id"),
            ("ParentId", "parent_id"),
            ("ReportType", "report_type"),
            ("Branch", "branch"),
            ("Section", "section_heading"),
            ("SystemPath", "system_path"),
        ):
            value = cls._sanitize_rerank_text(chunk.get(key))
            if value:
                metadata_lines.append(f"{label}: {cls._truncate_text(value, 220)}")
        page_start = chunk.get("page_start")
        page_end = chunk.get("page_end")
        if page_start not in (None, "") or page_end not in (None, ""):
            metadata_lines.append(f"Pages: {page_start or '?'}-{page_end or page_start or '?'}")
        chunk_no = chunk.get("chunk_no")
        total_chunks = chunk.get("total_chunks")
        if chunk_no not in (None, ""):
            metadata_lines.append(f"Chunk: {chunk_no}/{total_chunks or '?'}")
        quality_score = chunk.get("quality_score")
        if quality_score not in (None, ""):
            try:
                metadata_lines.append(f"QualityScore: {float(quality_score):.3f}")
            except Exception:
                metadata_lines.append(f"QualityScore: {quality_score}")
        neighbor_count = chunk.get("_neighbor_chunk_count")
        if neighbor_count not in (None, "", 0):
            metadata_lines.append(f"NeighborContextChunks: {neighbor_count}")

        full_text = content
        if metadata_lines:
            full_text = "\n".join(metadata_lines) + "\n\n" + content
        full_text = re.sub(r"[ \t]+", " ", full_text)
        full_text = re.sub(r"\n{3,}", "\n\n", full_text)
        return full_text.strip()

    @classmethod
    def _chunk_text_for_rerank(cls, chunk: Dict[str, Any], *, max_chars: int) -> str:
        full_text = cls._chunk_body_text_for_rerank(chunk)
        if not full_text:
            return ""
        if bool(getattr(settings, "rag_reranker_qwen3_template_enabled", True)):
            prefix = "<Document>: "
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            body_limit = max(1, int(max_chars) - len(prefix) - len(suffix))
            return f"{prefix}{cls._truncate_text(full_text, body_limit)}{suffix}"
        return cls._truncate_text(full_text, max_chars)

    @staticmethod
    def _query_text_for_rerank(query: str) -> str:
        clean_query = str(query or "").strip()
        if not clean_query:
            return ""
        if not bool(getattr(settings, "rag_reranker_use_instruction", True)):
            return clean_query
        instruction = str(getattr(settings, "rag_reranker_instruction", "") or "").strip()
        if not instruction:
            return clean_query
        lowered = clean_query.lstrip().lower()
        if lowered.startswith("<instruct>:") or lowered.startswith("instruct:"):
            return clean_query
        if bool(getattr(settings, "rag_reranker_qwen3_template_enabled", True)):
            prefix = (
                "<|im_start|>system\n"
                ' Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
                "<|im_end|>\n"
                "<|im_start|>user\n"
            )
            return f"{prefix}<Instruct>: {instruction}\n<Query>: {clean_query}\n"
        return f"<Instruct>: {instruction}\n<Query>: {clean_query}"

    @staticmethod
    def _chunk_raw_content(chunk: Dict[str, Any]) -> str:
        return str(
            chunk.get("content")
            or chunk.get("text")
            or chunk.get("chunk_text")
            or chunk.get("page_content")
            or ""
        ).strip()

    @staticmethod
    def _chunk_debug_id(chunk: Dict[str, Any]) -> str:
        base = (
            str(chunk.get("doc_id") or "").strip()
            or str(chunk.get("parent_id") or "").strip()
            or str(chunk.get("file_id") or "").strip()
            or "source"
        )
        suffix = (
            str(chunk.get("chunk_no") or "").strip()
            or str(chunk.get("chunk_id") or "").strip()
            or str(chunk.get("point_id") or "").strip()
        )
        return f"{base}:{suffix}" if suffix else base

    @classmethod
    def _chunk_origin_flags(cls, chunk: Dict[str, Any]) -> Tuple[bool, bool]:
        scope = str(chunk.get("_scope") or "").strip().lower()
        retrieval_queries = [
            str(value).strip().lower()
            for value in list(chunk.get("_retrieval_queries") or [])
            if str(value).strip()
        ]
        exact_score = cls._chunk_exact_score(chunk)
        has_exact = (
            scope.startswith("exact")
            or "exact" in retrieval_queries
            or any(value.startswith("exact_") for value in retrieval_queries)
            or exact_score > 0.0
        )
        has_semantic = scope == "semantic" or "semantic" in retrieval_queries
        return has_exact, has_semantic

    @classmethod
    def _chunk_origin_counts(cls, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        counts = {"total": len(list(chunks or [])), "exact": 0, "semantic": 0, "both": 0, "other": 0}
        for chunk in list(chunks or []):
            has_exact, has_semantic = cls._chunk_origin_flags(chunk)
            if has_exact:
                counts["exact"] += 1
            if has_semantic:
                counts["semantic"] += 1
            if has_exact and has_semantic:
                counts["both"] += 1
            if not has_exact and not has_semantic:
                counts["other"] += 1
        return counts

    def _build_rerank_pool(self, chunks: List[Dict[str, Any]], *, pool_size: int) -> List[Dict[str, Any]]:
        limit = max(1, int(pool_size))
        rows = list(chunks or [])
        if len(rows) <= limit:
            return rows

        semantic_target = max(0, int(getattr(settings, "rag_reranker_semantic_pool_size", 0)))
        exact_target = max(0, int(getattr(settings, "rag_reranker_exact_pool_size", 0)))
        if semantic_target + exact_target <= 0:
            semantic_target = int(math.ceil(limit * 0.80))
            exact_target = limit - semantic_target
        if semantic_target + exact_target > limit:
            total_target = max(1, semantic_target + exact_target)
            semantic_target = int(math.floor(limit * (semantic_target / total_target)))
            exact_target = limit - semantic_target

        selected: List[Dict[str, Any]] = []
        selected_keys: set[str] = set()

        def add_matching(predicate, target: int) -> None:
            for chunk in rows:
                if len(selected) >= limit or sum(1 for item in selected if predicate(item)) >= target:
                    break
                if not predicate(chunk):
                    continue
                key = self.support.chunk_dedupe_key(chunk)
                if key in selected_keys:
                    continue
                selected.append(chunk)
                selected_keys.add(key)

        # Prefer pure semantic chunks for the larger pool share, and reserve a
        # smaller lane for exact document hits. Rows that are both exact and
        # semantic count toward the exact lane to avoid starving exact recall.
        add_matching(
            lambda item: (lambda flags: flags[1] and not flags[0])(self._chunk_origin_flags(item)),
            semantic_target,
        )
        add_matching(
            lambda item: self._chunk_origin_flags(item)[0],
            exact_target,
        )

        for chunk in rows:
            if len(selected) >= limit:
                break
            key = self.support.chunk_dedupe_key(chunk)
            if key in selected_keys:
                continue
            selected.append(chunk)
            selected_keys.add(key)
        return selected

    @staticmethod
    def _prompt_chars_per_token() -> float:
        try:
            return max(1.0, float(getattr(settings, "rag_prompt_chars_per_token", 4.0)))
        except Exception:
            return 4.0

    @classmethod
    def _estimate_prompt_tokens(cls, text: Any) -> int:
        value = str(text or "")
        if not value:
            return 0
        return max(1, int(math.ceil(len(value) / cls._prompt_chars_per_token())))

    @classmethod
    def _truncate_text_to_prompt_tokens(cls, text: Any, max_tokens: int) -> str:
        value = str(text or "").strip()
        if max_tokens <= 0 or not value:
            return ""
        max_chars = max(1, int(max_tokens * cls._prompt_chars_per_token()))
        return cls._truncate_text(value, max_chars)

    def _prompt_chunk_token_cost(self, chunk: Dict[str, Any]) -> int:
        overhead = max(0, int(getattr(settings, "rag_prompt_chunk_overhead_tokens", 80)))
        return overhead + self._estimate_prompt_tokens(self._chunk_raw_content(chunk))

    def _with_prompt_content(self, chunk: Dict[str, Any], prompt_content: str) -> Dict[str, Any]:
        payload = dict(chunk)
        raw_content = self._chunk_raw_content(payload)
        normalized_raw = re.sub(r"\s+", " ", raw_content).strip()
        if prompt_content and prompt_content != normalized_raw:
            payload["_prompt_content"] = prompt_content
            payload["_prompt_content_truncated"] = True
            payload["_prompt_content_chars"] = len(prompt_content)
            payload["_raw_content_chars"] = len(raw_content)
        else:
            payload.pop("_prompt_content", None)
            payload["_prompt_content_truncated"] = False
            payload["_prompt_content_chars"] = len(raw_content)
            payload["_raw_content_chars"] = len(raw_content)
        return payload

    def _select_prompt_chunks_by_token_budget(
        self,
        chunks: List[Dict[str, Any]],
        *,
        token_budget_override: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        rows = list(chunks or [])
        configured_token_budget = int(
            token_budget_override
            if token_budget_override is not None
            else getattr(settings, "rag_prompt_token_budget", 30000)
        )
        if not rows:
            return [], {
                "selected_chunks": 0,
                "candidate_chunks": 0,
                "token_budget": configured_token_budget,
                "estimated_prompt_tokens": 0,
            }

        token_budget = max(1, configured_token_budget)
        static_overhead = max(0, int(getattr(settings, "rag_prompt_static_overhead_tokens", 1400)))
        content_budget = max(1, token_budget - static_overhead)
        chunk_overhead = max(0, int(getattr(settings, "rag_prompt_chunk_overhead_tokens", 80)))
        min_chunk_tokens = max(1, int(getattr(settings, "rag_prompt_min_chunk_tokens", 180)))
        hard_cap = int(getattr(settings, "rag_max_prompt_chunks", 0))
        max_chunks = len(rows) if hard_cap <= 0 else max(1, hard_cap)
        guaranteed_exact = max(0, int(getattr(settings, "rag_exact_match_guaranteed_chunks", 0)))

        selected: Dict[str, Tuple[int, Dict[str, Any], int]] = {}
        used_tokens = 0
        truncated_chunks = 0

        def add_chunk(index: int, chunk: Dict[str, Any], *, force: bool = False) -> bool:
            nonlocal used_tokens, truncated_chunks
            if len(selected) >= max_chunks:
                return False
            key = self.support.chunk_dedupe_key(chunk)
            if key in selected:
                return False

            raw_content = re.sub(r"\s+", " ", self._chunk_raw_content(chunk)).strip()
            if not raw_content:
                return False
            full_cost = chunk_overhead + self._estimate_prompt_tokens(raw_content)
            remaining = content_budget - used_tokens
            if full_cost <= remaining:
                selected[key] = (index, self._with_prompt_content(chunk, raw_content), full_cost)
                used_tokens += full_cost
                return True

            available_content_tokens = max(0, remaining - chunk_overhead)
            if available_content_tokens >= min_chunk_tokens or (force and available_content_tokens > 0 and not selected):
                prompt_content = self._truncate_text_to_prompt_tokens(raw_content, available_content_tokens)
                if not prompt_content:
                    return False
                cost = chunk_overhead + self._estimate_prompt_tokens(prompt_content)
                selected[key] = (index, self._with_prompt_content(chunk, prompt_content), cost)
                used_tokens += cost
                truncated_chunks += 1
                return True
            return False

        exact_added = 0
        if guaranteed_exact > 0:
            for index, chunk in enumerate(rows):
                if exact_added >= guaranteed_exact:
                    break
                if self._chunk_origin_flags(chunk)[0] and add_chunk(index, chunk, force=True):
                    exact_added += 1

        for index, chunk in enumerate(rows):
            if len(selected) >= max_chunks:
                break
            if content_budget - used_tokens <= chunk_overhead + min_chunk_tokens:
                break
            add_chunk(index, chunk)

        ordered = [item[1] for item in sorted(selected.values(), key=lambda item: item[0])]
        counts = self._chunk_origin_counts(ordered)
        budget_info = {
            "candidate_chunks": len(rows),
            "selected_chunks": len(ordered),
            "token_budget": token_budget,
            "static_overhead_tokens": static_overhead,
            "content_budget_tokens": content_budget,
            "estimated_prompt_tokens": used_tokens + static_overhead,
            "estimated_chunk_tokens": used_tokens,
            "max_prompt_chunks": hard_cap,
            "truncated_chunks": truncated_chunks,
            "guaranteed_exact": guaranteed_exact,
            "exact_guarantee_selected": exact_added,
            "chunk_counts": counts,
        }
        return ordered, budget_info

    @staticmethod
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

    @staticmethod
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

    def _normalize_text_list(self, raw: Any, *, max_items: int, max_chars: int = 180) -> List[str]:
        items = list(raw) if isinstance(raw, (list, tuple, set)) else [raw]
        out: List[str] = []
        seen: set[str] = set()
        for item in items:
            parts = re.split(r"[\r\n]+", str(item or "")) if isinstance(item, str) else [item]
            for part in parts:
                value = str(part or "").strip()
                if not value:
                    continue
                normalized = re.sub(r"\s+", " ", value).strip(" \t\r\n,.;:-")
                if not normalized:
                    continue
                key = normalized.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(self._truncate_text(normalized, max_chars))
                if len(out) >= max(1, int(max_items)):
                    return out
        return out

    @staticmethod
    def _completion_text(response: Any) -> str:
        try:
            if not response or not list(getattr(response, "choices", []) or []):
                return ""
            message = response.choices[0].message
            content = message.content
        except Exception:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content") or ""
                else:
                    text = getattr(item, "text", "") or getattr(item, "content", "")
                if text:
                    parts.append(str(text))
            return "".join(parts).strip()
        for attr_name in ("reasoning_content", "reasoning"):
            try:
                fallback = getattr(message, attr_name, "")
            except Exception:
                fallback = ""
            if fallback:
                return str(fallback or "").strip()
        return str(content or "").strip()

    async def _call_llm_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        model_name: str = "",
        json_mode: bool = False,
    ) -> str:
        request_payload = {
            "model": str(model_name or settings.model_name),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": float(temperature),
            "max_tokens": max(1, int(max_tokens)),
            "stream": False,
        }
        reasoning_effort = str(getattr(settings, "rag_utility_llm_reasoning_effort", "") or "").strip().lower()
        if reasoning_effort:
            request_payload["extra_body"] = {"reasoning_effort": reasoning_effort}
        try:
            if json_mode:
                response = await self.client.chat.completions.create(
                    **request_payload,
                    response_format={"type": "json_object"},
                )
            else:
                response = await self.client.chat.completions.create(**request_payload)
        except Exception as exc:
            if json_mode and any(
                token in str(exc).lower()
                for token in ("response_format", "json_object", "json schema", "extra_forbidden", "extra_body", "reasoning_effort")
            ):
                logger.info("RAG utility JSON mode or reasoning hint unsupported by backend; retrying without extras.")
                request_payload.pop("extra_body", None)
                response = await self.client.chat.completions.create(**request_payload)
            else:
                raise
        return self._completion_text(response)

    def _selector_chunk_text(self, chunk: Dict[str, Any], *, max_chars: int) -> str:
        text = str(chunk.get("_rerank_content") or self._chunk_raw_content(chunk)).strip()
        if not text:
            return ""
        return self._truncate_text(re.sub(r"\s+", " ", text), max_chars)

    def _build_evidence_selector_batches(
        self,
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        rows = list(chunks or [])
        if not rows:
            return {"batches": [], "id_to_chunk": {}, "candidate_count": 0}

        batch_token_budget = max(1, int(getattr(settings, "rag_evidence_selector_batch_token_budget", 30000)))
        static_overhead = max(0, int(getattr(settings, "rag_evidence_selector_static_overhead_tokens", 1000)))
        content_budget = max(1, batch_token_budget - static_overhead)
        chunk_overhead = max(0, int(getattr(settings, "rag_evidence_selector_chunk_overhead_tokens", 80)))
        min_chunk_tokens = max(1, int(getattr(settings, "rag_evidence_selector_min_chunk_tokens", 160)))
        max_batches = max(1, int(getattr(settings, "rag_evidence_selector_max_batches", 2)))
        max_chars_per_chunk = max(200, int(getattr(settings, "rag_evidence_selector_max_chars_per_chunk", 2200)))

        id_to_chunk = {index: chunk for index, chunk in enumerate(rows, start=1)}
        batches: List[Dict[str, Any]] = []
        next_index = 0

        for batch_number in range(1, max_batches + 1):
            if next_index >= len(rows):
                break

            start_rank = next_index + 1
            estimated_chunk_tokens = 0
            truncated_rows = 0
            batch_rows: List[Dict[str, Any]] = []

            while next_index < len(rows):
                chunk = rows[next_index]
                global_source_id = next_index + 1
                excerpt = self._selector_chunk_text(chunk, max_chars=max_chars_per_chunk)
                if not excerpt:
                    next_index += 1
                    continue

                full_cost = chunk_overhead + self._estimate_prompt_tokens(excerpt)
                remaining = content_budget - estimated_chunk_tokens
                selected_excerpt = excerpt
                was_truncated = False

                if full_cost <= remaining:
                    cost = full_cost
                else:
                    available_content_tokens = max(0, remaining - chunk_overhead)
                    if available_content_tokens < min_chunk_tokens and batch_rows:
                        break
                    selected_excerpt = self._truncate_text_to_prompt_tokens(excerpt, available_content_tokens)
                    if not selected_excerpt:
                        break
                    cost = chunk_overhead + self._estimate_prompt_tokens(selected_excerpt)
                    was_truncated = selected_excerpt != excerpt

                has_exact, has_semantic = self._chunk_origin_flags(chunk)
                batch_rows.append(
                    {
                        "global_source_id": global_source_id,
                        "rerank_rank": global_source_id,
                        "origin": (
                            "semantic+exact"
                            if has_exact and has_semantic
                            else "exact"
                            if has_exact
                            else "semantic"
                            if has_semantic
                            else str(chunk.get("_scope") or "other")
                        ),
                        "debug_id": self._chunk_debug_id(chunk),
                        "doc_id": str(chunk.get("doc_id") or "").strip(),
                        "parent_id": str(chunk.get("parent_id") or "").strip(),
                        "chunk_no": chunk.get("chunk_no"),
                        "report_type": chunk.get("report_type"),
                        "branch": chunk.get("branch"),
                        "score": self._chunk_score(chunk),
                        "sim_score": float(chunk.get("_sim_score", 0.0) or 0.0),
                        "exact_match_score": float(chunk.get("_exact_match_score", 0.0) or 0.0),
                        "content": selected_excerpt,
                    }
                )
                estimated_chunk_tokens += cost
                if was_truncated:
                    truncated_rows += 1
                next_index += 1
                if was_truncated:
                    break

            if not batch_rows:
                break

            batches.append(
                {
                    "batch_number": batch_number,
                    "start_rank": start_rank,
                    "end_rank": next_index,
                    "estimated_prompt_tokens": estimated_chunk_tokens + static_overhead,
                    "estimated_chunk_tokens": estimated_chunk_tokens,
                    "truncated_rows": truncated_rows,
                    "rows": batch_rows,
                }
            )

            if next_index >= len(rows):
                break

        return {"batches": batches, "id_to_chunk": id_to_chunk, "candidate_count": len(rows)}

    def _normalize_evidence_selector_decision(
        self,
        raw_payload: Dict[str, Any],
        *,
        valid_source_ids: set[int],
        max_ids: int,
    ) -> Dict[str, Any]:
        payload = dict(raw_payload or {})
        evidence_source_ids: List[int] = []
        seen: set[int] = set()

        def add_value(raw_value: Any) -> None:
            nonlocal evidence_source_ids
            if len(evidence_source_ids) >= max(1, int(max_ids)):
                return
            parsed_values: List[int] = []
            if isinstance(raw_value, (list, tuple, set)):
                for item in raw_value:
                    add_value(item)
                return
            if isinstance(raw_value, str):
                parsed_values.extend(int(match.group(1)) for match in re.finditer(r"\b(\d+)\b", raw_value))
            else:
                try:
                    parsed_values.append(int(raw_value))
                except Exception:
                    return
            for value in parsed_values:
                if value not in valid_source_ids or value in seen:
                    continue
                seen.add(value)
                evidence_source_ids.append(value)
                if len(evidence_source_ids) >= max(1, int(max_ids)):
                    return

        add_value(payload.get("evidence_source_ids"))

        return {
            "evidence_source_ids": evidence_source_ids,
            "insufficient_in_batch": bool(self._coerce_bool(payload.get("insufficient_in_batch"))),
        }

    async def _run_evidence_selector_batch(
        self,
        *,
        message_id: str,
        query: str,
        primary_query: str,
        retrieval_focus: str,
        batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        batch_rows = list(batch.get("rows") or [])
        if not batch_rows:
            return {
                "evidence_source_ids": [],
                "insufficient_in_batch": True,
            }

        valid_source_ids = {
            int(row.get("global_source_id"))
            for row in batch_rows
            if self._safe_int(row.get("global_source_id")) is not None
        }
        max_ids = max(1, int(getattr(settings, "rag_evidence_selector_max_evidence_ids_per_batch", 48)))
        system_prompt = (
            "You are an evidence selector for a document-grounded RAG system.\n"
            "You are not answering the user. You only select globally numbered reranked chunks that contain usable answer evidence.\n"
            "Usable answer evidence means distinct answer-bearing evidence, including direct evidence and necessary supporting evidence.\n"
            "Direct evidence means facts, disambiguators, names with org/date/location, quoted lines, identifiers, or chunk text that materially answers the query.\n"
            "Necessary supporting evidence means context that adds a distinct fact needed to explain, verify, connect, qualify, or disambiguate a direct answer.\n"
            "Favor recall over minimality, but select only chunks that add distinct evidence useful for answering the query.\n"
            "For person-detail queries, be exhaustive: select chunks with family, relatives, aliases, addresses, identifiers, phones, travel, roles, relationships, events, associates, and biographical details when they are relevant to the requested person.\n"
            "If multiple plausible entities share the requested name, select usable evidence for every plausible entity rather than forcing a single identity.\n"
            "For news or event queries, select chunks with concrete events, dates, actors, locations, claims, outcomes, or source/report details.\n"
            "Do not select generic background, nearby context, repeated exact-match hits, vague topic/name mentions, or duplicate chunks unless they add a new fact needed for the answer.\n"
            "Do not explain your reasoning, do not discuss chunks one by one, and do not write prose before the JSON.\n"
            "Return exactly one JSON object and no other text.\n"
            "Use this schema:\n"
            "{"
            "\"evidence_source_ids\":[],"
            "\"insufficient_in_batch\":false"
            "}\n"
            "Rules:\n- `evidence_source_ids` must contain only IDs from this batch.\n- Return only the IDs that contain distinct direct or necessary supporting evidence for the query.\n- Return IDs even when this batch alone is insufficient; partial usable evidence is still useful.\n- If this batch contains no distinct direct or necessary supporting answer evidence, return an empty list.\n- Set `insufficient_in_batch=true` only when this batch alone is not enough for a detailed answer."
        )
        user_prompt = "\n".join(
            [
                "## User Query",
                str(query or "").strip(),
                "",
                "## Resolved Retrieval Query",
                str(primary_query or query or "").strip(),
                "",
                "## Retrieval Focus",
                str(retrieval_focus or "").strip(),
                "",
                "## Batch Metadata",
                json.dumps(
                    {
                        "batch_number": int(batch.get("batch_number") or 0),
                        "start_rerank_rank": int(batch.get("start_rank") or 0),
                        "end_rerank_rank": int(batch.get("end_rank") or 0),
                        "estimated_prompt_tokens": int(batch.get("estimated_prompt_tokens") or 0),
                        "truncated_rows": int(batch.get("truncated_rows") or 0),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "",
                "## Reranked Batch Chunks",
                json.dumps(batch_rows, ensure_ascii=False, indent=2),
                "",
                "## Task",
                "Select all global_source_id values in this batch that contain distinct direct or necessary supporting evidence for an exhaustive answer to the user query.",
            ]
        ).strip()
        await self._append_selector_trace(
            message_id=message_id,
            stage=f"selector_batch_{int(batch.get('batch_number') or 0)}_prompt",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            metadata={
                "batch_number": int(batch.get("batch_number") or 0),
                "row_count": len(batch_rows),
                "start_rank": int(batch.get("start_rank") or 0),
                "end_rank": int(batch.get("end_rank") or 0),
            },
        )
        try:
            raw_output = await self._call_llm_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=float(getattr(settings, "rag_evidence_selector_temperature", 0.0)),
                max_tokens=int(getattr(settings, "rag_evidence_selector_max_tokens", 2048)),
                model_name=str(getattr(settings, "rag_evidence_selector_model_name", "") or "").strip(),
                json_mode=True,
            )
            parsed = self._extract_json_object(raw_output)
            normalized = self._normalize_evidence_selector_decision(
                parsed,
                valid_source_ids=valid_source_ids,
                max_ids=max_ids,
            )
            await self._append_selector_trace(
                message_id=message_id,
                stage=f"selector_batch_{int(batch.get('batch_number') or 0)}_result",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                metadata={
                    "batch_number": int(batch.get("batch_number") or 0),
                    "row_count": len(batch_rows),
                    "start_rank": int(batch.get("start_rank") or 0),
                    "end_rank": int(batch.get("end_rank") or 0),
                },
                raw_output=raw_output,
                parsed_payload=parsed,
                normalized_payload=normalized,
            )
            return normalized
        except Exception as exc:
            logger.warning(
                "RAG evidence selector failed message_id=%s batch=%s error=%s",
                message_id,
                int(batch.get("batch_number") or 0),
                exc,
            )
            await self._append_selector_trace(
                message_id=message_id,
                stage=f"selector_batch_{int(batch.get('batch_number') or 0)}_error",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                metadata={
                    "batch_number": int(batch.get("batch_number") or 0),
                    "row_count": len(batch_rows),
                    "start_rank": int(batch.get("start_rank") or 0),
                    "end_rank": int(batch.get("end_rank") or 0),
                },
                error=str(exc),
            )
            return {
                "evidence_source_ids": [],
                "insufficient_in_batch": True,
            }

    async def _maybe_apply_evidence_selector(
        self,
        *,
        message_id: str,
        query: str,
        primary_query: str,
        retrieval_focus: str,
        ranked_chunks: List[Dict[str, Any]],
        prompt_chunks: List[Dict[str, Any]],
        prompt_budget: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        if not bool(getattr(settings, "rag_evidence_selector_enabled", False)):
            return list(prompt_chunks or []), dict(prompt_budget or {}), {}

        batch_payload = self._build_evidence_selector_batches(ranked_chunks)
        batches = list(batch_payload.get("batches") or [])
        id_to_chunk = dict(batch_payload.get("id_to_chunk") or {})
        if not batches or not id_to_chunk:
            return list(prompt_chunks or []), dict(prompt_budget or {}), {}

        await self._set_ui_state(
            message_id=message_id,
            ui="planning",
            ui_detail="selecting strongest evidence from reranked chunks",
        )
        selector_results = await asyncio.gather(
            *[
                self._run_evidence_selector_batch(
                    message_id=message_id,
                    query=query,
                    primary_query=primary_query,
                    retrieval_focus=retrieval_focus,
                    batch=batch,
                )
                for batch in batches
            ]
        )

        selected_source_ids_set: set[int] = set()
        for result in selector_results:
            for source_id in list(result.get("evidence_source_ids") or []):
                source_value = self._safe_int(source_id)
                if source_value is None:
                    continue
                if source_value not in id_to_chunk:
                    continue
                selected_source_ids_set.add(int(source_value))

        selected_source_ids = [
            source_id for source_id in range(1, len(id_to_chunk) + 1) if source_id in selected_source_ids_set
        ]
        batch_summaries = []
        for batch, result in zip(batches, selector_results):
            batch_summaries.append(
                {
                    "batch_number": int(batch.get("batch_number") or 0),
                    "start_rank": int(batch.get("start_rank") or 0),
                    "end_rank": int(batch.get("end_rank") or 0),
                    "row_count": len(list(batch.get("rows") or [])),
                    "selected_source_ids": list(result.get("evidence_source_ids") or []),
                    "insufficient_in_batch": bool(result.get("insufficient_in_batch")),
                }
            )

        if not selected_source_ids:
            fallback_prompt_chunks, fallback_prompt_budget = self._select_prompt_chunks_by_token_budget(
                ranked_chunks,
                token_budget_override=max(
                    1,
                    int(getattr(settings, "rag_evidence_selector_fallback_prompt_token_budget", 20000)),
                ),
            )
            selector_meta = {
                "enabled": True,
                "applied": False,
                "candidate_count": int(batch_payload.get("candidate_count") or 0),
                "batch_count": len(batches),
                "selected_source_ids": [],
                "selected_chunk_count": 0,
                "prompt_chunk_count": len(fallback_prompt_chunks),
                "fallback_prompt_tokens_est": int(fallback_prompt_budget.get("estimated_prompt_tokens") or 0),
                "fallback_to_prompt_budget_selection": True,
                "fallback_reason": "selector_returned_no_valid_evidence_ids",
                "batches": batch_summaries,
            }
            logger.info(
                "RAG evidence selector message_id=%s candidate_count=%s batch_count=%s selected=%s fallback=%s fallback_prompt_chunks=%s fallback_prompt_tokens_est=%s",
                message_id,
                selector_meta["candidate_count"],
                selector_meta["batch_count"],
                selector_meta["selected_chunk_count"],
                True,
                selector_meta["prompt_chunk_count"],
                selector_meta["fallback_prompt_tokens_est"],
            )
            return fallback_prompt_chunks, fallback_prompt_budget, selector_meta

        selected_chunks = [id_to_chunk[source_id] for source_id in selected_source_ids if source_id in id_to_chunk]
        selected_prompt_chunks, selected_prompt_budget = self._select_prompt_chunks_by_token_budget(selected_chunks)
        if not selected_prompt_chunks:
            fallback_prompt_chunks, fallback_prompt_budget = self._select_prompt_chunks_by_token_budget(
                ranked_chunks,
                token_budget_override=max(
                    1,
                    int(getattr(settings, "rag_evidence_selector_fallback_prompt_token_budget", 20000)),
                ),
            )
            selector_meta = {
                "enabled": True,
                "applied": False,
                "candidate_count": int(batch_payload.get("candidate_count") or 0),
                "batch_count": len(batches),
                "selected_source_ids": selected_source_ids,
                "selected_chunk_count": len(selected_chunks),
                "prompt_chunk_count": len(fallback_prompt_chunks),
                "fallback_prompt_tokens_est": int(fallback_prompt_budget.get("estimated_prompt_tokens") or 0),
                "fallback_to_prompt_budget_selection": True,
                "fallback_reason": "selector_selected_chunks_could_not_fit_prompt_budget",
                "batches": batch_summaries,
            }
            logger.info(
                "RAG evidence selector message_id=%s candidate_count=%s batch_count=%s selected=%s fallback=%s fallback_prompt_chunks=%s fallback_prompt_tokens_est=%s",
                message_id,
                selector_meta["candidate_count"],
                selector_meta["batch_count"],
                selector_meta["selected_chunk_count"],
                True,
                selector_meta["prompt_chunk_count"],
                selector_meta["fallback_prompt_tokens_est"],
            )
            return fallback_prompt_chunks, fallback_prompt_budget, selector_meta

        selector_meta = {
            "enabled": True,
            "applied": True,
            "candidate_count": int(batch_payload.get("candidate_count") or 0),
            "batch_count": len(batches),
            "selected_source_ids": selected_source_ids,
            "selected_chunk_count": len(selected_chunks),
            "prompt_chunk_count": len(selected_prompt_chunks),
            "prompt_tokens_est": int(selected_prompt_budget.get("estimated_prompt_tokens") or 0),
            "fallback_to_prompt_budget_selection": False,
            "batches": batch_summaries,
        }
        logger.info(
            "RAG evidence selector message_id=%s candidate_count=%s batch_count=%s selected=%s prompt_chunks=%s prompt_tokens_est=%s",
            message_id,
            selector_meta["candidate_count"],
            selector_meta["batch_count"],
            selector_meta["selected_chunk_count"],
            selector_meta["prompt_chunk_count"],
            selector_meta["prompt_tokens_est"],
        )
        return selected_prompt_chunks, selected_prompt_budget, selector_meta

    def _build_coverage_chunk_rows(
        self,
        chunks: List[Dict[str, Any]],
        *,
        max_chunks: int,
        max_chars_per_chunk: int,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        chunk_limit = int(max_chunks)
        source_rows = list(chunks or []) if chunk_limit <= 0 else list(chunks or [])[: max(1, chunk_limit)]
        char_limit = int(max_chars_per_chunk)
        for idx, chunk in enumerate(source_rows, start=1):
            has_exact, has_semantic = self._chunk_origin_flags(chunk)
            if has_exact and has_semantic:
                origin = "semantic+exact"
            elif has_exact:
                origin = "exact"
            elif has_semantic:
                origin = "semantic"
            else:
                origin = str(chunk.get("_scope") or "other")
            content = str(chunk.get("_prompt_content") or self._chunk_raw_content(chunk)).strip()
            if char_limit > 0:
                content = self._truncate_text(content, max(200, char_limit))
            rows.append(
                {
                    "source": idx,
                    "origin": origin,
                    "debug_id": self._chunk_debug_id(chunk),
                    "doc_id": str(chunk.get("doc_id") or "").strip(),
                    "parent_id": str(chunk.get("parent_id") or "").strip(),
                    "chunk_no": chunk.get("chunk_no"),
                    "report_type": chunk.get("report_type"),
                    "branch": chunk.get("branch"),
                    "score": self._chunk_score(chunk),
                    "sim_score": float(chunk.get("_sim_score", 0.0) or 0.0),
                    "exact_match_score": float(chunk.get("_exact_match_score", 0.0) or 0.0),
                    "recovery_source": str(chunk.get("_recovery_source") or "").strip(),
                    "content": content,
                }
            )
        return rows

    def _normalize_coverage_decision(
        self,
        raw_payload: Dict[str, Any],
        *,
        primary_query: str,
        exact_terms: List[str],
    ) -> Dict[str, Any]:
        payload = dict(raw_payload or {})
        answerable = str(payload.get("answerable") or "").strip().lower()
        if answerable not in {"yes", "partial", "no"}:
            answerable = "yes"

        ambiguity_detected = bool(self._coerce_bool(payload.get("ambiguity_detected")))
        should_ask_clarification = bool(self._coerce_bool(payload.get("should_ask_clarification")))
        need_exact_retry = bool(self._coerce_bool(payload.get("need_exact_retry")))
        need_semantic_rescue = bool(self._coerce_bool(payload.get("need_semantic_rescue")))
        need_next_reranked_window = bool(self._coerce_bool(payload.get("need_next_reranked_window")))

        missing_aspects = self._normalize_text_list(payload.get("missing_aspects"), max_items=8, max_chars=180)
        suggested_queries = self.support.dedupe_queries(
            self._normalize_text_list(
                [payload.get("suggested_queries"), payload.get("recovery_queries")],
                max_items=max(1, int(getattr(settings, "rag_recovery_max_suggested_queries", 3))),
                max_chars=220,
            ),
            max_items=max(1, int(getattr(settings, "rag_recovery_max_suggested_queries", 3))),
        )
        suggested_exact_terms = self.support.dedupe_exact_terms(
            [
                *self._normalize_text_list(
                    [payload.get("suggested_exact_terms"), payload.get("exact_terms")],
                    max_items=max(1, int(getattr(settings, "rag_recovery_max_suggested_exact_terms", 4))),
                    max_chars=120,
                ),
                *list(exact_terms or []),
            ],
            max_items=max(1, int(getattr(settings, "rag_recovery_max_suggested_exact_terms", 4))),
        )
        clarification_question = self._truncate_text(
            str(payload.get("clarification_question") or "").strip(),
            220,
        )
        assessment = self._truncate_text(
            str(payload.get("assessment") or payload.get("reason") or "").strip(),
            320,
        )

        if answerable == "yes":
            should_ask_clarification = False
            need_exact_retry = False
            need_semantic_rescue = False
            need_next_reranked_window = False
        elif not (need_exact_retry or need_semantic_rescue or need_next_reranked_window) and not should_ask_clarification:
            need_semantic_rescue = True
            need_next_reranked_window = True

        if ambiguity_detected and answerable == "no" and clarification_question:
            should_ask_clarification = True

        if not suggested_queries and primary_query:
            suggested_queries = [str(primary_query).strip()]

        return {
            "judge_used": True,
            "answerable": answerable,
            "ambiguity_detected": ambiguity_detected,
            "should_ask_clarification": should_ask_clarification,
            "clarification_question": clarification_question,
            "missing_aspects": missing_aspects,
            "need_exact_retry": need_exact_retry,
            "need_semantic_rescue": need_semantic_rescue,
            "need_next_reranked_window": need_next_reranked_window,
            "suggested_queries": suggested_queries,
            "suggested_exact_terms": suggested_exact_terms,
            "assessment": assessment,
        }

    async def _judge_prompt_coverage(
        self,
        *,
        message_id: str,
        query: str,
        primary_query: str,
        retrieval_focus: str,
        exact_terms: List[str],
        prompt_chunks: List[Dict[str, Any]],
        post_rerank_chunks: List[Dict[str, Any]],
        pre_rerank_chunks: List[Dict[str, Any]],
        stage: str,
    ) -> Dict[str, Any]:
        if not bool(getattr(settings, "rag_coverage_judge_enabled", False)):
            return {"judge_used": False, "answerable": "yes"}
        rows = list(prompt_chunks or [])
        if not rows:
            return {"judge_used": False, "answerable": "yes"}

        max_chunks = int(getattr(settings, "rag_coverage_judge_max_chunks", 0))
        max_chars_per_chunk = int(getattr(settings, "rag_coverage_judge_max_chars_per_chunk", 0))
        coverage_rows = self._build_coverage_chunk_rows(
            rows,
            max_chunks=max_chunks,
            max_chars_per_chunk=max_chars_per_chunk,
        )
        selected_keys = {self.support.chunk_dedupe_key(chunk) for chunk in rows}
        additional_reranked = sum(
            1 for chunk in list(post_rerank_chunks or []) if self.support.chunk_dedupe_key(chunk) not in selected_keys
        )
        additional_semantic = sum(
            1
            for chunk in list(pre_rerank_chunks or [])
            if self._chunk_origin_flags(chunk)[1] and self.support.chunk_dedupe_key(chunk) not in selected_keys
        )
        system_prompt = (
            "You are a retrieval coverage judge for a document-grounded RAG system.\n"
            "You are not answering the user. You only decide whether the currently selected chunks are sufficient.\n"
            "Be conservative. If a required fact, disambiguator, date, organization, or direct evidence is missing, do not say the answer is fully answerable.\n"
            "Do not explain your reasoning, do not discuss chunks one by one, and do not write prose before the JSON.\n"
            "Return exactly one JSON object and no other text.\n"
            "Use this schema:\n"
            "{"
            "\"answerable\":\"yes|partial|no\","
            "\"ambiguity_detected\":false,"
            "\"should_ask_clarification\":false,"
            "\"clarification_question\":\"\","
            "\"missing_aspects\":[],"
            "\"need_exact_retry\":false,"
            "\"need_semantic_rescue\":false,"
            "\"need_next_reranked_window\":false,"
            "\"suggested_queries\":[],"
            "\"suggested_exact_terms\":[],"
            "\"assessment\":\"\""
            "}\n"
            "Guidance:\n"
            "- `need_exact_retry`: use for names, orgs, dates, phone numbers, identifiers, quoted text, or direct factual matches.\n"
            "- `need_semantic_rescue`: use when a semantically relevant chunk may have been buried or missed.\n"
            "- `need_next_reranked_window`: use when the current selected prompt likely clipped evidence that may already exist later in the current ranked set.\n"
            "- Prefer recovery flags and suggested queries/exact terms over clarification when another retrieval pass could resolve the gap.\n"
            "- `should_ask_clarification`: use only when the query is impossible, irrelevant to the evidence, or highly ambiguous in a way retrieval cannot resolve.\n"
            "- Do not assume the answer model should improvise missing evidence."
        )
        user_prompt = "\n".join(
            [
                "## User Query",
                str(query or "").strip(),
                "",
                "## Resolved Retrieval Query",
                str(primary_query or query or "").strip(),
                "",
                "## Retrieval Focus",
                str(retrieval_focus or "").strip(),
                "",
                "## Existing Exact Terms",
                json.dumps(list(exact_terms or []), ensure_ascii=False),
                "",
                "## Retrieval Capacity Metadata",
                json.dumps(
                    {
                        "selected_prompt_chunks": len(rows),
                        "additional_reranked_candidates_available": additional_reranked,
                        "additional_semantic_candidates_available": additional_semantic,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "",
                "## Selected Prompt Chunks",
                json.dumps(coverage_rows, ensure_ascii=False, indent=2),
                "",
                "## Task",
                "Judge whether these selected chunks are sufficient for a grounded answer right now.",
            ]
        ).strip()
        await self._append_judge_trace(
            message_id=message_id,
            stage=f"{stage}_prompt",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            metadata={
                "prompt_chunks": len(rows),
                "coverage_rows": len(coverage_rows),
                "additional_reranked_candidates_available": additional_reranked,
                "additional_semantic_candidates_available": additional_semantic,
            },
        )
        try:
            raw_output = await self._call_llm_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=float(getattr(settings, "rag_coverage_judge_temperature", 0.0)),
                max_tokens=int(getattr(settings, "rag_coverage_judge_max_tokens", 1024)),
                model_name=str(getattr(settings, "rag_coverage_judge_model_name", "") or "").strip(),
                json_mode=True,
            )
            parsed = self._extract_json_object(raw_output)
            normalized = self._normalize_coverage_decision(
                parsed,
                primary_query=primary_query,
                exact_terms=exact_terms,
            )
            await self._append_judge_trace(
                message_id=message_id,
                stage=f"{stage}_result",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                metadata={
                    "prompt_chunks": len(rows),
                    "coverage_rows": len(coverage_rows),
                    "additional_reranked_candidates_available": additional_reranked,
                    "additional_semantic_candidates_available": additional_semantic,
                },
                raw_output=raw_output,
                parsed_payload=parsed,
                normalized_payload=normalized,
            )
            return normalized
        except Exception as exc:
            logger.warning(
                "RAG coverage judge failed message_id=%s stage=%s error=%s",
                message_id,
                stage,
                exc,
            )
            await self._append_judge_trace(
                message_id=message_id,
                stage=f"{stage}_error",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                metadata={
                    "prompt_chunks": len(rows),
                    "coverage_rows": len(coverage_rows),
                    "additional_reranked_candidates_available": additional_reranked,
                    "additional_semantic_candidates_available": additional_semantic,
                },
                error=str(exc),
            )
            return {
                "judge_used": False,
                "answerable": "yes",
                "ambiguity_detected": False,
                "should_ask_clarification": False,
                "clarification_question": "",
                "missing_aspects": [],
                "need_exact_retry": False,
                "need_semantic_rescue": False,
                "need_next_reranked_window": False,
                "suggested_queries": [str(primary_query or "").strip()] if str(primary_query or "").strip() else [],
                "suggested_exact_terms": list(exact_terms or []),
                "assessment": "",
            }

    @staticmethod
    def _coverage_requires_recovery(coverage: Dict[str, Any]) -> bool:
        if not bool((coverage or {}).get("judge_used")):
            return False
        if str((coverage or {}).get("answerable") or "yes").strip().lower() in {"partial", "no"}:
            return True
        return any(
            bool((coverage or {}).get(key))
            for key in ("need_exact_retry", "need_semantic_rescue", "need_next_reranked_window")
        )

    def _select_semantic_rescue_chunks(
        self,
        chunks: List[Dict[str, Any]],
        *,
        exclude_keys: set[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        pure_semantic: List[Dict[str, Any]] = []
        mixed_semantic: List[Dict[str, Any]] = []
        for chunk in list(chunks or []):
            key = self.support.chunk_dedupe_key(chunk)
            if key in exclude_keys:
                continue
            has_exact, has_semantic = self._chunk_origin_flags(chunk)
            if not has_semantic:
                continue
            payload = dict(chunk)
            payload["_recovery_source"] = "semantic_rescue"
            if has_exact:
                mixed_semantic.append(payload)
            else:
                pure_semantic.append(payload)
        out = pure_semantic[: max(0, int(limit))]
        if len(out) < max(0, int(limit)):
            out.extend(mixed_semantic[: max(0, int(limit)) - len(out)])
        return out

    async def _fetch_relaxed_semantic_recovery_chunks(
        self,
        *,
        data: Dict[str, Any],
        primary_query: str,
        secondary_queries: List[str],
        retrieval_focus: str,
        filters: Dict[str, Any],
        exclude_keys: set[str],
        suggested_queries: List[str],
    ) -> List[Dict[str, Any]]:
        if not bool(getattr(settings, "rag_recovery_relaxed_semantic_enabled", True)):
            return []

        relaxed_top_n_per_query = max(
            1,
            int(getattr(settings, "rag_recovery_relaxed_semantic_top_n_per_query", 20)),
        )
        relaxed_total_limit = max(
            1,
            int(getattr(settings, "rag_recovery_relaxed_semantic_total_chunks", 30)),
        )
        relaxed_queries = self.support.dedupe_queries(
            [
                *list(suggested_queries or []),
                primary_query,
            ],
            max_items=max(1, int(getattr(settings, "rag_recovery_max_suggested_queries", 3))),
        )
        if not relaxed_queries:
            return []

        relaxed_filter_variants = self._expand_bigdata_filter_variants(self._relax_bigdata_filters(filters))
        relaxed_min_score = max(
            float(getattr(settings, "rag_relaxed_min_score", 0.18)),
            float(getattr(settings, "rag_bigdata_min_score_floor", 0.0)),
        )
        relaxed_pairs = await self._run_bigdata_batch(
            data,
            retrieval_queries=relaxed_queries,
            filter_variants=relaxed_filter_variants,
            top_n=relaxed_top_n_per_query,
            min_score=relaxed_min_score,
        )
        relaxed_rows = self.support.merge_chunks_by_identity(relaxed_pairs)
        if not relaxed_rows:
            return []
        ranked_relaxed = self.support.rank_chunks_for_query(
            relaxed_rows,
            primary_query,
            secondary_queries=secondary_queries,
            topic_hint=retrieval_focus,
        )
        out: List[Dict[str, Any]] = []
        seen = set(exclude_keys)
        for chunk in ranked_relaxed:
            key = self.support.chunk_dedupe_key(chunk)
            if key in seen:
                continue
            seen.add(key)
            payload = dict(chunk)
            payload["_recovery_source"] = "relaxed_semantic_rescue"
            out.append(payload)
            if len(out) >= relaxed_total_limit:
                break
        return out

    def _select_next_reranked_window_chunks(
        self,
        chunks: List[Dict[str, Any]],
        *,
        exclude_keys: set[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for chunk in list(chunks or []):
            key = self.support.chunk_dedupe_key(chunk)
            if key in exclude_keys:
                continue
            payload = dict(chunk)
            payload["_recovery_source"] = "next_reranked_window"
            out.append(payload)
            if len(out) >= max(0, int(limit)):
                break
        return out

    async def _run_targeted_recovery(
        self,
        *,
        data: Dict[str, Any],
        message_id: str,
        primary_query: str,
        secondary_queries: List[str],
        retrieval_focus: str,
        exact_terms: List[str],
        filters: Dict[str, Any],
        prompt_chunks: List[Dict[str, Any]],
        pre_rerank_chunks: List[Dict[str, Any]],
        post_rerank_chunks: List[Dict[str, Any]],
        coverage: Dict[str, Any],
    ) -> Dict[str, Any]:
        max_attempts = max(0, int(getattr(settings, "rag_recovery_max_attempts", 1)))
        if (
            not bool(getattr(settings, "rag_recovery_enabled", False))
            or max_attempts <= 0
            or not self._coverage_requires_recovery(coverage)
        ):
            return {
                "applied": False,
                "attempts": 0,
                "ranked_chunks": list(post_rerank_chunks or []),
                "prompt_chunks": list(prompt_chunks or []),
                "prompt_budget": self._select_prompt_chunks_by_token_budget(list(post_rerank_chunks or []))[1]
                if not prompt_chunks
                else {},
                "sources": {},
            }

        selected_keys = {self.support.chunk_dedupe_key(chunk) for chunk in list(prompt_chunks or [])}
        generic_recovery = str((coverage or {}).get("answerable") or "").strip().lower() in {"partial", "no"} and not any(
            bool((coverage or {}).get(key))
            for key in ("need_exact_retry", "need_semantic_rescue", "need_next_reranked_window")
        )
        exact_retry_rows: List[Dict[str, Any]] = []
        semantic_rescue_rows: List[Dict[str, Any]] = []
        relaxed_semantic_rows: List[Dict[str, Any]] = []
        next_window_rows: List[Dict[str, Any]] = []
        recovery_sources: Dict[str, Any] = {
            "attempted_exact_retry": False,
            "attempted_semantic_rescue": False,
            "attempted_relaxed_semantic_rescue": False,
            "attempted_next_reranked_window": False,
            "exact_retry_query": "",
            "exact_retry_terms": [],
            "exact_retry_added": 0,
            "semantic_rescue_added": 0,
            "relaxed_semantic_added": 0,
            "next_reranked_added": 0,
        }

        if bool((coverage or {}).get("need_exact_retry")):
            retry_queries = self.support.dedupe_queries(
                [
                    *list((coverage or {}).get("suggested_queries") or []),
                    primary_query,
                ],
                max_items=max(1, int(getattr(settings, "rag_recovery_max_suggested_queries", 3))),
            )
            retry_exact_terms = self.support.dedupe_exact_terms(
                [
                    *list((coverage or {}).get("suggested_exact_terms") or []),
                    *list((coverage or {}).get("missing_aspects") or []),
                    *list(exact_terms or []),
                ],
                max_items=max(1, int(getattr(settings, "rag_recovery_max_suggested_exact_terms", 4))),
            )
            retry_query = retry_queries[0] if retry_queries else str(primary_query or "").strip()
            if retry_query and retry_exact_terms and bool(getattr(settings, "rag_exact_match_enabled", True)):
                recovery_sources["attempted_exact_retry"] = True
                recovery_sources["exact_retry_query"] = retry_query
                recovery_sources["exact_retry_terms"] = list(retry_exact_terms)
                exact_retry_rows, exact_retry_meta = await self._fetch_bigdata_exact_context_multi(
                    data,
                    query=retry_query,
                    keywords=retry_exact_terms,
                    filters=filters,
                    top_n=max(1, int(getattr(settings, "rag_recovery_exact_match_top_n", 96))),
                )
                recovery_sources["exact_retry_error_count"] = int(exact_retry_meta.get("error_count") or 0)
                recovery_sources["exact_retry_all_failed"] = bool(exact_retry_meta.get("all_failed"))
                if exact_retry_meta.get("errors"):
                    recovery_sources["exact_retry_errors"] = list(exact_retry_meta.get("errors") or [])
                exact_retry_rows = self.support.rank_chunks_for_query(
                    exact_retry_rows,
                    primary_query,
                    secondary_queries=secondary_queries,
                    topic_hint=retrieval_focus,
                )
                for row in exact_retry_rows:
                    row["_recovery_source"] = "exact_retry"
                recovery_sources["exact_retry_added"] = len(
                    [row for row in exact_retry_rows if self.support.chunk_dedupe_key(row) not in selected_keys]
                )

        if bool((coverage or {}).get("need_semantic_rescue")) or generic_recovery:
            recovery_sources["attempted_semantic_rescue"] = True
            semantic_rescue_rows = self._select_semantic_rescue_chunks(
                pre_rerank_chunks,
                exclude_keys=selected_keys,
                limit=max(1, int(getattr(settings, "rag_recovery_semantic_rescue_chunks", 8))),
            )
            recovery_sources["semantic_rescue_added"] = len(semantic_rescue_rows)
            relaxed_recovery_exclude_keys = {
                *selected_keys,
                *{self.support.chunk_dedupe_key(chunk) for chunk in semantic_rescue_rows},
                *{self.support.chunk_dedupe_key(chunk) for chunk in exact_retry_rows},
            }
            recovery_sources["attempted_relaxed_semantic_rescue"] = bool(
                getattr(settings, "rag_recovery_relaxed_semantic_enabled", True)
            )
            relaxed_semantic_rows = await self._fetch_relaxed_semantic_recovery_chunks(
                data=data,
                primary_query=primary_query,
                secondary_queries=secondary_queries,
                retrieval_focus=retrieval_focus,
                filters=filters,
                exclude_keys=relaxed_recovery_exclude_keys,
                suggested_queries=list((coverage or {}).get("suggested_queries") or []),
            )
            recovery_sources["relaxed_semantic_added"] = len(relaxed_semantic_rows)

        recovery_exclude_keys = {
            *selected_keys,
            *{self.support.chunk_dedupe_key(chunk) for chunk in semantic_rescue_rows},
            *{self.support.chunk_dedupe_key(chunk) for chunk in relaxed_semantic_rows},
            *{self.support.chunk_dedupe_key(chunk) for chunk in exact_retry_rows},
        }
        if bool((coverage or {}).get("need_next_reranked_window")) or generic_recovery:
            recovery_sources["attempted_next_reranked_window"] = True
            next_window_rows = self._select_next_reranked_window_chunks(
                post_rerank_chunks,
                exclude_keys=recovery_exclude_keys,
                limit=max(1, int(getattr(settings, "rag_recovery_next_reranked_window_chunks", 8))),
            )
            recovery_sources["next_reranked_added"] = len(next_window_rows)

        if not exact_retry_rows and not semantic_rescue_rows and not relaxed_semantic_rows and not next_window_rows:
            return {
                "applied": False,
                "attempts": 0,
                "ranked_chunks": list(post_rerank_chunks or []),
                "prompt_chunks": list(prompt_chunks or []),
                "prompt_budget": {},
                "sources": recovery_sources,
            }

        base_rows = list(post_rerank_chunks or [])
        base_head_count = min(len(base_rows), max(6, min(12, len(list(prompt_chunks or [])) + 2)))
        ordered_candidates = self.support.merge_chunks_by_identity(
            [
                ("base_head", base_rows[:base_head_count]),
                ("exact_retry", exact_retry_rows),
                ("semantic_rescue", semantic_rescue_rows),
                ("relaxed_semantic_rescue", relaxed_semantic_rows),
                ("next_reranked_window", next_window_rows),
                ("base_tail", base_rows[base_head_count:]),
            ]
        )
        final_chunks, prompt_budget = self._select_prompt_chunks_by_token_budget(ordered_candidates)
        return {
            "applied": True,
            "attempts": 1,
            "ranked_chunks": ordered_candidates,
            "prompt_chunks": final_chunks,
            "prompt_budget": prompt_budget,
            "sources": recovery_sources,
        }

    def _build_coverage_guidance(self, coverage: Dict[str, Any]) -> str:
        payload = dict(coverage or {})
        if not bool(payload.get("judge_used")):
            return ""
        answerable = str(payload.get("answerable") or "yes").strip().lower()
        lines: List[str] = []
        if answerable == "yes":
            lines.append("Backend coverage check judged the selected context sufficient for a grounded answer.")
        elif answerable == "partial":
            lines.append("Backend coverage check judged the selected context only partially sufficient.")
        else:
            lines.append("Backend coverage check judged the selected context insufficient for a fully grounded answer.")
        missing_aspects = list(payload.get("missing_aspects") or [])
        if missing_aspects:
            lines.append("Potentially missing aspects: " + "; ".join(str(item) for item in missing_aspects[:6]) + ".")
        if bool(payload.get("ambiguity_detected")):
            lines.append("Possible entity/topic ambiguity remains in the retrieved evidence.")
        if bool(payload.get("should_ask_clarification")) and str(payload.get("clarification_question") or "").strip():
            lines.append(
                "If the ambiguity cannot be resolved from the provided chunks, ask this clarification question: "
                + str(payload.get("clarification_question")).strip()
            )
        if str(payload.get("assessment") or "").strip():
            lines.append(str(payload.get("assessment")).strip())
        return " ".join(lines).strip()

    @staticmethod
    def _should_run_coverage_judge_for_selector(selector_meta: Dict[str, Any]) -> bool:
        if not selector_meta:
            return True
        if not bool(selector_meta.get("enabled")):
            return True

        selected_source_ids = list(selector_meta.get("selected_source_ids") or [])
        selected_count = len(selected_source_ids) or int(selector_meta.get("selected_chunk_count") or 0)
        batches = list(selector_meta.get("batches") or [])
        if not bool(selector_meta.get("applied")) or bool(selector_meta.get("fallback_to_prompt_budget_selection")):
            return True
        if selected_count < 2:
            return True
        if len(batches) >= 2 and all(bool(batch.get("insufficient_in_batch")) for batch in batches):
            return True

        return False

    @staticmethod
    def _coverage_judge_should_use_baseline_prompt(selector_meta: Dict[str, Any]) -> bool:
        if not selector_meta or not bool(selector_meta.get("enabled")):
            return False
        selected_source_ids = list(selector_meta.get("selected_source_ids") or [])
        selected_count = len(selected_source_ids) or int(selector_meta.get("selected_chunk_count") or 0)
        return bool(selector_meta.get("applied")) and selected_count < 2

    @staticmethod
    def _coverage_judge_skip_meta(selector_meta: Dict[str, Any]) -> Dict[str, Any]:
        selected_source_ids = list((selector_meta or {}).get("selected_source_ids") or [])
        batches = list((selector_meta or {}).get("batches") or [])
        return {
            "skipped": True,
            "reason": "selector_gate_conditions_not_met",
            "policy": (
                "coverage judge runs after selector only when selector did not apply, returned no usable IDs, "
                "selected fewer than two evidence chunks, or at least two selector batches all report insufficient_in_batch=true"
            ),
            "selector_applied": bool((selector_meta or {}).get("applied")),
            "selector_fallback_to_prompt_budget_selection": bool(
                (selector_meta or {}).get("fallback_to_prompt_budget_selection")
            ),
            "selector_selected_source_ids": selected_source_ids,
            "selector_selected_count": len(selected_source_ids) or int((selector_meta or {}).get("selected_chunk_count") or 0),
            "selector_batch_count": len(batches),
            "selector_batches_insufficient": [
                bool(batch.get("insufficient_in_batch")) for batch in batches
            ],
        }

    async def _maybe_apply_coverage_recovery(
        self,
        *,
        data: Dict[str, Any],
        message_id: str,
        query: str,
        primary_query: str,
        retrieval_focus: str,
        secondary_queries: List[str],
        exact_terms: List[str],
        filters: Dict[str, Any],
        prompt_chunks: List[Dict[str, Any]],
        prompt_budget: Dict[str, Any],
        pre_rerank_chunks: List[Dict[str, Any]],
        post_rerank_chunks: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
        if not bool(getattr(settings, "rag_coverage_judge_enabled", False)) or not list(prompt_chunks or []):
            return list(post_rerank_chunks or []), dict(prompt_budget or {}), {}, list(prompt_chunks or [])

        await self._set_ui_state(
            message_id=message_id,
            ui="planning",
            ui_detail="checking answer coverage",
        )
        initial_coverage = await self._judge_prompt_coverage(
            message_id=message_id,
            query=query,
            primary_query=primary_query,
            retrieval_focus=retrieval_focus,
            exact_terms=exact_terms,
            prompt_chunks=prompt_chunks,
            post_rerank_chunks=post_rerank_chunks,
            pre_rerank_chunks=pre_rerank_chunks,
            stage="coverage_judge_initial",
        )
        final_coverage = dict(initial_coverage)
        ranked_chunks = list(post_rerank_chunks or [])
        final_prompt_chunks = list(prompt_chunks or [])
        final_prompt_budget = dict(prompt_budget or {})
        recovery_info = {
            "applied": False,
            "attempts": 0,
            "sources": {},
        }

        if self._coverage_requires_recovery(initial_coverage):
            await self._set_ui_state(
                message_id=message_id,
                ui="planning",
                ui_detail="recovering missing evidence before answering",
            )
            recovery_result = await self._run_targeted_recovery(
                data=data,
                message_id=message_id,
                primary_query=primary_query,
                secondary_queries=secondary_queries,
                retrieval_focus=retrieval_focus,
                exact_terms=exact_terms,
                filters=filters,
                prompt_chunks=prompt_chunks,
                pre_rerank_chunks=pre_rerank_chunks,
                post_rerank_chunks=post_rerank_chunks,
                coverage=initial_coverage,
            )
            if bool(recovery_result.get("applied")) and list(recovery_result.get("prompt_chunks") or []):
                ranked_chunks = list(recovery_result.get("ranked_chunks") or [])
                final_prompt_chunks = list(recovery_result.get("prompt_chunks") or [])
                final_prompt_budget = dict(recovery_result.get("prompt_budget") or {})
                recovery_info = {
                    "applied": True,
                    "attempts": int(recovery_result.get("attempts") or 0),
                    "sources": dict(recovery_result.get("sources") or {}),
                }
                final_coverage = await self._judge_prompt_coverage(
                    message_id=message_id,
                    query=query,
                    primary_query=primary_query,
                    retrieval_focus=retrieval_focus,
                    exact_terms=exact_terms,
                    prompt_chunks=final_prompt_chunks,
                    post_rerank_chunks=ranked_chunks,
                    pre_rerank_chunks=pre_rerank_chunks,
                    stage="coverage_judge_post_recovery",
                )

        coverage_meta = {
            "initial": initial_coverage,
            "final": final_coverage,
            "recovery": recovery_info,
            "guidance": self._build_coverage_guidance(final_coverage),
        }
        logger.info(
            "RAG coverage evaluation message_id=%s initial=%s final=%s recovery_applied=%s recovery_sources=%s",
            message_id,
            {k: v for k, v in initial_coverage.items() if k not in {"missing_aspects", "suggested_queries", "suggested_exact_terms", "clarification_question", "assessment"}},
            {k: v for k, v in final_coverage.items() if k not in {"missing_aspects", "suggested_queries", "suggested_exact_terms", "clarification_question", "assessment"}},
            bool(recovery_info.get("applied")),
            dict(recovery_info.get("sources") or {}),
        )
        return ranked_chunks, final_prompt_budget, coverage_meta, final_prompt_chunks

    async def _apply_bigdata_reranker(
        self,
        *,
        message_id: str,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if self._reranker is None or not bool(getattr(settings, "rag_reranker_enabled", False)):
            return chunks

        candidate_count = len(chunks)
        min_candidates = max(1, int(getattr(settings, "rag_reranker_min_candidates", 8)))
        if candidate_count < min_candidates:
            logger.info(
                "RAG reranker skipped message_id=%s candidates=%s min_candidates=%s",
                message_id,
                candidate_count,
                min_candidates,
            )
            return chunks

        pool = self._build_rerank_pool(
            chunks,
            pool_size=max(1, int(getattr(settings, "rag_reranker_candidate_pool_size", 48))),
        )
        pool_counts = self._chunk_origin_counts(pool)
        doc_max_chars = int(getattr(settings, "rag_reranker_doc_max_chars", 1200))
        pool_entries: List[Tuple[int, Dict[str, Any], str]] = []
        for pool_index, chunk in enumerate(pool):
            chunk.setdefault("_heuristic_score", self._chunk_score(chunk))
            chunk.setdefault("_final_score", float(chunk.get("_heuristic_score", 0.0) or 0.0))
            text = self._chunk_text_for_rerank(chunk, max_chars=doc_max_chars)
            if not text:
                continue
            pool_entries.append((pool_index, chunk, text))

        if len(pool_entries) < min_candidates:
            logger.info(
                "RAG reranker skipped message_id=%s usable_candidates=%s min_candidates=%s pool_counts=%s",
                message_id,
                len(pool_entries),
                min_candidates,
                pool_counts,
            )
            return chunks

        pre_top = [self._chunk_debug_id(chunk) for chunk in chunks[:5]]
        rerank_query = self._query_text_for_rerank(query)
        rerank_instruction_enabled = bool(rerank_query and rerank_query != str(query or "").strip())
        pre_rerank_scores: Dict[int, Dict[str, Any]] = {}
        for local_index, (_pool_index, chunk, _text) in enumerate(pool_entries):
            has_exact, has_semantic = self._chunk_origin_flags(chunk)
            pre_rerank_scores[local_index] = {
                "debug_id": self._chunk_debug_id(chunk),
                "origin": (
                    "semantic+exact"
                    if has_exact and has_semantic
                    else "exact"
                    if has_exact
                    else "semantic"
                    if has_semantic
                    else "other"
                ),
                "pre_rank": local_index + 1,
                "pre_final_score": float(chunk.get("_final_score", 0.0) or 0.0),
                "pre_heuristic_score": float(chunk.get("_heuristic_score", 0.0) or 0.0),
                "pre_sim_score": float(chunk.get("_sim_score", 0.0) or 0.0),
                "pre_exact_score": float(chunk.get("_exact_match_score", 0.0) or 0.0),
            }

        await self._set_ui_state(
            message_id=message_id,
            ui="reranking",
            ui_detail=(
                f"sending {len(pool_entries)} candidates to reranker "
                f"(semantic={pool_counts['semantic']}, exact={pool_counts['exact']})"
            ),
        )

        try:
            trace_path = self._resolve_service_log_path("logs/reranker_trace.log")
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            semantic_entries = [
                (local_index, chunk, text)
                for local_index, (_pool_index, chunk, text) in enumerate(pool_entries)
                if self._chunk_origin_flags(chunk)[1]
            ]
            top_20_semantic = semantic_entries[:20]
            with open(trace_path, "a", encoding="utf-8") as f:
                f.write(f"\n[{datetime.now(timezone.utc).isoformat()}] --- RERANKER TRACE message_id={message_id} ---\n")
                f.write(f"RAW_QUERY: {query}\n")
                f.write(f"RERANK_QUERY_SENT: {rerank_query}\n")
                f.write(f"INSTRUCTION_ENABLED: {rerank_instruction_enabled}\n")
                f.write(f"QWEN3_TEMPLATE_ENABLED: {bool(getattr(settings, 'rag_reranker_qwen3_template_enabled', True))}\n")
                f.write(f"RERANK_QUERY_TOKENS_EST: {self._estimate_prompt_tokens(rerank_query)}\n")
                f.write(f"RERANK_DOCUMENT_COUNT: {len(pool_entries)}\n")
                f.write(f"RERANK_DOCUMENT_TOKENS_EST: {sum(self._estimate_prompt_tokens(text) for _, _, text in pool_entries)}\n")
                for i, (local_index, c, rerank_input) in enumerate(top_20_semantic, start=1):
                    before = pre_rerank_scores.get(local_index, {})
                    debug_id = self._chunk_debug_id(c)
                    rerank_body = self._truncate_text(
                        self._chunk_body_text_for_rerank(c),
                        max_chars=max(1, int(doc_max_chars)),
                    )
                    f.write(
                        f"\n--- PRE-RERANK SEMANTIC CANDIDATE {i} "
                        f"(overall_pre_rank={local_index + 1}, ID: {debug_id}) ---\n"
                    )
                    f.write(
                        f"origin={before.get('origin', 'semantic')} "
                        f"pre_final={float(before.get('pre_final_score', 0.0) or 0.0):.6f} "
                        f"pre_sim={float(before.get('pre_sim_score', 0.0) or 0.0):.6f} "
                        f"pre_exact={float(before.get('pre_exact_score', 0.0) or 0.0):.6f} "
                        f"quality_score={c.get('quality_score', 'N/A')} "
                        f"neighbor_chunks={c.get('_neighbor_chunk_count', 0)} "
                        f"section={c.get('section_heading', '')}\n"
                    )
                    f.write("\n[RERANK_BODY_BEFORE_TEMPLATE]\n")
                    f.write(f"{rerank_body}\n")
                    f.write("\n[RERANK_INPUT_SENT]\n")
                    f.write(f"{rerank_input}\n")
                f.write("-" * 80 + "\n")
        except Exception as e:
            logger.warning("Failed to write to reranker_trace_log: %s", e)

        started = time.perf_counter()
        try:
            rerank_rows = await self._reranker.rerank(
                query=rerank_query,
                documents=[text for _, _, text in pool_entries],
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            logger.warning(
                "RAG reranker failed message_id=%s candidates=%s pool_size=%s pool_counts=%s instruction_enabled=%s latency_ms=%.1f fallback=%s error=%s pre_top=%s post_top=%s",
                message_id,
                candidate_count,
                len(pool_entries),
                pool_counts,
                rerank_instruction_enabled,
                elapsed_ms,
                True,
                exc,
                pre_top,
                pre_top,
            )
            return chunks

        score_by_index: Dict[int, float] = {}
        for row in rerank_rows:
            try:
                index = int(row.get("index"))
                score = float(row.get("score"))
            except Exception:
                continue
            if 0 <= index < len(pool_entries):
                score_by_index[index] = score

        pool_keys = {self.support.chunk_dedupe_key(chunk) for _, chunk, _ in pool_entries}
        reranked_pool: List[Tuple[int, Dict[str, Any]]] = []
        for local_index, (_pool_index, chunk, _text) in enumerate(pool_entries):
            if local_index in score_by_index:
                chunk["_rerank_score"] = score_by_index[local_index]
                chunk["_final_score"] = score_by_index[local_index]
            else:
                chunk["_final_score"] = float(chunk.get("_heuristic_score", 0.0) or 0.0)
            reranked_pool.append((local_index, chunk))

        reranked_pool.sort(
            key=lambda item: (
                1 if "_rerank_score" in item[1] else 0,
                float(item[1].get("_rerank_score", -1e9)),
                float(item[1].get("_heuristic_score", 0.0) or 0.0),
                -item[0],
            ),
            reverse=True,
        )
        reranked_chunks = [chunk for _, chunk in reranked_pool]
        post_rank_by_index = {local_index: rank for rank, (local_index, _chunk) in enumerate(reranked_pool, start=1)}

        try:
            trace_path = self._resolve_service_log_path("logs/reranker_trace.log")
            if trace_path.exists():
                with open(trace_path, "a", encoding="utf-8") as f:
                    f.write(f"\n--- RERANK SCORE COMPARISON for message_id={message_id} docs_sent={len(pool_entries)} ---\n")
                    f.write(
                        "Format: pre_rank -> post_rank | origin | pre_final | pre_sim | pre_exact | pre_heuristic | rerank_score | delta_vs_pre_final | id\n"
                    )
                    for local_index, (_pool_index, chunk, _text) in enumerate(pool_entries):
                        before = pre_rerank_scores.get(local_index, {})
                        rerank_score = score_by_index.get(local_index)
                        pre_final = float(before.get("pre_final_score", 0.0) or 0.0)
                        if rerank_score is None:
                            rerank_text = "N/A"
                            delta_text = "N/A"
                        else:
                            rerank_text = f"{float(rerank_score):.6f}"
                            delta_text = f"{float(rerank_score) - pre_final:+.6f}"
                        f.write(
                            f"{int(before.get('pre_rank', local_index + 1)):04d} -> {post_rank_by_index.get(local_index, 0):04d} | "
                            f"{before.get('origin', 'other')} | "
                            f"{float(before.get('pre_final_score', 0.0) or 0.0):.6f} | "
                            f"{float(before.get('pre_sim_score', 0.0) or 0.0):.6f} | "
                            f"{float(before.get('pre_exact_score', 0.0) or 0.0):.6f} | "
                            f"{float(before.get('pre_heuristic_score', 0.0) or 0.0):.6f} | "
                            f"{rerank_text} | {delta_text} | {before.get('debug_id') or self._chunk_debug_id(chunk)}\n"
                        )
                    f.write(f"\n--- POST-RERANK SCORES for message_id={message_id} ---\n")
                    for r_index, chunk in enumerate(reranked_chunks[:20]):
                        debug_id = self._chunk_debug_id(chunk)
                        score = chunk.get("_rerank_score", chunk.get("_final_score", "N/A"))
                        try:
                            score_fmt = f"{float(score):.4f}"
                        except Exception:
                            score_fmt = str(score)
                        f.write(f"Rank {r_index+1} | Score: {score_fmt} | ID: {debug_id}\n")
                    f.write(f"\n--- POST-RERANK TOP 20 SEMANTIC SCORES for message_id={message_id} ---\n")
                    semantic_rank = 0
                    for overall_rank, chunk in enumerate(reranked_chunks, start=1):
                        if not self._chunk_origin_flags(chunk)[1]:
                            continue
                        semantic_rank += 1
                        debug_id = self._chunk_debug_id(chunk)
                        score = chunk.get("_rerank_score", chunk.get("_final_score", "N/A"))
                        try:
                            score_fmt = f"{float(score):.6f}"
                        except Exception:
                            score_fmt = str(score)
                        f.write(
                            f"Semantic Rank {semantic_rank} | Overall Rank {overall_rank} | Score: {score_fmt} | ID: {debug_id}\n"
                        )
                        if semantic_rank >= 20:
                            break
                    f.write("-" * 80 + "\n")
        except Exception as e:
            logger.warning("Failed to write to reranker_trace_log post-scores: %s", e)
        remainder = [chunk for chunk in chunks if self.support.chunk_dedupe_key(chunk) not in pool_keys]
        final_order = [*reranked_chunks, *remainder]

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        post_top = [self._chunk_debug_id(chunk) for chunk in final_order[:5]]
        await self._set_ui_state(
            message_id=message_id,
            ui="reranking",
            ui_detail=f"reranked {len(pool_entries)} candidates in {elapsed_ms:.1f} ms",
        )
        logger.info(
            "RAG reranker applied message_id=%s candidates=%s pool_size=%s pool_counts=%s semantic_target=%s exact_target=%s instruction_enabled=%s latency_ms=%.1f fallback=%s pre_top=%s post_top=%s",
            message_id,
            candidate_count,
            len(pool_entries),
            pool_counts,
            int(getattr(settings, "rag_reranker_semantic_pool_size", 0)),
            int(getattr(settings, "rag_reranker_exact_pool_size", 0)),
            rerank_instruction_enabled,
            elapsed_ms,
            False,
            pre_top,
            post_top,
        )
        return final_order

    async def _fetch_chat_context(
        self,
        data: Dict[str, Any],
        *,
        query: str,
        enable_semantic_search: bool,
        top_n: int,
    ) -> Dict[str, Any]:
        semantic_threshold = None
        if enable_semantic_search:
            semantic_threshold = max(0.0, float(getattr(settings, "rag_chat_context_semantic_threshold", 0.18)))
        try:
            return await self.cf.fetch_conv_context(
                user_id=str(data.get("user_id")),
                chat_id=str(data.get("chat_id")),
                message_id=str(data.get("message_id")),
                query=query,
                semantic_threshold=semantic_threshold,
                top_n=max(1, int(top_n)),
                enable_semantic_search=enable_semantic_search,
            )
        except Exception as exc:
            logger.warning(
                "RAG chat context fetch failed chat_id=%s semantic=%s error=%s",
                data.get("chat_id"),
                enable_semantic_search,
                exc,
            )
            return {}

    async def _fetch_attachment_context(
        self,
        data: Dict[str, Any],
        attachment_ids: List[str],
        *,
        retrieval_query: str,
    ) -> List[Dict[str, Any]]:
        query = str(retrieval_query or self._normalize_query(data)).strip()
        file_id_value = ",".join(attachment_ids) if attachment_ids else None
        retrieval_mode = "full_file" if attachment_ids or bool(data.get("has_attachments")) else "semantic"
        top_n = int(settings.rag_file_chunk_cap) if retrieval_mode == "full_file" else int(settings.rag_top_n_docs)
        try:
            response = await self.cf.fetch_file_context(
                user_id=str(data.get("user_id")),
                chat_id=str(data.get("chat_id")),
                query=query,
                top_n=max(1, top_n),
                min_score=0.0 if retrieval_mode == "full_file" else float(settings.rag_min_score),
                file_id=file_id_value,
                retrieval_mode=retrieval_mode,
            )
            return self._order_file_chunks(
                list((response or {}).get("file_context") or []),
                attachment_ids,
            )
        except Exception as exc:
            logger.warning(
                "RAG file-context fetch failed chat_id=%s retrieval_mode=%s error=%s",
                data.get("chat_id"),
                retrieval_mode,
                exc,
            )
            return []

    async def _fetch_bigdata_context_for_query(
        self,
        data: Dict[str, Any],
        *,
        query: str,
        filters: Dict[str, Any],
        top_n: int,
        min_score: float,
    ) -> List[Dict[str, Any]]:
        try:
            return await self.cf.fetch_bigdata_context(
                query=query,
                top_n=max(1, int(top_n)),
                min_score=float(min_score),
                **filters,
            )
        except Exception as exc:
            logger.warning(
                "RAG big-data fetch failed chat_id=%s query=%s top_n=%s min_score=%s error=%s",
                data.get("chat_id"),
                self._truncate_text(query, 120),
                top_n,
                min_score,
                exc,
            )
            return []

    async def _fetch_bigdata_exact_context(
        self,
        data: Dict[str, Any],
        *,
        query: str,
        keywords: List[str],
        filters: Dict[str, Any],
        top_n: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        exact_filters = {k: v for k, v in dict(filters or {}).items() if k != "report_types"}
        try:
            rows = await self.cf.fetch_bigdata_exact_context(
                query=query,
                keywords=keywords,
                top_n=max(1, int(top_n if top_n is not None else settings.rag_exact_match_top_n)),
                elasticsearch_base_url=self.support.clean_filter_text(data.get("elasticsearch_base_url")),
                elasticsearch_index=self.support.clean_filter_text(data.get("elasticsearch_index")),
                **exact_filters,
            )
            return list(rows or []), None
        except Exception as exc:
            error_meta = {
                "query": self._truncate_text(query, 180),
                "keywords": list(keywords or []),
                "filters": {k: v for k, v in exact_filters.items() if self.support.has_filter_value(v)},
                "error": self._truncate_text(str(exc), 500),
            }
            logger.warning(
                "RAG exact big-data fetch failed chat_id=%s message_id=%s keywords=%s error=%s",
                data.get("chat_id"),
                data.get("message_id"),
                keywords,
                exc,
            )
            return [], error_meta

    async def _fetch_bigdata_exact_context_multi(
        self,
        data: Dict[str, Any],
        *,
        query: str,
        keywords: List[str],
        filters: Dict[str, Any],
        top_n: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        variants = self._expand_bigdata_filter_variants(filters)
        tasks = [
            self._fetch_bigdata_exact_context(
                data,
                query=query,
                keywords=keywords,
                filters=variant,
                top_n=top_n,
            )
            for variant in variants
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        pairs: List[Tuple[str, List[Dict[str, Any]]]] = []
        errors: List[Dict[str, Any]] = []
        for idx, result in enumerate(results, start=1):
            if isinstance(result, Exception):
                errors.append({"variant": idx, "error": self._truncate_text(str(result), 500)})
                logger.warning(
                    "RAG exact big-data multi-variant failure chat_id=%s variant=%s error=%s",
                    data.get("chat_id"),
                    idx,
                    result,
                )
                continue
            rows, error_meta = result
            if error_meta:
                errors.append({"variant": idx, **dict(error_meta)})
            pairs.append((f"exact_variant_{idx}", list(rows or [])))
        merged = self.support.merge_chunks_by_identity(pairs)
        meta = {
            "attempted": True,
            "variants": len(variants),
            "keywords": list(keywords or []),
            "candidate_count": len(merged),
            "error_count": len(errors),
            "errors": errors,
            "all_failed": bool(errors and len(errors) >= len(variants) and not merged),
        }
        logger.info(
            "RAG exact big-data retrieval chat_id=%s variants=%s candidates=%s keywords=%s error_count=%s all_failed=%s",
            data.get("chat_id"),
            len(variants),
            len(merged),
            keywords,
            meta["error_count"],
            meta["all_failed"],
        )
        return merged, meta

    async def _build_retrieval_plan(self, data: Dict[str, Any]) -> Tuple[RetrievalPlan, Dict[str, Any]]:
        query = self._normalize_query(data)
        recent_top_n = max(8, int(getattr(settings, "rag_chat_context_top_n", 14)))
        recent_chat_context = await self._fetch_chat_context(
            data,
            query=query,
            enable_semantic_search=False,
            top_n=recent_top_n,
        )
        explicit_filters = {
            "report_types": self.support.normalize_report_type_values(data.get("report_types") or data.get("report_type")),
            "branch": self.support.clean_filter_text(data.get("branch")),
            "doc_id": self.support.clean_filter_text(data.get("doc_id")),
            "parent_id": self.support.clean_filter_text(data.get("parent_id")),
            "lang": self.support.clean_filter_text(data.get("lang")),
            "is_attachment": self.support.parse_bool(data.get("is_attachment")),
            "chunk_no": self.support.safe_int(data.get("chunk_no")),
            "document_date_gte": self.support.safe_int(data.get("document_date_gte")),
            "document_date_lte": self.support.safe_int(data.get("document_date_lte")),
            "ingestion_date_gte": self.support.safe_int(data.get("ingestion_date_gte")),
            "ingestion_date_lte": self.support.safe_int(data.get("ingestion_date_lte")),
        }
        allow_history_expansion = bool(getattr(settings, "rag_query_planner_allow_history_expansion", True))    # True right now: planner may trigger one semantic chat-history expansion pass when recent turns are insufficient
        plan = await self.planner.plan(
            query=query,
            chat_context=recent_chat_context,
            explicit_filters=explicit_filters,
            allow_history_expansion=allow_history_expansion,
            expanded_history_used=False,
            chat_id=str(data.get("chat_id") or ""),
            message_id=str(data.get("message_id") or ""),
        )
        if not allow_history_expansion or not plan.needs_history_expansion:
            return plan, recent_chat_context

        hybrid_top_n = max(recent_top_n, int(getattr(settings, "rag_query_planner_hybrid_chat_top_n", 18)))
        hybrid_chat_context = await self._fetch_chat_context(
            data,
            query=plan.standalone_query or query,
            enable_semantic_search=True,
            top_n=hybrid_top_n,
        )
        if self.support.chat_turn_count(hybrid_chat_context) <= self.support.chat_turn_count(recent_chat_context):
            return plan, recent_chat_context

        expanded_plan = await self.planner.plan(
            query=query,
            chat_context=hybrid_chat_context,
            explicit_filters=explicit_filters,
            allow_history_expansion=allow_history_expansion,
            expanded_history_used=True,
            chat_id=str(data.get("chat_id") or ""),
            message_id=str(data.get("message_id") or ""),
        )
        return expanded_plan, hybrid_chat_context

    def _build_bigdata_filters(self, data: Dict[str, Any], *, plan: RetrievalPlan) -> Dict[str, Any]:
        query = self._normalize_query(data)
        inferred = self.support.extract_bigdata_filters_from_query(query)
        planner_filters = dict(plan.filters or {})

        explicit_report_types = self.support.normalize_report_type_values(data.get("report_types"))
        if not explicit_report_types:
            explicit_report_types = self.support.normalize_report_type_values(data.get("report_type"))
        planner_report_types = self.support.normalize_report_type_values(planner_filters.get("report_types"))
        inferred_report_types = self.support.normalize_report_type_values(inferred.get("report_types"))
        report_types = explicit_report_types or planner_report_types or inferred_report_types
        report_type = report_types[0] if report_types else None

        branch = self.support.clean_filter_text(data.get("branch")) or self.support.clean_filter_text(planner_filters.get("branch")) or self.support.clean_filter_text(inferred.get("branch"))
        doc_id = self.support.clean_filter_text(data.get("doc_id")) or self.support.clean_filter_text(planner_filters.get("doc_id")) or self.support.clean_filter_text(inferred.get("doc_id"))
        parent_id = self.support.clean_filter_text(data.get("parent_id")) or self.support.clean_filter_text(planner_filters.get("parent_id")) or self.support.clean_filter_text(inferred.get("parent_id"))
        lang = self.support.clean_filter_text(data.get("lang")) or self.support.clean_filter_text(planner_filters.get("lang")) or self.support.clean_filter_text(inferred.get("lang"))

        is_attachment = self.support.parse_bool(data.get("is_attachment"))
        if is_attachment is None:
            is_attachment = self.support.parse_bool(planner_filters.get("is_attachment"))
        if is_attachment is None:
            is_attachment = self.support.parse_bool(inferred.get("is_attachment"))

        chunk_no = self.support.safe_int(data.get("chunk_no"))
        if chunk_no is None:
            chunk_no = self.support.safe_int(planner_filters.get("chunk_no"))
        if chunk_no is None:
            chunk_no = self.support.safe_int(inferred.get("chunk_no"))

        one_day_ms = 86_400_000
        doc_date = self.support.safe_int(data.get("document_date"))
        ingest_date = self.support.safe_int(data.get("ingestion_date"))
        document_date_gte = self.support.safe_int(data.get("document_date_gte")) or (doc_date - one_day_ms if doc_date else None)
        document_date_lte = self.support.safe_int(data.get("document_date_lte")) or (doc_date + one_day_ms if doc_date else None)
        ingestion_date_gte = self.support.safe_int(data.get("ingestion_date_gte"))
        ingestion_date_lte = self.support.safe_int(data.get("ingestion_date_lte"))
        if ingest_date is not None and ingestion_date_gte is None and ingestion_date_lte is None:
            ingestion_date_gte = ingest_date - one_day_ms
            ingestion_date_lte = ingest_date + one_day_ms

        time_filter = self.support.resolve_time_filter(query, plan)
        if time_filter and not any(value is not None for value in (document_date_gte, document_date_lte, ingestion_date_gte, ingestion_date_lte)):
            field = str(time_filter.get("field") or getattr(settings, "rag_default_time_filter_field", "ingestion_date"))
            if field == "document_date":
                document_date_gte = self.support.safe_int(time_filter.get("start_ms"))
                document_date_lte = self.support.safe_int(time_filter.get("end_ms"))
            else:
                ingestion_date_gte = self.support.safe_int(time_filter.get("start_ms"))
                ingestion_date_lte = self.support.safe_int(time_filter.get("end_ms"))

        filters = {
            "report_type": report_type,
            "report_types": report_types,
            "branch": branch,
            "doc_id": doc_id,
            "parent_id": parent_id,
            "lang": lang,
            "is_attachment": is_attachment,
            "chunk_no": chunk_no,
            "document_date_gte": document_date_gte,
            "document_date_lte": document_date_lte,
            "ingestion_date_gte": ingestion_date_gte,
            "ingestion_date_lte": ingestion_date_lte,
            "collection_name": self.support.clean_filter_text(data.get("collection_name")),
        }
        logger.info(
            "RAG effective filters chat_id=%s message_id=%s planner_filters=%s inferred_filters=%s time_filter=%s effective=%s",
            data.get("chat_id"),
            data.get("message_id"),
            planner_filters,
            inferred,
            time_filter,
            {k: v for k, v in filters.items() if self.support.has_filter_value(v)},
        )
        return filters

    async def _run_bigdata_batch(
        self,
        data: Dict[str, Any],
        *,
        retrieval_queries: List[str],
        filter_variants: List[Dict[str, Any]],
        top_n: int,
        min_score: float,
    ) -> List[Tuple[str, List[Dict[str, Any]]]]:
        work: List[Tuple[str, Dict[str, Any]]] = []
        tasks = []
        for query in retrieval_queries:
            for variant in filter_variants:
                work.append((query, variant))
                tasks.append(
                    self._fetch_bigdata_context_for_query(
                        data,
                        query=query,
                        filters=variant,
                        top_n=top_n,
                        min_score=min_score,
                    )
                )
        results = await asyncio.gather(*tasks, return_exceptions=True)
        pairs: List[Tuple[str, List[Dict[str, Any]]]] = []
        for (query, _variant), result in zip(work, results):
            if isinstance(result, Exception):
                logger.warning(
                    "RAG big-data retrieval raised exception chat_id=%s query=%s error=%s",
                    data.get("chat_id"),
                    self._truncate_text(query, 120),
                    result,
                )
                continue
            pairs.append((query, list(result or [])))
        return pairs

    @staticmethod
    def _relax_bigdata_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
        relaxed = dict(filters)
        relaxed["doc_id"] = None
        relaxed["parent_id"] = None
        relaxed["chunk_no"] = None
        return relaxed

    @staticmethod
    def _expand_bigdata_filter_variants(filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        base = dict(filters)
        report_types = [str(value).strip() for value in list(base.pop("report_types", []) or []) if str(value).strip()]
        if not report_types:
            return [base]
        variants: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for report_type in report_types:
            key = report_type.lower()
            if key in seen:
                continue
            seen.add(key)
            variant = dict(base)
            variant["report_type"] = report_type
            variants.append(variant)
        return variants or [base]

    async def _fetch_bigdata_context(
        self,
        data: Dict[str, Any],
        *,
        message_id: str,
        retrieval_queries: List[str],
        exact_terms: List[str],
        filters: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        filter_variants = self._expand_bigdata_filter_variants(filters)
        semantic_query_limit = max(
            1,
            int(
                getattr(
                    settings,
                    "rag_semantic_query_variants",
                    getattr(settings, "rag_retrieval_query_variants", 2),
                )
            ),
        )
        semantic_queries = list(retrieval_queries[:semantic_query_limit]) or list(retrieval_queries[:1])
        retrieval_call_count = max(1, len(semantic_queries) * len(filter_variants))
        strict_top_n_per_query = max(1, int(getattr(settings, "rag_bigdata_semantic_top_n_per_query", 200)))
        exact_task = None
        exact_meta: Dict[str, Any] = {
            "attempted": False,
            "keywords": list(exact_terms or []),
            "candidate_count": 0,
            "error_count": 0,
            "errors": [],
            "all_failed": False,
        }
        if exact_terms and bool(getattr(settings, "rag_exact_match_enabled", True)):
            exact_task = asyncio.create_task(
                self._fetch_bigdata_exact_context_multi(
                    data,
                    query=semantic_queries[0],
                    keywords=exact_terms,
                    filters=filters,
                )
            )

        strict_min_score = max(
            float(settings.rag_min_score),
            float(getattr(settings, "rag_bigdata_min_score_floor", 0.0)),
        )
        await self._set_ui_state(
            message_id=message_id,
            ui="semantic_fetch",
            ui_detail=(
                f"fetching semantic candidates across {retrieval_call_count} retrieval calls "
                f"(top_n_per_query={strict_top_n_per_query}, min_score={strict_min_score:.3f})"
            ),
        )
        strict_pairs = await self._run_bigdata_batch(
            data,
            retrieval_queries=semantic_queries,
            filter_variants=filter_variants,
            top_n=strict_top_n_per_query,
            min_score=strict_min_score,
        )
        strict_rows = self.support.merge_chunks_by_identity(strict_pairs)
        await self._set_ui_state(
            message_id=message_id,
            ui="semantic_fetch",
            ui_detail=(
                f"{len(strict_rows)} semantic candidates merged from "
                f"{retrieval_call_count} retrieval calls"
            ),
        )
        if strict_rows:
            logger.info(
                "RAG strict big-data retrieval chat_id=%s query_variants=%s semantic_query_variants=%s filter_variants=%s top_n_per_query=%s strict_min_score=%s candidates=%s",
                data.get("chat_id"),
                len(retrieval_queries),
                len(semantic_queries),
                len(filter_variants),
                strict_top_n_per_query,
                strict_min_score,
                len(strict_rows),
            )

        exact_rows: List[Dict[str, Any]] = []
        if exact_task is not None:
            exact_rows, exact_meta = await exact_task
            await self._set_ui_state(
                message_id=message_id,
                ui="exact_candidates",
                ui_detail=f"{len(exact_rows)} exact candidates found",
            )

        merged_strict = self.support.merge_chunks_by_identity(
            [("semantic", strict_rows), ("exact", exact_rows)]
        )
        await self._set_ui_state(
            message_id=message_id,
            ui="semantic_fetch",
            ui_detail=f"{len(merged_strict)} merged candidates ready for reranking",
        )
        retrieval_meta = {
            "semantic": {
                "query_count": len(semantic_queries),
                "filter_variant_count": len(filter_variants),
                "retrieval_call_count": retrieval_call_count,
                "top_n_per_query": strict_top_n_per_query,
                "min_score": strict_min_score,
                "candidate_count": len(strict_rows),
            },
            "exact": exact_meta,
            "merged_candidate_count": len(merged_strict),
        }
        return merged_strict, retrieval_meta

    @classmethod
    def _should_use_chat_context_only(
        cls,
        *,
        query: str,
        plan: RetrievalPlan,
        chat_ctx: Dict[str, Any],
        search_modes: set[str],
        attachment_ids: List[str],
        has_attachments: bool,
    ) -> bool:
        del search_modes
        if attachment_ids or bool(has_attachments):
            return False
        if str(plan.retrieval_action or "").strip() != "reuse_previous_topic":
            return False
        if not bool(plan.context_dependent or plan.followup):
            return False
        if not cls._chat_context_has_answer(chat_ctx):
            return False
        answer_intent = str(plan.answer_intent or "").strip().lower()
        if answer_intent in {"summarize", "format", "continue"}:
            return True
        return bool(cls._CHAT_ONLY_TRANSFORM_PATTERN.search(str(query or "")))

    @staticmethod
    def _chat_context_has_answer(chat_ctx: Dict[str, Any]) -> bool:
        ordered_turns = list((chat_ctx or {}).get("ordered_turns") or [])
        for turn in ordered_turns:
            if str((turn or {}).get("assistant_reply") or "").strip():
                return True
        recent_pairs = list((chat_ctx or {}).get("recent_conversations") or [])
        for pair in recent_pairs:
            if str((pair or {}).get("assistant_reply") or "").strip():
                return True
        history = list((chat_ctx or {}).get("conversation_history") or [])
        for item in history:
            role = str((item or {}).get("role") or "").strip().lower()
            if role in {"assistant", "system"} and str((item or {}).get("text") or (item or {}).get("content") or "").strip():
                return True
        return False

    async def _fetch_rag_context(self, data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        message_id = str(data.get("message_id") or "").strip()
        user_query = self._normalize_query(data)
        attachment_ids = self.support.extract_attachment_ids(data)
        has_uploaded_file = bool(attachment_ids or data.get("has_attachments"))
        search_modes = self.support.search_mode_tokens(data.get("search_mode"))
        deep_research_enabled = bool({"deep_research", "deep-research", "deepresearch"} & search_modes)
        uploaded_file_chunks: List[Dict[str, Any]] = []
        uploaded_file_anchor: Dict[str, Any] = {}
        planner_data = dict(data)
        if has_uploaded_file:
            await self._set_ui_state(
                message_id=message_id,
                ui="semantic_fetch",
                ui_detail="fetching uploaded file context for BigData query anchor",
            )
            uploaded_file_chunks = self._mark_uploaded_file_chunks(
                await self._fetch_attachment_context(
                    data,
                    attachment_ids,
                    retrieval_query=user_query,
                )
            )
            uploaded_file_anchor = self._build_uploaded_file_anchor(uploaded_file_chunks)
            planner_data = self._build_file_anchored_planner_payload(
                data,
                user_query=user_query,
                file_anchor=uploaded_file_anchor,
            )
            logger.info(
                "RAG uploaded-file anchor chat_id=%s message_id=%s attachment_ids=%s file_chunks=%s anchor_chars=%s anchor_terms=%s",
                data.get("chat_id"),
                data.get("message_id"),
                attachment_ids,
                len(uploaded_file_chunks),
                len(str(uploaded_file_anchor.get("text") or "")),
                list(uploaded_file_anchor.get("terms") or [])[:12],
            )
        await self._set_ui_state(
            message_id=message_id,
            ui="planning",
            ui_detail="building retrieval plan",
        )
        plan, chat_ctx = await self._build_retrieval_plan(planner_data)
        if uploaded_file_chunks:
            plan = self._sanitize_file_anchored_plan(
                plan,
                user_query=user_query,
                file_anchor=uploaded_file_anchor,
            )
        chat_context_only = self._should_use_chat_context_only(
            query=user_query,
            plan=plan,
            chat_ctx=chat_ctx,
            search_modes=search_modes,
            attachment_ids=attachment_ids,
            has_attachments=bool(data.get("has_attachments")),
        )
        retrieval_queries = self.support.dedupe_queries(
            [
                plan.standalone_query or user_query,
                *list(plan.query_variants or []),
                *([] if plan.context_dependent or plan.retrieval_action == "reuse_previous_topic" else [user_query]),
            ],
            max_items=max(1, int(settings.rag_retrieval_query_variants)),
        )
        exact_term_query = (
            plan.standalone_query
            if plan.context_dependent or plan.retrieval_action == "reuse_previous_topic"
            else user_query
        )
        exact_terms = self.support.fallback_exact_terms(
            exact_term_query,
            plan,
            max_items=max(1, int(settings.rag_exact_match_max_terms)),
        )

        kb_chunks: List[Dict[str, Any]] = []
        retrieval_diagnostics: Dict[str, Any] = {}
        used_attachment_context = False
        if chat_context_only and not uploaded_file_chunks:
            retrieval_plan = {
                "primary_query": retrieval_queries[0],
                "queries": retrieval_queries,
                "focus_hint": str(plan.focus_hint or "").strip(),
                "focus_subject": str(plan.focus_subject or "").strip(),
                "answer_intent": str(plan.answer_intent or "").strip(),
                "retrieval_action": str(plan.retrieval_action or "").strip(),
                "context_dependent": bool(plan.context_dependent),
                "followup": bool(plan.followup),
                "exact_terms": exact_terms,
                "time_filter": self.support.resolve_time_filter(user_query, plan),
                "planner_used": bool(plan.planner_used),
                "search_modes": sorted(search_modes),
                "deep_research": deep_research_enabled,
                "filters": {},
                "prompt_budget": {
                    "candidate_chunks": 0,
                    "selected_chunks": 0,
                    "token_budget": int(getattr(settings, "rag_prompt_token_budget", 30000)),
                    "estimated_prompt_tokens": 0,
                    "chat_context_only": True,
                    "skip_vector_retrieval_reason": "reuse_previous_topic_transform",
                },
                "chat_context_only": True,
                "skip_vector_retrieval_reason": "reuse_previous_topic_transform",
            }
            logger.info(
                "RAG chat-context-only followup chat_id=%s message_id=%s answer_intent=%s retrieval_action=%s query=%s",
                data.get("chat_id"),
                data.get("message_id"),
                retrieval_plan["answer_intent"],
                retrieval_plan["retrieval_action"],
                self._truncate_text(user_query, 120),
            )
            await self._set_ui_state(
                message_id=message_id,
                ui="sending_to_llm",
                ui_detail="using prior chat context only",
            )
            return [], chat_ctx, retrieval_plan

        filters = self._build_bigdata_filters(data, plan=plan)
        kb_chunks, retrieval_diagnostics = await self._fetch_bigdata_context(
            data,
            message_id=message_id,
            retrieval_queries=retrieval_queries,
            exact_terms=exact_terms,
            filters=filters,
        )
        kb_chunks = self._mark_bigdata_chunks(kb_chunks)

        secondary_query_inputs: List[str] = [*retrieval_queries[1:]]
        if not plan.context_dependent and plan.retrieval_action != "reuse_previous_topic":
            secondary_query_inputs.append(user_query)
        secondary_query_inputs.extend([plan.focus_hint, plan.focus_subject])
        secondary_queries = self.support.dedupe_queries(
            [query for query in secondary_query_inputs if query],
            max_items=max(4, len(retrieval_queries) + 2),
        )
        ranked = self.support.rank_chunks_for_query(
            kb_chunks,
            retrieval_queries[0],
            secondary_queries=secondary_queries,
            topic_hint=str(plan.focus_hint or plan.focus_subject or ""),
        )

        if (
            bool(getattr(settings, "rag_contextual_retry_enabled", True))
            and plan.context_dependent
            and (plan.focus_subject or plan.focus_hint)
            and not attachment_ids
            and not bool(data.get("has_attachments"))
        ):
            topic_hint = str(plan.focus_hint or plan.focus_subject or "")
            alignment = self.support.topic_alignment_score(ranked, topic_hint, top_k=6)
            threshold = float(getattr(settings, "rag_contextual_retry_topic_overlap_threshold", 0.18))
            if alignment < threshold:
                topic_queries = self.support.dedupe_queries(
                    [f"{retrieval_queries[0]} {topic_hint}", f"{topic_hint} {retrieval_queries[0]}", topic_hint, user_query],
                    max_items=4,
                )
                retry_chunks, retry_retrieval_diagnostics = await self._fetch_bigdata_context(
                    data,
                    message_id=message_id,
                    retrieval_queries=topic_queries,
                    exact_terms=exact_terms,
                    filters=filters,
                )
                if retry_retrieval_diagnostics:
                    retrieval_diagnostics["contextual_retry"] = retry_retrieval_diagnostics
                if retry_chunks:
                    ranked = self.support.rank_chunks_for_query(
                        self.support.merge_chunks_by_identity([("initial", kb_chunks), ("topic_retry", retry_chunks)]),
                        retrieval_queries[0],
                        secondary_queries=self.support.dedupe_queries([*secondary_queries, *topic_queries], max_items=8),
                        topic_hint=topic_hint,
                    )

        pre_rerank_ranked = list(ranked)
        if not used_attachment_context:
            ranked = await self._apply_bigdata_reranker(
                message_id=str(data.get("message_id") or "").strip(),
                query=retrieval_queries[0],
                chunks=ranked,
            )

        post_rerank_ranked = list(ranked)
        baseline_prompt_chunks, baseline_prompt_budget = self._select_prompt_chunks_by_token_budget(ranked)
        final_chunks = list(baseline_prompt_chunks)
        prompt_budget = dict(baseline_prompt_budget)
        selector_meta: Dict[str, Any] = {}
        if final_chunks and not used_attachment_context and deep_research_enabled:
            final_chunks, prompt_budget, selector_meta = await self._maybe_apply_evidence_selector(
                message_id=message_id,
                query=user_query,
                primary_query=retrieval_queries[0],
                retrieval_focus=str(plan.focus_subject or plan.focus_hint or "").strip(),
                ranked_chunks=post_rerank_ranked,
                prompt_chunks=final_chunks,
                prompt_budget=prompt_budget,
            )
        elif final_chunks and not used_attachment_context:
            selector_meta = {
                "enabled": bool(getattr(settings, "rag_evidence_selector_enabled", False)),
                "applied": False,
                "skipped": True,
                "reason": "deep_research_not_selected",
            }
        coverage_meta: Dict[str, Any] = {}
        if final_chunks and not used_attachment_context and deep_research_enabled:
            if self._should_run_coverage_judge_for_selector(selector_meta):
                use_baseline_for_judge = self._coverage_judge_should_use_baseline_prompt(selector_meta)
                judge_prompt_chunks = list(baseline_prompt_chunks) if use_baseline_for_judge else list(final_chunks)
                judge_prompt_budget = dict(baseline_prompt_budget) if use_baseline_for_judge else dict(prompt_budget)
                ranked, prompt_budget, coverage_meta, final_chunks = await self._maybe_apply_coverage_recovery(
                    data=data,
                    message_id=message_id,
                    query=user_query,
                    primary_query=retrieval_queries[0],
                    retrieval_focus=str(plan.focus_subject or plan.focus_hint or "").strip(),
                    secondary_queries=secondary_queries,
                    exact_terms=exact_terms,
                    filters=filters,
                    prompt_chunks=judge_prompt_chunks,
                    prompt_budget=judge_prompt_budget,
                    pre_rerank_chunks=pre_rerank_ranked,
                    post_rerank_chunks=post_rerank_ranked,
                )
                coverage_meta["judge_input_source"] = (
                    "baseline_reranked_prompt_budget"
                    if use_baseline_for_judge
                    else "selector_selected_prompt_budget"
                )
            else:
                coverage_meta = self._coverage_judge_skip_meta(selector_meta)
                logger.info(
                    "RAG coverage judge skipped by selector gate message_id=%s selected=%s batch_count=%s insufficient_flags=%s",
                    message_id,
                    coverage_meta.get("selector_selected_count"),
                    coverage_meta.get("selector_batch_count"),
                    coverage_meta.get("selector_batches_insufficient"),
                )
        elif final_chunks and not used_attachment_context:
            coverage_meta = {
                "skipped": True,
                "reason": "deep_research_not_selected",
            }
        if uploaded_file_chunks:
            final_chunks = [*uploaded_file_chunks, *self._mark_bigdata_chunks(final_chunks)]
            prompt_budget = {
                **dict(prompt_budget or {}),
                "uploaded_file_chunks": len(uploaded_file_chunks),
                "bigdata_prompt_chunks": len(final_chunks) - len(uploaded_file_chunks),
                "file_anchor_chars": len(str(uploaded_file_anchor.get("text") or "")),
            }
        prompt_chunk_counts = self._chunk_origin_counts(final_chunks)
        await self._set_ui_state(
            message_id=message_id,
            ui="sending_to_llm",
            ui_detail=(
                f"{len(final_chunks)} chunks selected for answer prompt "
                f"({int(prompt_budget.get('estimated_prompt_tokens') or 0)} tokens est.)"
            ),
        )
        retrieval_plan = {
            "primary_query": retrieval_queries[0],
            "queries": retrieval_queries,
            "focus_hint": str(plan.focus_hint or "").strip(),
            "focus_subject": str(plan.focus_subject or "").strip(),
            "answer_intent": str(plan.answer_intent or "").strip(),
            "retrieval_action": str(plan.retrieval_action or "").strip(),
            "context_dependent": bool(plan.context_dependent),
            "followup": bool(plan.followup),
            "exact_terms": exact_terms,
            "time_filter": self.support.resolve_time_filter(user_query, plan),
            "planner_used": bool(plan.planner_used),
            "search_modes": sorted(search_modes),
            "deep_research": deep_research_enabled,
            "filters": filters,
            "retrieval_diagnostics": retrieval_diagnostics,
            "prompt_budget": prompt_budget,
            "evidence_selector": selector_meta,
            "coverage": coverage_meta,
            "coverage_guidance": str((coverage_meta or {}).get("guidance") or "").strip(),
            "uploaded_file_context": {
                "used": bool(uploaded_file_chunks),
                "attachment_ids": attachment_ids,
                "chunk_count": len(uploaded_file_chunks),
                "anchor_terms": list(uploaded_file_anchor.get("terms") or []),
                "anchor_chars": len(str(uploaded_file_anchor.get("text") or "")),
            },
        }
        logger.info(
            "RAG retrieval plan chat_id=%s message_id=%s followup=%s context_dependent=%s answer_intent=%s retrieval_action=%s planner_used=%s primary=%s variants=%s exact_terms=%s",
            data.get("chat_id"),
            data.get("message_id"),
            retrieval_plan["followup"],
            retrieval_plan["context_dependent"],
            retrieval_plan["answer_intent"],
            retrieval_plan["retrieval_action"],
            retrieval_plan["planner_used"],
            self._truncate_text(retrieval_plan["primary_query"], 120),
            [self._truncate_text(item, 120) for item in retrieval_plan["queries"]],
            exact_terms,
        )
        logger.info(
            "RAG prompt chunk mix chat_id=%s message_id=%s total=%s exact=%s semantic=%s both=%s other=%s token_budget=%s estimated_prompt_tokens=%s truncated_chunks=%s max_prompt_chunks=%s exact_top_n=%s exact_guaranteed=%s",
            data.get("chat_id"),
            data.get("message_id"),
            prompt_chunk_counts["total"],
            prompt_chunk_counts["exact"],
            prompt_chunk_counts["semantic"],
            prompt_chunk_counts["both"],
            prompt_chunk_counts["other"],
            prompt_budget.get("token_budget"),
            prompt_budget.get("estimated_prompt_tokens"),
            prompt_budget.get("truncated_chunks"),
            int(settings.rag_max_prompt_chunks),
            int(getattr(settings, "rag_exact_match_top_n", 0)),
            int(getattr(settings, "rag_exact_match_guaranteed_chunks", 0)),
        )
        return final_chunks, chat_ctx, retrieval_plan

    async def _append_llm_trace(
        self,
        *,
        message_id: str,
        system_prompt: str = "",
        user_query: str = "",
        stage: str = "rag_answer_prompt",
        metadata: Optional[Dict[str, Any]] = None,
        extra_sections: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not bool(getattr(settings, "llm_trace_enabled", False)):
            return
        timestamp = datetime.now(tz=timezone.utc).isoformat()
        divider = "=" * 120
        system_tokens_est = self._estimate_prompt_tokens(system_prompt)
        user_tokens_est = self._estimate_prompt_tokens(user_query)
        trace_metadata = {
            "stage": stage,
            "message_id": message_id,
            "system_prompt_chars": len(system_prompt or ""),
            "user_query_chars": len(user_query or ""),
            "system_prompt_tokens_est": system_tokens_est,
            "user_query_tokens_est": user_tokens_est,
            "input_tokens_est": system_tokens_est + user_tokens_est,
            "current_message_total_tokens_est": system_tokens_est + user_tokens_est,
        }
        trace_metadata.update(dict(metadata or {}))
        sections = [
            f"\n{divider}\n",
            f"timestamp_utc: {timestamp}\n",
            f"{divider}\n",
            "[TRACE_METADATA]\n",
            f"{self._format_trace_json(trace_metadata)}\n",
        ]
        if system_prompt:
            sections.extend([f"\n[SYSTEM_PROMPT]\n", f"{system_prompt}\n"])
        if user_query:
            sections.extend([f"\n[USER_QUERY]\n", f"{user_query}\n"])
        for label, value in dict(extra_sections or {}).items():
            if value in (None, "", [], {}):
                continue
            sections.extend([f"\n[{str(label).upper()}]\n", f"{self._format_trace_json(value)}\n"])
        sections.append(f"{divider}\n")
        block = "".join(sections)
        try:
            async with self._trace_lock:
                await asyncio.to_thread(self._write_trace_sync, block)
        except Exception as exc:
            logger.warning("Failed to append RAG LLM trace message_id=%s error=%s", message_id, exc)

    def _write_trace_sync(self, block: str) -> None:
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self._trace_path.open("a", encoding="utf-8") as handle:
            handle.write(block)

    @staticmethod
    def _format_trace_json(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        except Exception:
            return str(value)

    async def _append_planner_trace(self, trace: Dict[str, Any]) -> None:
        if not bool(getattr(settings, "rag_planner_trace_enabled", True)):
            return
        timestamp = datetime.now(tz=timezone.utc).isoformat()
        divider = "=" * 120
        planner_system_tokens_est = self._estimate_prompt_tokens(trace.get("system_prompt"))
        planner_user_tokens_est = self._estimate_prompt_tokens(trace.get("user_prompt"))
        metadata = {
            "stage": trace.get("stage"),
            "chat_id": trace.get("chat_id"),
            "message_id": trace.get("message_id"),
            "expanded_history_used": bool(trace.get("expanded_history_used")),
            "deterministic_override_applied": bool(trace.get("deterministic_override_applied")),
            "system_prompt_tokens_est": planner_system_tokens_est,
            "user_prompt_tokens_est": planner_user_tokens_est,
            "input_tokens_est": planner_system_tokens_est + planner_user_tokens_est,
            "current_message_total_tokens_est": planner_system_tokens_est + planner_user_tokens_est,
            "raw_output_tokens_est": self._estimate_prompt_tokens(trace.get("raw_output")),
        }
        sections = [
            f"\n{divider}\n",
            f"timestamp_utc: {timestamp}\n",
            f"{divider}\n",
            "[TRACE_METADATA]\n",
            f"{self._format_trace_json(metadata)}\n",
        ]
        field_sections = [
            ("system_prompt", "SYSTEM_PROMPT"),
            ("user_prompt", "USER_PROMPT"),
            ("raw_output", "RAW_OUTPUT"),
            ("parsed_payload", "PARSED_PAYLOAD"),
            ("deterministic_payload", "DETERMINISTIC_PAYLOAD"),
            ("final_plan", "FINAL_PLAN"),
            ("error", "ERROR"),
        ]
        for key, label in field_sections:
            value = trace.get(key)
            if value in (None, "", [], {}):
                continue
            sections.extend([f"\n[{label}]\n", f"{self._format_trace_json(value)}\n"])
        sections.append(f"{divider}\n")
        block = "".join(sections)
        try:
            async with self._planner_trace_lock:
                await asyncio.to_thread(self._write_planner_trace_sync, block)
        except Exception as exc:
            logger.warning(
                "Failed to append planner trace message_id=%s stage=%s error=%s",
                trace.get("message_id"),
                trace.get("stage"),
                exc,
            )

    def _write_planner_trace_sync(self, block: str) -> None:
        self._planner_trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self._planner_trace_path.open("a", encoding="utf-8") as handle:
            handle.write(block)

    async def _append_judge_trace(
        self,
        *,
        message_id: str,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        metadata: Optional[Dict[str, Any]] = None,
        raw_output: Any = None,
        parsed_payload: Any = None,
        normalized_payload: Any = None,
        error: Any = None,
    ) -> None:
        if not bool(getattr(settings, "rag_judge_trace_enabled", True)):
            return
        timestamp = datetime.now(tz=timezone.utc).isoformat()
        divider = "=" * 120
        trace_metadata = {
            "stage": stage,
            "message_id": message_id,
            "system_prompt_chars": len(system_prompt or ""),
            "user_prompt_chars": len(user_prompt or ""),
            "system_prompt_tokens_est": self._estimate_prompt_tokens(system_prompt),
            "user_prompt_tokens_est": self._estimate_prompt_tokens(user_prompt),
            "input_tokens_est": self._estimate_prompt_tokens(system_prompt) + self._estimate_prompt_tokens(user_prompt),
        }
        trace_metadata.update(dict(metadata or {}))
        sections = [
            f"\n{divider}\n",
            f"timestamp_utc: {timestamp}\n",
            f"{divider}\n",
            "[TRACE_METADATA]\n",
            f"{self._format_trace_json(trace_metadata)}\n",
            "\n[SYSTEM_PROMPT]\n",
            f"{system_prompt}\n",
            "\n[USER_PROMPT]\n",
            f"{user_prompt}\n",
        ]
        if raw_output not in (None, "", [], {}):
            sections.extend(["\n[RAW_OUTPUT]\n", f"{self._format_trace_json(raw_output)}\n"])
        if parsed_payload not in (None, "", [], {}):
            sections.extend(["\n[PARSED_PAYLOAD]\n", f"{self._format_trace_json(parsed_payload)}\n"])
        if normalized_payload not in (None, "", [], {}):
            sections.extend(["\n[NORMALIZED_PAYLOAD]\n", f"{self._format_trace_json(normalized_payload)}\n"])
        if error not in (None, "", [], {}):
            sections.extend(["\n[ERROR]\n", f"{self._format_trace_json(error)}\n"])
        sections.append(f"{divider}\n")
        block = "".join(sections)
        try:
            async with self._judge_trace_lock:
                await asyncio.to_thread(self._write_judge_trace_sync, block)
        except Exception as exc:
            logger.warning("Failed to append RAG judge trace message_id=%s stage=%s error=%s", message_id, stage, exc)

    def _write_judge_trace_sync(self, block: str) -> None:
        self._judge_trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self._judge_trace_path.open("a", encoding="utf-8") as handle:
            handle.write(block)

    async def _append_selector_trace(
        self,
        *,
        message_id: str,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        metadata: Optional[Dict[str, Any]] = None,
        raw_output: Any = None,
        parsed_payload: Any = None,
        normalized_payload: Any = None,
        error: Any = None,
    ) -> None:
        if not bool(getattr(settings, "rag_selector_trace_enabled", True)):
            return
        timestamp = datetime.now(tz=timezone.utc).isoformat()
        divider = "=" * 120
        trace_metadata = {
            "stage": stage,
            "message_id": message_id,
            "system_prompt_chars": len(system_prompt or ""),
            "user_prompt_chars": len(user_prompt or ""),
            "system_prompt_tokens_est": self._estimate_prompt_tokens(system_prompt),
            "user_prompt_tokens_est": self._estimate_prompt_tokens(user_prompt),
            "input_tokens_est": self._estimate_prompt_tokens(system_prompt) + self._estimate_prompt_tokens(user_prompt),
        }
        trace_metadata.update(dict(metadata or {}))
        sections = [
            f"\n{divider}\n",
            f"timestamp_utc: {timestamp}\n",
            f"{divider}\n",
            "[TRACE_METADATA]\n",
            f"{self._format_trace_json(trace_metadata)}\n",
            "\n[SYSTEM_PROMPT]\n",
            f"{system_prompt}\n",
            "\n[USER_PROMPT]\n",
            f"{user_prompt}\n",
        ]
        if raw_output not in (None, "", [], {}):
            sections.extend(["\n[RAW_OUTPUT]\n", f"{self._format_trace_json(raw_output)}\n"])
        if parsed_payload not in (None, "", [], {}):
            sections.extend(["\n[PARSED_PAYLOAD]\n", f"{self._format_trace_json(parsed_payload)}\n"])
        if normalized_payload not in (None, "", [], {}):
            sections.extend(["\n[NORMALIZED_PAYLOAD]\n", f"{self._format_trace_json(normalized_payload)}\n"])
        if error not in (None, "", [], {}):
            sections.extend(["\n[ERROR]\n", f"{self._format_trace_json(error)}\n"])
        sections.append(f"{divider}\n")
        block = "".join(sections)
        try:
            async with self._selector_trace_lock:
                await asyncio.to_thread(self._write_selector_trace_sync, block)
        except Exception as exc:
            logger.warning(
                "Failed to append RAG selector trace message_id=%s stage=%s error=%s",
                message_id,
                stage,
                exc,
            )

    def _write_selector_trace_sync(self, block: str) -> None:
        self._selector_trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self._selector_trace_path.open("a", encoding="utf-8") as handle:
            handle.write(block)

    def _context_trace_rows(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(list(chunks or []), start=1):
            content = self.support.chunk_content(chunk)
            prompt_content = str(chunk.get("_prompt_content") or content or "").strip()
            rerank_content = str(chunk.get("_rerank_content") or "").strip()
            has_exact, has_semantic = self._chunk_origin_flags(chunk)
            rows.append(
                {
                    "source": idx,
                    "origin": (
                        "semantic+exact"
                        if has_exact and has_semantic
                        else "exact"
                        if has_exact
                        else "semantic"
                        if has_semantic
                        else str(chunk.get("_scope") or "unknown")
                    ),
                    "scope": str(chunk.get("_scope") or "").strip(),
                    "retrieval_queries": list(chunk.get("_retrieval_queries") or []),
                    "doc_id": str(chunk.get("doc_id") or "").strip(),
                    "parent_id": str(chunk.get("parent_id") or "").strip(),
                    "chunk_no": chunk.get("chunk_no"),
                    "chunk_id": chunk.get("chunk_id"),
                    "file_id": chunk.get("file_id"),
                    "file_name": chunk.get("file_name"),
                    "report_type": chunk.get("report_type"),
                    "branch": chunk.get("branch"),
                    "section_heading": chunk.get("section_heading"),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "quality_score": chunk.get("quality_score"),
                    "score": chunk.get("score"),
                    "sim_score": chunk.get("_sim_score", chunk.get("sim_score")),
                    "rerank_score": chunk.get("_rerank_score"),
                    "final_score": chunk.get("_final_score"),
                    "heuristic_score": chunk.get("_heuristic_score"),
                    "exact_match_score": chunk.get("_exact_match_score"),
                    "elastic_score_raw": chunk.get("_elastic_score_raw"),
                    "main_score_boost": chunk.get("_main_score_boost"),
                    "matched_keywords": list(chunk.get("_matched_keywords") or []),
                    "access_branches": chunk.get("access_branches"),
                    "access_groups": chunk.get("access_groups"),
                    "is_attachment": chunk.get("is_attachment"),
                    "has_been_embedded": chunk.get("has_been_embedded"),
                    "qdrant_collection": chunk.get("qdrant_collection"),
                    "recovery_source": str(chunk.get("_recovery_source") or "").strip(),
                    "neighbor_chunk_count": chunk.get("_neighbor_chunk_count", 0),
                    "neighbor_offsets": list(chunk.get("_neighbor_offsets") or []),
                    "content_chars": len(content),
                    "prompt_content_chars": len(prompt_content),
                    "rerank_content_chars": len(rerank_content),
                    "content_tokens_est": self._estimate_prompt_tokens(content),
                    "prompt_content_tokens_est": self._estimate_prompt_tokens(prompt_content),
                    "rerank_content_tokens_est": self._estimate_prompt_tokens(rerank_content),
                    "prompt_content_truncated": bool(chunk.get("_prompt_content_truncated")),
                    "raw_content_chars": chunk.get("_raw_content_chars", len(content)),
                    "content_excerpt": self._truncate_text(content, 700),
                    "prompt_content_excerpt": self._truncate_text(prompt_content, 700)
                    if prompt_content != content
                    else "",
                }
            )
        return rows

    async def _append_context_trace(
        self,
        *,
        message_id: str,
        query: str,
        user_prompt: str,
        kb_chunks: List[Dict[str, Any]],
        retrieval_plan: Dict[str, Any],
    ) -> None:
        if not bool(getattr(settings, "rag_context_trace_enabled", True)):
            return
        rows = self._context_trace_rows(kb_chunks)
        plan = dict(retrieval_plan or {})
        counts = self._chunk_origin_counts(kb_chunks)
        exact_meta = dict((plan.get("retrieval_diagnostics") or {}).get("exact") or {})
        selector_meta = dict(plan.get("evidence_selector") or {})
        coverage_meta = dict(plan.get("coverage") or {})
        trace_payload = {
            "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
            "message_id": message_id,
            "query": self._truncate_text(query, 500),
            "user_prompt_tokens_est": self._estimate_prompt_tokens(user_prompt),
            "selected_chunk_count": len(rows),
            "selected_chunk_counts": counts,
            "retrieval_plan": {
                "primary_query": plan.get("primary_query"),
                "queries": list(plan.get("queries") or []),
                "focus_hint": plan.get("focus_hint"),
                "focus_subject": plan.get("focus_subject"),
                "answer_intent": plan.get("answer_intent"),
                "retrieval_action": plan.get("retrieval_action"),
                "context_dependent": bool(plan.get("context_dependent")),
                "followup": bool(plan.get("followup")),
                "exact_terms": list(plan.get("exact_terms") or []),
                "time_filter": dict(plan.get("time_filter") or {}),
                "filters": dict(plan.get("filters") or {}),
                "prompt_budget": dict(plan.get("prompt_budget") or {}),
                "retrieval_diagnostics": dict(plan.get("retrieval_diagnostics") or {}),
            },
            "exact_retrieval": {
                "attempted": bool(exact_meta.get("attempted")),
                "candidate_count": int(exact_meta.get("candidate_count") or 0),
                "error_count": int(exact_meta.get("error_count") or 0),
                "all_failed": bool(exact_meta.get("all_failed")),
                "errors": list(exact_meta.get("errors") or []),
            },
            "evidence_selector": {
                "enabled": bool(selector_meta.get("enabled")),
                "applied": bool(selector_meta.get("applied")),
                "fallback_to_prompt_budget_selection": bool(
                    selector_meta.get("fallback_to_prompt_budget_selection")
                ),
                "selected_source_ids": list(selector_meta.get("selected_source_ids") or []),
                "selected_chunk_count": int(selector_meta.get("selected_chunk_count") or 0),
                "prompt_chunk_count": int(selector_meta.get("prompt_chunk_count") or 0),
                "batches": list(selector_meta.get("batches") or []),
            },
            "coverage": coverage_meta,
            "selected_chunks": rows,
        }
        block = self._format_trace_json(trace_payload) + "\n"
        try:
            async with self._context_trace_lock:
                await asyncio.to_thread(self._write_context_trace_sync, block)
        except Exception as exc:
            logger.warning("Failed to append RAG context trace message_id=%s error=%s", message_id, exc)

    def _write_context_trace_sync(self, block: str) -> None:
        self._context_trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self._context_trace_path.open("a", encoding="utf-8") as handle:
            handle.write(block)

    async def _call_llm_stream(
        self,
        *,
        system_prompt: str,
        user_query: str,
        message_id: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        chunk_count = 0
        content_chunk_count = 0
        empty_delta_count = 0
        no_choice_count = 0
        finish_reasons: List[str] = []
        start_timeout_s = max(1.0, float(getattr(settings, "rag_answer_stream_start_timeout_s", 120.0)))
        idle_timeout_s = max(1.0, float(getattr(settings, "rag_answer_stream_idle_timeout_s", 90.0)))
        try:
            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=settings.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_query},
                        ],
                        temperature=float(settings.rag_temperature if temperature is None else temperature),
                        max_tokens=int(
                            max_tokens
                            if max_tokens is not None
                            else max(1, int(getattr(settings, "rag_answer_max_tokens", 2048)))
                        ),
                        stream=True,
                    ),
                    timeout=start_timeout_s,
                )
            except asyncio.TimeoutError as exc:
                logger.error(
                    "RAG LLM stream request timed out before response message_id=%s timeout_s=%s",
                    message_id,
                    start_timeout_s,
                )
                raise RuntimeError("Answer model did not start streaming in time") from exc

            stream_iterator = response.__aiter__()
            while True:
                try:
                    chunk = await asyncio.wait_for(stream_iterator.__anext__(), timeout=idle_timeout_s)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError as exc:
                    logger.error(
                        "RAG LLM stream idle timeout message_id=%s timeout_s=%s chunks=%s content_chunks=%s",
                        message_id,
                        idle_timeout_s,
                        chunk_count,
                        content_chunk_count,
                    )
                    raise RuntimeError("Answer model stream stalled") from exc

                chunk_count += 1
                choices = list(getattr(chunk, "choices", []) or [])
                if not choices:
                    no_choice_count += 1
                    continue
                choice = choices[0]
                finish_reason = str(getattr(choice, "finish_reason", "") or "").strip()
                if finish_reason:
                    finish_reasons.append(finish_reason)
                delta = getattr(choice, "delta", None)
                content = getattr(delta, "content", None) if delta is not None else None
                if content:
                    content_chunk_count += 1
                    yield str(content)
                else:
                    empty_delta_count += 1
        finally:
            if content_chunk_count == 0:
                logger.warning(
                    "RAG LLM stream produced no content chunks message_id=%s chunks=%s empty_delta_chunks=%s no_choice_chunks=%s finish_reasons=%s",
                    message_id,
                    chunk_count,
                    empty_delta_count,
                    no_choice_count,
                    finish_reasons,
                )

    async def _remember_final_llm_request(
        self,
        *,
        message_id: str,
        system_prompt: str,
        user_query: str,
        temperature: float,
        max_tokens: int,
        mode: str,
    ) -> None:
        normalized_message_id = str(message_id or "").strip()
        if not normalized_message_id:
            return
        async with self._llm_log_lock:
            self._llm_log_contexts.setdefault(normalized_message_id, []).append(
                {
                    "messages": [
                        {"role": "system", "content": str(system_prompt or "")},
                        {"role": "user", "content": str(user_query or "")},
                    ],
                    "temperature": float(temperature),
                    "max_tokens": int(max_tokens),
                    "mode": mode,
                }
            )

    async def _pop_last_final_llm_request(self, message_id: str) -> Dict[str, Any]:
        normalized_message_id = str(message_id or "").strip()
        if not normalized_message_id:
            return {}
        async with self._llm_log_lock:
            calls = self._llm_log_contexts.pop(normalized_message_id, [])
        return dict(calls[-1]) if calls else {}

    async def _enqueue_final_llm_response_log(
        self,
        *,
        payload: Dict[str, Any],
        output: Any,
        assistant_message_id: str,
        llm_request: Optional[Dict[str, Any]] = None,
    ) -> None:
        stream_name = str(getattr(settings, "stream_llm_response", "") or "").strip()
        if self.redis is None or not stream_name:
            return

        request_context = dict(llm_request or {})
        messages = list(request_context.get("messages") or [])
        if not messages:
            messages = [
                {"role": "system", "content": f"{settings.worker_name} final RAG response"},
                {"role": "user", "content": str(payload.get("content") or "")},
            ]

        llm_payload = {
            "created_at": str(datetime.now(timezone.utc)),
            "model": settings.model_name,
            "messages": messages,
            "temperature": request_context.get("temperature", float(settings.rag_temperature)),
            "max_tokens": request_context.get(
                "max_tokens",
                max(1, int(getattr(settings, "rag_answer_max_tokens", 2048))),
            ),
            "total_tokens": None,
            "finish_reason": None,
            "output": str(output or ""),
            "worker": settings.worker_name,
            "worker_type": "rag_agent_service",
            "message_id": str(payload.get("message_id") or "").strip(),
            "assistant_message_id": str(assistant_message_id or "").strip(),
            "chat_id": str(payload.get("chat_id") or "").strip(),
            "user_id": str(payload.get("user_id") or "").strip(),
            "search_mode": str(payload.get("search_mode") or "").strip(),
            "llm_call_mode": str(request_context.get("mode") or "").strip(),
        }
        try:
            await self.redis.xadd(
                stream_name,
                {"data": json.dumps(llm_payload, ensure_ascii=False)},
            )
        except Exception as exc:
            logger.warning(
                "Failed to enqueue final RAG LLM response log message_id=%s stream=%s error=%s",
                payload.get("message_id"),
                stream_name,
                exc,
            )

    def _schedule_final_llm_response_log(
        self,
        *,
        payload: Dict[str, Any],
        output: Any,
        assistant_message_id: str,
        llm_request: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            asyncio.create_task(
                self._enqueue_final_llm_response_log(
                    payload=dict(payload or {}),
                    output=output,
                    assistant_message_id=assistant_message_id,
                    llm_request=dict(llm_request or {}),
                )
            )
        except RuntimeError as exc:
            logger.warning(
                "Failed to schedule final RAG LLM response log message_id=%s error=%s",
                (payload or {}).get("message_id"),
                exc,
            )

    async def _finalize_stream(self, *, message_id: str, stream: str) -> None:
        await self.redis.xadd(stream, {"end": "1"}, maxlen=1000, approximate=True)
        await self.redis.expire(stream, self._response_stream_ttl_seconds())
        await self._hset_task(
            message_id,
            {
                "status": "finished",
                "updated_at": self._task_timestamp(),
            },
        )

    async def _emit_source_documents(self, *, stream: str, source_documents: List[Dict[str, Any]]) -> None:
        if not source_documents:
            return
        await self.redis.xadd(
            stream,
            {"sources": json.dumps(source_documents, ensure_ascii=False)},
            maxlen=10000,
            approximate=True,
        )

    @staticmethod
    def _bigdata_answer_heading() -> str:
        return "## Results from BigData\n\n"

    @classmethod
    def _with_bigdata_answer_heading(cls, text: Any) -> str:
        value = str(text or "")
        heading = cls._bigdata_answer_heading()
        if value.startswith(heading):
            return value
        return heading + value.lstrip()

    async def _emit_plain_response(self, message_id: str, stream: str, text: str, finalize: bool = True) -> str:
        response_text = self._with_bigdata_answer_heading(text)
        tokens = re.findall(r"\S+\s*", response_text.strip())
        if not tokens and response_text:
            tokens = [response_text]
        await self._set_ui_state(
            message_id=message_id,
            ui="sending_to_llm",
            ui_detail="streaming response",
        )
        await self._hset_task(
            message_id,
            {
                "status": "completed",
                "updated_at": self._task_timestamp(),
            },
        )
        for token in tokens:
            await self.redis.xadd(stream, {"data": token}, maxlen=10000, approximate=True)
            await asyncio.sleep(0)
        if finalize:
            await self._finalize_stream(message_id=message_id, stream=stream)
        return response_text

    @staticmethod
    def _dedupe_source_numbers(values: List[int], *, max_sources: int) -> List[int]:
        out: List[int] = []
        seen: set[int] = set()
        limit = max(0, int(max_sources))
        for raw in values:
            try:
                value = int(raw)
            except Exception:
                continue
            if value < 1 or value > limit or value in seen:
                continue
            seen.add(value)
            out.append(value)
        return out

    @classmethod
    def _extract_used_source_numbers_from_block(cls, hidden_text: str, *, max_sources: int) -> List[int]:
        text = str(hidden_text or "")
        start_idx = text.find(USED_SOURCES_JSON_START)
        if start_idx < 0:
            return []
        start_idx += len(USED_SOURCES_JSON_START)
        end_idx = text.find(USED_SOURCES_JSON_END, start_idx)
        if end_idx < 0:
            return []

        payload = re.sub(r"<\|[^>]+?\|>", " ", text[start_idx:end_idx]).strip()
        if not payload:
            return []

        try:
            parsed = json.loads(payload)
        except Exception:
            parsed = None

        if isinstance(parsed, dict):
            raw_values = parsed.get("used_sources")
            if isinstance(raw_values, list):
                return cls._dedupe_source_numbers(raw_values, max_sources=max_sources)

        fallback_values = [int(match.group(1)) for match in re.finditer(r"\b(\d+)\b", payload)]
        return cls._dedupe_source_numbers(fallback_values, max_sources=max_sources)

    @classmethod
    def _extract_inline_source_numbers(cls, visible_text: str, *, max_sources: int) -> List[int]:
        matches = [int(match.group(1)) for match in re.finditer(r"\[source:(\d+)\]", str(visible_text or ""))]
        return cls._dedupe_source_numbers(matches, max_sources=max_sources)

    @staticmethod
    def _select_chunks_by_source_numbers(
        prompt_chunks: List[Dict[str, Any]],
        source_numbers: List[int],
    ) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        for source_number in source_numbers:
            index = int(source_number) - 1
            if index < 0 or index >= len(prompt_chunks):
                continue
            selected.append(prompt_chunks[index])
        return selected

    def _select_top_scored_source_chunks(
        self,
        prompt_chunks: List[Dict[str, Any]],
        *,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        rows = list(prompt_chunks or [])
        if not rows:
            return []
        ranked = sorted(
            enumerate(rows),
            key=lambda item: (-self._chunk_score(item[1]), item[0]),
        )
        return [chunk for _idx, chunk in ranked[: max(1, int(limit))]]

    @classmethod
    def _has_substantive_visible_answer(cls, visible_answer: Any, *, prefix_text: str = "") -> bool:
        text = str(visible_answer or "").strip()
        prefix = str(prefix_text or "").strip()
        if prefix and text.startswith(prefix):
            text = text[len(prefix):].strip()
        text = re.sub(r"^#+\s*Results\s+from\s+BigData\s*", " ", text, flags=re.IGNORECASE).strip()
        return bool(text)

    async def _llm_generate(
        self,
        *,
        message_id: str,
        stream: str,
        system_prompt: str,
        user_query: str,
        prompt_chunks: Optional[List[Dict[str, Any]]] = None,
        user_access_codes: Optional[List[str] | set[str]] = None,
        access_override: bool = False,
        emit_sources: bool = True,
        finalize: bool = True,
        prefix_text: str = "",
    ) -> Tuple[str, List[Dict[str, Any]]]:
        prompt_rows = list(prompt_chunks or [])
        await self._append_llm_trace(
            message_id=message_id,
            stage="rag_answer_prompt",
            system_prompt=system_prompt,
            user_query=user_query,
            metadata={
                "prompt_chunks": len(prompt_rows),
                "emit_sources": bool(emit_sources),
                "finalize": bool(finalize),
            },
        )
        await self._remember_final_llm_request(
            message_id=message_id,
            system_prompt=system_prompt,
            user_query=user_query,
            temperature=float(settings.rag_temperature),
            max_tokens=max(1, int(getattr(settings, "rag_answer_max_tokens", 2048))),
            mode="rag_answer_stream",
        )
        visible_answer = ""
        hidden_output = ""
        pending = ""
        stream_started = False
        hidden_started = False
        marker_keep = max(0, len(USED_SOURCES_JSON_START) - 1)
        prefix = str(prefix_text or "")
        if prefix:
            visible_answer += prefix
            stream_started = True
            await self._hset_task(
                message_id,
                {
                    "status": "completed",
                    "updated_at": self._task_timestamp(),
                },
            )
            await self.redis.xadd(stream, {"data": prefix}, maxlen=10000, approximate=True)
        streamed_content_chunks = 0
        async for token in self._call_llm_stream(
            system_prompt=system_prompt,
            user_query=user_query,
            message_id=message_id,
        ):
            if not token:
                continue
            streamed_content_chunks += 1
            if await self._is_task_cancelled(message_id):
                await self._delete_cancelled_response_stream(message_id=message_id, stream=stream)
                raise RuntimeError("Task cancelled during RAG answer streaming")
            if not stream_started:
                stream_started = True
                await self._hset_task(
                    message_id,
                    {
                        "status": "completed",
                        "updated_at": self._task_timestamp(),
                    },
                )
            if hidden_started:
                hidden_output += token
                continue

            pending += token
            marker_index = pending.find(USED_SOURCES_JSON_START)
            if marker_index >= 0:
                visible_chunk = pending[:marker_index]
                if visible_chunk:
                    visible_answer += visible_chunk
                    await self.redis.xadd(stream, {"data": visible_chunk}, maxlen=10000, approximate=True)
                hidden_output = pending[marker_index:]
                pending = ""
                hidden_started = True
                continue

            safe_len = len(pending) - marker_keep
            if safe_len <= 0:
                continue
            visible_chunk = pending[:safe_len]
            pending = pending[safe_len:]
            if visible_chunk:
                visible_answer += visible_chunk
                await self.redis.xadd(stream, {"data": visible_chunk}, maxlen=10000, approximate=True)

        if pending and not hidden_started:
            visible_answer += pending
            await self.redis.xadd(stream, {"data": pending}, maxlen=10000, approximate=True)
        if not stream_started:
            await self._hset_task(
                message_id,
                {
                    "status": "completed",
                    "updated_at": self._task_timestamp(),
                },
            )

        empty_answer_fallback_used = False
        if not self._has_substantive_visible_answer(visible_answer, prefix_text=prefix):
            empty_answer_fallback_used = True
            fallback_text = (
                "I could not generate a response from the answer model for this request. "
                "Please try again."
            )
            visible_answer += fallback_text
            await self.redis.xadd(stream, {"data": fallback_text}, maxlen=10000, approximate=True)
            await self._hset_task(
                message_id,
                {
                    "status": "completed",
                    "updated_at": self._task_timestamp(),
                },
            )
            logger.error(
                "RAG empty visible answer fallback emitted message_id=%s stream=%s prompt_chunks=%s streamed_content_chunks=%s hidden_started=%s hidden_chars=%s pending_chars=%s prefix_chars=%s",
                message_id,
                stream,
                len(prompt_rows),
                streamed_content_chunks,
                hidden_started,
                len(hidden_output),
                len(pending),
                len(prefix),
            )

        source_documents: List[Dict[str, Any]] = []
        used_source_numbers: List[int] = []
        selected_chunks: List[Dict[str, Any]] = []
        fallback_to_top_scored_prompt_chunks = False
        fallback_source_limit = 0
        filtered_uploaded_source_chunks = 0
        if prompt_rows and not empty_answer_fallback_used:
            used_source_numbers = self._extract_used_source_numbers_from_block(
                hidden_output,
                max_sources=len(prompt_rows),
            )
            if not used_source_numbers:
                used_source_numbers = self._extract_inline_source_numbers(
                    visible_answer,
                    max_sources=len(prompt_rows),
                )
            raw_selected_chunks = self._select_chunks_by_source_numbers(prompt_rows, used_source_numbers)
            selected_chunks = self._filter_bigdata_source_chunks(raw_selected_chunks)
            filtered_uploaded_source_chunks = len(raw_selected_chunks) - len(selected_chunks)
            if not selected_chunks:
                bigdata_prompt_rows = self._filter_bigdata_source_chunks(prompt_rows)
                fallback_to_top_scored_prompt_chunks = True
                fallback_source_limit = min(5, len(bigdata_prompt_rows))
                selected_chunks = self._select_top_scored_source_chunks(
                    bigdata_prompt_rows,
                    limit=fallback_source_limit,
                )
            source_documents = build_source_documents(
                selected_chunks,
                limit=int(settings.rag_emit_source_documents_limit),
                user_access_codes=user_access_codes,
                access_override=access_override,
            )
            if emit_sources:
                await self._emit_source_documents(stream=stream, source_documents=source_documents)
            logger.info(
                "RAG used-source resolution message_id=%s prompt_chunks=%s used_sources=%s emitted_sources=%s hidden_block=%s visible_answer_tokens_est=%s hidden_output_tokens_est=%s fallback_top_scored=%s fallback_limit=%s filtered_uploaded_source_chunks=%s",
                message_id,
                len(prompt_rows),
                used_source_numbers,
                len(source_documents),
                bool(hidden_output),
                self._estimate_prompt_tokens(visible_answer),
                self._estimate_prompt_tokens(hidden_output),
                fallback_to_top_scored_prompt_chunks,
                fallback_source_limit,
                filtered_uploaded_source_chunks,
            )
        elif prompt_rows and empty_answer_fallback_used:
            logger.warning(
                "RAG source emission skipped because answer model returned empty visible content message_id=%s prompt_chunks=%s",
                message_id,
                len(prompt_rows),
            )

        await self._append_llm_trace(
            message_id=message_id,
            stage="rag_answer_result",
            metadata={
                "prompt_chunks": len(prompt_rows),
                "used_sources": used_source_numbers,
                "selected_chunk_count": len(selected_chunks),
                "emitted_source_documents": len(source_documents),
                "hidden_block_found": bool(hidden_output),
                "visible_answer_chars": len(visible_answer),
                "visible_answer_tokens_est": self._estimate_prompt_tokens(visible_answer),
                "hidden_output_chars": len(hidden_output),
                "hidden_output_tokens_est": self._estimate_prompt_tokens(hidden_output),
                "output_tokens_est": self._estimate_prompt_tokens(visible_answer) + self._estimate_prompt_tokens(hidden_output),
                "fallback_to_top_scored_prompt_chunks": fallback_to_top_scored_prompt_chunks,
                "fallback_source_limit": fallback_source_limit,
                "filtered_uploaded_source_chunks": filtered_uploaded_source_chunks,
                "empty_answer_fallback_used": empty_answer_fallback_used,
                "streamed_content_chunks": streamed_content_chunks,
            },
            extra_sections={
                "hidden_output": hidden_output,
            },
        )

        if finalize:
            await self._finalize_stream(message_id=message_id, stream=stream)
        return visible_answer, source_documents

    def _should_add_general_background(
        self,
        *,
        query: str,
        db_answer: str,
        retrieval_plan: Dict[str, Any],
        has_kb_context: bool,
        search_modes: set[str],
    ) -> Tuple[bool, str]:
        if not bool(getattr(settings, "rag_general_background_enabled", True)):
            return False, "disabled"
        normalized_search_modes = {
            str(value or "").strip().lower()
            for value in set(search_modes or set())
            if str(value or "").strip()
        }
        if not ({"opensource", "open_source", "open-source"} & normalized_search_modes):
            return False, "opensource_not_selected"

        query_text = str(query or "").strip()
        if not query_text:
            return False, "empty_query"

        explicit_request = bool(self._GENERAL_BACKGROUND_EXPLICIT_PATTERN.search(query_text))
        explicit_no_general = bool(self._NO_GENERAL_BACKGROUND_PATTERN.search(query_text))
        internal_only = bool(self._INTERNAL_ONLY_QUERY_PATTERN.search(query_text))
        report_output_request = bool(self._REPORT_OUTPUT_PATTERN.search(query_text))
        current_or_verified = bool(self._CURRENT_OR_VERIFIED_QUERY_PATTERN.search(query_text))
        insufficient_answer = bool(self._INSUFFICIENT_ANSWER_PATTERN.search(str(db_answer or "")))
        answer_intent = str((retrieval_plan or {}).get("answer_intent") or "").strip().lower()

        if explicit_no_general:
            return False, "user_explicitly_disabled_general_background"
        if report_output_request:
            return False, "report_or_formatted_output_request"
        if answer_intent in {"format", "summarize", "continue"}:
            return False, f"transform_intent={answer_intent}"
        if current_or_verified:
            return False, "requires_current_or_verified_retrieval"
        if internal_only and not explicit_request:
            return False, "internal_only_query"
        if explicit_request and not internal_only:
            return True, "user_explicitly_requested_general_background"
        if explicit_request and internal_only:
            return False, "explicit_request_conflicts_with_internal_identifier_query"

        trigger_mode = str(getattr(settings, "rag_general_background_trigger_mode", "default_on") or "").strip().lower()
        if trigger_mode in {"off", "disabled", "never"}:
            return False, f"trigger_mode={trigger_mode}"
        if trigger_mode in {"default_on", "default-on"}:
            return True, "default_general_background"
        if trigger_mode in {"broad", "most", "auto", "always_general"}:
            return True, "broad_mode_general_query"

        if trigger_mode in {"explicit_or_insufficient", "conservative"}:
            if (
                not has_kb_context
                and bool(getattr(settings, "rag_general_background_on_no_context", True))
                and not internal_only
            ):
                return True, "no_internal_context_for_general_query"
            if (
                has_kb_context
                and insufficient_answer
                and bool(getattr(settings, "rag_general_background_on_insufficient_context", True))
                and not internal_only
            ):
                return True, "internal_answer_indicates_insufficient_context"
            if answer_intent in {"background", "explain"} and not internal_only:
                return True, f"planner_answer_intent={answer_intent}"
            return False, "not_requested_conservative_mode"

        return True, "default_general_background"

    async def _append_general_background(
        self,
        *,
        message_id: str,
        stream: str,
        query: str,
        db_answer: str,
        reason: str,
        retrieval_plan: Optional[Dict[str, Any]] = None,
    ) -> str:
        system_prompt = build_general_background_system_prompt()
        plan = dict(retrieval_plan or {})
        user_prompt = build_general_background_prompt(
            query=query,
            db_answer=db_answer,
            reason=reason,
            resolved_query=str(plan.get("primary_query") or "").strip(),
            retrieval_focus=str(plan.get("focus_subject") or plan.get("focus_hint") or "").strip(),
        )
        await self._append_llm_trace(
            message_id=message_id,
            stage="general_background_prompt",
            system_prompt=system_prompt,
            user_query=user_prompt,
            metadata={
                "reason": reason,
                "resolved_query": str(plan.get("primary_query") or "").strip(),
                "retrieval_focus": str(plan.get("focus_subject") or plan.get("focus_hint") or "").strip(),
                "max_tokens": max(1, int(getattr(settings, "rag_general_background_max_tokens", 700))),
                "temperature": float(getattr(settings, "rag_general_background_temperature", 0.2)),
            },
        )
        await self._remember_final_llm_request(
            message_id=message_id,
            system_prompt=system_prompt,
            user_query=user_prompt,
            temperature=float(getattr(settings, "rag_general_background_temperature", 0.2)),
            max_tokens=max(1, int(getattr(settings, "rag_general_background_max_tokens", 700))),
            mode="general_background_stream",
        )
        header = (
            "\n\n## Data From Open Source ( till April 2024 )\n"
            "This section is not from the retrieved internal documents and may need verification for facts after April 2024.\n\n"
        )
        collected = ""
        stream_started = False
        try:
            async for token in self._call_llm_stream(
                system_prompt=system_prompt,
                user_query=user_prompt,
                message_id=message_id,
                temperature=float(getattr(settings, "rag_general_background_temperature", 0.2)),
                max_tokens=max(1, int(getattr(settings, "rag_general_background_max_tokens", 700))),
            ):
                if not token:
                    continue
                if await self._is_task_cancelled(message_id):
                    await self._delete_cancelled_response_stream(message_id=message_id, stream=stream)
                    raise RuntimeError("Task cancelled during RAG general-background streaming")
                if not stream_started:
                    stream_started = True
                    collected += header
                    await self.redis.xadd(stream, {"data": header}, maxlen=10000, approximate=True)
                    await self._hset_task(
                        message_id,
                        {
                            "status": "completed",
                            "updated_at": self._task_timestamp(),
                        },
                    )
                collected += token
                await self.redis.xadd(stream, {"data": token}, maxlen=10000, approximate=True)
        except Exception as exc:
            logger.warning(
                "RAG general-background append failed message_id=%s reason=%s error=%s",
                message_id,
                reason,
                exc,
            )
            return ""

        if stream_started:
            logger.info(
                "RAG general-background appended message_id=%s reason=%s chars=%s output_tokens_est=%s",
                message_id,
                reason,
                len(collected),
                self._estimate_prompt_tokens(collected),
            )
        await self._append_llm_trace(
            message_id=message_id,
            stage="general_background_result",
            metadata={
                "reason": reason,
                "emitted": bool(stream_started),
                "output_chars": len(collected),
                "output_tokens_est": self._estimate_prompt_tokens(collected),
            },
            extra_sections={
                "general_background_output": collected,
            },
        )
        return collected

    @classmethod
    def _sanitize_assistant_source_documents(cls, value: Any) -> List[Dict[str, Any]]:
        sanitized: List[Dict[str, Any]] = []
        if not isinstance(value, list):
            return sanitized
        for item in value:
            if not isinstance(item, dict):
                continue
            payload = dict(item)
            payload["doc_id"] = str(payload.get("doc_id") or "")
            payload["parent_id"] = str(payload.get("parent_id") or "")
            payload["report_type"] = str(payload.get("report_type") or "")
            payload["branch"] = str(payload.get("branch") or "")
            payload["url"] = str(payload.get("url") or "")
            payload["excerpt"] = str(payload.get("excerpt") or "")
            payload["document_date"] = cls._safe_int(payload.get("document_date")) or 0
            try:
                payload["score"] = float(payload.get("score") or 0.0)
            except Exception:
                payload["score"] = 0.0
            payload["access"] = bool(payload.get("access"))
            sanitized.append(payload)
        return sanitized

    @classmethod
    def _sanitize_assistant_response_payload(cls, assistant_response: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(assistant_response or {})
        payload["source_documents"] = cls._sanitize_assistant_source_documents(payload.get("source_documents"))
        return payload

    async def save_assistant_response(self, assistant_response: Dict[str, Any]) -> bool:
        safe_assistant_response = self._sanitize_assistant_response_payload(assistant_response)
        retries = max(0, int(getattr(settings, "rag_save_assistant_response_retries", 2)))
        timeout = aiohttp.ClientTimeout(
            total=float(getattr(settings, "rag_save_assistant_response_timeout_s", 10.0))
        )
        last_error: Any = None
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for attempt in range(retries + 1):
                try:
                    async with session.post(settings.assistant_resp_endpoint, json=safe_assistant_response) as response:
                        raw_body = await response.text()
                        if response.status != 200:
                            last_error = f"status={response.status} body={raw_body[:400]}"
                            logger.warning(
                                "Failed to save assistant response message_id=%s attempt=%s/%s status=%s body=%s",
                                safe_assistant_response.get("message_id"),
                                attempt + 1,
                                retries + 1,
                                response.status,
                                raw_body[:400],
                            )
                            if response.status not in {429, 500, 502, 503, 504}:
                                return False
                            if attempt < retries:
                                await asyncio.sleep(0.2 * (attempt + 1))
                                continue
                            return False

                        try:
                            result = json.loads(raw_body or "{}")
                        except Exception:
                            last_error = f"invalid_json body={raw_body[:400]}"
                            logger.warning(
                                "Invalid save assistant response payload message_id=%s body=%s",
                                safe_assistant_response.get("message_id"),
                                raw_body[:400],
                            )
                            return False

                        ok = bool(isinstance(result, dict) and result.get("status") == "ok")
                        if ok:
                            if attempt > 0:
                                logger.info(
                                    "Saved assistant response after retry message_id=%s attempt=%s/%s",
                                    safe_assistant_response.get("message_id"),
                                    attempt + 1,
                                    retries + 1,
                                )
                            return True

                        last_error = f"unexpected_payload={raw_body[:400]}"
                        logger.warning(
                            "Assistant response save endpoint returned non-ok payload message_id=%s attempt=%s/%s body=%s",
                            safe_assistant_response.get("message_id"),
                            attempt + 1,
                            retries + 1,
                            raw_body[:400],
                        )
                        if attempt < retries:
                            await asyncio.sleep(0.2 * (attempt + 1))
                            continue
                        return False
                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        "Failed to save assistant response message_id=%s attempt=%s/%s error=%s",
                        safe_assistant_response.get("message_id"),
                        attempt + 1,
                        retries + 1,
                        exc,
                    )
                    if attempt < retries:
                        await asyncio.sleep(0.2 * (attempt + 1))
                        continue
                    break
        logger.error(
            "Assistant response save failed permanently message_id=%s attempts=%s last_error=%s",
            safe_assistant_response.get("message_id"),
            retries + 1,
            last_error,
        )
        return False

    def _extract_access_codes(self, payload: Any) -> List[str]:
        # changed: parse the user's access-code list from the new gateway endpoint response.
        if isinstance(payload, dict):
            for key in ("access_branches", "branches", "data", "result", "value"):
                value = payload.get(key)
                if isinstance(value, dict):
                    nested = self._extract_access_codes(value)
                    if nested:
                        return nested
                elif isinstance(value, list):
                    codes = [str(item).strip() for item in value if str(item).strip()]
                    if codes:
                        return codes
                elif isinstance(value, str) and value.strip():
                    try:
                        decoded = json.loads(value)
                    except Exception:
                        decoded = [item.strip() for item in value.split(",") if item.strip()]
                    nested = self._extract_access_codes(decoded)
                    if nested:
                        return nested
        if isinstance(payload, list):
            return [str(item).strip() for item in payload if str(item).strip()]
        if isinstance(payload, str) and payload.strip():
            return [item.strip() for item in payload.split(",") if item.strip()]
        return []

    def _extract_user_level(self, payload: Any) -> str:
        if isinstance(payload, dict):
            for key in ("user_level", "userLevel", "level"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            for value in payload.values():
                nested = self._extract_user_level(value)
                if nested:
                    return nested
        if isinstance(payload, list):
            for item in payload:
                nested = self._extract_user_level(item)
                if nested:
                    return nested
        return ""

    @staticmethod
    def _is_privileged_user_level(user_level: Any) -> bool:
        normalized = re.sub(r"[_\-\s]+", " ", str(user_level or "").strip().lower())
        return normalized in {"super user", "super admin", "superuser", "superadmin"}

    async def _fetch_user_access_codes(self, *, user_id: str) -> Tuple[set[str], bool]:
        # changed: fetch the user's access codes once per answer and reuse them for every emitted source document.
        normalized_user_id = str(user_id or "").strip()
        if not normalized_user_id:
            logger.warning("RAG source access skipped because user_id is empty")
            return set(), False

        endpoint = str(getattr(settings, "rag_user_access_branches_endpoint", "") or "").strip()
        if not endpoint:
            logger.warning("RAG user access branches endpoint is not configured; defaulting to empty access set")
            return set(), False

        payload = {"user_id": normalized_user_id}
        retries = max(0, int(getattr(settings, "rag_file_access_retries", 1)))
        timeout = aiohttp.ClientTimeout(total=float(getattr(settings, "rag_file_access_timeout_s", 6.0)))
        last_error: Exception | None = None

        async with aiohttp.ClientSession(timeout=timeout) as session:
            for attempt in range(retries + 1):
                try:
                    async with session.post(endpoint, params=payload) as response:
                        raw_body = await response.text()
                        response.raise_for_status()
                        try:
                            parsed_body = json.loads(raw_body)
                        except Exception:
                            parsed_body = raw_body.strip()
                        user_level = self._extract_user_level(parsed_body)
                        access_override = self._is_privileged_user_level(user_level)
                        access_codes = {
                            str(value).strip().lower()
                            for value in self._extract_access_codes(parsed_body)
                            if str(value).strip()
                        }
                        logger.info(
                            "RAG user access codes fetched user_id=%s user_level=%s override=%s count=%s sample=%s",
                            normalized_user_id,
                            user_level,
                            access_override,
                            len(access_codes),
                            sorted(access_codes)[:10],
                        )
                        return access_codes, access_override
                except Exception as exc:
                    last_error = exc
                    if attempt < retries:
                        await asyncio.sleep(0.15 * (attempt + 1))

        logger.warning(
            "RAG user access code lookup failed user_id=%s error=%s defaulting to empty access set",
            normalized_user_id,
            last_error,
        )
        return set(), False

    async def _rag(self, payload: Dict[str, Any], stream_name: str) -> Tuple[str, List[Dict[str, Any]]]:
        query = self._normalize_query(payload)
        search_modes = self.support.search_mode_tokens(payload.get("search_mode"))
        social_meta_kind = self._classify_social_meta_query(query)
        if social_meta_kind:
            response_text = self._social_meta_response(social_meta_kind)
            logger.info(
                "RAG social-meta short-circuit message_id=%s kind=%s query=%s",
                payload.get("message_id"),
                social_meta_kind,
                self._truncate_text(query, 120),
            )
            await self._set_ui_state(
                message_id=str(payload.get("message_id") or "").strip(),
                ui="sending_to_llm",
                ui_detail="responding without retrieval",
            )
            streamed_text = await self._emit_plain_response(
                payload["message_id"],
                stream_name,
                response_text,
                finalize=True,
            )
            return streamed_text, []
        kb_chunks, chat_ctx, retrieval_plan = await self._fetch_rag_context(payload)
        if not kb_chunks:
            if bool((retrieval_plan or {}).get("chat_context_only")):
                system_prompt = build_chat_context_system_prompt()
                user_prompt = build_chat_context_answer_text(
                    query=query,
                    chat_context=chat_ctx,
                    max_chat_messages=int(settings.rag_max_chat_messages),
                    resolved_query=str(retrieval_plan.get("primary_query") or "").strip(),
                    retrieval_focus=str(
                        retrieval_plan.get("focus_subject")
                        or retrieval_plan.get("focus_hint")
                        or ""
                    ).strip(),
                )
                await self._append_context_trace(
                    message_id=payload["message_id"],
                    query=query,
                    user_prompt=user_prompt,
                    kb_chunks=[],
                    retrieval_plan={
                        **dict(retrieval_plan or {}),
                        "_context_trace_note": "Chat-context-only transform follow-up; vector retrieval was skipped.",
                    },
                )
                await self._set_ui_state(
                    message_id=str(payload.get("message_id") or "").strip(),
                    ui="sending_to_llm",
                    ui_detail="answering from recent chat context",
                )
                answer, source_documents = await self._llm_generate(
                    message_id=payload["message_id"],
                    stream=stream_name,
                    system_prompt=system_prompt,
                    user_query=user_prompt,
                    prompt_chunks=[],
                    emit_sources=False,
                    finalize=True,
                    prefix_text=self._bigdata_answer_heading(),
                )
                return answer, source_documents
            text = self.support.build_no_context_message(retrieval_plan=retrieval_plan)
            await self._append_context_trace(
                message_id=payload["message_id"],
                query=query,
                user_prompt="",
                kb_chunks=[],
                retrieval_plan={
                    **dict(retrieval_plan or {}),
                    "_context_trace_note": "No KB chunks retrieved; DB-grounded answer LLM prompt was not built.",
                },
            )
            add_background, background_reason = self._should_add_general_background(
                query=query,
                db_answer=text,
                retrieval_plan=retrieval_plan,
                has_kb_context=False,
                search_modes=search_modes,
            )
            logger.info(
                "RAG general-background decision message_id=%s add=%s reason=%s has_kb_context=%s",
                payload["message_id"],
                add_background,
                background_reason,
                False,
            )
            await self._set_ui_state(
                message_id=str(payload.get("message_id") or "").strip(),
                ui="sending_to_llm",
                ui_detail="preparing response from retrieved context",
            )
            streamed_text = await self._emit_plain_response(
                payload["message_id"],
                stream_name,
                text,
                finalize=not add_background,
            )
            if add_background:
                streamed_text += await self._append_general_background(
                    message_id=payload["message_id"],
                    stream=stream_name,
                    query=query,
                    db_answer=text,
                    reason=background_reason,
                    retrieval_plan=retrieval_plan,
                )
                await self._finalize_stream(message_id=payload["message_id"], stream=stream_name)
            return streamed_text, []
        system_prompt = build_rag_system_prompt()
        user_prompt = build_rag_context_text(
            query=query,
            kb_chunks=kb_chunks,
            chat_context=chat_ctx,
            max_chunk_chars=int(settings.rag_max_chunk_chars),
            max_chat_messages=int(settings.rag_max_chat_messages),
            resolved_query=str(retrieval_plan.get("primary_query") or "").strip(),
            retrieval_focus=str(retrieval_plan.get("focus_subject") or retrieval_plan.get("focus_hint") or "").strip(),
            applied_time_filter=str((retrieval_plan.get("time_filter") or {}).get("label") or "").strip(),
            coverage_guidance=str(retrieval_plan.get("coverage_guidance") or "").strip(),
        )
        await self._append_context_trace(
            message_id=payload["message_id"],
            query=query,
            user_prompt=user_prompt,
            kb_chunks=kb_chunks,
            retrieval_plan=retrieval_plan,
        )
        # changed: compute source-document access from the chunk ACL fields using one fetched user access-code set.
        user_access_codes, access_override = await self._fetch_user_access_codes(
            user_id=str(payload.get("user_id") or "").strip()
        )
        await self._set_ui_state(
            message_id=str(payload.get("message_id") or "").strip(),
            ui="sending_to_llm",
            ui_detail=f"sending {len(kb_chunks)} chunks to answer model",
        )
        answer, source_documents = await self._llm_generate(
            message_id=payload["message_id"],
            stream=stream_name,
            system_prompt=system_prompt,
            user_query=user_prompt,
            prompt_chunks=kb_chunks,
            user_access_codes=user_access_codes,
            access_override=access_override,
            emit_sources=False,
            finalize=False,
            prefix_text=self._bigdata_answer_heading(),
        )
        add_background, background_reason = self._should_add_general_background(
            query=query,
            db_answer=answer,
            retrieval_plan=retrieval_plan,
            has_kb_context=True,
            search_modes=search_modes,
        )
        logger.info(
            "RAG general-background decision message_id=%s add=%s reason=%s has_kb_context=%s",
            payload["message_id"],
            add_background,
            background_reason,
            True,
        )
        if add_background:
            answer += await self._append_general_background(
                message_id=payload["message_id"],
                stream=stream_name,
                query=query,
                db_answer=answer,
                reason=background_reason,
                retrieval_plan=retrieval_plan,
            )
        await self._emit_source_documents(stream=stream_name, source_documents=source_documents)
        await self._finalize_stream(message_id=payload["message_id"], stream=stream_name)
        return answer, source_documents

    async def process(self, job_id: str, data: Dict[str, Any]) -> bool:
        del job_id
        payload: Dict[str, Any] = {}
        stream_name = ""
        message_id = ""
        try:
            wrapper = json.loads(data.get("data", "{}"))
            if not isinstance(wrapper, dict):
                raise RuntimeError("Invalid job wrapper; expected JSON object")
            payload = wrapper.get("payload") or {}
            if not payload and {"message_id", "user_id", "chat_id"} <= set(wrapper.keys()):
                payload = wrapper
            if not payload:
                raise RuntimeError("Missing payload in routed message")
            message_id = str(payload.get("message_id") or "").strip()
            if not message_id:
                raise RuntimeError("Payload missing message_id")
            stream_name = await self._ensure_task_record(message_id=message_id, payload=payload)
            if await self._is_task_cancelled(message_id):
                await self._delete_cancelled_response_stream(message_id=message_id, stream=stream_name)
                return False

            payload.setdefault("content", "")
            payload.setdefault("attachments", [])
            payload.setdefault("has_attachments", bool(payload.get("attachments")))
            payload.setdefault("search_mode", "assistant")
            payload.setdefault("seq", 0)

            system_response_id = str(uuid.uuid4())
            await self._hset_task(
                message_id,
                {
                    "system_resp_id": system_response_id,
                    "status": "answering",
                    "current_stage": "rag_agent",
                    "ui": "planning",
                    "ui_detail": "building retrieval plan",
                    "ui_detailed": "building retrieval plan",
                    "stream": stream_name,
                    "updated_at": self._task_timestamp(),
                },
            )

            rec = await self.redis.hgetall(f"task:{message_id}")
            stream_name = str(rec.get("stream", "") or "").strip() or stream_name or f"streaming:resp:{message_id}"
            if not stream_name:
                raise RuntimeError(f"Stream name not found for {message_id}")

            assistant_resp, source_documents = await self._rag(payload, stream_name)
            llm_request = await self._pop_last_final_llm_request(message_id)
            if llm_request:
                self._schedule_final_llm_response_log(
                    payload=payload,
                    output=assistant_resp,
                    assistant_message_id=system_response_id,
                    llm_request=llm_request,
                )
            saved = await self.save_assistant_response(
                {
                    "user_id": payload["user_id"],
                    "chat_id": payload["chat_id"],
                    "message_id": system_response_id,
                    "role": "system",
                    "content": assistant_resp,
                    "seq": int(payload["seq"]) + 1,
                    "search_mode": payload["search_mode"],
                    "source_documents": source_documents,
                }
            )
            if not saved:
                logger.error(
                    "RAG assistant response streamed but failed to persist user_message_id=%s assistant_message_id=%s content_chars=%s source_documents=%s",
                    message_id,
                    system_response_id,
                    len(str(assistant_resp or "")),
                    len(list(source_documents or [])),
                )
                await self._hset_task(
                    message_id,
                    {
                        "ui_detail": "response streamed but failed to save",
                        "ui_detailed": "response streamed but failed to save",
                        "updated_at": self._task_timestamp(),
                    },
                )
            return True
        except Exception as exc:
            if message_id and await self._is_task_cancelled(message_id):
                await self._delete_cancelled_response_stream(message_id=message_id, stream=stream_name)
                try:
                    await self._pop_last_final_llm_request(message_id)
                except Exception:
                    pass
                return False
            logger.exception("RAGWorker process failed message_id=%s: %s", message_id or "na", exc)
            if message_id:
                await self._hset_task(
                    message_id,
                    {
                        "status": "failed",
                        "ui": "failed",
                        "ui_detail": "request failed",
                        "ui_detailed": "request failed",
                        "updated_at": self._task_timestamp(),
                    },
                )
            if message_id:
                stream_name = str(stream_name or "").strip() or f"streaming:resp:{message_id}"
                try:
                    await self._emit_plain_response(
                        message_id,
                        stream_name,
                        "I hit an internal issue while processing your retrieval request. Please try again.",
                        finalize=True,
                    )
                    await self._hset_task(
                        message_id,
                        {
                            "status": "failed",
                            "ui": "failed",
                            "ui_detail": "request failed",
                            "ui_detailed": "request failed",
                            "updated_at": self._task_timestamp(),
                        },
                    )
                except Exception as emit_exc:
                    logger.error("Failed to stream fallback for message_id=%s: %s", message_id, emit_exc)
            if message_id:
                try:
                    await self._pop_last_final_llm_request(message_id)
                except Exception:
                    pass
            return False
