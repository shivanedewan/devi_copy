import asyncio
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp
from qdrant_client import QdrantClient
from qdrant_client import models as qm

logger = logging.getLogger(__name__)
_EARLIEST_CHAT_QUERY_RE = re.compile(
    r"\b("
    r"first|earliest|oldest|initial|starting|start(?:ed)?"
    r")\b.{0,80}\b("
    r"question|query|message|thing|prompt|ask(?:ed)?|said|conversation|chat"
    r")\b|"
    r"\b("
    r"question|query|message|thing|prompt|ask(?:ed)?|said"
    r")\b.{0,80}\b("
    r"first|earliest|oldest|initial"
    r")\b",
    flags=re.IGNORECASE,
)
_CHAT_MATCH_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9._/-]{2,}", flags=re.IGNORECASE)
_CHAT_MATCH_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    flags=re.IGNORECASE,
)
_CHAT_MATCH_QUOTED_RE = re.compile(r'"([^"\n]{4,160})"')
_CHAT_REFERENCE_QUERY_RE = re.compile(
    r"\b("
    r"pasted|paste|mentioned|mention|sent|shared|wrote|typed|said|"
    r"above|earlier|before|previous|prior|that|this"
    r")\b",
    flags=re.IGNORECASE,
)
_CHAT_LEXICAL_STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "above",
    "about",
    "below",
    "before",
    "brief",
    "briefly",
    "bullet",
    "bullets",
    "continue",
    "detail",
    "details",
    "earlier",
    "explain",
    "for",
    "from",
    "give",
    "his",
    "her",
    "how",
    "i",
    "in",
    "into",
    "is",
    "it",
    "its",
    "line",
    "lines",
    "me",
    "mention",
    "mentioned",
    "more",
    "my",
    "of",
    "on",
    "or",
    "our",
    "paragraph",
    "pasted",
    "please",
    "previous",
    "prior",
    "query",
    "question",
    "regarding",
    "relation",
    "relevance",
    "said",
    "same",
    "sent",
    "shared",
    "show",
    "some",
    "summary",
    "summarize",
    "summarise",
    "tell",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "us",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "without",
    "wrote",
    "you",
}


def _env_int(name: str, default: int) -> int:
    try:
        value = str(os.getenv(name, "")).strip()
        return int(value) if value else int(default)
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        value = str(os.getenv(name, "")).strip()
        return float(value) if value else float(default)
    except Exception:
        return float(default)


def _recency_boost(created_at_epoch: int, now_epoch: int) -> float:
    age_seconds = max(0, now_epoch - created_at_epoch)
    return 1.0 / (1.0 + (age_seconds / 3600.0))


def created_at_to_epoch(created_at: Any) -> int:
    if created_at is None:
        return 0

    try:
        if isinstance(created_at, (int, float)):
            return int(created_at)

        value = str(created_at).strip()
        if not value:
            return 0

        # Be tolerant to "YYYY-MM-DD HH:MM:SS..." payloads in older runtimes.
        if " " in value and "T" not in value:
            value = value.replace(" ", "T", 1)
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return 0


def _seq_to_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _message_order_key(payload: Dict[str, Any]) -> tuple:
    seq = _seq_to_int(payload.get("seq"))
    if seq is not None:
        return (0, seq, created_at_to_epoch(payload.get("created_at")), str(payload.get("message_id", "")))
    return (1, created_at_to_epoch(payload.get("created_at")), str(payload.get("message_id", "")))


def _is_junk_text(text: str, min_chars: int = 8) -> bool:
    content = (text or "").strip()
    return (not content) or (len(content) < min_chars)


def _normalize_match_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _extract_match_tokens(text: Any) -> List[str]:
    return [match.group(0).lower() for match in _CHAT_MATCH_TOKEN_RE.finditer(str(text or ""))]


class Embedder:
    def __init__(self, embed_endpoint: str, model_name: str = "qwen-embed_8b"):
        self.embed_endpoint = embed_endpoint
        self.model_name = str(model_name or "qwen-embed_8b")
        self.timeout = aiohttp.ClientTimeout(total=15)
        self.use_query_instruction = os.getenv(
            "EMBED_QUERY_INSTRUCTION_ENABLED",
            "true",
        ).strip().lower() in {"1", "true", "t", "yes", "y"}
        self.query_instruction = str(
            os.getenv(
                "EMBED_QUERY_INSTRUCTION",
                "Given a user query, retrieve relevant passages from internal documents, uploaded files, or conversation history that answer the query.",
            )
            or ""
        ).strip()

    def _prepare_query_texts(self, texts: List[str]) -> List[str]:
        if not self.use_query_instruction or not self.query_instruction:
            return texts
        prepared: List[str] = []
        for text in texts:
            normalized = str(text or "").strip()
            if not normalized:
                continue
            lowered = normalized.lower()
            if lowered.startswith("instruct:") or lowered.startswith("<instruct>:"):
                prepared.append(normalized)
                continue
            prepared.append(f"Instruct: {self.query_instruction}\nQuery: {normalized}")
        return prepared

    async def generate_embeddings(self, payload: Dict[str, Any]) -> Optional[List[float]]:
        try:
            raw_input = payload.get("input")
            if isinstance(raw_input, list):
                texts = [str(item).strip() for item in raw_input if str(item or "").strip()]
            else:
                text = str(
                    payload.get("content")
                    or payload.get("text")
                    or payload.get("query")
                    or raw_input
                    or ""
                ).strip()
                texts = [text] if text else []

            if not texts:
                logger.error("Embedding request payload missing input text")
                return None
            texts = self._prepare_query_texts(texts)

            request_payload = {
                "model": str(payload.get("model") or self.model_name),
                "input": texts,
            }

            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(self.embed_endpoint, json=request_payload) as response:
                    if response.status != 200:
                        body = await response.text()
                        logger.error("Embedding request failed status=%s body=%s", response.status, body)
                        return None

                    result = await response.json()
                    if (
                        isinstance(result, dict)
                        and isinstance(result.get("data"), list)
                        and result["data"]
                        and isinstance(result["data"][0], dict)
                        and isinstance(result["data"][0].get("embedding"), list)
                    ):
                        return result["data"][0]["embedding"]

                    logger.error("Embedding response format invalid")
                    return None
        except Exception as exc:
            logger.exception("Embedding request error: %s", exc)
            return None


@dataclass
class Config:
    chat_collection: str = "chat_messages_production_1"
    semantic_threshold: float = 0.4
    top_n_chats: int = 12
    tail_n: int = 6
    semantic_trigger_count: int = 10
    recent_window_size: int = 10
    max_session_messages: int = 1000
    tail_max_turns: int = _env_int("CHAT_CONTEXT_TAIL_MAX_TURNS", 10)
    tail_token_budget: int = _env_int("CHAT_CONTEXT_TAIL_TOKEN_BUDGET", 8000)
    semantic_history_max_turns: int = _env_int("CHAT_CONTEXT_SEMANTIC_HISTORY_MAX_TURNS", 8)
    semantic_history_token_budget: int = _env_int("CHAT_CONTEXT_SEMANTIC_HISTORY_TOKEN_BUDGET", 4000)
    total_token_budget: int = _env_int("CHAT_CONTEXT_TOTAL_TOKEN_BUDGET", 12000)
    chars_per_token: float = _env_float("CHAT_CONTEXT_CHARS_PER_TOKEN", 4.0)
    include_chronological_anchors: bool = os.getenv(
        "CHAT_CONTEXT_INCLUDE_CHRONOLOGICAL_ANCHORS",
        "true",
    ).strip().lower() in {"1", "true", "t", "yes", "y"}
    chronological_anchor_turns: int = _env_int("CHAT_CONTEXT_CHRONOLOGICAL_ANCHOR_TURNS", 2)


class ContextFetcher:
    def __init__(self, qdrant: QdrantClient, embedder: Embedder, cfg: Optional[Config] = None):
        self.qdrant = qdrant
        self.embedder = embedder
        self.cfg = cfg or Config()

    async def fetch(
        self,
        *,
        user_id: str,
        chat_id: str,
        message_id: str,
        query: str,
        top_n: int = 9,
        semantic_threshold: Optional[float] = None,
        enable_semantic_search: bool = True,
        include_docs: bool = False,
        include_history: bool = False,
        include_memories: bool = False,
        tail_max_turns: Optional[int] = None,
        tail_token_budget: Optional[int] = None,
        semantic_history_max_turns: Optional[int] = None,
        semantic_history_token_budget: Optional[int] = None,
        total_token_budget: Optional[int] = None,
        chars_per_token: Optional[float] = None,
        include_chronological_anchors: Optional[bool] = None,
        chronological_anchor_turns: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        del include_docs, include_history, include_memories

        if not user_id or not chat_id or not message_id:
            raise ValueError("user_id, chat_id and message_id are required")
        normalized_query = str(query or "").strip()
        if top_n < 1:
            raise ValueError("top_n must be >= 1")
        if semantic_threshold is not None and semantic_threshold < 0:
            raise ValueError("semantic_threshold must be >= 0")
        effective_enable_semantic_search = bool(enable_semantic_search and normalized_query)

        logger.info(
            "chat_context fetch request user_id=%s chat_id=%s message_id=%s query_len=%s top_n=%s semantic_threshold=%s enable_semantic_search=%s",
            user_id,
            chat_id,
            message_id,
            len(normalized_query),
            top_n,
            semantic_threshold,
            effective_enable_semantic_search,
        )

        budget_cfg = self._resolve_context_budget(
            top_n=top_n,
            tail_max_turns=tail_max_turns,
            tail_token_budget=tail_token_budget,
            semantic_history_max_turns=semantic_history_max_turns,
            semantic_history_token_budget=semantic_history_token_budget,
            total_token_budget=total_token_budget,
            chars_per_token=chars_per_token,
            include_chronological_anchors=include_chronological_anchors,
            chronological_anchor_turns=chronological_anchor_turns,
        )
        recent_budget = max(self.cfg.recent_window_size, top_n, budget_cfg["tail_max_turns"] * 2)
        if not effective_enable_semantic_search:
            tail_msgs = await self._fetch_tail_by_created_at(
                user_id=user_id,
                chat_id=chat_id,
                message_id=message_id,
                tail_n=min(
                    self.cfg.max_session_messages,
                    max(recent_budget, budget_cfg["tail_max_turns"] * 4, 40),
                ),
            )
            recent_turns = self._take_recent_turns_by_budget(
                self._build_turns(tail_msgs)[0],
                max_turns=budget_cfg["tail_max_turns"],
                token_budget=min(budget_cfg["tail_token_budget"], budget_cfg["total_token_budget"]),
                chars_per_token=budget_cfg["chars_per_token"],
            )
            tagged_recent_turns = self._tag_turns(
                recent_turns,
                scope="recent",
                is_recent=True,
            )
            ordered_recent = self._flatten_turns(tagged_recent_turns)
            logger.info(
                "Returning recent-only chat context chat_id=%s recent_turns=%s returned_messages=%s",
                chat_id,
                len(tagged_recent_turns),
                len(ordered_recent),
            )
            return ordered_recent

        all_msgs = await self._fetch_all_session_messages(
            user_id=user_id,
            chat_id=chat_id,
            message_id=message_id,
            limit=self.cfg.max_session_messages,
        )
        total_msgs = len(all_msgs)
        logger.info(
            "Context fetch start user_id=%s chat_id=%s message_id=%s total_msgs=%s",
            user_id,
            chat_id,
            message_id,
            total_msgs,
        )

        turns, msg_to_turn = self._build_turns(all_msgs)
        logger.info(
            "Built turns chat_id=%s turns=%s messages=%s",
            chat_id,
            len(turns),
            len(all_msgs),
        )

        recent_token_budget = min(budget_cfg["tail_token_budget"], budget_cfg["total_token_budget"])
        recent_turns = self._tag_turns(
            self._take_recent_turns_by_budget(
                turns,
                max_turns=budget_cfg["tail_max_turns"],
                token_budget=recent_token_budget,
                chars_per_token=budget_cfg["chars_per_token"],
            ),
            scope="recent",
            is_recent=True,
        )
        recent_token_count = self._turns_token_count(recent_turns, chars_per_token=budget_cfg["chars_per_token"])
        ordered_recent = self._flatten_turns(recent_turns)
        if total_msgs <= self.cfg.semantic_trigger_count:
            logger.info(
                "Returning recent-only context chat_id=%s total_msgs=%s threshold=%s recent_turns=%s recent_tokens_est=%s returned_messages=%s",
                chat_id,
                total_msgs,
                self.cfg.semantic_trigger_count,
                len(recent_turns),
                recent_token_count,
                len(ordered_recent),
            )
            return ordered_recent

        qvec = await self.embedder.generate_embeddings({"input": [normalized_query]})
        if not qvec:
            raise RuntimeError("Failed to generate embeddings for query")

        recent_turn_ids = {
            int(turn.get("_turn_id"))
            for turn in recent_turns
            if turn.get("_turn_id") is not None
        }
        chronological_turns: List[Dict[str, Any]] = []
        if (
            bool(budget_cfg["include_chronological_anchors"])
            and budget_cfg["chronological_anchor_turns"] > 0
            and self._query_needs_earliest_turns(query)
        ):
            chronological_turns = self._tag_turns(
                self._select_earliest_turns(
                    turns,
                    exclude_turn_ids=recent_turn_ids,
                    max_turns=budget_cfg["chronological_anchor_turns"],
                    token_budget=min(
                        budget_cfg["semantic_history_token_budget"],
                        max(0, budget_cfg["total_token_budget"] - recent_token_count),
                    ),
                    chars_per_token=budget_cfg["chars_per_token"],
                ),
                scope="history",
                is_recent=False,
            )
        chronological_token_count = self._turns_token_count(
            chronological_turns,
            chars_per_token=budget_cfg["chars_per_token"],
        )
        history_exclude_turn_ids = {
            *recent_turn_ids,
            *{
                int(turn.get("_turn_id"))
                for turn in chronological_turns
                if turn.get("_turn_id") is not None
            },
        }
        semantic_hits = await self._fetch_session_chats(
            user_id=user_id,
            chat_id=chat_id,
            message_id=message_id,
            qvec=qvec,
            top_n=max(top_n * 3, budget_cfg["semantic_history_max_turns"] * 4, 20),
            semantic_threshold=semantic_threshold,
        )
        logger.info(
            "chat_context semantic candidates chat_id=%s candidates=%s recent_turns=%s",
            chat_id,
            len(semantic_hits),
            len(recent_turns),
        )
        lexical_turn_hits = self._build_lexical_turn_hits(
            turns=turns,
            exclude_turn_ids=history_exclude_turn_ids,
            query=query,
        )
        logger.info(
            "chat_context lexical candidates chat_id=%s candidates=%s recent_turns=%s",
            chat_id,
            len(lexical_turn_hits),
            len(recent_turns),
        )
        semantic_turns = self._select_semantic_turns(
            turns=turns,
            msg_to_turn=msg_to_turn,
            semantic_hits=semantic_hits,
            lexical_turn_hits=lexical_turn_hits,
            exclude_turn_ids=history_exclude_turn_ids,
            max_turns=budget_cfg["semantic_history_max_turns"],
            token_budget=min(
                budget_cfg["semantic_history_token_budget"],
                max(0, budget_cfg["total_token_budget"] - recent_token_count - chronological_token_count),
            ),
            chars_per_token=budget_cfg["chars_per_token"],
        )
        history_turns = [*chronological_turns, *semantic_turns]
        combined_turns = self._merge_turns(recent_turns=recent_turns, semantic_turns=history_turns)
        final_msgs = self._flatten_turns(combined_turns)
        logger.info(
            "Returning hybrid turn-level context chat_id=%s recent_turns=%s chronological_turns=%s semantic_turns=%s lexical_candidate_turns=%s recent_tokens_est=%s chronological_tokens_est=%s semantic_tokens_est=%s total_tokens_est=%s returned_messages=%s",
            chat_id,
            len(recent_turns),
            len(chronological_turns),
            len(semantic_turns),
            len(lexical_turn_hits),
            recent_token_count,
            chronological_token_count,
            self._turns_token_count(semantic_turns, chars_per_token=budget_cfg["chars_per_token"]),
            self._turns_token_count(combined_turns, chars_per_token=budget_cfg["chars_per_token"]),
            len(final_msgs),
        )
        return final_msgs

    async def _count_session_messages(
        self,
        *,
        user_id: str,
        chat_id: str,
        message_id: str,
    ) -> int:
        chat_filter = qm.Filter(
            must=[
                qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id)),
                qm.FieldCondition(key="chat_id", match=qm.MatchValue(value=chat_id)),
            ],
            must_not=[qm.FieldCondition(key="message_id", match=qm.MatchValue(value=message_id))],
        )

        try:
            result = await asyncio.to_thread(
                self.qdrant.count,
                collection_name=self.cfg.chat_collection,
                count_filter=chat_filter,
                exact=True,
            )
        except Exception as exc:
            logger.exception("Qdrant count query failed: %s", exc)
            raise RuntimeError("Failed to count chat messages") from exc

        return int(getattr(result, "count", 0) or 0)

    async def _fetch_session_chats(
        self,
        *,
        user_id: str,
        chat_id: str,
        message_id: str,
        qvec: List[float],
        top_n: int,
        semantic_threshold: Optional[float] = None,
        exclude_message_ids: Optional[set[str]] = None,
    ) -> List[Dict[str, Any]]:
        if top_n <= 0:
            return []
        excluded_ids = {str(i) for i in (exclude_message_ids or set())}
        threshold = self.cfg.semantic_threshold if semantic_threshold is None else float(semantic_threshold)

        chat_filter = qm.Filter(
            must=[
                qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id)),
                qm.FieldCondition(key="chat_id", match=qm.MatchValue(value=chat_id)),
            ],
            must_not=[qm.FieldCondition(key="message_id", match=qm.MatchValue(value=message_id))],
        )

        try:
            hits = await asyncio.to_thread(
                self.qdrant.query_points,
                collection_name=self.cfg.chat_collection,
                query=qvec,
                query_filter=chat_filter,
                limit=top_n,
                with_vectors=False,
                with_payload=True,
            )
        except Exception as exc:
            logger.exception("Qdrant semantic query failed: %s", exc)
            raise RuntimeError("Failed to fetch semantic chat context") from exc

        raw_points = list(getattr(hits, "points", []) or [])
        raw_top_score = float(raw_points[0].score) if raw_points else 0.0
        semantic: List[Dict[str, Any]] = []
        for point in raw_points:
            score = float(point.score)
            if score < threshold:
                continue

            payload = dict(point.payload or {})
            msg_id = payload.get("message_id")
            if msg_id and str(msg_id) in excluded_ids:
                continue
            payload["_sim_score"] = score
            payload["_scope"] = "semantic"
            semantic.append(payload)
            if len(semantic) >= top_n:
                break
        logger.info(
            "chat_context semantic raw_hits=%s accepted=%s threshold=%s raw_top_score=%.6f",
            len(raw_points),
            len(semantic),
            threshold,
            raw_top_score,
        )

        return semantic

    async def _fetch_tail_by_created_at(
        self,
        *,
        user_id: str,
        chat_id: str,
        message_id: str,
        tail_n: int,
        exclude_message_ids: Optional[set[str]] = None,
    ) -> List[Dict[str, Any]]:
        if tail_n <= 0:
            return []
        excluded_ids = {str(i) for i in (exclude_message_ids or set())}

        chat_filter = qm.Filter(
            must=[
                qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id)),
                qm.FieldCondition(key="chat_id", match=qm.MatchValue(value=chat_id)),
            ],
            must_not=[qm.FieldCondition(key="message_id", match=qm.MatchValue(value=message_id))],
        )

        try:
            hits, _ = await asyncio.to_thread(
                self.qdrant.scroll,
                collection_name=self.cfg.chat_collection,
                limit=tail_n,
                order_by=qm.OrderBy(key="created_at", direction=qm.Direction.DESC),
                scroll_filter=chat_filter,
            )
        except Exception as exc:
            logger.exception("Qdrant tail scroll failed: %s", exc)
            raise RuntimeError("Failed to fetch recent chat context") from exc

        ordered: List[Dict[str, Any]] = []
        for hit in hits:
            payload = dict(hit.payload or {})
            msg_id = payload.get("message_id")
            if msg_id and str(msg_id) in excluded_ids:
                continue
            ordered.append(payload)
            if len(ordered) >= tail_n:
                break
        return ordered

    async def _fetch_all_session_messages(
        self,
        *,
        user_id: str,
        chat_id: str,
        message_id: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []

        chat_filter = qm.Filter(
            must=[
                qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id)),
                qm.FieldCondition(key="chat_id", match=qm.MatchValue(value=chat_id)),
            ],
            must_not=[qm.FieldCondition(key="message_id", match=qm.MatchValue(value=message_id))],
        )

        try:
            hits, _ = await asyncio.to_thread(
                self.qdrant.scroll,
                collection_name=self.cfg.chat_collection,
                limit=limit,
                order_by=qm.OrderBy(key="created_at", direction=qm.Direction.DESC),
                scroll_filter=chat_filter,
            )
        except Exception as exc:
            logger.exception("Qdrant session scroll failed: %s", exc)
            raise RuntimeError("Failed to fetch chat session messages") from exc

        msgs: List[Dict[str, Any]] = []
        for hit in hits:
            payload = dict(hit.payload or {})
            message_key = payload.get("message_id")
            if not message_key:
                logger.warning("Skipping payload without message_id chat_id=%s payload=%s", chat_id, payload)
                continue
            if _is_junk_text(payload.get("content", "")):
                continue
            payload["_scope"] = "tail"
            payload["_sim_score"] = 0.0
            msgs.append(payload)

        if len(msgs) >= limit:
            logger.warning(
                "Session messages reached cap chat_id=%s cap=%s fetched=%s older_messages_truncated=true",
                chat_id,
                limit,
                len(msgs),
            )
        return msgs

    def _merge_dedup(
        self,
        *,
        tail_msgs: List[Dict[str, Any]],
        semantic_msgs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}

        for msg in tail_msgs:
            payload = dict(msg)
            if _is_junk_text(payload.get("content", "")):
                continue

            key = payload.get("message_id")
            if not key:
                continue
            payload["_scope"] = "tail"
            payload["_sim_score"] = 0.0
            merged[key] = payload

        for msg in semantic_msgs:
            payload = dict(msg)
            if _is_junk_text(payload.get("content", "")):
                continue

            key = payload.get("message_id")
            if not key:
                continue
            payload["_scope"] = payload.get("_scope", "semantic")
            payload["_sim_score"] = float(payload.get("_sim_score", 0.0))

            if key in merged:
                merged[key]["_sim_score"] = max(
                    float(merged[key].get("_sim_score", 0.0)),
                    float(payload.get("_sim_score", 0.0)),
                )
            else:
                merged[key] = payload

        return list(merged.values())

    def _rank_messages(
        self,
        msgs: List[Dict[str, Any]],
        w_sim: float = 1.0,
        w_recency: float = 0.6,
    ) -> List[Dict[str, Any]]:
        now_ts = int(time.time())

        for payload in msgs:
            sim = float(payload.get("_sim_score", 0.0))
            created_at = created_at_to_epoch(payload.get("created_at"))
            payload["_final_score"] = (w_sim * sim) + (w_recency * _recency_boost(created_at, now_ts))

        msgs.sort(key=lambda p: float(p.get("_final_score", 0.0)), reverse=True)
        return msgs

    def _build_turns(
        self,
        msgs: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
        sorted_msgs = sorted(
            msgs,
            key=_message_order_key,
        )

        turns: List[Dict[str, Any]] = []
        msg_to_turn: Dict[str, int] = {}
        idx = 0
        turn_id = 0
        while idx < len(sorted_msgs):
            current = dict(sorted_msgs[idx])
            role = str(current.get("role", "")).lower()
            next_msg = dict(sorted_msgs[idx + 1]) if idx + 1 < len(sorted_msgs) else None
            next_role = str((next_msg or {}).get("role", "")).lower()
            turn_msgs: List[Dict[str, Any]] = []

            if role == "user" and next_msg and next_role in {"system", "assistant"}:
                turn_msgs.append(current)
                turn_msgs.append(next_msg)
                idx += 2
            else:
                turn_msgs.append(current)
                idx += 1
                logger.debug(
                    "Unpaired message in turn build message_id=%s role=%s",
                    current.get("message_id"),
                    role,
                )

            user_anchor = next(
                (created_at_to_epoch(m.get("created_at")) for m in turn_msgs if str(m.get("role", "")).lower() == "user"),
                None,
            )
            turn_created_at = user_anchor if user_anchor is not None else max(
                created_at_to_epoch(m.get("created_at")) for m in turn_msgs
            )
            turn_payload: Dict[str, Any] = {
                "_turn_id": turn_id,
                "_sim_score": 0.0,
                "_created_at_epoch": turn_created_at,
                "messages": turn_msgs,
            }
            turns.append(turn_payload)
            for m in turn_msgs:
                msg_id = m.get("message_id")
                if msg_id:
                    msg_to_turn[str(msg_id)] = turn_id
            turn_id += 1

        return turns, msg_to_turn

    def _resolve_context_budget(
        self,
        *,
        top_n: int,
        tail_max_turns: Optional[int],
        tail_token_budget: Optional[int],
        semantic_history_max_turns: Optional[int],
        semantic_history_token_budget: Optional[int],
        total_token_budget: Optional[int],
        chars_per_token: Optional[float],
        include_chronological_anchors: Optional[bool],
        chronological_anchor_turns: Optional[int],
    ) -> Dict[str, Any]:
        def int_value(raw: Optional[int], fallback: int, *, minimum: int = 0) -> int:
            try:
                value = int(raw if raw is not None else fallback)
            except Exception:
                value = int(fallback)
            return max(minimum, value)

        def float_value(raw: Optional[float], fallback: float, *, minimum: float = 1.0) -> float:
            try:
                value = float(raw if raw is not None else fallback)
            except Exception:
                value = float(fallback)
            return max(minimum, value)

        tail_turns = int_value(tail_max_turns, self.cfg.tail_max_turns, minimum=1)
        semantic_turns = int_value(semantic_history_max_turns, self.cfg.semantic_history_max_turns, minimum=0)
        return {
            "tail_max_turns": tail_turns,
            "tail_token_budget": int_value(tail_token_budget, self.cfg.tail_token_budget, minimum=1),
            "semantic_history_max_turns": semantic_turns,
            "semantic_history_token_budget": int_value(
                semantic_history_token_budget,
                self.cfg.semantic_history_token_budget,
                minimum=0,
            ),
            "total_token_budget": int_value(total_token_budget, self.cfg.total_token_budget, minimum=1),
            "chars_per_token": float_value(chars_per_token, self.cfg.chars_per_token, minimum=1.0),
            "include_chronological_anchors": (
                bool(include_chronological_anchors)
                if include_chronological_anchors is not None
                else bool(self.cfg.include_chronological_anchors)
            ),
            "chronological_anchor_turns": int_value(
                chronological_anchor_turns,
                self.cfg.chronological_anchor_turns,
                minimum=0,
            ),
            "legacy_top_n": max(1, int(top_n)),
        }

    @staticmethod
    def _query_needs_earliest_turns(query: Any) -> bool:
        return bool(_EARLIEST_CHAT_QUERY_RE.search(str(query or "")))

    def _extract_lexical_query_cues(self, query: Any) -> Dict[str, Any]:
        raw_query = str(query or "").strip()
        normalized_query = _normalize_match_text(raw_query)
        raw_tokens = _extract_match_tokens(raw_query)
        terms: List[str] = []
        seen_terms: set[str] = set()
        for token in raw_tokens:
            lowered = token.lower()
            if lowered in _CHAT_LEXICAL_STOPWORDS:
                continue
            if len(lowered) < 3 and not any(ch.isdigit() for ch in lowered):
                continue
            if lowered in seen_terms:
                continue
            seen_terms.add(lowered)
            terms.append(lowered)
        if len(terms) > 10:
            terms = terms[:10]

        phrases: List[str] = []
        seen_phrases: set[str] = set()
        for phrase in _CHAT_MATCH_QUOTED_RE.findall(raw_query):
            normalized_phrase = _normalize_match_text(phrase)
            if normalized_phrase and normalized_phrase not in seen_phrases:
                seen_phrases.add(normalized_phrase)
                phrases.append(normalized_phrase)
        if len(terms) >= 2:
            for size in (3, 2):
                for start in range(0, len(terms) - size + 1):
                    phrase = " ".join(terms[start : start + size]).strip()
                    if len(phrase) < 7 or phrase in seen_phrases:
                        continue
                    seen_phrases.add(phrase)
                    phrases.append(phrase)
                    if len(phrases) >= 8:
                        break
                if len(phrases) >= 8:
                    break

        uuids = [match.group(0).lower() for match in _CHAT_MATCH_UUID_RE.finditer(raw_query)]
        return {
            "normalized_query": normalized_query,
            "terms": terms,
            "phrases": phrases,
            "uuids": uuids,
            "has_reference_language": bool(_CHAT_REFERENCE_QUERY_RE.search(raw_query)),
        }

    def _build_lexical_turn_hits(
        self,
        *,
        turns: List[Dict[str, Any]],
        exclude_turn_ids: set[int],
        query: Any,
    ) -> Dict[int, Dict[str, Any]]:
        cues = self._extract_lexical_query_cues(query)
        terms = list(cues.get("terms") or [])
        phrases = list(cues.get("phrases") or [])
        uuids = list(cues.get("uuids") or [])
        normalized_query = str(cues.get("normalized_query") or "")
        has_reference_language = bool(cues.get("has_reference_language"))
        if not terms and not phrases and not uuids:
            return {}

        hits: Dict[int, Dict[str, Any]] = {}
        for turn in turns:
            try:
                turn_id = int(turn.get("_turn_id"))
            except Exception:
                continue
            if turn_id in exclude_turn_ids:
                continue

            text = " ".join(str(msg.get("content") or "").strip() for msg in list(turn.get("messages") or []))
            normalized_text = _normalize_match_text(text)
            if not normalized_text:
                continue
            token_set = set(_extract_match_tokens(normalized_text))
            matched_terms = [term for term in terms if term in token_set]
            matched_phrases = [phrase for phrase in phrases if phrase in normalized_text]
            matched_uuids = [value for value in uuids if value in normalized_text]
            exact_query_match = bool(normalized_query and len(normalized_query) >= 18 and normalized_query in normalized_text)
            match_count = len(matched_terms)
            term_ratio = (match_count / max(1, len(terms))) if terms else 0.0
            max_term_len = max((len(term) for term in matched_terms), default=0)

            qualifies = bool(
                matched_uuids
                or matched_phrases
                or exact_query_match
                or (match_count >= 2 and term_ratio >= 0.34)
                or (has_reference_language and max_term_len >= 8)
            )
            if not qualifies:
                continue

            lexical_score = 0.0
            if exact_query_match:
                lexical_score += 0.45
            lexical_score += min(0.45, 0.22 * len(matched_uuids))
            lexical_score += min(0.40, 0.18 * len(matched_phrases))
            lexical_score += min(0.55, 0.55 * term_ratio)
            if has_reference_language and max_term_len >= 8:
                lexical_score += 0.10

            if lexical_score <= 0.0:
                continue
            hits[turn_id] = {
                "score": lexical_score,
                "term_hits": matched_terms,
                "phrase_hits": matched_phrases,
                "uuid_hits": matched_uuids,
                "exact_query_match": exact_query_match,
            }
        return hits

    @staticmethod
    def _estimate_text_tokens(text: Any, *, chars_per_token: float) -> int:
        value = str(text or "").strip()
        if not value:
            return 0
        return max(1, int(math.ceil(len(value) / max(1.0, float(chars_per_token)))))

    def _turn_token_count(self, turn: Dict[str, Any], *, chars_per_token: float) -> int:
        total = 0
        for msg in list(turn.get("messages") or []):
            total += self._estimate_text_tokens(msg.get("content"), chars_per_token=chars_per_token)
        if total > 0:
            total += 8 * len(list(turn.get("messages") or []))
        return total

    def _turns_token_count(self, turns: List[Dict[str, Any]], *, chars_per_token: float) -> int:
        return sum(self._turn_token_count(turn, chars_per_token=chars_per_token) for turn in turns)

    def _take_recent_turns_by_budget(
        self,
        turns: List[Dict[str, Any]],
        *,
        max_turns: int,
        token_budget: int,
        chars_per_token: float,
    ) -> List[Dict[str, Any]]:
        if max_turns <= 0 or token_budget <= 0:
            return []
        selected: List[Dict[str, Any]] = []
        used_tokens = 0
        for turn in reversed(turns):
            turn_messages = turn.get("messages", [])
            if not turn_messages:
                continue
            turn_tokens = self._turn_token_count(turn, chars_per_token=chars_per_token)
            if selected and len(selected) >= max_turns:
                break
            if selected and (used_tokens + turn_tokens) > token_budget:
                break
            selected.append(turn)
            used_tokens += turn_tokens
            if len(selected) >= max_turns or used_tokens >= token_budget:
                break
        selected.reverse()
        return selected

    def _select_earliest_turns(
        self,
        turns: List[Dict[str, Any]],
        *,
        exclude_turn_ids: set[int],
        max_turns: int,
        token_budget: int,
        chars_per_token: float,
    ) -> List[Dict[str, Any]]:
        if max_turns <= 0 or token_budget <= 0:
            return []
        selected: List[Dict[str, Any]] = []
        used_tokens = 0
        for turn in turns:
            try:
                turn_id = int(turn.get("_turn_id"))
            except Exception:
                continue
            if turn_id in exclude_turn_ids:
                continue
            if not turn.get("messages"):
                continue
            turn_tokens = self._turn_token_count(turn, chars_per_token=chars_per_token)
            if selected and len(selected) >= max_turns:
                break
            if selected and (used_tokens + turn_tokens) > token_budget:
                break
            selected.append(turn)
            used_tokens += turn_tokens
            if len(selected) >= max_turns or used_tokens >= token_budget:
                break
        return selected

    def _take_recent_turns(
        self,
        turns: List[Dict[str, Any]],
        *,
        max_messages: int,
    ) -> List[Dict[str, Any]]:
        if max_messages <= 0:
            return []
        selected: List[Dict[str, Any]] = []
        budget = 0
        for turn in reversed(turns):
            turn_messages = turn.get("messages", [])
            if not turn_messages:
                continue
            msg_count = len(turn_messages)
            if selected and (budget + msg_count) > max_messages:
                break
            if not selected and msg_count > max_messages:
                selected.append(turn)
                break
            selected.append(turn)
            budget += msg_count
            if budget >= max_messages:
                break
        selected.reverse()
        return selected

    def _tag_turns(
        self,
        turns: List[Dict[str, Any]],
        *,
        scope: str,
        is_recent: bool,
    ) -> List[Dict[str, Any]]:
        tagged: List[Dict[str, Any]] = []
        for turn in turns:
            payload = dict(turn)
            payload["_scope"] = scope
            payload["_is_recent"] = bool(is_recent)
            tagged.append(payload)
        return tagged

    def _select_semantic_turns(
        self,
        *,
        turns: List[Dict[str, Any]],
        msg_to_turn: Dict[str, int],
        semantic_hits: List[Dict[str, Any]],
        lexical_turn_hits: Optional[Dict[int, Dict[str, Any]]] = None,
        exclude_turn_ids: set[int],
        max_turns: int,
        token_budget: int,
        chars_per_token: float,
    ) -> List[Dict[str, Any]]:
        if max_turns <= 0 or token_budget <= 0:
            return []

        lexical_hits = dict(lexical_turn_hits or {})
        scored_turns: Dict[int, float] = {}
        for hit in semantic_hits:
            msg_id = hit.get("message_id")
            if not msg_id:
                continue
            turn_id = msg_to_turn.get(str(msg_id))
            if turn_id is None or turn_id in exclude_turn_ids:
                continue
            sim_score = float(hit.get("_sim_score", 0.0))
            prev_score = scored_turns.get(turn_id, 0.0)
            if sim_score > prev_score:
                scored_turns[turn_id] = sim_score

        candidates: List[Dict[str, Any]] = []
        now_ts = int(time.time())
        for turn in turns:
            turn_id = int(turn.get("_turn_id"))
            semantic_score = scored_turns.get(turn_id, 0.0)
            lexical_meta = lexical_hits.get(turn_id) or {}
            lexical_score = float(lexical_meta.get("score", 0.0) or 0.0)
            if semantic_score <= 0.0 and lexical_score <= 0.0:
                continue
            created_at = int(turn.get("_created_at_epoch", 0))
            final_score = (
                max(semantic_score, lexical_score)
                + (0.25 * min(semantic_score, lexical_score))
                + (0.6 * _recency_boost(created_at, now_ts))
            )
            if semantic_score > 0.0 and lexical_score > 0.0:
                scope = "semantic+lexical"
            elif lexical_score > 0.0:
                scope = "lexical"
            else:
                scope = "semantic"
            payload = dict(turn)
            payload["_sim_score"] = semantic_score
            payload["_lexical_score"] = lexical_score
            payload["_final_score"] = final_score
            payload["_scope"] = scope
            payload["_is_recent"] = False
            payload["_lexical_term_hits"] = list(lexical_meta.get("term_hits") or [])
            payload["_lexical_phrase_hits"] = list(lexical_meta.get("phrase_hits") or [])
            payload["_lexical_uuid_hits"] = list(lexical_meta.get("uuid_hits") or [])
            for msg in payload.get("messages", []):
                msg["_scope"] = scope
                msg["_sim_score"] = semantic_score
                msg["_lexical_score"] = lexical_score
            candidates.append(payload)

        candidates.sort(key=lambda t: float(t.get("_final_score", 0.0)), reverse=True)
        selected: List[Dict[str, Any]] = []
        used_tokens = 0
        for turn in candidates:
            if not turn.get("messages"):
                continue
            turn_tokens = self._turn_token_count(turn, chars_per_token=chars_per_token)
            if selected and len(selected) >= max_turns:
                break
            if selected and (used_tokens + turn_tokens) > token_budget:
                break
            selected.append(turn)
            used_tokens += turn_tokens
            if len(selected) >= max_turns or used_tokens >= token_budget:
                break
        selected.sort(key=lambda t: int(t.get("_created_at_epoch", 0)))
        return selected

    def _merge_turns(
        self,
        *,
        recent_turns: List[Dict[str, Any]],
        semantic_turns: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        merged: Dict[int, Dict[str, Any]] = {}
        for turn in recent_turns:
            turn_id = int(turn.get("_turn_id"))
            merged[turn_id] = dict(turn)
            merged[turn_id]["_scope"] = "recent"
            merged[turn_id]["_is_recent"] = True
            for msg in merged[turn_id].get("messages", []):
                msg["_scope"] = "tail"
                msg["_sim_score"] = 0.0

        for turn in semantic_turns:
            turn_id = int(turn.get("_turn_id"))
            if turn_id in merged:
                continue
            merged[turn_id] = dict(turn)
            merged[turn_id]["_scope"] = str(turn.get("_scope") or "semantic")
            merged[turn_id]["_is_recent"] = bool(turn.get("_is_recent"))

        out = list(merged.values())
        out.sort(key=lambda t: int(t.get("_created_at_epoch", 0)))
        return out

    def _flatten_turns(self, turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for turn in turns:
            turn_id = turn.get("_turn_id")
            turn_scope = str(turn.get("_scope") or "")
            turn_is_recent = bool(turn.get("_is_recent"))
            turn_created_at_epoch = int(turn.get("_created_at_epoch", 0) or 0)
            for msg in turn.get("messages", []):
                payload = dict(msg)
                payload["_turn_id"] = turn_id
                payload["_turn_scope"] = turn_scope
                payload["_turn_is_recent"] = turn_is_recent
                payload["_turn_created_at_epoch"] = turn_created_at_epoch
                out.append(payload)
        return out

    def _order_messages_as_pairs(
        self,
        msgs: List[Dict[str, Any]],
        max_messages: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not msgs:
            return []

        sorted_msgs = sorted(
            msgs,
            key=_message_order_key,
        )
        out: List[Dict[str, Any]] = []
        idx = 0
        while idx < len(sorted_msgs):
            current = sorted_msgs[idx]
            current_role = str(current.get("role", "")).lower()
            next_msg = sorted_msgs[idx + 1] if idx + 1 < len(sorted_msgs) else None
            next_role = str((next_msg or {}).get("role", "")).lower()

            if current_role == "user" and next_msg and next_role in {"system", "assistant"}:
                if max_messages is not None and len(out) >= max_messages:
                    break
                out.append(current)
                if max_messages is not None and len(out) >= max_messages:
                    break
                out.append(next_msg)
                idx += 2
                continue

            if max_messages is not None and len(out) >= max_messages:
                break
            out.append(current)
            idx += 1

        return out
