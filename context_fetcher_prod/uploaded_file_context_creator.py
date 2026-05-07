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


def _created_at_to_epoch(value: Any) -> int:
    if value is None:
        return 0
    try:
        if isinstance(value, (int, float)):
            return int(value)
        parsed = datetime.fromisoformat(str(value).strip())
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp())
    except Exception:
        return 0


def _recency_boost(created_at_epoch: int, now_epoch: int) -> float:
    age_seconds = max(0, now_epoch - created_at_epoch)
    # Strongly favor newer uploads. Half-life is ~2 hours.
    return 1.0 / (1.0 + (age_seconds / 7200.0))


def _build_payload_filter(
    *,
    user_id: str,
    chat_id: str,
    file_ids: List[str]
) -> qm.Filter:
    must: List[qm.FieldCondition] = [
        qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id)),
        qm.FieldCondition(key="chat_id", match=qm.MatchValue(value=chat_id)),
    ]

    if file_ids:
        must.append(qm.FieldCondition(key="file_id", match=qm.MatchAny(any=file_ids)))

    return qm.Filter(must=must)


def _normalize_file_ids(file_id: Optional[str]) -> List[str]:
    if not file_id:
        return []
    values = [part.strip() for part in re.split(r"[,\n;]+", str(file_id))]
    return [value for value in values if value]


def _sort_full_file_context_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _sort_key(payload: Dict[str, Any]) -> tuple:
        file_id = str(payload.get("file_id") or "")
        # changed
        # Full-file attachment context must follow original document chunk order.
        # Prefer chunk_no whenever it is present, then fall back to older keys.
        chunk_no = payload.get("chunk_no")
        has_chunk_no = chunk_no is not None and str(chunk_no).strip() != ""
        created_at_epoch = _created_at_to_epoch(payload.get("created_at"))
        chunk_id = str(payload.get("chunk_id") or "")
        try:
            normalized_chunk_no = int(chunk_no) if has_chunk_no else 10**9
        except Exception:
            normalized_chunk_no = 10**9
        return (file_id, 0 if has_chunk_no else 1, normalized_chunk_no, created_at_epoch, chunk_id)

    return sorted(rows, key=_sort_key)


def _extract_content(payload: Dict[str, Any]) -> str:
    for key in ("content", "text", "chunk_text", "page_content"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    return ""


def _extract_file_name(payload: Dict[str, Any]) -> str:
    for key in ("file_name", "title", "filename", "name"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    return ""


def _row_identity(payload: Dict[str, Any]) -> tuple[str, str]:
    file_id = str(payload.get("file_id") or "").strip()
    chunk_id = str(payload.get("chunk_id") or "").strip()
    if not chunk_id:
        chunk_no = payload.get("chunk_no")
        chunk_id = str(chunk_no).strip() if chunk_no is not None else ""
    if not chunk_id:
        chunk_id = _extract_content(payload)[:80]
    return file_id, chunk_id


def _estimate_text_tokens(text: Any, *, chars_per_token: float) -> int:
    value = str(text or "").strip()
    if not value:
        return 0
    return max(1, int(math.ceil(len(value) / max(1.0, float(chars_per_token)))))


class Embedder:
    def __init__(self, embed_endpoint: str, model_name: str = "qwen-embed_8b"):
        self.embed_endpoint = embed_endpoint
        self.model_name = str(model_name or "qwen-embed")
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
    uploaded_file_collection: str = "chat_attachment_chunks_production_1"  # always change
    top_n_docs: int = 12
    semantic_threshold: float = 0.3
    candidate_multiplier: int = 6
    weight_similarity: float = 1.0
    weight_file_recency: float = 2.2
    weight_chunk_recency: float = 0.6
    scoped_weight_file_recency: float = _env_float("UPLOADED_FILE_SCOPED_WEIGHT_FILE_RECENCY", 0.25)
    scoped_weight_chunk_recency: float = _env_float("UPLOADED_FILE_SCOPED_WEIGHT_CHUNK_RECENCY", 0.10)
    full_file_scroll_page_size: int = 256
    latest_file_resolver_scan_limit: int = 128
    semantic_neighbor_radius: int = _env_int("UPLOADED_FILE_SEMANTIC_NEIGHBOR_RADIUS", 1)
    semantic_neighbor_seed_limit: int = _env_int("UPLOADED_FILE_SEMANTIC_NEIGHBOR_SEED_LIMIT", 8)
    full_file_token_budget: int = _env_int("UPLOADED_FILE_FULL_FILE_TOKEN_BUDGET", 22000)
    full_file_chars_per_token: float = _env_float("UPLOADED_FILE_FULL_FILE_CHARS_PER_TOKEN", 4.0)
    oversize_full_file_semantic_candidate_limit: int = _env_int(
        "UPLOADED_FILE_OVERSIZE_SEMANTIC_CANDIDATE_LIMIT",
        64,
    )


class UploadedFileContextFetcher:
    def __init__(self, qdrant: QdrantClient, embedder: Embedder, cfg: Optional[Config] = None):
        self.qdrant = qdrant
        self.embedder = embedder
        self.cfg = cfg or Config()

    def _rows_token_count(self, rows: List[Dict[str, Any]], *, chars_per_token: float) -> int:
        return sum(_estimate_text_tokens(_extract_content(row), chars_per_token=chars_per_token) for row in rows)

    @staticmethod
    def _extract_query_terms(query: Any) -> List[str]:
        terms = [token.lower() for token in re.findall(r"[a-z0-9][a-z0-9._/-]{2,}", str(query or ""), flags=re.IGNORECASE)]
        if not terms:
            return []
        return list(dict.fromkeys(terms))

    def _lexical_row_score(self, payload: Dict[str, Any], *, query: Any) -> float:
        content = _extract_content(payload).lower()
        if not content:
            return 0.0
        query_text = str(query or "").strip().lower()
        terms = self._extract_query_terms(query_text)
        if not terms and not query_text:
            return 0.0
        overlap_count = sum(1 for term in terms if term in content) if terms else 0
        lexical_score = (overlap_count / max(1, len(terms))) if terms else 0.0
        phrase_bonus = 0.20 if query_text and len(query_text) >= 8 and query_text in content else 0.0
        return lexical_score + phrase_bonus

    @staticmethod
    def _row_chunk_no(payload: Dict[str, Any]) -> Optional[int]:
        raw = payload.get("chunk_no")
        if raw in (None, ""):
            return None
        try:
            return int(raw)
        except Exception:
            return None

    def _dedupe_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for row in rows:
            key = _row_identity(row)
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
        return out

    def _select_rows_under_token_budget(
        self,
        rows: List[Dict[str, Any]],
        *,
        token_budget: int,
        chars_per_token: float,
    ) -> List[Dict[str, Any]]:
        if token_budget <= 0:
            return []
        selected: List[Dict[str, Any]] = []
        used_tokens = 0
        for row in rows:
            content = _extract_content(row)
            if not content:
                continue
            row_tokens = _estimate_text_tokens(content, chars_per_token=chars_per_token)
            if selected and used_tokens + row_tokens > token_budget:
                break
            selected.append(row)
            used_tokens += row_tokens
            if used_tokens >= token_budget:
                break
        return selected

    def _select_relevant_full_file_rows(
        self,
        rows: List[Dict[str, Any]],
        *,
        query: Any,
        token_budget: int,
    ) -> List[Dict[str, Any]]:
        scored: List[tuple[float, Dict[str, Any]]] = []
        for row in rows:
            lexical_score = self._lexical_row_score(row, query=query)
            if lexical_score <= 0.0:
                continue
            scored.append((lexical_score, dict(row)))
        if not scored:
            return self._select_rows_under_token_budget(
                rows,
                token_budget=token_budget,
                chars_per_token=self.cfg.full_file_chars_per_token,
            )
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = self._select_rows_under_token_budget(
            [row for _score, row in scored],
            token_budget=token_budget,
            chars_per_token=self.cfg.full_file_chars_per_token,
        )
        for row in selected:
            row["_scope"] = "lexical_fallback"
            row["_full_file_oversize_fallback"] = True
            row["_sim_score"] = 0.0
            row["_file_recency_score"] = 0.0
            row["_chunk_recency_score"] = 0.0
            row["_final_score"] = float(self._lexical_row_score(row, query=query))
        return selected

    async def _fetch_neighbor_rows(
        self,
        *,
        collection_name: str,
        payload_filter: qm.Filter,
        targets: set[tuple[str, int]],
    ) -> List[Dict[str, Any]]:
        if not targets:
            return []

        remaining = set(targets)
        rows: List[Dict[str, Any]] = []
        next_offset = None
        page_limit = max(32, int(self.cfg.full_file_scroll_page_size))

        while remaining:
            hits, next_offset = await asyncio.to_thread(
                self.qdrant.scroll,
                collection_name=collection_name,
                limit=page_limit,
                scroll_filter=payload_filter,
                offset=next_offset,
            )
            if not hits:
                break

            for hit in hits:
                payload = dict(hit.payload or {})
                file_id = str(payload.get("file_id") or "").strip()
                chunk_no = self._row_chunk_no(payload)
                if not file_id or chunk_no is None:
                    continue
                key = (file_id, chunk_no)
                if key not in remaining:
                    continue
                content = _extract_content(payload)
                if not content:
                    continue
                payload["content"] = content
                payload["file_name"] = _extract_file_name(payload) or "Unknown"
                payload["_chunk_created_at_epoch"] = _created_at_to_epoch(payload.get("created_at"))
                payload["_scope"] = "semantic_neighbor"
                payload["_sim_score"] = float(payload.get("_sim_score", 0.0) or 0.0)
                payload["_final_score"] = float(payload.get("_final_score", 0.0) or 0.0)
                rows.append(payload)
                remaining.discard(key)

            if next_offset is None:
                break

        return rows

    async def _expand_rows_with_neighbors(
        self,
        *,
        rows: List[Dict[str, Any]],
        collection_name: str,
        payload_filter: qm.Filter,
        result_limit: Optional[int] = None,
        token_budget: Optional[int] = None,
        chars_per_token: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        radius = max(0, int(self.cfg.semantic_neighbor_radius))
        if radius <= 0 or not rows:
            return rows

        seed_limit = max(1, int(self.cfg.semantic_neighbor_seed_limit))
        seeds = [row for row in rows[:seed_limit] if self._row_chunk_no(row) is not None and str(row.get("file_id") or "").strip()]
        if not seeds:
            return rows

        existing_keys = {
            (str(row.get("file_id") or "").strip(), self._row_chunk_no(row))
            for row in rows
            if str(row.get("file_id") or "").strip() and self._row_chunk_no(row) is not None
        }
        neighbor_targets: set[tuple[str, int]] = set()
        for row in seeds:
            file_id = str(row.get("file_id") or "").strip()
            chunk_no = self._row_chunk_no(row)
            if not file_id or chunk_no is None:
                continue
            for delta in range(-radius, radius + 1):
                candidate_no = chunk_no + delta
                if candidate_no < 0:
                    continue
                key = (file_id, candidate_no)
                if key in existing_keys:
                    continue
                neighbor_targets.add(key)

        neighbor_rows = await self._fetch_neighbor_rows(
            collection_name=collection_name,
            payload_filter=payload_filter,
            targets=neighbor_targets,
        )
        if not neighbor_rows:
            return rows

        all_rows = {(_row_identity(row)): dict(row) for row in [*rows, *neighbor_rows]}
        expanded: List[Dict[str, Any]] = []
        added: set[tuple[str, str]] = set()
        used_tokens = 0

        def append_row(payload: Dict[str, Any]) -> bool:
            nonlocal used_tokens
            key = _row_identity(payload)
            if key in added:
                return True
            if result_limit is not None and result_limit > 0 and len(expanded) >= result_limit:
                return False
            if token_budget is not None and chars_per_token is not None:
                row_tokens = _estimate_text_tokens(_extract_content(payload), chars_per_token=chars_per_token)
                if expanded and used_tokens + row_tokens > token_budget:
                    return False
                used_tokens += row_tokens
            added.add(key)
            expanded.append(payload)
            return True

        for seed in seeds:
            file_id = str(seed.get("file_id") or "").strip()
            chunk_no = self._row_chunk_no(seed)
            if not file_id or chunk_no is None:
                continue
            for candidate_no in range(chunk_no - radius, chunk_no + radius + 1):
                if candidate_no < 0:
                    continue
                for row in all_rows.values():
                    if str(row.get("file_id") or "").strip() == file_id and self._row_chunk_no(row) == candidate_no:
                        if not append_row(row):
                            return expanded or rows
                        break

        for row in rows:
            if not append_row(row):
                break

        return expanded or rows

    async def _build_semantic_rows(
        self,
        *,
        collection_name: str,
        payload_filter: qm.Filter,
        query: str,
        threshold: float,
        final_top_n: int,
        explicit_file_scope: bool,
        expand_neighbors: bool,
        result_token_budget: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        candidate_limit = max(
            final_top_n,
            final_top_n * self.cfg.candidate_multiplier,
        )
        if result_token_budget is not None:
            candidate_limit = max(candidate_limit, int(self.cfg.oversize_full_file_semantic_candidate_limit))

        qvec = await self.embedder.generate_embeddings({"input": [query]})
        if not qvec:
            raise RuntimeError("Failed to generate embeddings for query")

        try:
            hits = await asyncio.to_thread(
                self.qdrant.query_points,
                collection_name=collection_name,
                query=qvec,
                query_filter=payload_filter,
                limit=candidate_limit,
                with_vectors=False,
                with_payload=True,
            )
        except Exception as exc:
            logger.exception("Qdrant query failed for uploaded file context: %s", exc)
            raise RuntimeError("Failed to fetch uploaded file context from Qdrant") from exc

        logger.info(
            "uploaded_file_fetch semantic candidates points=%s threshold=%s explicit_file_scope=%s",
            len(hits.points),
            threshold,
            explicit_file_scope,
        )

        now_ts = int(time.time())
        contexts: List[Dict[str, Any]] = []
        file_latest_ts: Dict[str, int] = {}
        for point in hits.points:
            sim_score = float(point.score)
            if sim_score < threshold:
                continue
            payload = dict(point.payload or {})
            content = _extract_content(payload)
            if not content:
                continue
            payload["content"] = content
            payload["file_name"] = _extract_file_name(payload) or "Unknown"
            created_at_epoch = _created_at_to_epoch(payload.get("created_at"))
            fid = str(payload.get("file_id", ""))
            if fid:
                file_latest_ts[fid] = max(file_latest_ts.get(fid, 0), created_at_epoch)
            payload["_sim_score"] = sim_score
            payload["_chunk_created_at_epoch"] = created_at_epoch
            contexts.append(payload)

        if not contexts:
            return []

        file_recency_weight = self.cfg.scoped_weight_file_recency if explicit_file_scope else self.cfg.weight_file_recency
        chunk_recency_weight = self.cfg.scoped_weight_chunk_recency if explicit_file_scope else self.cfg.weight_chunk_recency
        for payload in contexts:
            fid = str(payload.get("file_id", ""))
            sim_score = float(payload.get("_sim_score", 0.0))
            file_ts = file_latest_ts.get(fid, payload.get("_chunk_created_at_epoch", 0))
            chunk_ts = int(payload.get("_chunk_created_at_epoch", 0))
            file_recency = _recency_boost(file_ts, now_ts)
            chunk_recency = _recency_boost(chunk_ts, now_ts)

            payload["_scope"] = "semantic"
            payload["_file_recency_score"] = file_recency
            payload["_chunk_recency_score"] = chunk_recency
            payload["_final_score"] = (
                self.cfg.weight_similarity * sim_score
                + file_recency_weight * file_recency
                + chunk_recency_weight * chunk_recency
            )

        contexts.sort(key=lambda p: float(p.get("_final_score", 0.0)), reverse=True)
        if expand_neighbors:
            contexts = await self._expand_rows_with_neighbors(
                rows=contexts,
                collection_name=collection_name,
                payload_filter=payload_filter,
                result_limit=None if result_token_budget is not None else final_top_n,
                token_budget=result_token_budget,
                chars_per_token=self.cfg.full_file_chars_per_token if result_token_budget is not None else None,
            )

        if result_token_budget is not None:
            contexts = self._select_rows_under_token_budget(
                contexts,
                token_budget=result_token_budget,
                chars_per_token=self.cfg.full_file_chars_per_token,
            )
        trimmed = contexts[:final_top_n] if result_token_budget is None else contexts
        if result_token_budget is not None:
            for payload in trimmed:
                payload["_scope"] = "semantic_fallback"
                payload["_full_file_oversize_fallback"] = True
        return self._dedupe_rows(trimmed)

    async def _legacy_file_id_resolver_unused(self) -> List[str]:
        del self
        return []

        # ------- Basic‑auth credentials ------------------------------------
    async def fetch(
        self,
        *,
        user_id: str,
        chat_id: str,
        query: str,
        top_n: Optional[int] = None,
        min_score: Optional[float] = None,
        retrieval_mode: str = "semantic",
        file_id: Optional[str] = None,
        resolve_latest_file: bool = False,
        latest_file_limit: Optional[int] = None,
        before_created_at: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not user_id or not chat_id:
            raise ValueError("user_id and chat_id are required")
        retrieval_mode = str(retrieval_mode or "semantic").strip().lower()
        if retrieval_mode not in {"semantic", "full_file"}:
            raise ValueError("retrieval_mode must be 'semantic' or 'full_file'")
        if retrieval_mode != "full_file" and (not query or not query.strip()):
            raise ValueError("query is required")
        if top_n is not None and top_n < 0:
            raise ValueError("top_n must be >= 0")

        if retrieval_mode == "full_file":
            final_top_n = None if top_n in {None, 0} else int(top_n)
        else:
            final_top_n = int(top_n) if top_n is not None else self.cfg.top_n_docs
            if final_top_n < 1:
                raise ValueError("top_n must be >= 1")

        threshold = self.cfg.semantic_threshold if min_score is None else float(min_score)
        if threshold < 0:
            raise ValueError("min_score must be >= 0")

        logger.info(
            "uploaded_file_fetch start user_id=%s chat_id=%s mode=%s top_n=%s min_score=%s file_id=%s resolve_latest_file=%s latest_file_limit=%s before_created_at=%s collection=%s",
            user_id,
            chat_id,
            retrieval_mode,
            final_top_n,
            threshold,
            file_id,
            resolve_latest_file,
            latest_file_limit,
            before_created_at,
            collection_name or self.cfg.uploaded_file_collection,
        )

        selected_collection = collection_name or self.cfg.uploaded_file_collection
        file_ids = _normalize_file_ids(file_id)
        if not file_ids and bool(resolve_latest_file):
            file_ids = await self._resolve_latest_file_ids_from_qdrant(
                user_id=user_id,
                chat_id=chat_id,
                collection_name=selected_collection,
                limit=max(1, int(latest_file_limit or 1)),
                before_created_at=before_created_at,
            )
            if not file_ids:
                logger.info(
                    "uploaded_file_fetch latest-file resolve found no files chat_id=%s before_created_at=%s",
                    chat_id,
                    before_created_at,
                )
                return []
        payload_filter = _build_payload_filter(user_id=user_id, chat_id=chat_id, file_ids=file_ids)

        if retrieval_mode == "full_file":
            try:
                rows: List[Dict[str, Any]] = []
                page_limit = max(1, int(self.cfg.full_file_scroll_page_size))
                next_offset = None
                while True:
                    hits, next_offset = await asyncio.to_thread(
                        self.qdrant.scroll,
                        collection_name=selected_collection,
                        limit=page_limit,
                        scroll_filter=payload_filter,
                        offset=next_offset,
                    )
                    if not hits:
                        break
                    for hit in hits:
                        payload = dict(hit.payload or {})
                        content = _extract_content(payload)
                        if not content:
                            continue
                        payload["content"] = content
                        payload["file_name"] = _extract_file_name(payload) or "Unknown"
                        rows.append(payload)
                    if next_offset is None:
                        break
            except Exception as exc:
                logger.exception("Qdrant full-file query failed for uploaded file context: %s", exc)
                raise RuntimeError("Failed to fetch full-file context from Qdrant") from exc

            rows = _sort_full_file_context_rows(rows)
            for payload in rows:
                payload["_scope"] = "full_file"
                payload["_sim_score"] = float(payload.get("_sim_score", payload.get("score", 1.0)))
                payload["_chunk_recency_score"] = 0.0
                payload["_file_recency_score"] = 0.0
                payload["_final_score"] = float(payload.get("_final_score", 1.0))

            total_tokens_est = self._rows_token_count(
                rows,
                chars_per_token=self.cfg.full_file_chars_per_token,
            )
            if rows and total_tokens_est > max(1, int(self.cfg.full_file_token_budget)):
                logger.info(
                    "uploaded_file_fetch full_file oversize chat_id=%s file_id=%s chunks=%s tokens_est=%s budget=%s; falling back to top relevant chunks",
                    chat_id,
                    file_id,
                    len(rows),
                    total_tokens_est,
                    int(self.cfg.full_file_token_budget),
                )
                if query and query.strip():
                    try:
                        semantic_rows = await self._build_semantic_rows(
                            collection_name=selected_collection,
                            payload_filter=payload_filter,
                            query=query,
                            threshold=max(0.0, min(threshold, self.cfg.semantic_threshold)),
                            final_top_n=max(1, min(len(rows), int(self.cfg.oversize_full_file_semantic_candidate_limit))),
                            explicit_file_scope=bool(file_ids),
                            expand_neighbors=bool(file_ids),
                            result_token_budget=int(self.cfg.full_file_token_budget),
                        )
                    except Exception as exc:
                        logger.warning(
                            "uploaded_file_fetch full_file oversize semantic fallback failed chat_id=%s file_id=%s error=%s",
                            chat_id,
                            file_id,
                            exc,
                        )
                        semantic_rows = []
                    if semantic_rows:
                        logger.info(
                            "uploaded_file_fetch full_file oversize fallback result chat_id=%s file_id=%s chunks=%s tokens_est=%s",
                            chat_id,
                            file_id,
                            len(semantic_rows),
                            self._rows_token_count(
                                semantic_rows,
                                chars_per_token=self.cfg.full_file_chars_per_token,
                            ),
                        )
                        return semantic_rows
                rows = self._select_relevant_full_file_rows(
                    rows,
                    query=query,
                    token_budget=int(self.cfg.full_file_token_budget),
                )

            logger.info(
                "uploaded_file_fetch full_file result chat_id=%s file_id=%s chunks=%s requested_top_n=%s effective_limit=%s tokens_est=%s",
                chat_id,
                file_id,
                len(rows),
                top_n,
                final_top_n,
                self._rows_token_count(rows, chars_per_token=self.cfg.full_file_chars_per_token),
            )
            return rows if final_top_n is None else rows[:final_top_n]
        trimmed = await self._build_semantic_rows(
            collection_name=selected_collection,
            payload_filter=payload_filter,
            query=query,
            threshold=threshold,
            final_top_n=final_top_n,
            explicit_file_scope=bool(file_ids),
            expand_neighbors=bool(file_ids),
        )
        if not trimmed:
            logger.info("uploaded_file_fetch semantic result chat_id=%s chunks=0", chat_id)
            return []
        top_score = float(trimmed[0].get("_final_score", 0.0)) if trimmed else 0.0
        logger.info(
            "uploaded_file_fetch semantic result chat_id=%s chunks=%s top_score=%.4f explicit_file_scope=%s",
            chat_id,
            len(trimmed),
            top_score,
            bool(file_ids),
        )
        return trimmed

    async def _resolve_latest_file_ids_from_qdrant(
        self,
        *,
        user_id: str,
        chat_id: str,
        collection_name: str,
        limit: int,
        before_created_at: Optional[str] = None,
    ) -> List[str]:
        if limit <= 0:
            return []

        before_epoch = _created_at_to_epoch(before_created_at)
        total_scan_limit = max(int(self.cfg.latest_file_resolver_scan_limit), limit * 8)
        page_limit = min(64, total_scan_limit)
        payload_filter = _build_payload_filter(user_id=user_id, chat_id=chat_id, file_ids=[])

        resolved_ids: List[str] = []
        seen: set[str] = set()
        next_offset = None
        scanned = 0

        while scanned < total_scan_limit and len(resolved_ids) < limit:
            current_limit = min(page_limit, total_scan_limit - scanned)
            try:
                hits, next_offset = await asyncio.to_thread(
                    self.qdrant.scroll,
                    collection_name=collection_name,
                    limit=current_limit,
                    order_by=qm.OrderBy(key="created_at", direction=qm.Direction.DESC),
                    scroll_filter=payload_filter,
                    offset=next_offset,
                )
            except Exception as exc:
                logger.exception("Qdrant latest-file resolve failed for uploaded file context: %s", exc)
                raise RuntimeError("Failed to resolve latest uploaded file from Qdrant") from exc

            if not hits:
                break

            scanned += len(hits)
            for hit in hits:
                payload = dict(hit.payload or {})
                file_id = str(payload.get("file_id") or "").strip()
                if not file_id or file_id in seen:
                    continue

                created_at_epoch = _created_at_to_epoch(payload.get("created_at"))
                # When the router drops current attachment_ids, QA falls back to the most recent
                # uploaded file visible in this chat before the current message timestamp.
                if before_epoch and created_at_epoch and created_at_epoch > before_epoch:
                    continue

                seen.add(file_id)
                resolved_ids.append(file_id)
                if len(resolved_ids) >= limit:
                    break

            if next_offset is None:
                break

        logger.info(
            "uploaded_file_fetch latest-file resolve chat_id=%s resolved_files=%s scanned_points=%s before_created_at=%s",
            chat_id,
            len(resolved_ids),
            scanned,
            before_created_at,
        )
        return resolved_ids
