import asyncio
import logging
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from aiohttp import BasicAuth, TCPConnector
from qdrant_client import QdrantClient
from qdrant_client import models as qm

logger = logging.getLogger(__name__)
_CONTENT_KEYS = ("content", "text", "chunk_text", "page_content")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_DIGIT_PATTERN = re.compile(r"\d+")
_MOJIBAKE_MARKERS = (
    chr(0x00C3),
    chr(0x00C2),
    chr(0x00E2) + chr(0x20AC),
    chr(0x00E2) + chr(0x0080),
)
_DASH_OR_QUOTE_VARIANTS = "".join(
    chr(codepoint)
    for codepoint in (
        *range(0x2010, 0x2016),
        0x2212,
        0xFE58,
        0xFE63,
        0xFF0D,
        0x002D,
        0x0027,
        0x0060,
        0x00B4,
        0x2018,
        0x2019,
        0x201A,
        0x201B,
        0x2032,
    )
)
_MOJIBAKE_MARKER_PATTERN = re.compile("|".join(re.escape(marker) for marker in _MOJIBAKE_MARKERS))
_DASH_OR_QUOTE_VARIANT_PATTERN = re.compile(f"[{re.escape(_DASH_OR_QUOTE_VARIANTS)}]")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "t", "yes", "y"}


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _to_optional_int(value: Any) -> Optional[int]:
    parsed = _to_int(value)
    return parsed if parsed else None


def _to_optional_epoch_millis(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str):
        text = str(value).strip()
        if not text:
            return None
        if re.fullmatch(r"-?\d+", text):
            parsed = _to_optional_int(text)
            if parsed is None:
                return None
            if parsed < 10_000_000_000:
                return parsed * 1000
            return parsed
        try:
            normalized = text.replace("Z", "+00:00")
            parsed_dt = datetime.fromisoformat(normalized)
            if parsed_dt.tzinfo is None:
                parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
            return int(parsed_dt.timestamp() * 1000)
        except Exception:
            return None

    parsed = _to_optional_int(value)
    if parsed is None:
        return None
    # Convert likely-seconds epoch into millis.
    if parsed < 10_000_000_000:
        return parsed * 1000
    return parsed


def _to_optional_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return None


def _extract_content(payload: Dict[str, Any]) -> str:
    for key in _CONTENT_KEYS:
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    return ""


def _normalize_exact_keyword(value: Any) -> str:
    text = _WHITESPACE_PATTERN.sub(" ", str(value or "").strip())
    return text[:160].strip()


def _repair_common_mojibake(value: Any) -> str:
    text = str(value or "")
    if not text or not _MOJIBAKE_MARKER_PATTERN.search(text):
        return text
    try:
        repaired = text.encode("cp1252", errors="strict").decode("utf-8", errors="strict")
    except Exception:
        return text
    # Keep the repaired form only when it plausibly fixed mojibake instead of
    # transforming valid non-ASCII text into something worse.
    if _MOJIBAKE_MARKER_PATTERN.search(repaired):
        return text
    return repaired


def _normalize_text_for_match(value: Any) -> str:
    text = _repair_common_mojibake(value)
    text = unicodedata.normalize("NFKC", text)
    text = _DASH_OR_QUOTE_VARIANT_PATTERN.sub(" ", text)
    text = re.sub(r"[/_.:,;()\[\]{}]+", " ", text)
    return _WHITESPACE_PATTERN.sub(" ", text.strip().lower())


def _exact_keyword_variants(keyword: str) -> List[str]:
    base = _normalize_exact_keyword(keyword)
    if not base:
        return []
    normalized = _normalize_text_for_match(base)
    variants = [base]
    if normalized and normalized.lower() != base.lower():
        variants.append(normalized)
    compact = re.sub(r"[^a-z0-9]+", "", normalized.lower())
    if compact and compact != normalized.lower():
        variants.append(compact)
    return _dedupe_keywords(variants, max_items=3)


def _truncate_words(value: Any, *, max_words: int) -> str:
    text = _WHITESPACE_PATTERN.sub(" ", str(value or "").strip())
    if not text:
        return ""
    limit = max(1, int(max_words))
    words = text.split(" ")
    if len(words) <= limit:
        return text
    return " ".join(words[:limit]).strip()


def _numeric_signature(value: Any) -> str:
    return "".join(_DIGIT_PATTERN.findall(str(value or "")))


def _dedupe_keywords(keywords: List[str], *, max_items: int) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for raw in keywords:
        keyword = _normalize_exact_keyword(raw)
        if not keyword:
            continue
        key = keyword.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(keyword)
        if len(out) >= max(1, int(max_items)):
            break
    return out


def _keyword_match_strength(content: str, keyword: str) -> float:
    normalized_content = _normalize_text_for_match(content)
    normalized_keyword = _normalize_text_for_match(keyword)
    if not normalized_content or not normalized_keyword:
        return 0.0

    numeric_keyword = _numeric_signature(keyword)
    if len(numeric_keyword) >= 5:
        numeric_content = _numeric_signature(content)
        if numeric_keyword and numeric_keyword in numeric_content:
            return 1.0

    if normalized_keyword in normalized_content:
        if " " in normalized_keyword or len(normalized_keyword) >= 8:
            return 1.0
        return 0.82

    keyword_tokens = [token for token in normalized_keyword.split() if token]
    if len(keyword_tokens) >= 2 and all(token in normalized_content for token in keyword_tokens):
        return 0.72

    compact_content = re.sub(r"[^a-z0-9]+", "", normalized_content)
    compact_keyword = re.sub(r"[^a-z0-9]+", "", normalized_keyword)
    if len(compact_keyword) >= 8 and compact_keyword in compact_content:
        return 0.90
    return 0.0


def _score_exact_payload(content: str, keywords: List[str]) -> Dict[str, Any]:
    matched_keywords: List[str] = []
    total_strength = 0.0

    for keyword in keywords:
        strength = _keyword_match_strength(content, keyword)
        if strength <= 0:
            continue
        matched_keywords.append(keyword)
        total_strength += strength

    if not matched_keywords:
        return {"matched_keywords": [], "exact_match_score": 0.0}

    exact_match_score = min(1.0, total_strength / max(1.0, float(len(keywords))))
    return {
        "matched_keywords": matched_keywords,
        "exact_match_score": exact_match_score,
    }


def _metadata_exact_should_clauses_for_keyword(keyword: str) -> List[Dict[str, Any]]:
    clauses: List[Dict[str, Any]] = []
    for variant in _exact_keyword_variants(keyword):
        clauses.extend(
            [
                {"match_phrase": {"content": {"query": variant, "boost": 8.0}}},
                {"match_phrase": {"text": {"query": variant, "boost": 4.0}}},
            ]
        )

        normalized_keyword = _normalize_text_for_match(variant)
        keyword_tokens = [token for token in normalized_keyword.split() if token]
        if keyword_tokens:
            clauses.append(
                {"match": {"content": {"query": variant, "operator": "and", "boost": 2.5}}}
            )
            clauses.append(
                {"match": {"text": {"query": variant, "operator": "and", "boost": 1.2}}}
            )

        numeric_keyword = _numeric_signature(variant)
        if len(numeric_keyword) >= 5 and numeric_keyword != normalized_keyword:
            clauses.append(
                {"match_phrase": {"content": {"query": numeric_keyword, "boost": 7.5}}}
            )
            clauses.append(
                {"match": {"content": {"query": numeric_keyword, "operator": "and", "boost": 3.2}}}
            )

    return clauses


def _build_metadata_exact_filter_clauses(
    *,
    report_type: Optional[str],
    branch: Optional[str],
    doc_id: Optional[str],
    parent_id: Optional[str],
    lang: Optional[str],
    is_attachment: Optional[bool],
    chunk_no: Optional[int],
    document_date_gte: Optional[int],
    document_date_lte: Optional[int],
    ingestion_date_gte: Optional[int],
    ingestion_date_lte: Optional[int],
    collection_name: Optional[str],
) -> List[Dict[str, Any]]:
    filters: List[Dict[str, Any]] = []
    if report_type:
        filters.append({"term": {"report_type": report_type}})
    if branch:
        filters.append({"term": {"branch": branch}})
    if doc_id:
        filters.append({"term": {"doc_id": doc_id}})
    if parent_id:
        filters.append({"term": {"parent_id": parent_id}})
    if lang:
        filters.append({"term": {"lang": lang}})
    if is_attachment is not None:
        filters.append({"term": {"is_attachment": bool(is_attachment)}})
    if chunk_no is not None:
        filters.append({"term": {"chunk_no": int(chunk_no)}})
    if collection_name:
        filters.append({"term": {"qdrant_collection": collection_name}})

    document_range: Dict[str, Any] = {}
    if document_date_gte is not None:
        document_range["gte"] = int(document_date_gte)
    if document_date_lte is not None:
        document_range["lte"] = int(document_date_lte)
    if document_range:
        filters.append({"range": {"document_date": document_range}})

    ingestion_range: Dict[str, Any] = {}
    if ingestion_date_gte is not None:
        ingestion_range["gte"] = int(ingestion_date_gte)
    if ingestion_date_lte is not None:
        ingestion_range["lte"] = int(ingestion_date_lte)
    if ingestion_range:
        filters.append({"range": {"ingestion_date": ingestion_range}})

    return filters


def _recency_boost_ms(sort_time_ms: int, now_ms: int) -> float:
    if sort_time_ms <= 0:
        return 0.0
    age_ms = max(0, now_ms - sort_time_ms)
    age_days = age_ms / 86_400_000.0
    # Smoothly decay older content while still preserving strong semantic matches.
    return 1.0 / (1.0 + (age_days / 30.0))


def _build_payload_filter(
    *,
    report_type: Optional[str],
    branch: Optional[str],
    doc_id: Optional[str],
    parent_id: Optional[str],
    lang: Optional[str],
    is_attachment: Optional[bool],
    chunk_no: Optional[int],
    document_date_gte: Optional[int],
    document_date_lte: Optional[int],
    ingestion_date_gte: Optional[int],
    ingestion_date_lte: Optional[int],
) -> Optional[qm.Filter]:
    must: List[qm.FieldCondition] = []

    if report_type:
        must.append(qm.FieldCondition(key="report_type", match=qm.MatchValue(value=report_type)))
    if branch:
        must.append(qm.FieldCondition(key="branch", match=qm.MatchValue(value=branch)))
    if doc_id:
        must.append(qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id)))
    if parent_id:
        must.append(qm.FieldCondition(key="parent_id", match=qm.MatchValue(value=parent_id)))
    if lang:
        must.append(qm.FieldCondition(key="lang", match=qm.MatchValue(value=lang)))
    if is_attachment is not None:
        must.append(qm.FieldCondition(key="is_attachment", match=qm.MatchValue(value=is_attachment)))
    if chunk_no is not None:
        must.append(qm.FieldCondition(key="chunk_no", match=qm.MatchValue(value=chunk_no)))

    # changed: `document_date` and `ingestion_date` are indexed as integer in
    # `document_chunks`, so use numeric range directly.
    if document_date_gte is not None or document_date_lte is not None:
        must.append(
            qm.FieldCondition(
                key="document_date",
                range=qm.Range(gte=document_date_gte, lte=document_date_lte),
            )
        )
    if ingestion_date_gte is not None or ingestion_date_lte is not None:
        must.append(
            qm.FieldCondition(
                key="ingestion_date",
                range=qm.Range(gte=ingestion_date_gte, lte=ingestion_date_lte),
            )
        )

    return qm.Filter(must=must) if must else None


class Embedder:
    def __init__(self, embed_endpoint: str, model_name: str = "qwen-embed"):
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
    big_data_collection: str = os.getenv(
        "BIGDATA_QDRANT_COLLECTION",
        os.getenv("REPORT_QDRANT_COLLECTION", "document_chunks_test_new_clean_v1"),
    )
    elasticsearch_base_url: str = os.getenv("ELASTICSEARCH_BASE_URL", "https://192.168.10.236:9200")
    elasticsearch_index: str = os.getenv("ELASTICSEARCH_BIGDATA_INDEX", "document_chunks_v2")  # legacy request field; exact retrieval uses chunk_metadata_index.
    chunk_metadata_index: str = os.getenv(
        "BIGDATA_CHUNK_METADATA_INDEX",
        os.getenv("REPORT_CHUNK_ELASTICSEARCH_INDEX", "document_chunks_test_new_clean_metadata"),
    )
    elasticsearch_username: str = os.getenv("ELASTICSEARCH_USER", "elastic")
    elasticsearch_password: str = os.getenv(
        "ELASTICSEARCH_PASS",
        os.getenv("ELASTICSEARCH_PASSWORD", "Elastic@123"),
    )
    elasticsearch_disable_ssl_verify: bool = os.getenv(
        "ELASTICSEARCH_DISABLE_SSL_VERIFY",
        "true",
    ).strip().lower() in {"1", "true", "t", "yes", "y"}
    top_n_docs: int = _env_int("BIGDATA_TOP_N_DOCS", 8)
    semantic_threshold: float = _env_float("BIGDATA_SEMANTIC_THRESHOLD", 0.22)
    semantic_min_score_floor: float = _env_float("BIGDATA_MIN_SCORE_FLOOR", 0.18)
    candidate_multiplier: int = _env_int("BIGDATA_CANDIDATE_MULTIPLIER", 1)
    max_qdrant_query_limit: int = _env_int("BIGDATA_QDRANT_MAX_QUERY_LIMIT", 800)
    qdrant_query_timeout_seconds: int = _env_int("BIGDATA_QDRANT_QUERY_TIMEOUT_SECONDS", 60)
    semantic_neighbor_enabled: bool = _env_bool("BIGDATA_SEMANTIC_NEIGHBOR_ENABLED", True)
    semantic_neighbor_window: int = _env_int("BIGDATA_SEMANTIC_NEIGHBOR_WINDOW", 1)
    semantic_neighbor_max_contexts: int = _env_int("BIGDATA_SEMANTIC_NEIGHBOR_MAX_CONTEXTS", 200)
    semantic_neighbor_max_words: int = _env_int("BIGDATA_SEMANTIC_NEIGHBOR_MAX_WORDS", 160)
    semantic_neighbor_timeout_seconds: int = _env_int("BIGDATA_SEMANTIC_NEIGHBOR_TIMEOUT_SECONDS", 20)
    weight_similarity: float = 1.0
    weight_recency: float = 0.0
    exact_top_n_docs: int = 6
    exact_max_keywords: int = 6
    weight_exact_match: float = 1.05
    weight_exact_semantic: float = 0.18
    weight_exact_recency: float = 0.10
    exact_main_score_boost: float = _env_float("BIGDATA_EXACT_MAIN_SCORE_BOOST", 0.03)
    exact_candidate_multiplier: int = _env_int("BIGDATA_EXACT_CANDIDATE_MULTIPLIER", 3)
    exact_max_candidates_per_index: int = _env_int("BIGDATA_EXACT_MAX_CANDIDATES_PER_INDEX", 800)
    exact_elasticsearch_timeout_seconds: int = _env_int("BIGDATA_EXACT_ELASTIC_TIMEOUT_SECONDS", 45)


class BigDataDocumentsContextFetcher:
    def __init__(self, qdrant: QdrantClient, embedder: Embedder, cfg: Optional[Config] = None):
        self.qdrant = qdrant
        self.embedder = embedder
        self.cfg = cfg or Config()
        self.http_timeout = aiohttp.ClientTimeout(
            total=max(5, int(self.cfg.exact_elasticsearch_timeout_seconds))
        )

    @staticmethod
    def _semantic_neighbor_key(payload: Dict[str, Any], *, chunk_no: Optional[int] = None) -> Optional[Tuple[str, str, str, int]]:
        doc_id = str(payload.get("doc_id") or "").strip()
        parent_id = str(payload.get("parent_id") or "").strip()
        resolved_chunk_no = _to_optional_int(chunk_no if chunk_no is not None else payload.get("chunk_no"))
        if not doc_id or resolved_chunk_no is None:
            return None
        is_attachment = _to_optional_bool(payload.get("is_attachment"))
        attachment_key = "attachment" if is_attachment is True else "main" if is_attachment is False else "unknown"
        return (attachment_key, doc_id, parent_id, int(resolved_chunk_no))

    def _semantic_neighbor_clauses(self, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        window = max(0, int(self.cfg.semantic_neighbor_window))
        clauses: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str, str, str]] = set()
        if window <= 0:
            return clauses

        for context in contexts[: max(0, int(self.cfg.semantic_neighbor_max_contexts))]:
            doc_id = str(context.get("doc_id") or "").strip()
            if not doc_id:
                continue
            chunk_no = _to_optional_int(context.get("chunk_no"))
            if chunk_no is None:
                continue
            neighbor_chunk_numbers = [
                int(value)
                for value in range(int(chunk_no) - window, int(chunk_no) + window + 1)
                if value > 0 and value != int(chunk_no)
            ]
            if not neighbor_chunk_numbers:
                continue

            parent_id = str(context.get("parent_id") or "").strip()
            is_attachment = _to_optional_bool(context.get("is_attachment"))
            dedupe_key = (
                doc_id,
                parent_id,
                "attachment" if is_attachment is True else "main" if is_attachment is False else "unknown",
                ",".join(str(value) for value in neighbor_chunk_numbers),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            filters: List[Dict[str, Any]] = [
                {"term": {"doc_id": doc_id}},
                {"terms": {"chunk_no": neighbor_chunk_numbers}},
            ]
            if parent_id:
                filters.append({"term": {"parent_id": parent_id}})
            if is_attachment is not None:
                filters.append({"term": {"is_attachment": is_attachment}})
            clauses.append({"bool": {"filter": filters}})
        return clauses

    async def _fetch_semantic_neighbor_rows(
        self,
        contexts: List[Dict[str, Any]],
    ) -> Dict[Tuple[str, str, str, int], Dict[str, Any]]:
        if not bool(self.cfg.semantic_neighbor_enabled):
            return {}
        metadata_index = str(self.cfg.chunk_metadata_index or "").strip()
        if not metadata_index:
            return {}

        clauses = self._semantic_neighbor_clauses(contexts)
        if not clauses:
            return {}

        base_url = str(self.cfg.elasticsearch_base_url or "").rstrip("/")
        if not base_url:
            return {}

        search_url = f"{base_url}/{metadata_index}/_search"
        max_contexts = max(1, int(self.cfg.semantic_neighbor_max_contexts))
        window = max(1, int(self.cfg.semantic_neighbor_window))
        body = {
            "size": min(max_contexts * window * 2, max(1, len(clauses) * window * 2)),
            "_source": [
                "content",
                "text",
                "doc_id",
                "parent_id",
                "is_attachment",
                "chunk_no",
                "chunk_id",
                "total_chunks",
                "section_heading",
                "page_start",
                "page_end",
                "quality_score",
                "has_been_embedded",
            ],
            "query": {
                "bool": {
                    "should": clauses,
                    "minimum_should_match": 1,
                }
            },
            "sort": [
                {"doc_id": {"order": "asc", "unmapped_type": "keyword"}},
                {"parent_id": {"order": "asc", "unmapped_type": "keyword"}},
                {"chunk_no": {"order": "asc", "unmapped_type": "long"}},
            ],
        }
        connector = TCPConnector(ssl=not bool(self.cfg.elasticsearch_disable_ssl_verify))
        auth = BasicAuth(
            login=self.cfg.elasticsearch_username,
            password=self.cfg.elasticsearch_password,
        )
        timeout = aiohttp.ClientTimeout(total=max(5, int(self.cfg.semantic_neighbor_timeout_seconds)))

        try:
            async with aiohttp.ClientSession(connector=connector, auth=auth, timeout=timeout) as session:
                async with session.post(search_url, json=body) as response:
                    if response.status != 200:
                        text = await response.text()
                        logger.warning(
                            "big_data_semantic_neighbor_fetch skipped index=%s status=%s body=%s",
                            metadata_index,
                            response.status,
                            text[:500],
                        )
                        return {}
                    data = await response.json()
        except Exception as exc:
            logger.warning(
                "big_data_semantic_neighbor_fetch failed index=%s clauses=%s error=%s",
                metadata_index,
                len(clauses),
                exc,
            )
            return {}

        rows: Dict[Tuple[str, str, str, int], Dict[str, Any]] = {}
        for hit in (((data or {}).get("hits") or {}).get("hits") or []):
            source = dict(hit.get("_source") or {})
            if source.get("has_been_embedded") is False:
                continue
            content = _extract_content(source)
            if not content:
                continue
            key = self._semantic_neighbor_key(source)
            if key is None:
                continue
            source["content"] = content
            rows[key] = source
        return rows

    def _attach_semantic_neighbor_context_from_rows(
        self,
        contexts: List[Dict[str, Any]],
        neighbor_rows: Dict[Tuple[str, str, str, int], Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not neighbor_rows:
            return contexts
        window = max(0, int(self.cfg.semantic_neighbor_window))
        max_words = max(1, int(self.cfg.semantic_neighbor_max_words))
        enriched = 0

        for index, context in enumerate(contexts):
            if index >= max(0, int(self.cfg.semantic_neighbor_max_contexts)):
                break
            chunk_no = _to_optional_int(context.get("chunk_no"))
            if chunk_no is None:
                continue
            offsets: List[int] = []
            parts_by_offset: Dict[int, str] = {}
            for offset in range(-window, window + 1):
                neighbor_chunk_no = int(chunk_no) + offset
                if neighbor_chunk_no <= 0:
                    continue
                if offset == 0:
                    text = _extract_content(context)
                    if not text:
                        continue
                    label = "Current chunk"
                else:
                    key = self._semantic_neighbor_key(context, chunk_no=neighbor_chunk_no)
                    row = neighbor_rows.get(key) if key is not None else None
                    text = _extract_content(row or {})
                    if not text:
                        continue
                    offsets.append(offset)
                    direction = "Previous" if offset < 0 else "Next"
                    label = f"{direction} chunk {neighbor_chunk_no}"
                    text = _truncate_words(text, max_words=max_words)
                parts_by_offset[offset] = f"[{label}]\n{text}"
            ordered_offsets = [0]
            for distance in range(1, window + 1):
                ordered_offsets.extend([-distance, distance])
            parts = [parts_by_offset[offset] for offset in ordered_offsets if offset in parts_by_offset]
            if not offsets or not parts:
                continue
            context["_neighbor_chunk_count"] = len(offsets)
            context["_neighbor_offsets"] = offsets
            context["_rerank_content"] = "\n\n".join(parts)
            enriched += 1

        if enriched:
            logger.info(
                "big_data_semantic_neighbor_context enriched=%s metadata_index=%s window=%s max_contexts=%s",
                enriched,
                self.cfg.chunk_metadata_index,
                window,
                self.cfg.semantic_neighbor_max_contexts,
            )
        return contexts

    async def _attach_semantic_neighbor_context(self, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not contexts:
            return contexts
        neighbor_rows = await self._fetch_semantic_neighbor_rows(contexts)
        return self._attach_semantic_neighbor_context_from_rows(contexts, neighbor_rows)

    async def fetch(
        self,
        *,
        query: str,
        top_n: Optional[int] = None,
        min_score: Optional[float] = None,
        report_type: Optional[str] = None,
        branch: Optional[str] = None,
        doc_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        lang: Optional[str] = None,
        is_attachment: Optional[bool] = None,
        chunk_no: Optional[int] = None,
        document_date_gte: Optional[int] = None,
        document_date_lte: Optional[int] = None,
        ingestion_date_gte: Optional[int] = None,
        ingestion_date_lte: Optional[int] = None,
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("query is required")

        final_top_n = top_n if top_n is not None else self.cfg.top_n_docs
        if final_top_n < 1:
            raise ValueError("top_n must be >= 1")

        requested_threshold = self.cfg.semantic_threshold if min_score is None else float(min_score)
        if requested_threshold < 0:
            raise ValueError("min_score must be >= 0")
        score_floor = max(0.0, float(getattr(self.cfg, "semantic_min_score_floor", 0.0)))
        threshold = max(float(requested_threshold), score_floor)
        if threshold > requested_threshold:
            logger.info(
                "big_data_semantic_min_score_floor applied requested=%s effective=%s",
                requested_threshold,
                threshold,
            )

        qvec = await self.embedder.generate_embeddings({"input": [query]})
        if not qvec:
            raise RuntimeError("Failed to generate embeddings for query")

        selected_collection = collection_name or self.cfg.big_data_collection
        normalized_document_date_gte = _to_optional_epoch_millis(document_date_gte)
        normalized_document_date_lte = _to_optional_epoch_millis(document_date_lte)
        normalized_ingestion_date_gte = _to_optional_epoch_millis(ingestion_date_gte)
        normalized_ingestion_date_lte = _to_optional_epoch_millis(ingestion_date_lte)
        normalized_is_attachment = _to_optional_bool(is_attachment)
        normalized_chunk_no = _to_optional_int(chunk_no)
        payload_filter = _build_payload_filter(
            report_type=report_type,
            branch=branch,
            doc_id=doc_id,
            parent_id=parent_id,
            lang=lang,
            is_attachment=normalized_is_attachment,
            chunk_no=normalized_chunk_no,
            document_date_gte=normalized_document_date_gte,
            document_date_lte=normalized_document_date_lte,
            ingestion_date_gte=normalized_ingestion_date_gte,
            ingestion_date_lte=normalized_ingestion_date_lte,
        )

        raw_limit = max(final_top_n, final_top_n * max(1, int(self.cfg.candidate_multiplier)))
        max_query_limit = max(1, int(getattr(self.cfg, "max_qdrant_query_limit", 800)))
        limit = min(raw_limit, max_query_limit)
        if limit < final_top_n:
            logger.warning(
                "big_data_semantic_top_n capped requested_top_n=%s qdrant_limit=%s max_qdrant_query_limit=%s",
                final_top_n,
                limit,
                max_query_limit,
            )
        has_date_filters = any(
            value is not None
            for value in (
                normalized_document_date_gte,
                normalized_document_date_lte,
                normalized_ingestion_date_gte,
                normalized_ingestion_date_lte,
            )
        )
        try:
            logger.info(
                "big_data_semantic_qdrant_query collection=%s top_n=%s qdrant_limit=%s requested_min_score=%s effective_min_score=%s candidate_multiplier=%s timeout_s=%s",
                selected_collection,
                final_top_n,
                limit,
                requested_threshold,
                threshold,
                self.cfg.candidate_multiplier,
                self.cfg.qdrant_query_timeout_seconds,
            )
            hits = await asyncio.to_thread(
                self.qdrant.query_points,
                collection_name=selected_collection,
                query=qvec,
                query_filter=payload_filter,
                limit=limit,
                score_threshold=threshold,
                with_vectors=False,
                with_payload=True,
                timeout=max(1, int(self.cfg.qdrant_query_timeout_seconds)),
            )
        except Exception as exc:
            logger.exception("Qdrant query failed for big data context: %s", exc)
            raise RuntimeError("Failed to fetch big data document context") from exc

        qdrant_points = list(getattr(hits, "points", []) or [])

        contexts: List[Dict[str, Any]] = []
        for point in qdrant_points:
            sim_score = float(point.score)
            if sim_score < threshold:
                continue

            payload = dict(point.payload or {})
            content = _extract_content(payload)
            if not content:
                continue
            payload["content"] = content
            payload["_scope"] = "semantic"
            payload["_sim_score"] = sim_score

            sort_time_ms = max(
                _to_optional_epoch_millis(payload.get("ingestion_date")) or 0,
                _to_optional_epoch_millis(payload.get("document_date")) or 0,
            )
            payload["_sort_time"] = sort_time_ms
            payload["_recency_score"] = 0.0
            payload["_final_score"] = sim_score
            contexts.append(payload)

        contexts.sort(
            key=lambda p: (
                float(p.get("_sim_score", 0.0)),
                float(p.get("quality_score", 0.0) or 0.0),
                float(p.get("_final_score", 0.0)),
            ),
            reverse=True,
        )
        if has_date_filters:
            logger.info(
                "big_data_semantic_date_filter collection=%s mode=numeric_range raw_candidates=%s kept=%s",
                selected_collection,
                len(qdrant_points),
                len(contexts),
            )
        contexts = contexts[:final_top_n]
        return await self._attach_semantic_neighbor_context(contexts)

    @staticmethod
    def _point_identity(payload: Dict[str, Any]) -> str:
        doc_id = str(payload.get("doc_id") or "").strip()
        parent_id = str(payload.get("parent_id") or "").strip()
        chunk_no = str(payload.get("chunk_no") or "").strip()
        point_id = str(payload.get("point_id") or "").strip()
        content = _extract_content(payload)
        if doc_id or parent_id or chunk_no:
            return f"{doc_id}|{parent_id}|{chunk_no}"
        if point_id:
            return f"point:{point_id}"
        return _normalize_text_for_match(content)[:512]

    def _build_exact_payload(self, payload: Dict[str, Any], *, keywords: List[str], now_ms: int) -> Optional[Dict[str, Any]]:
        item = dict(payload or {})
        content = _extract_content(item)
        if not content:
            return None

        match_info = _score_exact_payload(content, keywords)
        exact_match_score = float(match_info.get("exact_match_score", 0.0))
        if exact_match_score <= 0:
            return None

        item["content"] = content
        item["_matched_keywords"] = list(match_info.get("matched_keywords") or [])
        item["_exact_match_score"] = exact_match_score
        sort_time_ms = max(
            _to_optional_epoch_millis(item.get("ingestion_date")) or 0,
            _to_optional_epoch_millis(item.get("document_date")) or 0,
        )
        item["_sort_time"] = sort_time_ms
        item["_recency_score"] = _recency_boost_ms(sort_time_ms, now_ms)
        return item

    def _finalize_exact_contexts(self, rows: List[Dict[str, Any]], *, top_n: int) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            key = self._point_identity(row)
            if key not in merged:
                merged[key] = row
                continue

            existing = merged[key]
            existing["_exact_match_score"] = max(
                float(existing.get("_exact_match_score", 0.0)),
                float(row.get("_exact_match_score", 0.0)),
            )
            existing["_sim_score"] = max(
                float(existing.get("_sim_score", 0.0)),
                float(row.get("_sim_score", 0.0)),
            )
            existing["_final_score"] = max(
                float(existing.get("_final_score", 0.0)),
                float(row.get("_final_score", 0.0)),
            )
            matches = list(existing.get("_matched_keywords") or [])
            seen = {str(value).lower() for value in matches}
            for value in list(row.get("_matched_keywords") or []):
                key_value = str(value).lower()
                if key_value in seen:
                    continue
                seen.add(key_value)
                matches.append(value)
            existing["_matched_keywords"] = matches

        ordered = list(merged.values())
        ordered.sort(
            key=lambda p: (
                float(p.get("_final_score", 0.0)),
                float(p.get("_exact_match_score", 0.0)),
                float(p.get("_sim_score", 0.0)),
                int(p.get("_sort_time", 0)),
            ),
            reverse=True,
        )
        return ordered[: max(1, int(top_n))]

    def _metadata_exact_hit_to_payload(
        self,
        hit: Dict[str, Any],
        *,
        keywords: List[str],
        now_ms: int,
    ) -> Optional[Dict[str, Any]]:
        source = dict(hit.get("_source") or {})
        raw_content = _extract_content(source)
        if not raw_content:
            return None

        payload: Dict[str, Any] = {
            "content": raw_content,
            "text": str(source.get("text") or raw_content).strip(),
            "doc_id": str(source.get("doc_id") or "").strip(),
            "parent_id": str(source.get("parent_id") or "").strip(),
            "report_type": str(source.get("report_type") or "").strip(),
            "branch": str(source.get("branch") or "").strip(),
            "document_date": source.get("document_date"),
            "ingestion_date": source.get("ingestion_date"),
            "system_path": str(source.get("system_path") or "").strip(),
            "access_branches": str(source.get("access_branches") or "").strip(),
            "access_groups": str(source.get("access_groups") or "").strip(),
            "lang": str(source.get("lang") or "").strip(),
            "is_attachment": _to_optional_bool(source.get("is_attachment")),
            "chunk_no": _to_optional_int(source.get("chunk_no")),
            "total_chunks": _to_optional_int(source.get("total_chunks")),
            "chunk_id": str(source.get("chunk_id") or "").strip(),
            "point_id": str(source.get("point_id") or hit.get("_id") or "").strip(),
            "quality_score": source.get("quality_score"),
            "section_heading": str(source.get("section_heading") or "").strip(),
            "page_start": _to_optional_int(source.get("page_start")),
            "page_end": _to_optional_int(source.get("page_end")),
            "chunk_token_estimate": _to_optional_int(source.get("chunk_token_estimate")),
            "chunking_strategy": str(source.get("chunking_strategy") or "").strip(),
            "qdrant_eligible": _to_optional_bool(source.get("qdrant_eligible")),
            "qdrant_skip_reason": str(source.get("qdrant_skip_reason") or "").strip(),
            "has_been_embedded": _to_optional_bool(source.get("has_been_embedded")),
            "source_index": str(source.get("source_index") or "").strip(),
            "source_es_id": str(source.get("source_es_id") or "").strip(),
            "source_doc_key": str(source.get("source_doc_key") or "").strip(),
            "qdrant_collection": str(source.get("qdrant_collection") or "").strip(),
        }

        payload = self._build_exact_payload(payload, keywords=keywords, now_ms=now_ms)
        if payload is None:
            return None
        return payload

    async def _query_metadata_exact_index(
        self,
        session: aiohttp.ClientSession,
        *,
        base_url: str,
        index_name: str,
        keywords: List[str],
        candidate_size: int,
        report_type: Optional[str],
        branch: Optional[str],
        doc_id: Optional[str],
        parent_id: Optional[str],
        lang: Optional[str],
        is_attachment: Optional[bool],
        chunk_no: Optional[int],
        document_date_gte: Optional[int],
        document_date_lte: Optional[int],
        ingestion_date_gte: Optional[int],
        ingestion_date_lte: Optional[int],
        collection_name: Optional[str],
        now_ms: int,
    ) -> List[Dict[str, Any]]:
        search_url = f"{base_url}/{index_name}/_search"
        filter_clauses = _build_metadata_exact_filter_clauses(
            report_type=report_type,
            branch=branch,
            doc_id=doc_id,
            parent_id=parent_id,
            lang=lang,
            is_attachment=is_attachment,
            chunk_no=chunk_no,
            document_date_gte=document_date_gte,
            document_date_lte=document_date_lte,
            ingestion_date_gte=ingestion_date_gte,
            ingestion_date_lte=ingestion_date_lte,
            collection_name=collection_name,
        )
        should_clauses: List[Dict[str, Any]] = []
        for keyword in keywords:
            should_clauses.extend(_metadata_exact_should_clauses_for_keyword(keyword))

        body = {
            "size": candidate_size,
            "_source": [
                "point_id",
                "content",
                "text",
                "doc_id",
                "parent_id",
                "report_type",
                "branch",
                "document_date",
                "ingestion_date",
                "system_path",
                "access_branches",
                "access_groups",
                "lang",
                "is_attachment",
                "chunk_no",
                "total_chunks",
                "chunk_id",
                "chunk_token_estimate",
                "chunking_strategy",
                "page_start",
                "page_end",
                "section_heading",
                "quality_score",
                "qdrant_eligible",
                "qdrant_skip_reason",
                "has_been_embedded",
                "source_index",
                "source_es_id",
                "source_doc_key",
                "qdrant_collection",
            ],
            "query": {
                "bool": {
                    "filter": filter_clauses,
                    "should": should_clauses,
                    "minimum_should_match": 1,
                }
            },
            "sort": [
                {"_score": {"order": "desc"}},
                {"quality_score": {"order": "desc", "unmapped_type": "float"}},
                {"document_date": {"order": "desc", "unmapped_type": "date"}},
                {"ingestion_date": {"order": "desc", "unmapped_type": "date"}},
            ],
        }

        async with session.post(search_url, json=body) as response:
            if response.status != 200:
                text = await response.text()
                raise RuntimeError(
                    f"Elasticsearch metadata exact lookup failed index={index_name} status={response.status} body={text}"
                )
            data = await response.json()

        hits = ((data or {}).get("hits") or {}).get("hits") or []
        max_elastic_score = max((float(hit.get("_score") or 0.0) for hit in hits), default=0.0)
        rows: List[Dict[str, Any]] = []
        for hit in hits:
            payload = self._metadata_exact_hit_to_payload(
                hit,
                keywords=keywords,
                now_ms=now_ms,
            )
            if payload is None:
                continue
            elastic_score = float(hit.get("_score") or 0.0)
            normalized_score = (elastic_score / max_elastic_score) if max_elastic_score > 0 else 0.0
            main_score_boost = (
                float(self.cfg.exact_main_score_boost)
                if _to_optional_bool(payload.get("is_attachment")) is False
                else 0.0
            )
            payload["_scope"] = "exact_chunk_metadata"
            payload["_sim_score"] = normalized_score
            payload["_elastic_score_raw"] = elastic_score
            payload["_main_score_boost"] = main_score_boost
            payload["_final_score"] = (
                self.cfg.weight_exact_match * float(payload.get("_exact_match_score", 0.0))
                + self.cfg.weight_exact_semantic * normalized_score
                + self.cfg.weight_exact_recency * float(payload.get("_recency_score", 0.0))
                + main_score_boost
            )
            rows.append(payload)

        logger.info(
            "big_data_exact_metadata_fetch index=%s keywords=%s candidates=%s matched=%s filters=%s",
            index_name,
            keywords,
            len(hits),
            len(rows),
            {k: v for k, v in {
                "report_type": report_type,
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
                "collection_name": collection_name,
            }.items() if v not in (None, "", [])},
        )
        return rows

    async def _fetch_exact_metadata_matches(
        self,
        *,
        elasticsearch_base_url: Optional[str],
        keywords: List[str],
        top_n: int,
        report_type: Optional[str],
        branch: Optional[str],
        doc_id: Optional[str],
        parent_id: Optional[str],
        lang: Optional[str],
        is_attachment: Optional[bool],
        chunk_no: Optional[int],
        document_date_gte: Optional[int],
        document_date_lte: Optional[int],
        ingestion_date_gte: Optional[int],
        ingestion_date_lte: Optional[int],
        collection_name: Optional[str],
    ) -> List[Dict[str, Any]]:
        base_url = (elasticsearch_base_url or self.cfg.elasticsearch_base_url).rstrip("/")
        index_name = str(self.cfg.chunk_metadata_index or "").strip()
        if not index_name:
            raise RuntimeError("Chunk metadata Elasticsearch index is not configured")
        now_ms = int(time.time() * 1000)
        final_top_n = max(1, int(top_n))
        max_candidates_per_index = max(0, int(self.cfg.exact_max_candidates_per_index))
        raw_candidate_size = max(
            final_top_n * max(1, int(self.cfg.exact_candidate_multiplier)),
            len(list(keywords or [])) * 12,
            48,
        )
        candidate_size = (
            min(raw_candidate_size, max_candidates_per_index)
            if max_candidates_per_index > 0
            else raw_candidate_size
        )
        if candidate_size < raw_candidate_size:
            logger.info(
                "big_data_exact_metadata_candidate_size_capped index=%s top_n=%s raw_candidate_size=%s capped_candidate_size=%s max_candidates_per_index=%s multiplier=%s",
                index_name,
                final_top_n,
                raw_candidate_size,
                candidate_size,
                max_candidates_per_index,
                self.cfg.exact_candidate_multiplier,
            )

        normalized_document_date_gte = _to_optional_epoch_millis(document_date_gte)
        normalized_document_date_lte = _to_optional_epoch_millis(document_date_lte)
        normalized_is_attachment = _to_optional_bool(is_attachment)
        normalized_chunk_no = _to_optional_int(chunk_no)
        normalized_ingestion_date_gte = _to_optional_epoch_millis(ingestion_date_gte)
        normalized_ingestion_date_lte = _to_optional_epoch_millis(ingestion_date_lte)

        connector = TCPConnector(ssl=not bool(self.cfg.elasticsearch_disable_ssl_verify))
        auth = BasicAuth(
            login=self.cfg.elasticsearch_username,
            password=self.cfg.elasticsearch_password,
        )

        try:
            async with aiohttp.ClientSession(
                connector=connector,
                auth=auth,
                timeout=self.http_timeout,
            ) as session:
                rows = await self._query_metadata_exact_index(
                    session,
                    base_url=base_url,
                    index_name=index_name,
                    keywords=keywords,
                    candidate_size=max(1, int(candidate_size)),
                    report_type=report_type,
                    branch=branch,
                    doc_id=doc_id,
                    parent_id=parent_id,
                    lang=lang,
                    is_attachment=normalized_is_attachment,
                    chunk_no=normalized_chunk_no,
                    document_date_gte=normalized_document_date_gte,
                    document_date_lte=normalized_document_date_lte,
                    ingestion_date_gte=normalized_ingestion_date_gte,
                    ingestion_date_lte=normalized_ingestion_date_lte,
                    collection_name=collection_name,
                    now_ms=now_ms,
                )
        except Exception as exc:
            logger.exception(
                "Elasticsearch metadata exact lookup failed index=%s keywords=%s error=%s",
                index_name,
                keywords,
                exc,
            )
            raise RuntimeError("Failed to fetch exact chunk metadata context from Elasticsearch") from exc

        logger.info(
            "big_data_exact_metadata_result index=%s keywords=%s candidates_requested=%s returned=%s",
            index_name,
            keywords,
            candidate_size,
            len(rows),
        )
        return self._finalize_exact_contexts(rows, top_n=final_top_n)

    async def fetch_exact(
        self,
        *,
        query: str,
        keywords: Optional[List[str]] = None,
        top_n: Optional[int] = None,
        min_score: Optional[float] = None,
        elasticsearch_base_url: Optional[str] = None,
        elasticsearch_index: Optional[str] = None,
        report_type: Optional[str] = None,
        branch: Optional[str] = None,
        doc_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        lang: Optional[str] = None,
        is_attachment: Optional[bool] = None,
        chunk_no: Optional[int] = None,
        document_date_gte: Optional[int] = None,
        document_date_lte: Optional[int] = None,
        ingestion_date_gte: Optional[int] = None,
        ingestion_date_lte: Optional[int] = None,
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("query is required")

        exact_keywords = _dedupe_keywords(
            list(keywords or []),
            max_items=int(self.cfg.exact_max_keywords),
        )
        if not exact_keywords:
            raise ValueError("keywords are required for exact retrieval")

        final_top_n = top_n if top_n is not None else self.cfg.exact_top_n_docs
        if final_top_n < 1:
            raise ValueError("top_n must be >= 1")

        del min_score
        if str(elasticsearch_index or "").strip():
            logger.info(
                "big_data_exact_fetch ignoring legacy elasticsearch_index=%s and using chunk_metadata_index=%s",
                elasticsearch_index,
                self.cfg.chunk_metadata_index,
            )

        rows = await self._fetch_exact_metadata_matches(
            elasticsearch_base_url=elasticsearch_base_url,
            keywords=exact_keywords,
            top_n=final_top_n,
            report_type=report_type,
            branch=branch,
            doc_id=doc_id,
            parent_id=parent_id,
            lang=lang,
            is_attachment=is_attachment,
            chunk_no=chunk_no,
            document_date_gte=document_date_gte,
            document_date_lte=document_date_lte,
            ingestion_date_gte=ingestion_date_gte,
            ingestion_date_lte=ingestion_date_lte,
            collection_name=collection_name,
        )
        logger.info(
            "big_data_exact_fetch query_len=%s keywords=%s returned=%s chunk_metadata_index=%s collection_name=%s",
            len(str(query or "")),
            exact_keywords,
            len(rows),
            self.cfg.chunk_metadata_index,
            collection_name,
        )
        return rows
