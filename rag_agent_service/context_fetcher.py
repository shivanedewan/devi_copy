from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


class ContextFetcher:
    def __init__(
        self,
        conv_context_url: str,
        file_context_url: str,
        big_data_context_url: str,
        big_data_exact_context_url: str = "",
        *,
        timeout_seconds: float = 12.0,
        retries: int = 2,
    ):
        self._conv_context_url = conv_context_url
        self._file_context_url = file_context_url
        self._big_data_context_url = big_data_context_url
        self._big_data_exact_context_url = big_data_exact_context_url
        self._timeout_seconds = timeout_seconds
        self._retries = max(0, retries)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout_seconds),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post(
        self,
        url: str,
        payload: Dict[str, Any],
        *,
        prefer_query_params: bool = True,
    ) -> Dict[str, Any] | list[Dict[str, Any]]:
        if not url:
            raise RuntimeError("Context endpoint URL is not configured")

        params = {k: v for k, v in payload.items() if v is not None}
        last_exc: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                if prefer_query_params:
                    response = await self._client.post(url, params=params)
                    if response.status_code in {400, 405, 415, 422}:
                        alt_response = await self._client.post(url, json=params)
                        if alt_response.status_code < 400:
                            response = alt_response
                else:
                    response = await self._client.post(url, json=params)
                    if response.status_code in {400, 405, 415, 422}:
                        alt_response = await self._client.post(url, params=params)
                        if alt_response.status_code < 400:
                            response = alt_response

                response.raise_for_status()
                data = response.json()
                if isinstance(data, (dict, list)):
                    return data
                raise RuntimeError(f"Invalid context response type: {type(data).__name__}")
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Context request failed attempt=%s/%s url=%s error=%s",
                    attempt + 1,
                    self._retries + 1,
                    url,
                    exc,
                )
                if attempt < self._retries:
                    await asyncio.sleep(0.2 * (attempt + 1))

        raise RuntimeError(f"Context request failed for {url}: {last_exc}")

    async def fetch_conv_context(
        self,
        *,
        user_id: str,
        chat_id: str,
        message_id: str,
        query: str,
        semantic_threshold: Optional[float] = None,
        top_n: Optional[int] = None,
        enable_semantic_search: Optional[bool] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "user_id": user_id,
            "chat_id": chat_id,
            "message_id": message_id,
            "query": query,
            "semantic_threshold": semantic_threshold,
            "top_n": top_n,
            "enable_semantic_search": enable_semantic_search,
        }
        data = await self._post(self._conv_context_url, payload, prefer_query_params=True)
        if isinstance(data, dict):
            return data
        return {}

    async def fetch_file_context(
        self,
        *,
        user_id: str,
        chat_id: str,
        query: str,
        top_n: int,
        min_score: float = 0.0,
        file_id: Optional[str] = None,
        retrieval_mode: str = "semantic",
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "user_id": user_id,
            "chat_id": chat_id,
            "query": query,
            "top_n": top_n,
            "min_score": min_score,
            "file_id": file_id,
            "retrieval_mode": retrieval_mode,
        }
        data = await self._post(self._file_context_url, payload, prefer_query_params=True)
        if isinstance(data, dict):
            return data
        return {"file_context": []}

    async def fetch_bigdata_context(
        self,
        *,
        query: str,
        top_n: int,
        min_score: float = 0.0,
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
    ) -> list[Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "query": query,
            "top_n": top_n,
            "min_score": min_score,
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
        }
        # Send JSON body first to avoid very long query-string URLs for metadata-heavy retrieval.
        data = await self._post(self._big_data_context_url, payload, prefer_query_params=False)
        if isinstance(data, list):
            return data
        return []

    async def fetch_bigdata_exact_context(
        self,
        *,
        query: str,
        keywords: list[str],
        top_n: int,
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
    ) -> list[Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "query": query,
            "keywords": list(keywords or []),
            "top_n": top_n,
            "elasticsearch_base_url": elasticsearch_base_url,
            "elasticsearch_index": elasticsearch_index,
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
        }
        data = await self._post(self._big_data_exact_context_url, payload, prefer_query_params=False)
        if isinstance(data, list):
            return data
        return []
