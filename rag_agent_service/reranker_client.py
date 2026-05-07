from __future__ import annotations

import logging
from typing import Any, Dict, List

import httpx

logger = logging.getLogger(__name__)


class NativeRerankerClient:
    def __init__(self, base_url: str, *, timeout_seconds: float = 6.0, model_name: str = "") -> None:
        self._base_url = str(base_url or "").rstrip("/")
        self._model_name = str(model_name or "").strip()
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(max(0.5, float(timeout_seconds))),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _endpoint_candidates(self) -> List[str]:
        if not self._base_url:
            return []
        lowered = self._base_url.lower()
        if lowered.endswith("/rerank") or lowered.endswith("/v1/rerank"):
            return [self._base_url]
        if lowered.endswith("/v1"):
            root_base_url = self._base_url[:-3].rstrip("/")
            return [f"{root_base_url}/rerank", f"{self._base_url}/rerank"]
        return [f"{self._base_url}/rerank", f"{self._base_url}/v1/rerank"]

    @staticmethod
    def _coerce_score(value: Any) -> float | None:
        try:
            return float(value)
        except Exception:
            return None

    @classmethod
    def _normalize_results(cls, payload: Any, *, doc_count: int) -> List[Dict[str, Any]]:
        raw_items: Any = payload
        if isinstance(payload, dict):
            if isinstance(payload.get("results"), list):
                raw_items = payload.get("results")
            elif isinstance(payload.get("data"), list):
                raw_items = payload.get("data")
            elif isinstance(payload.get("output"), list):
                raw_items = payload.get("output")
        if not isinstance(raw_items, list):
            raise RuntimeError(f"Unsupported reranker response type: {type(payload).__name__}")

        rows: List[Dict[str, Any]] = []
        seen: set[int] = set()
        for fallback_index, item in enumerate(raw_items):
            if not isinstance(item, dict):
                continue
            try:
                raw_index = item.get("index", item.get("document_index", item.get("id")))
                index = fallback_index if raw_index is None else int(raw_index)
            except Exception:
                continue
            if index < 0 or index >= max(0, int(doc_count)) or index in seen:
                continue
            score = cls._coerce_score(
                item.get("score", item.get("relevance_score", item.get("relevance", item.get("logit"))))
            )
            if score is None:
                continue
            seen.add(index)
            rows.append({"index": int(index), "score": float(score)})
        if not rows:
            raise RuntimeError("Reranker response did not contain any usable scores")
        rows.sort(key=lambda item: (float(item["score"]), -int(item["index"])), reverse=True)
        return rows

    async def rerank(self, *, query: str, documents: List[str]) -> List[Dict[str, Any]]:
        clean_query = str(query or "").strip()
        docs = [str(item or "").strip() for item in list(documents or [])]
        if not clean_query:
            raise ValueError("query is required for reranking")
        if not docs:
            return []

        payload_candidates: List[Dict[str, Any]] = [
            {"query": clean_query, "documents": docs},
            {"query": clean_query, "documents": [{"text": doc} for doc in docs]},
            {"query": clean_query, "texts": docs},
            {"query": clean_query, "input": docs},
        ]
        if self._model_name:
            payload_candidates = [
                {"model": self._model_name, **payload}
                for payload in payload_candidates
            ]
        last_error: Exception | None = None

        for url in self._endpoint_candidates():
            for payload in payload_candidates:
                try:
                    response = await self._client.post(url, json=payload)
                    if response.status_code in {400, 404, 405, 415, 422}:
                        last_error = RuntimeError(
                            f"Reranker rejected request status={response.status_code} url={url}"
                        )
                        if response.status_code == 404:
                            break
                        continue
                    response.raise_for_status()
                    return self._normalize_results(response.json(), doc_count=len(docs))
                except httpx.TimeoutException as exc:
                    raise RuntimeError(f"Reranker request timed out url={url}") from exc
                except httpx.HTTPStatusError as exc:
                    status_code = getattr(exc.response, "status_code", 0)
                    if status_code in {400, 404, 405, 415, 422}:
                        last_error = exc
                        if status_code == 404:
                            break
                        continue
                    raise RuntimeError(f"Reranker request failed status={status_code} url={url}") from exc
                except httpx.RequestError as exc:
                    raise RuntimeError(f"Reranker request transport error url={url}: {exc}") from exc
                except Exception as exc:
                    last_error = exc
                    logger.debug("Reranker request candidate failed url=%s error=%s", url, exc)

        raise RuntimeError(f"Reranker request failed: {last_error}")
