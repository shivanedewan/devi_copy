from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


def created_at_to_epoch(value: Any) -> int:
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


def recency_boost(created_at_epoch: int, now_epoch: int, half_life_seconds: float) -> float:
    age_seconds = max(0, now_epoch - created_at_epoch)
    divisor = half_life_seconds if half_life_seconds > 0 else 3600.0
    return 1.0 / (1.0 + (age_seconds / divisor))


class SharedEmbedder:
    def __init__(
        self,
        embed_endpoint: str,
        *,
        model_name: str = "qwen-embed_8b",
        timeout_seconds: float = 15.0,
        retries: int = 2,
        backoff_base_seconds: float = 0.2,
    ):
        self.embed_endpoint = embed_endpoint
        self.model_name = str(model_name or "qwen-embed_8b")
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self.retries = max(0, retries)
        self.backoff_base_seconds = max(0.05, backoff_base_seconds)
        self._session: Optional[aiohttp.ClientSession] = None
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

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

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
        last_exc: Optional[Exception] = None
        for attempt in range(self.retries + 1):
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

                session = await self._get_session()
                async with session.post(self.embed_endpoint, json=request_payload) as response:
                    if response.status != 200:
                        body = await response.text()
                        logger.error(
                            "Embedding request failed status=%s body=%s attempt=%s",
                            response.status,
                            body,
                            attempt + 1,
                        )
                        if attempt < self.retries:
                            await asyncio.sleep(self.backoff_base_seconds * (2**attempt))
                            continue
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
                last_exc = exc
                logger.exception("Embedding request error attempt=%s: %s", attempt + 1, exc)
                if attempt < self.retries:
                    await asyncio.sleep(self.backoff_base_seconds * (2**attempt))

        if last_exc is not None:
            logger.error("Embedding request failed after retries: %s", last_exc)
        return None
