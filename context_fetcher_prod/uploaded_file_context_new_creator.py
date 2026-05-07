import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from uploaded_file_context_creator import UploadedFileContextFetcher

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)


@dataclass
class Config:
    max_content_chars_per_chunk_semantic: int = _env_int("UPLOADED_FILE_CONTEXT_SEMANTIC_MAX_CHARS", 0)
    max_content_chars_per_chunk_full_file: int = _env_int("UPLOADED_FILE_CONTEXT_FULL_FILE_MAX_CHARS", 0)
    default_top_n_semantic: int = 5
    context_preview_chunks: int = 8
    source_documents_limit: int = 50


class UploadedFileContextNewFetcher:
    def __init__(
        self,
        uploaded_file_fetcher: UploadedFileContextFetcher,
        cfg: Optional[Config] = None,
    ):
        self.uploaded_file_fetcher = uploaded_file_fetcher
        self.cfg = cfg or Config()

    async def fetch(
        self,
        *,
        user_id: str,
        chat_id: str,
        query: str,
        top_n: Optional[int] = None,
        min_score: float = 0.5,
        retrieval_mode: str = "semantic",
        file_id: Optional[str] = None,
        resolve_latest_file: bool = False,
        latest_file_limit: Optional[int] = None,
        before_created_at: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not user_id or not chat_id:
            raise ValueError("user_id and chat_id are required")
        if retrieval_mode != "full_file" and (not query or not query.strip()):
            raise ValueError("query is required")
        if top_n is not None and top_n < 0:
            raise ValueError("top_n must be >= 0")
        if min_score < 0:
            raise ValueError("min_score must be >= 0")

        effective_top_n: Optional[int]
        if retrieval_mode == "full_file":
            effective_top_n = top_n
        else:
            effective_top_n = self.cfg.default_top_n_semantic if top_n is None else top_n
            if effective_top_n < 1:
                raise ValueError("top_n must be >= 1")

        logger.info(
            "uploaded_file_context_new fetch start user_id=%s chat_id=%s mode=%s requested_top_n=%s effective_top_n=%s min_score=%s file_id=%s resolve_latest_file=%s latest_file_limit=%s before_created_at=%s",
            user_id,
            chat_id,
            retrieval_mode,
            top_n,
            effective_top_n,
            min_score,
            file_id,
            resolve_latest_file,
            latest_file_limit,
            before_created_at,
        )

        chunks = await self.uploaded_file_fetcher.fetch(
            user_id=user_id,
            chat_id=chat_id,
            query=query,
            top_n=effective_top_n,
            min_score=min_score,
            retrieval_mode=retrieval_mode,
            file_id=file_id,
            resolve_latest_file=resolve_latest_file,
            latest_file_limit=latest_file_limit,
            before_created_at=before_created_at,
            collection_name=collection_name,
        )

        file_context = self._build_file_context(chunks, retrieval_mode=retrieval_mode)
        context = self._render_context(file_context=file_context)
        source_documents = self._build_source_documents(file_context=file_context)

        logger.info(
            "uploaded_file_context_new built chat_id=%s mode=%s chunks=%s source_documents=%s",
            chat_id,
            retrieval_mode,
            len(file_context),
            len(source_documents),
        )

        return {
            "context": context,
            "file_context": file_context,
            "file_context_count": len(file_context),
            "source_documents": source_documents,
        }

    def _trim_content(self, text: str, *, max_chars: int) -> str:
        value = (text or "").strip()
        if max_chars > 0 and len(value) > max_chars:
            return value[:max_chars].rstrip() + "..."
        return value

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except Exception:
            return default

    def _build_file_context(self, chunks: List[Dict[str, Any]], *, retrieval_mode: str) -> List[Dict[str, Any]]:
        max_chars = (
            self.cfg.max_content_chars_per_chunk_full_file
            if retrieval_mode == "full_file"
            else self.cfg.max_content_chars_per_chunk_semantic
        )
        out: List[Dict[str, Any]] = []
        for chunk in chunks:
            content = self._trim_content(
                str(
                    chunk.get("content")
                    or chunk.get("text")
                    or chunk.get("chunk_text")
                    or chunk.get("page_content")
                    or ""
                ),
                max_chars=max_chars,
            )
            if not content:
                continue

            out.append(
                {
                    "file_id": chunk.get("file_id"),
                    "chunk_id": chunk.get("chunk_id"),
                    # changed
                    # Preserve chunk_no so downstream services can keep document order
                    # when building report prompts and traces.
                    "chunk_no": chunk.get("chunk_no"),
                    "file_name": str(
                        chunk.get("file_name")
                        or chunk.get("title")
                        or chunk.get("filename")
                        or chunk.get("name")
                        or "Unknown"
                    ).strip() or "Unknown",
                    "content": content,
                    "score": self._safe_float(chunk.get("score", chunk.get("_final_score", 0.0))),
                    "sim_score": self._safe_float(chunk.get("sim_score", chunk.get("_sim_score", chunk.get("score", 0.0)))),
                    "chunk_recency_score": self._safe_float(chunk.get("chunk_recency_score", chunk.get("_chunk_recency_score", 0.0))),
                    "file_recency_score": self._safe_float(chunk.get("file_recency_score", chunk.get("_file_recency_score", 0.0))),
                    "scope": str(chunk.get("_scope", "semantic")),
                    "full_file_oversize_fallback": bool(
                        chunk.get("full_file_oversize_fallback", chunk.get("_full_file_oversize_fallback", False))
                    ),
                    "created_at": chunk.get("created_at"),
                }
            )
        return out

    def _build_source_documents(self, file_context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        documents: List[Dict[str, Any]] = []
        for chunk in file_context[: max(1, self.cfg.source_documents_limit)]:
            documents.append(
                {
                    "file_id": chunk.get("file_id"),
                    "chunk_id": chunk.get("chunk_id"),
                    "file_name": chunk.get("file_name"),
                    "score": chunk.get("score"),
                    "excerpt": self._trim_content(
                        chunk.get("content", ""),
                        max_chars=self.cfg.max_content_chars_per_chunk_semantic,
                    )[:240],
                }
            )
        return documents

    def _render_context(self, *, file_context: List[Dict[str, Any]]) -> str:
        lines: List[str] = ["### File Context", f"- Total chunks: {len(file_context)}"]
        if not file_context:
            lines.append("- No additional file context provided.")
            return "\n".join(lines)

        preview_limit = max(1, self.cfg.context_preview_chunks)
        for idx, chunk in enumerate(file_context[:preview_limit], 1):
            snippet = str(chunk.get("content", "")).strip().replace("\n", " ")
            if len(snippet) > 160:
                snippet = snippet[:160].rstrip() + "..."
            lines.append(
                f"- [{idx}] file={chunk.get('file_name')} score={float(chunk.get('score', 0.0)):.4f}"
            )
            lines.append(f"  snippet: {snippet}")
        remaining = len(file_context) - preview_limit
        if remaining > 0:
            lines.append(f"- ... and {remaining} more chunks")
        return "\n".join(lines)
