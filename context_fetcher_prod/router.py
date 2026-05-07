
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, status
from minio import Minio
from minio.error import S3Error
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client import models as qm

from big_data_documents_context_creator import (
    BigDataDocumentsContextFetcher,
    Config as BigDataConfig,
    Embedder as BigDataEmbedder,
)
from chat_context_new_creator import ChatContextNewFetcher
from context_creator import Config, ContextFetcher, Embedder
from qdrant_utils import QdrantUtils
from uploaded_file_context_creator import (
    Config as UploadedFileConfig,
    Embedder as UploadedFileEmbedder,
    UploadedFileContextFetcher,
)
from uploaded_file_context_new_creator import UploadedFileContextNewFetcher

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ContextBuilder"])

QDRANT_URL = os.getenv("QDRANT_URL", "http://192.168.10.32:6333")
QDRANT_TIMEOUT_SECONDS = float(os.getenv("QDRANT_TIMEOUT_SECONDS", "60"))
EMBED_URL = os.getenv("EMBED_URL", "http://192.168.10.210:9084/v1/embeddings")
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen-embed_8b")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "192.168.10.35:9198")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "user1")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "VMware@123")
FILE_UPLOADS_BUCKET = os.getenv("FILE_UPLOADS_BUCKET", "app-uploads")
UPLOADED_FILE_COLLECTION = os.getenv(
    "UPLOADED_FILE_COLLECTION",
    "chat_attachment_chunks_production_1",
)
CHAT_MESSAGES_COLLECTION = os.getenv(
    "CHAT_MESSAGES_COLLECTION",
    "chat_messages_production_1",
)

qdrant = QdrantClient(url=QDRANT_URL, timeout=QDRANT_TIMEOUT_SECONDS)
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)
fetcher = ContextFetcher(
    qdrant=qdrant,
    embedder=Embedder(EMBED_URL, EMBED_MODEL),
    cfg=Config(),
)
chat_context_new_fetcher = ChatContextNewFetcher(context_fetcher=fetcher)
big_data_fetcher = BigDataDocumentsContextFetcher(
    qdrant=qdrant,
    embedder=BigDataEmbedder(EMBED_URL, EMBED_MODEL),
    cfg=BigDataConfig(),
)
uploaded_file_fetcher = UploadedFileContextFetcher(
    qdrant=qdrant,
    embedder=UploadedFileEmbedder(EMBED_URL, EMBED_MODEL),
    cfg=UploadedFileConfig(),
)
uploaded_file_context_new_fetcher = UploadedFileContextNewFetcher(
    uploaded_file_fetcher=uploaded_file_fetcher
)

qutils = QdrantUtils(qdrant=qdrant)


class BigDataRequest(BaseModel):
    query: str
    top_n: int = 6
    min_score: float = 0.30
    report_type: Optional[str] = None
    branch: Optional[str] = None
    doc_id: Optional[str] = None
    parent_id: Optional[str] = None
    lang: Optional[str] = None
    is_attachment: Optional[bool] = None
    chunk_no: Optional[int] = None
    document_date_gte: Optional[int] = None
    document_date_lte: Optional[int] = None
    ingestion_date_gte: Optional[int] = None
    ingestion_date_lte: Optional[int] = None
    collection_name: Optional[str] = None


class BigDataExactRequest(BigDataRequest):
    keywords: list[str] = Field(default_factory=list)
    elasticsearch_base_url: Optional[str] = None
    elasticsearch_index: Optional[str] = None


class FullFilesRequest(BaseModel):
    user_id: str
    chat_id: str
    attachment_id: List[str] = Field(default_factory=list)

    def file_ids(self) -> List[str]:
        seen = set()
        values: List[str] = []
        for file_id in self.attachment_id:
            clean_id = str(file_id or "").strip()
            if clean_id and clean_id not in seen:
                seen.add(clean_id)
                values.append(clean_id)
        return values


class MessageContentRequest(BaseModel):
    user_id: str
    chat_id: str
    message_id: str


def _raise_internal_error(message: str, exc: Exception) -> None:
    logger.exception("%s: %s", message, exc)
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=message,
    )


def _attachment_filter(user_id: str, chat_id: str, file_ids: List[str]) -> qm.Filter:
    return qm.Filter(
        must=[
            qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id)),
            qm.FieldCondition(key="chat_id", match=qm.MatchValue(value=chat_id)),
            qm.FieldCondition(key="file_id", match=qm.MatchAny(any=file_ids)),
        ]
    )


def _message_filter(user_id: str, chat_id: str, message_id: str) -> qm.Filter:
    return qm.Filter(
        must=[
            qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id)),
            qm.FieldCondition(key="chat_id", match=qm.MatchValue(value=chat_id)),
            qm.FieldCondition(key="message_id", match=qm.MatchValue(value=message_id)),
        ]
    )


def _get_message_payloads_from_qdrant(
    *,
    user_id: str,
    chat_id: str,
    message_id: str,
) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    next_offset = None

    while True:
        points, next_offset = qdrant.scroll(
            collection_name=CHAT_MESSAGES_COLLECTION,
            scroll_filter=_message_filter(user_id, chat_id, message_id),
            limit=64,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = dict(point.payload or {})
            payload["_point_id"] = str(point.id)
            payloads.append(payload)

        if next_offset is None:
            break

    chunk_order = {
        str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{message_id}_{idx}")): idx
        for idx in range(len(payloads))
    }
    return sorted(
        payloads,
        key=lambda item: chunk_order.get(str(item.get("_point_id") or ""), 10**9),
    )


def _get_file_titles_from_qdrant(
    *,
    user_id: str,
    chat_id: str,
    file_ids: List[str],
) -> Dict[str, Dict[str, Any]]:
    found: Dict[str, Dict[str, Any]] = {}
    next_offset = None

    while len(found) < len(file_ids):
        points, next_offset = qdrant.scroll(
            collection_name=UPLOADED_FILE_COLLECTION,
            scroll_filter=_attachment_filter(user_id, chat_id, file_ids),
            limit=64,
            offset=next_offset,
            with_payload=["file_id", "title", "bucket_name"],
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}
            file_id = str(payload.get("file_id") or "").strip()
            title = str(payload.get("title") or "").strip()
            if file_id and title and file_id not in found:
                found[file_id] = {
                    "file_name": title,
                    "bucket_name": payload.get("bucket_name") or FILE_UPLOADS_BUCKET,
                }

        if next_offset is None:
            break

    return found


def _read_extracted_file_from_minio(
    *,
    user_id: str,
    file_id: str,
    file_name: str,
    bucket_name: str,
) -> str:
    root, _ = os.path.splitext(file_name)
    markdown_file_name = f"{root}.md"
    object_name = f"{user_id}/{file_id}/extracted_text/{markdown_file_name}"

    try:
        response = minio_client.get_object(bucket_name, object_name)
        try:
            return response.read().decode("utf-8")
        finally:
            response.close()
            response.release_conn()
    except S3Error as exc:
        raise RuntimeError(f"Unable to read {object_name} from bucket {bucket_name}") from exc


@router.post("/")
async def get_context(
    user_id: str,
    chat_id: str,
    message_id: str,
    query: str = "",
    semantic_threshold: Optional[float] = None,
):
    logger.info(
        "context request user_id=%s chat_id=%s message_id=%s query_len=%s semantic_threshold=%s",
        user_id,
        chat_id,
        message_id,
        len(str(query or "")),
        semantic_threshold,
    )
    try:
        return await fetcher.fetch(
            user_id=user_id,
            chat_id=chat_id,
            message_id=message_id,
            query=query,
            semantic_threshold=semantic_threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except RuntimeError as exc:
        _raise_internal_error(str(exc), exc)
    except Exception as exc:
        _raise_internal_error("Unexpected error while building chat context", exc)


@router.post("/big_data_documents_context")
async def get_big_data_documents_context(
    request: Optional[BigDataRequest] = Body(default=None),
    query: Optional[str] = None,
    top_n: int = 6,
    min_score: float = 0.30,
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
):
    payload = (
        request.model_dump()
        if request is not None
        else {
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
    )
    if not str(payload.get("query") or "").strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="query is required")

    logger.info(
        "big_data_documents_context request top_n=%s min_score=%s made_query=%s query_len=%s collection_name=%s",
        payload.get("top_n"),
        payload.get("min_score"),
        payload.get("query"),
        len(str(payload.get("query") or "")),
        payload.get("collection_name"),
    )
    try:
        return await big_data_fetcher.fetch(**payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except RuntimeError as exc:
        _raise_internal_error(str(exc), exc)
    except Exception as exc:
        _raise_internal_error("Unexpected error while building big-data document context", exc)


@router.post("/big_data_documents_exact_context")
async def get_big_data_documents_exact_context(request: BigDataExactRequest):
    payload = request.model_dump()
    if not str(payload.get("query") or "").strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="query is required")
    if not list(payload.get("keywords") or []):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="keywords are required")

    logger.info(
        "big_data_documents_exact_context request top_n=%s made_query=%s query_len=%s keywords=%s collection_name=%s legacy_elasticsearch_index=%s",
        payload.get("top_n"),
        payload.get("query"),
        len(str(payload.get("query") or "")),
        list(payload.get("keywords") or []),
        payload.get("collection_name"),
        payload.get("elasticsearch_index"),
    )
    try:
        return await big_data_fetcher.fetch_exact(**payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except RuntimeError as exc:
        _raise_internal_error(str(exc), exc)
    except Exception as exc:
        _raise_internal_error("Unexpected error while building big-data exact context", exc)


@router.post("/chat_context_new")
async def get_chat_context_new(
    user_id: str,
    chat_id: str,
    message_id: str,
    query: str = "",
    top_n: int = 9,
    semantic_threshold: Optional[float] = None,
    enable_semantic_search: bool = True,
    tail_max_turns: Optional[int] = None,
    tail_token_budget: Optional[int] = None,
    semantic_history_max_turns: Optional[int] = None,
    semantic_history_token_budget: Optional[int] = None,
    total_token_budget: Optional[int] = None,
    chars_per_token: Optional[float] = None,
    include_chronological_anchors: Optional[bool] = None,
    chronological_anchor_turns: Optional[int] = None,
):
    logger.info(
        "chat_context_new request user_id=%s chat_id=%s message_id=%s query_len=%s top_n=%s semantic_threshold=%s enable_semantic_search=%s",
        user_id,
        chat_id,
        message_id,
        len(str(query or "")),
        top_n,
        semantic_threshold,
        enable_semantic_search,
    )
    try:
        return await chat_context_new_fetcher.fetch(
            user_id=user_id,
            chat_id=chat_id,
            message_id=message_id,
            query=query,
            top_n=top_n,
            semantic_threshold=semantic_threshold,
            enable_semantic_search=enable_semantic_search,
            tail_max_turns=tail_max_turns,
            tail_token_budget=tail_token_budget,
            semantic_history_max_turns=semantic_history_max_turns,
            semantic_history_token_budget=semantic_history_token_budget,
            total_token_budget=total_token_budget,
            chars_per_token=chars_per_token,
            include_chronological_anchors=include_chronological_anchors,
            chronological_anchor_turns=chronological_anchor_turns,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except RuntimeError as exc:
        _raise_internal_error(str(exc), exc)
    except Exception as exc:
        _raise_internal_error("Unexpected error while building chat context (new template)", exc)


@router.post("/uploaded_file_context")
async def get_uploaded_file_context(
    user_id: str,
    chat_id: str,
    query: str = "",
    top_n: int = 5,
    min_score: float = 0.5,
    retrieval_mode: str = "semantic",
    file_id: Optional[str] = None,
    resolve_latest_file: bool = False,
    latest_file_limit: Optional[int] = None,
    before_created_at: Optional[str] = None,
    collection_name: Optional[str] = None,
):
    logger.info(
        "uploaded_file_context request user_id=%s chat_id=%s query_len=%s top_n=%s min_score=%s retrieval_mode=%s file_id=%s resolve_latest_file=%s latest_file_limit=%s before_created_at=%s",
        user_id,
        chat_id,
        len(str(query or "")),
        top_n,
        min_score,
        retrieval_mode,
        file_id,
        resolve_latest_file,
        latest_file_limit,
        before_created_at,
    )
    try:
        return await uploaded_file_fetcher.fetch(
            user_id=user_id,
            chat_id=chat_id,
            query=query,
            top_n=top_n,
            min_score=min_score,
            retrieval_mode=retrieval_mode,
            file_id=file_id,
            resolve_latest_file=resolve_latest_file,
            latest_file_limit=latest_file_limit,
            before_created_at=before_created_at,
            collection_name=collection_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except RuntimeError as exc:
        _raise_internal_error(str(exc), exc)
    except Exception as exc:
        _raise_internal_error("Unexpected error while building uploaded-file context", exc)


@router.post("/uploaded_file_context_new")
async def get_uploaded_file_context_new(
    user_id: str,
    chat_id: str,
    query: str = "",
    top_n: Optional[int] = None,
    min_score: float = 0.0,
    retrieval_mode: str = "semantic",
    file_id: Optional[str] = None,
    resolve_latest_file: bool = False,
    latest_file_limit: Optional[int] = None,
    before_created_at: Optional[str] = None,
    collection_name: Optional[str] = None,
):
    logger.info(
        "uploaded_file_context_new request user_id=%s chat_id=%s query_len=%s top_n=%s min_score=%s retrieval_mode=%s file_id=%s resolve_latest_file=%s latest_file_limit=%s before_created_at=%s",
        user_id,
        chat_id,
        len(str(query or "")),
        top_n,
        min_score,
        retrieval_mode,
        file_id,
        resolve_latest_file,
        latest_file_limit,
        before_created_at,
    )
    try:
        return await uploaded_file_context_new_fetcher.fetch(
            user_id=user_id,
            chat_id=chat_id,
            query=query,
            top_n=top_n,
            min_score=min_score,
            retrieval_mode=retrieval_mode,
            file_id=file_id,
            resolve_latest_file=resolve_latest_file,
            latest_file_limit=latest_file_limit,
            before_created_at=before_created_at,
            collection_name=collection_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except RuntimeError as exc:
        _raise_internal_error(str(exc), exc)
    except Exception as exc:
        _raise_internal_error("Unexpected error while building uploaded-file context (new template)", exc)


#  new endpoints for new contracts








# {
#   "user_id": "user123",
#   "chat_id": "chat456",
#   "attachement_id": ["file_id_abcd", "file_id_bdfg"]
# }


# request body example for /get_full_files_for_each_ids:
# {
#   "files": {
#     "file_id_abcd": {
#       "file_name": "abcd.pdf",
#       "content": "full extracted markdown text..."
#     }
#   },
#   "missing_file_ids": []
# }

@router.post("/get_full_files_for_each_ids")
async def get_full_files_for_each_ids(request: FullFilesRequest):
    file_ids = request.file_ids()
    if not file_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="attachment_id is required",
        )

    logger.info(
        "get_full_files_for_each_ids request user_id=%s chat_id=%s file_count=%s",
        request.user_id,
        request.chat_id,
        len(file_ids),
    )

    try:
        file_metadata = _get_file_titles_from_qdrant(
            user_id=request.user_id,
            chat_id=request.chat_id,
            file_ids=file_ids,
        )

        files: Dict[str, Dict[str, str]] = {}
        missing_file_ids: List[str] = []

        for file_id in file_ids:
            metadata = file_metadata.get(file_id)
            if not metadata:
                missing_file_ids.append(file_id)
                continue

            file_name = str(metadata["file_name"])
            bucket_name = str(metadata["bucket_name"])
            files[file_id] = {
                "file_name": file_name,
                "content": _read_extracted_file_from_minio(
                    user_id=request.user_id,
                    file_id=file_id,
                    file_name=file_name,
                    bucket_name=bucket_name,
                ),
            }

        return {
            "files": files,
            "missing_file_ids": missing_file_ids,
        }
    except RuntimeError as exc:
        _raise_internal_error(str(exc), exc)
    except Exception as exc:
        _raise_internal_error("Unexpected error while reading full uploaded files", exc)


@router.post("/get_message_content")
async def get_message_content(request: MessageContentRequest):

#  request body example:
#     {
#   "user_id": "user123",
#   "chat_id": "chat456",
#   "message_id": "msg789"
# }



# {
#   "content": "message text",
#   "metadata": [
#     {
#       "created_at": "...",
#       "message_id": "...",
#       "user_id": "...",
#       "has_attachments": true,
#       "chat_id": "...",
#       "role": "user",
#       "seq": 1,
#       "attachment_ids": [],
#       "_point_id": "..."
#     }
#   ],
#   "chunk_count": 1
# }


    logger.info(
        "get_message_content request user_id=%s chat_id=%s message_id=%s",
        request.user_id,
        request.chat_id,
        request.message_id,
    )

    try:
        payloads = _get_message_payloads_from_qdrant(
            user_id=request.user_id,
            chat_id=request.chat_id,
            message_id=request.message_id,
        )

        if not payloads:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="message not found",
            )

        content_parts = [str(payload.get("content") or "") for payload in payloads]
        metadata = []
        for payload in payloads:
            item = dict(payload)
            item.pop("content", None)
            metadata.append(item)

        return {
            "content": "".join(content_parts),
            "metadata": metadata,
            "chunk_count": len(payloads),
        }
    except HTTPException:
        raise
    except Exception as exc:
        _raise_internal_error("Unexpected error while reading message content", exc)
