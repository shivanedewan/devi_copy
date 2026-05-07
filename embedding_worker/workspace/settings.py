from __future__ import annotations

from typing import Optional, List
from pydantic import BaseModel, Field, HttpUrl, PositiveInt, validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Redis ────────────────────────────────────────────────────────
    redis_url: str = Field(
        default='redis://:Redis@123@192.168.10.35:6379/0',
        description="Redis connection URL (redis://host:port/db).",
    )
    listening_stream: str = Field(
        default="tasks.embed",
        description="Name of the redis stream",
    )
    
    stream_router: str = Field(
        default="tasks.router",
        description="Router stream",
    )
    
    stream_reports: str = Field(
        default="tasks.reports",
        description="Report generator worker",
    )
    
    dead_letter_queue: str = Field(
        default="tasks.dlq",
        description="Failed jobs will be pushed to dlq",
    )
    
    group_name: str = Field(
        default="embed",
        description="Name of the Redis group",
    )
    
    worker_name: str = Field(
        default="embed_01",
        description="Name of the worker",
    )
    
    # Minio ────────────────────────────────────────────────────────
    minio_endpoint: str = Field(
        default="192.168.10.35:9198",
        description="Router stream",
    )
    
    minio_access_key: str = Field(
        default="user1",
        description="Embedding stream",
    )
    
    minio_secret: str = Field(
        default="VMware@123",
        description="Translate stream",
    )
    
    file_uploads_bucket: str = Field(
        default="app-uploads",
        description="minio bucket for file uploads",
    )
    
    
    # Qdrant ────────────────────────────────────────────────────────
    qdrant_url: str = Field(
        default="http://192.168.10.32:6333",
        description="Url for connecting to qdrant",
    )
    
    collection_file_uploads: str = Field(
        default="chat_attachment_chunks_production_1",
        description="chat attachment chunks collection",
    )
    
    collection_chat_messages: str = Field(
        default="chat_messages_production_1",
        description="chat message embedding index",
    )


    # Embedding ────────────────────────────────────────────────────────
    embedding_url: str = Field(
        default="http://192.168.10.210:9000/v1/embeddings",
        description="Url for creating embeddings of text",
    )
    
    chunk_size: int = Field(
        default=3072,
        description="Chunk size",
    )
    
    # Max tokens for translation should be greater than chunk size
    max_tokens: int = Field(
        default=4096,
        description="Max generation tokens",
    )
    
    # ── LLM ────────────────────────────────────────────────────────────
    vllm_base_url: str = Field(
        default="http://192.168.10.210:8000/v1",
        description="Full URL of the LLM chat-completion endpoint.",
    )
    
    vllm_api_key: str = Field(
        default="EMPTY",
        description="VLLM API key",
    )
    
    model_name: str = Field(
        default="openai/gpt-oss-120b",
        description="Model name",
    )
    
    semaphore: int = Field(
        default=50,
        description="Number of concurrent jobs can run in parallel",
    )

# instantiate ONCE
settings = Settings()