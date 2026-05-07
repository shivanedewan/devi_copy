import json
import aiohttp
import uuid 
import os
from pathlib import Path
import asyncio
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, Sequence
import redis.asyncio as redis

from settings import settings
from logger import logger
from agent.minio_adapter import MinioAdapter
from agent.qdrant_api import QdrantAPI
from agent.utils import chunk_text, clean_tmp

PointId = Union[int, str]
Vector = Sequence[float]
Payload = Dict[str, Any]

class EmbeddingAgent():
    def __init__(self, rc: redis):
        self.qdrant = QdrantAPI(settings.qdrant_url)
        self.embedding_url = settings.embedding_url
        self.redis = rc
        self.minio = MinioAdapter(
            settings.minio_endpoint,
            access_key = settings.minio_access_key,
            secret_key = settings.minio_secret
        )
    
    async def generate_embeddings(self, content: str):
        payload = {
            "model": "qwen-embed",
            "input": [content],
            # "embed_normalize": 2
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.embedding_url, json=payload) as response:
                # print(f"Embedding response: {response}")
                if response.status == 200:
                    result = await response.json()
                    # print(result)
                    return result["data"][0]["embedding"]
        
        return None
        
    async def _embed_any_text_chunk(
        self,
        collection_name: str, 
        job_id: str,
        text: str, 
        puuid: str,
        qd_payload: Dict   
    ):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, puuid))
        vec = await self.generate_embeddings(text)
            
        if vec == None:
            logger.error("Embedding is not returned!")
            return False
            
        self.qdrant.upsert_points(
            collection=collection_name,
            points=[
                (point_id, vec, qd_payload)
            ]
        )  
        return True
        
        
    async def _process_chat_message(self, payload):
        await self.redis.hset(f"task:{payload["message_id"]}", "ui", "Analyzing query")
        chunks = chunk_text(payload["content"])
        for idx, chunk in enumerate(chunks):
            qd_payload = {
                "content": chunk,
                "created_at": payload["created_at"],
                "message_id": payload["message_id"],
                "user_id": payload["user_id"],
                "has_attachments": payload["has_attachments"],
                "chat_id": payload["chat_id"],
                "role": payload["role"],
                "seq": payload["seq"],
                "attachment_ids": [attachment.get("attachment_id") for attachment in payload["attachments"] if attachment.get("attachment_id")]
            }
            
            if not await self._embed_any_text_chunk(
                collection_name=settings.collection_chat_messages,
                job_id=payload["message_id"],
                text=chunk,
                puuid=f"{payload["message_id"]}_{idx}",
                qd_payload=qd_payload
            ):
                await self.redis.hset(f"task:{payload["message_id"]}", "status", "failed")
                await self.redis.hset(f"task:{payload["message_id"]}", "ui", "Vector db connection issue")  
                raise RuntimeError("Error in creating embedding or translation of the text!")
        return True
        
    async def _process_attachments(self, payload):
        await self.redis.hset(f"task:{payload["message_id"]}", "ui", f"Analyzing files")  
        for att in payload["attachments"]:  
            file_name = att["title"]
            root, ext = os.path.splitext(file_name)
            plain_text_file_name = f"{root}.md"
            await self.redis.hset(f"task:{payload["message_id"]}", "status", "embedding_file")    
            await self.redis.hset(f"task:{payload["message_id"]}", "ui_detailed", f"Analyzing file: {file_name}")  
              
            object_name = f'{payload["user_id"]}/{att["attachment_id"]}/extracted_text/{plain_text_file_name}'
            
            # download extracted file
            file_contents = self.minio.read_file_from_minio(
                bucket=settings.file_uploads_bucket,
                object_name=object_name
            )
            
            if not file_contents:
                await self.redis.hset(f"task:{payload["message_id"]}", "ui", "Unable to retreive file")  
                await self.redis.hset(f"task:{payload["message_id"]}", "ui_detailed", "File doesnot exists on upload server")  
                await self.redis.hset(f"task:{payload["message_id"]}", "status", "failed")
                await self.redis.hset(f"file_upload:{att["attachment_id"]}", "status", "embedding_failed")
                logger.error(f"Error in reading object [{object_name}] from bucket [{settings.file_uploads_bucket}]")
                raise RuntimeError("Error in reading file from minio!")
                
            # store locally with name attachment_id.md
            self.minio.store_locally_sync(file_contents, f"tmp/{att["attachment_id"]}.md")
            # read only text
            with open(f"tmp/{att["attachment_id"]}.md", "r") as file:
                plain_text = file.read()
                # print(plain_text)
                
            file_path = Path(f"tmp/{att["attachment_id"]}.md")
            
            if file_path.exists():
                file_path.unlink()
                
            if not plain_text:    
                await self.redis.hset(f"task:{payload["message_id"]}", "ui", "Unable to extract file")  
                await self.redis.hset(f"task:{payload["message_id"]}", "ui_detailed", "Plain text file not created")  
                await self.redis.hset(f"task:{payload["message_id"]}", "status", "failed")  
                await self.redis.hset(f"file_upload:{att["attachment_id"]}", "status", "failed")
                logger.error("extraction of file failed! text not found!")
                raise RuntimeError("Extraction of file failed!")
                
            plain_text = plain_text.strip()
            chunks = chunk_text(plain_text)
            
            for idx, chunk in enumerate(chunks):
                qd_payload = {
                    "content": chunk,
                    "created_at": payload["created_at"],
                    "user_id": payload["user_id"],
                    "file_id": att["attachment_id"],
                    "chat_id": payload["chat_id"],
                    "bucket_name": settings.file_uploads_bucket,
                    "title": att["title"],
                    "chunk_no": idx
                }
                
                if not await self._embed_any_text_chunk(
                    collection_name=settings.collection_file_uploads,
                    job_id=payload["message_id"],
                    text=chunk,
                    puuid=f"{att["attachment_id"]}_{idx}",
                    qd_payload=qd_payload
                ):
                    await self.redis.hset(f"task:{payload["message_id"]}", "status", "failed")
                    await self.redis.hset(f"file_upload:{att["attachment_id"]}", "status", "embedding_failed")
                    await self.redis.hset(f"task:{payload["message_id"]}", "ui", "Embedding error")    
                    raise RuntimeError("Error in creating embedding or translation of the file!")
            
            await self.redis.hset(f"file_upload:{att["attachment_id"]}", "status", "embedding_success")  
        return True
                 
    async def process(self, job_id, payload: Dict):
        try:    
            # print(payload)
            await self.redis.hset(f"task:{payload["message_id"]}", "status", "embedding")    
                
            await self._process_chat_message(payload)
            if payload["role"] == "user":
                # only user attachments will be embedded
                await self._process_attachments(payload)    
                payload["attachments"] = [attachment.get("attachment_id") for attachment in payload["attachments"] if attachment.get("attachment_id")]
                # send user query to router
                await self.redis.xadd(settings.stream_router, {
                    "data": json.dumps(payload)
                }, maxlen=50000, approximate=True)
                await self.redis.hset(f"task:{payload["message_id"]}", "status", "embedding_success")  
                await self.redis.hset(f"task:{payload["message_id"]}", "ui", "Sending to router")
                await self.redis.hset(f"task:{payload["message_id"]}", "ui_detailed", "")
        except Exception as e:
            raise RuntimeError(f"Error in process function: {e}")
    