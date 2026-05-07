import time
import uuid
import asyncio
import aiohttp
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

from qdrant_client import QdrantClient 
from qdrant_client import models as qm 

@dataclass
class Config:
    attachment_collection: str = "chat_attachment_chunks_production_1"
    chat_collection: str = "chat_messages_production_1"
    
class QdrantUtils:
    def __init__(self, qdrant: QdrantClient, cfg: Optional[Config] = None):
        self.qdrant=qdrant
        self.cfg = cfg or Config
        
    def _match_filter_and(self, field1: str, value1: Union[str, int, float, bool], field2: str, value2: Union[str, int, float, bool]) -> qm.Filter:
        return qm.Filter(
            must=[
                qm.FieldCondition(
                    key=field1,
                    match=qm.MatchValue(value=value1),
                ),
                qm.FieldCondition(
                    key=field2,
                    match=qm.MatchValue(value=value2),
                )
            ]
        )
        
    def _delete_by_filter(self, collection_name: str, qfilter: qm.Filter, wait: bool=True) -> qm.UpdateResult:
        selector = qm.FilterSelector(filter=qfilter)
        res = self.qdrant.delete(
            collection_name=collection_name,
            points_selector=selector,
            wait=wait
        )

        return res
    
    def delete_points_by_file_id(self, user_id: str, file_id: str) -> bool:
        qfilter = self._match_filter_and("file_id", file_id, "user_id", user_id)
        return self._delete_by_filter(self.cfg.attachment_collection, qfilter, True)
    
    def delete_points_by_chat_id(self, user_id: str, chat_id: str) -> bool:
        qfilter = self._match_filter_and("chat_id", chat_id, "user_id", user_id)
        return self._delete_by_filter(self.cfg.chat_collection, qfilter, True)

async def main():
    qdrant = QdrantClient(url="http://192.168.10.32:6333")
    qutils = QdrantUtils(qdrant=qdrant, cfg=Config)
    
    status = qutils.delete_points_by_file_id(collection_name = Config.attachment_collection, file_id="22cb74b4-abf1-450f-99ea-ae24ca61d230")
    print(status)
    status = qutils.delete_points_by_chat_id(collection_name = Config.chat_collection, chat_id="6b34bbb2-30d9-4091-a9b0-a946831bb388")
    print(status)
    # bundle = await fetcher.fetch(
    #     user_id="superuser",
    #     chat_id="d6ccd3a6-58cd-4711-b34a-8f8bab84f409",
    #     query="give me a python code for elasticsearch"
    # )
    
    
    
    
if __name__ == "__main__":
    asyncio.run(main())