import time
import uuid
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, Sequence

from qdrant_client import QdrantClient 
from qdrant_client import models as qm 

PointId = Union[int, str]
Vector = Sequence[float]
Payload = Dict[str, Any]
    
class QdrantAPI:
    def __init__(self, qdrant_url: str):
        self._client = QdrantClient(url=qdrant_url)
        
    def upsert_points(self, collection: str, points: Sequence[Tuple[PointId, Vector, Payload]], *, wait: bool=True) -> qm.UpdateResult:
        qpoints: List[qm.PointStruct] = []
        for pid, vec, payload in points:
            vectors = list(vec)
            qpoints.append(qm.PointStruct(id=pid, vector=vectors, payload=payload))
                
        return self._client.upsert(
            collection_name=collection,
            points=qpoints,
            wait=wait
        )
        
    def delete_by_filter_two_field_and(self, collection: str, *, field1: str, value1: str, field2: str, value2: str, wait: bool=True) -> qm.UpdateResult:
        qfilter = qm.Filter(
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
        
        selector = qm.FilterSelector(filter=qfilter)
        res = self._client.delete(
            collection_name=collection,
            points_selector=selector,
            wait=wait
        )

        return res

