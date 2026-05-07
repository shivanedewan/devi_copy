import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List
import redis.asyncio as redis

from logger import logger
from settings import settings

from agent.embedding_agent import EmbeddingAgent

class AIWorker:
    def __init__(self):
        self.redis_url = settings.redis_url
        self.stream = settings.listening_stream
        self.group = settings.group_name
        self.worker = settings.worker_name
        self.dlq = settings.dead_letter_queue
        self.sem = asyncio.Semaphore(settings.semaphore)
        self.active_tasks: set[asyncio.Task[None]] = set()
        self.redis = None
        self.agent = None
        
    async def ensure_group(self) -> None:
        try:
            await self.redis.xgroup_create(
                name=self.stream,
                groupname=self.group,
                id="0-0",
                mkstream=True,
            )
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise RuntimeError(f"Error in group {e}")
            
    async def test_redis_connection(self):    
        if await self.redis.ping():
            return True
        else:
            return False
        
    async def xreadgroup(
        self,
        streams: Dict,
        count: int = 50,
        block_ms: int = 2000,
    ):
        return await self.redis.xreadgroup(
            groupname = self.group,
            consumername = self.worker,
            streams = streams,
            count=count,
            block=block_ms,
        )
    
    # initialize the worker and agent
    async def initialize(self):
        self.redis = redis.from_url(settings.redis_url, decode_responses=True)
        if not await self.test_redis_connection():
            print("Error in connecting to redis")            
        await self.ensure_group()
        self.agent = EmbeddingAgent(self.redis)
        
    async def _process_one(self, job_id, data):
        start = time.time()
        try:
            # print(data)
            payload = json.loads(data["data"])
            print(f"Got a Job: [{job_id}]")
            await self.agent.process(job_id, payload)
            await self.redis.xack(self.stream, self.group, job_id)          
            print(f"[{job_id}] processed successfully in [{round(time.time() - start, 3)}]s")
        except Exception as exc:                  
            await self.redis.hset(f"task:{payload["message_id"]}", "status", "failed")
            await self.redis.xadd(settings.dead_letter_queue, data, maxlen=50000, approximate=True)  
            await self.redis.xack(self.stream, self.group, job_id)           
            logger.error(f"FAILED {job_id}: {exc}| Pushed to {self.dlq}| Error: {exc}")
            print(f"FAILED {job_id}: {exc}| Pushed to dlq")
        finally:
            self.sem.release()
                
    async def _handle_batch(self, msgs: list[tuple[bytes, dict]]) -> None:
        for job_id, data in msgs:
            await self.sem.acquire()
            task = asyncio.create_task(self._process_one(job_id, data))
            self.active_tasks.add(task)
            task.add_done_callback(lambda t: self.active_tasks.discard(t))
        
    async def run(self):
        logger.debug(f"Starting {self.worker} listening on {self.stream}")
        
        pending = await self.xreadgroup(streams = {self.stream: "0"},count=200)
        
        if pending:
            for _, msgs in pending:
                asyncio.create_task(self._handle_batch(msgs))
                
        try:
            while True:
                # read new messages
                new_msgs = await self.xreadgroup({self.stream: ">"})
                if new_msgs:
                    for _, msgs in new_msgs:
                        asyncio.create_task(self._handle_batch(msgs))
                        
                await asyncio.sleep(0.01)
        finally:
            self.sem.release()
            await self._shutdown()

    async def _shutdown(self) -> None:
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks, return_exceptions=False)
        await self.redis.close()
        print("Connections closed - processed:")
        

async def main():
    qw = AIWorker()
    await qw.initialize()
    await qw.run()

    
if __name__ == "__main__":
    logger.debug(f"Starting {settings.worker_name} worker!")
    asyncio.run(main())