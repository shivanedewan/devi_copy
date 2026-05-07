from __future__ import annotations

import asyncio
import contextlib
import json
import time

import redis.asyncio as redis

from context_fetcher import ContextFetcher
from logger import logger
from rag_worker import RAGWorker
from settings import settings


class AIWorker:
    def __init__(self, stream: str, group: str, worker: str, dlq: str, processor: RAGWorker):
        self.stream = stream
        self.group = group
        self.worker = worker
        self.dlq = dlq
        self.processor = processor
        self.redis = None
        self.max_concurrency = max(1, int(getattr(settings, "rag_worker_max_concurrency", 1)))
        self.read_count = max(1, int(getattr(settings, "rag_worker_read_count", self.max_concurrency)))
        self.read_block_ms = max(1, int(getattr(settings, "rag_worker_read_block_ms", 2000)))
        self.idle_sleep_s = max(0.0, float(getattr(settings, "rag_worker_idle_sleep_s", 0.05)))
        self.job_timeout_s = max(0.0, float(getattr(settings, "rag_worker_job_timeout_s", 0.0)))
        self.sem = asyncio.Semaphore(self.max_concurrency)
        self.active_tasks: set[asyncio.Task[None]] = set()
        self.active_job_ids: set[str] = set()
        self.active_message_ids: set[str] = set()

    async def initialize(self) -> None:
        self.redis = redis.from_url(settings.redis_url, decode_responses=True)
        if not await self.redis.ping():
            raise RuntimeError("Redis is not connected")
        try:
            await self.redis.xgroup_create(
                name=self.stream,
                groupname=self.group,
                id="0-0",
                mkstream=True,
            )
        except redis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise RuntimeError(f"Error creating group: {exc}")
        self.processor.setup_redis(self.redis)

    def _available_slots(self) -> int:
        return max(0, self.max_concurrency - len(self.active_tasks))

    @staticmethod
    def _task_ttl_seconds() -> int:
        return max(60, int(getattr(settings, "rag_task_ttl_s", 1800)))

    @staticmethod
    def _response_stream_ttl_seconds() -> int:
        return max(60, int(getattr(settings, "rag_response_stream_ttl_s", 3600)))

    def _track_task(self, task: asyncio.Task[None]) -> None:
        self.active_tasks.add(task)
        task.add_done_callback(self.active_tasks.discard)

    @staticmethod
    def _non_empty_batches(response) -> list[tuple[str, list[tuple[str, dict]]]]:
        return [(stream_name, messages) for stream_name, messages in list(response or []) if messages]

    @staticmethod
    def _message_id_from_data(data: dict) -> str:
        try:
            wrapper = json.loads(str((data or {}).get("data") or "{}"))
        except Exception:
            return ""
        if not isinstance(wrapper, dict):
            return ""
        payload = wrapper.get("payload") if isinstance(wrapper.get("payload"), dict) else wrapper
        return str((payload or {}).get("message_id") or "").strip()

    async def _mark_job_timeout(self, job_id: str, data: dict) -> None:
        message_id = self._message_id_from_data(data)
        if not message_id or self.redis is None:
            return
        stream_name = f"streaming:resp:{message_id}"
        with contextlib.suppress(Exception):
            task_record = await self.redis.hgetall(f"task:{message_id}")
            stream_name = str((task_record or {}).get("stream") or "").strip() or stream_name
        with contextlib.suppress(Exception):
            await self.redis.hset(
                f"task:{message_id}",
                mapping={
                    "status": "failed",
                    "ui": "failed",
                    "ui_detail": "request timed out",
                    "ui_detailed": "request timed out",
                    "stream": stream_name,
                    "updated_at": str(time.time()),
                },
            )
            await self.redis.expire(f"task:{message_id}", self._task_ttl_seconds())
        with contextlib.suppress(Exception):
            await self.redis.xadd(
                stream_name,
                {"data": "The request timed out while processing. Please try again."},
                maxlen=10000,
                approximate=True,
            )
            await self.redis.xadd(stream_name, {"end": "1"}, maxlen=1000, approximate=True)
            await self.redis.expire(stream_name, self._response_stream_ttl_seconds())
        logger.error(
            "RAG job timed out job_id=%s message_id=%s stream=%s timeout_s=%s",
            job_id,
            message_id,
            stream_name,
            self.job_timeout_s,
        )

    async def _mark_job_failed(self, job_id: str, data: dict, exc: Exception) -> None:
        message_id = self._message_id_from_data(data)
        if not message_id or self.redis is None:
            return
        stream_name = f"streaming:resp:{message_id}"
        with contextlib.suppress(Exception):
            task_record = await self.redis.hgetall(f"task:{message_id}")
            stream_name = str((task_record or {}).get("stream") or "").strip() or stream_name
        with contextlib.suppress(Exception):
            await self.redis.hset(
                f"task:{message_id}",
                mapping={
                    "status": "failed",
                    "ui": "failed",
                    "ui_detail": "request failed",
                    "ui_detailed": "request failed",
                    "stream": stream_name,
                    "updated_at": str(time.time()),
                },
            )
            await self.redis.expire(f"task:{message_id}", self._task_ttl_seconds())
        with contextlib.suppress(Exception):
            await self.redis.xadd(
                stream_name,
                {"data": "I hit an internal issue while processing the request. Please try again."},
                maxlen=10000,
                approximate=True,
            )
            await self.redis.xadd(stream_name, {"end": "1"}, maxlen=1000, approximate=True)
            await self.redis.expire(stream_name, self._response_stream_ttl_seconds())
        logger.error(
            "RAG job failed before worker fallback job_id=%s message_id=%s stream=%s error=%s",
            job_id,
            message_id,
            stream_name,
            exc,
        )

    async def _process_one(self, job_id: str, data: dict) -> None:
        try:
            started = float(time.time())
            if self.job_timeout_s > 0:
                await asyncio.wait_for(self.processor.process(job_id, data), timeout=self.job_timeout_s)
            else:
                await self.processor.process(job_id, data)
            await self.redis.xack(self.stream, self.group, job_id)
            logger.info("Processed job_id=%s latency=%.3fs", job_id, float(time.time()) - started)
        except asyncio.TimeoutError:
            await self._mark_job_timeout(job_id, data)
            await self.redis.xadd(self.dlq, data, maxlen=10000, approximate=True)
            await self.redis.xack(self.stream, self.group, job_id)
        except Exception as exc:
            await self._mark_job_failed(job_id, data, exc)
            await self.redis.xadd(self.dlq, data, maxlen=10000, approximate=True)
            await self.redis.xack(self.stream, self.group, job_id)
            logger.error("FAILED job_id=%s error=%s pushed_to=%s", job_id, exc, self.dlq)
        finally:
            self.active_job_ids.discard(str(job_id))
            message_id = self._message_id_from_data(data)
            if message_id:
                self.active_message_ids.discard(message_id)
            self.sem.release()

    async def _handle_messages(self, messages) -> None:
        for job_id, data in messages:
            normalized_job_id = str(job_id)
            if normalized_job_id in self.active_job_ids:
                logger.warning("Skipping duplicate in-process RAG job_id=%s", normalized_job_id)
                continue
            message_id = self._message_id_from_data(data)
            if message_id and message_id in self.active_message_ids:
                logger.warning(
                    "Acking duplicate in-process RAG message_id=%s duplicate_job_id=%s",
                    message_id,
                    normalized_job_id,
                )
                await self.redis.xack(self.stream, self.group, normalized_job_id)
                continue
            await self.sem.acquire()
            self.active_job_ids.add(normalized_job_id)
            if message_id:
                self.active_message_ids.add(message_id)
            task = asyncio.create_task(self._process_one(normalized_job_id, data))
            self._track_task(task)

    async def _read_stream(self, cursor: str, *, count: int, block_ms: int):
        return await self.redis.xreadgroup(
            groupname=self.group,
            consumername=self.worker,
            streams={self.stream: cursor},
            count=max(1, int(count)),
            block=max(1, int(block_ms)),
        )

    async def _drain_pending_once(self) -> None:
        cursor = "0-0"
        scheduled_pending_ids: set[str] = set()
        while True:
            available_slots = self._available_slots()
            if available_slots <= 0:
                await asyncio.sleep(self.idle_sleep_s)
                continue
            pending = self._non_empty_batches(
                await self._read_stream(
                    cursor,
                    count=min(self.read_count, available_slots),
                    block_ms=1,
                )
            )
            if not pending:
                return
            newest_id = cursor
            for _, messages in pending:
                fresh_messages = []
                for job_id, payload in messages:
                    normalized_job_id = str(job_id)
                    newest_id = normalized_job_id
                    if normalized_job_id in scheduled_pending_ids:
                        logger.warning("Skipping duplicate pending RAG job_id=%s during startup drain", normalized_job_id)
                        continue
                    scheduled_pending_ids.add(normalized_job_id)
                    fresh_messages.append((normalized_job_id, payload))
                if fresh_messages:
                    await self._handle_messages(fresh_messages)
            if newest_id == cursor:
                return
            cursor = newest_id

    async def run(self) -> None:
        logger.info(
            "Starting RAG worker=%s stream=%s max_concurrency=%s read_count=%s",
            self.worker,
            self.stream,
            self.max_concurrency,
            self.read_count,
        )
        try:
            await self._drain_pending_once()
            while True:
                available_slots = self._available_slots()
                if available_slots <= 0:
                    await asyncio.sleep(self.idle_sleep_s)
                    continue
                new_messages = self._non_empty_batches(
                    await self._read_stream(
                        ">",
                        count=min(self.read_count, available_slots),
                        block_ms=self.read_block_ms,
                    )
                )
                if new_messages:
                    for _, messages in new_messages:
                        await self._handle_messages(messages)
                await asyncio.sleep(self.idle_sleep_s)
        finally:
            if self.active_tasks:
                await asyncio.gather(*self.active_tasks, return_exceptions=True)
            await self.redis.close()


async def main() -> None:
    cf = ContextFetcher(
        conv_context_url=settings.conv_context_url,
        file_context_url=settings.file_context_url,
        big_data_context_url=settings.big_data_context_url,
        big_data_exact_context_url=settings.big_data_exact_context_url,
        timeout_seconds=settings.context_fetch_timeout_s,
        retries=settings.context_fetch_retries,
    )
    processor = None
    try:
        processor = RAGWorker(cf)
        worker = AIWorker(
            stream=settings.stream_name,
            group=settings.group_name,
            worker=settings.worker_name,
            dlq=settings.dead_letter_queue,
            processor=processor,
        )
        await worker.initialize()
        await worker.run()
    finally:
        if processor is not None:
            await processor.aclose()
        await cf.aclose()


if __name__ == "__main__":
    asyncio.run(main())
