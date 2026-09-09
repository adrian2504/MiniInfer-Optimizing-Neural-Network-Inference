"""One model worker, a bounded queue, and batches of compatible requests."""

import asyncio
import math
from collections import Counter, deque
from dataclasses import dataclass
import time

from miniinfer.metrics import percentile


class Overloaded(Exception):
    pass


class Closed(Exception):
    pass


@dataclass
class Job:
    ids: list[int]
    new_tokens: int
    cached: bool
    future: asyncio.Future
    enqueued: float

    @property
    def key(self):
        return (len(self.ids), self.new_tokens, self.cached)


class Batcher:
    def __init__(self, infer, capacity=128, max_batch=8, wait_ms=5):
        if capacity < 1 or max_batch < 1 or not math.isfinite(wait_ms) or wait_ms < 0:
            raise ValueError("Queue capacity and batch size must be positive; wait must be nonnegative")
        self.infer = infer
        self.capacity, self.max_batch, self.wait_ms = capacity, max_batch, wait_ms
        self.pending = deque()
        self.ready = asyncio.Event()
        self.closing = False
        self.task = None
        self.active = 0
        self.counts = Counter()
        self.batch_sizes = Counter()
        self.recent = deque(maxlen=2048)
        self.started = time.perf_counter()

    def start(self):
        self.task = asyncio.create_task(self._worker())

    def submit(self, ids, new_tokens, cached):
        if self.closing:
            raise Closed("Server is shutting down")
        self.pending = deque(job for job in self.pending if not job.future.done())
        if len(self.pending) >= self.capacity:
            self.counts["rejected"] += 1
            raise Overloaded("Request queue is full")
        future = asyncio.get_running_loop().create_future()
        job = Job(ids, new_tokens, cached, future, time.perf_counter())
        self.pending.append(job)
        self.counts["accepted"] += 1
        self.ready.set()
        return job

    def cancel(self, job):
        if not job.future.done():
            job.future.cancel()
            self.counts["cancelled"] += 1
        try:
            self.pending.remove(job)
        except ValueError:
            pass  # An already-running batch finishes; its result is discarded.

    async def close(self):
        self.closing = True
        for job in self.pending:
            if not job.future.done():
                job.future.set_exception(Closed("Server is shutting down"))
        self.pending.clear()
        self.ready.set()
        if self.task:
            await self.task  # Never launch another model call while a thread is finishing.

    async def _worker(self):
        while not self.closing:
            await self.ready.wait()
            if self.closing:
                break
            self.pending = deque(job for job in self.pending if not job.future.done())
            if not self.pending:
                self.ready.clear()
                continue
            first = self.pending[0]
            if first.cached and self.wait_ms:
                await asyncio.sleep(self.wait_ms / 1000)
            if self.closing:
                break
            selected, remaining = [], deque()
            limit = self.max_batch if first.cached else 1
            for job in self.pending:
                if job.future.done():
                    continue
                if job.key == first.key and len(selected) < limit:
                    selected.append(job)
                else:
                    remaining.append(job)
            self.pending = remaining
            if not remaining:
                self.ready.clear()
            if not selected:
                continue
            dispatch = time.perf_counter()
            self.active = len(selected)
            self.batch_sizes[len(selected)] += 1
            try:
                result = await asyncio.to_thread(self.infer, [job.ids for job in selected],
                                                first.new_tokens, first.cached)
                if len(result["output_ids"]) != len(selected):
                    raise RuntimeError("Model returned an unexpected batch size")
                for job, tokens in zip(selected, result["output_ids"]):
                    if job.future.done():
                        continue
                    timing = {"queue_ms": (dispatch - job.enqueued) * 1000,
                              "server_ms": (time.perf_counter() - job.enqueued) * 1000,
                              "model_batch_ms": result["latency_ms"], "model_ttft_ms": result["ttft_ms"]}
                    self.recent.append(timing)
                    self.counts["completed"] += 1
                    self.counts["output_tokens"] += len(tokens)
                    job.future.set_result({"output_ids": tokens, "batch_size": len(selected),
                                           "kv_cache": first.cached, **timing})
            except Exception as exc:
                for job in selected:
                    if not job.future.done():
                        self.counts["failed"] += 1
                        job.future.set_exception(exc)
            finally:
                self.active = 0

    def metrics(self):
        elapsed = time.perf_counter() - self.started
        values = [row["server_ms"] for row in self.recent]
        return {"counts": dict(self.counts), "queue_depth": len(self.pending), "active_requests": self.active,
                "batch_sizes": dict(self.batch_sizes), "uptime_seconds": elapsed,
                "completed_requests_per_second_since_start": self.counts["completed"] / elapsed,
                "output_tokens_per_second_since_start": self.counts["output_tokens"] / elapsed,
                "recent_completed_count": len(values),
                "server_latency_ms": {name: percentile(values, q) if values else None
                                      for name, q in (("p50", .5), ("p95", .95), ("p99", .99))}}
