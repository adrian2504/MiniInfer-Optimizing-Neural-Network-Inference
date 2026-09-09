import asyncio
import threading
import time

import pytest

from miniinfer.batching import Batcher, Closed, Overloaded


def inference(rows, new_tokens, cached):
    return {"output_ids": [[row[0]] * new_tokens for row in rows], "latency_ms": 1.0, "ttft_ms": .5}


def test_batch_compatibility_and_baseline_singletons():
    async def scenario():
        calls = []
        def infer(rows, n, cache):
            calls.append((rows, n, cache))
            return inference(rows, n, cache)
        batcher = Batcher(infer, max_batch=3, wait_ms=2)
        batcher.start()
        jobs = [batcher.submit([i, 1], 2, True) for i in range(5)]
        jobs += [batcher.submit([8], 1, True), batcher.submit([9, 1], 2, False), batcher.submit([10, 1], 2, False)]
        results = await asyncio.gather(*(job.future for job in jobs))
        await batcher.close()
        assert results[0]["batch_size"] == 3
        assert results[0]["output_ids"] == [0, 0]
        assert results[-1]["batch_size"] == results[-2]["batch_size"] == 1
        assert all(len({len(row) for row in rows}) == 1 for rows, _, _ in calls)
        assert batcher.metrics()["counts"]["completed"] == 8
    asyncio.run(scenario())


def test_overload_cancel_and_shutdown():
    async def scenario():
        started, release = threading.Event(), threading.Event()
        def blocking(*args):
            started.set()
            release.wait(5)
            return inference(*args)
        batcher = Batcher(blocking, capacity=1, wait_ms=0)
        first = batcher.submit([1], 1, False)
        with pytest.raises(Overloaded):
            batcher.submit([2], 1, False)
        batcher.start()
        await asyncio.to_thread(started.wait, 2)
        queued = batcher.submit([2], 1, False)
        batcher.cancel(queued)
        assert queued.future.cancelled()
        queued = batcher.submit([3], 1, False)
        closing = asyncio.create_task(batcher.close())
        await asyncio.sleep(.01)
        with pytest.raises(Closed):
            await queued.future
        with pytest.raises(Closed):
            batcher.submit([4], 1, False)
        release.set()
        assert (await first.future)["output_ids"] == [1]
        await closing
    asyncio.run(scenario())


def test_inference_failure_does_not_kill_worker():
    async def scenario():
        def infer(rows, n, cache):
            if rows[0][0] == 0:
                raise ValueError("bad batch")
            return inference(rows, n, cache)
        batcher = Batcher(infer, wait_ms=0)
        batcher.start()
        failed = batcher.submit([0], 1, False)
        good = batcher.submit([1], 1, False)
        with pytest.raises(ValueError):
            await failed.future
        assert (await good.future)["output_ids"] == [1]
        await batcher.close()
    asyncio.run(scenario())


def test_http_routes_validation_and_timeout():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from miniinfer.server import Settings, create_app
    class Runner:
        vocab_size, context_limit, info = 16, 32, {"smoke_only": True}
        def __call__(self, rows, n, cache):
            if rows[0][0] == 15:
                time.sleep(.1)
            return inference(rows, n, cache)
    with TestClient(create_app(Settings(timeout_seconds=.03), Runner())) as client:
        for route in ("baseline", "optimized", "benchmark"):
            response = client.post(f"/{route}", json={"input_ids": [3, 4], "new_tokens": 2})
            assert response.status_code == 200
            assert response.json()["output_ids"] == [3, 3]
        for ids in ([20], [-1], [True], [1.2], []):
            assert client.post("/optimized", json={"input_ids": ids}).status_code == 422
        assert client.post("/baseline", json={"input_ids": [15], "new_tokens": 1}).status_code == 504
        assert client.get("/metrics").json()["counts"]["cancelled"] == 1


def test_real_model_serving_parity_and_load_measurement():
    pytest.importorskip("fastapi")
    import httpx
    from miniinfer.load_test import measure
    from miniinfer.server import Settings, create_app
    async def scenario():
        app = create_app(Settings(smoke=True, batch_wait_ms=10))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                payload = {"input_ids": [3, 4, 5], "new_tokens": 3}
                baseline = (await client.post("/baseline", json=payload)).json()
                report = await measure(client, "/optimized", payload, 10, 20)
                assert report["status_counts"] == {"200": 20}
                assert report["output_tokens_per_second"] > 0
                assert all(row["response"]["output_ids"] == baseline["output_ids"] for row in report["samples"])
                assert max(row["response"]["batch_size"] for row in report["samples"]) > 1
    asyncio.run(scenario())
