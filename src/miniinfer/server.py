"""A small token-ID inference server for batching experiments."""

import argparse
import asyncio
import math
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, StrictInt
import torch

from miniinfer.batching import Batcher, Closed, Overloaded
from miniinfer.benchmark import build_parser
from miniinfer.generation import generate_trial
from miniinfer.optimization import load_model


class GenerationRequest(BaseModel):
    input_ids: list[StrictInt] = Field(min_length=1, max_length=4096)
    new_tokens: int = Field(default=32, ge=1, le=256, strict=True)


class BenchmarkRequest(GenerationRequest):
    mode: Literal["baseline", "optimized"] = "optimized"


@dataclass
class Settings:
    model: str = "openai-community/gpt2"
    revision: str = "main"
    device: str = "cpu"
    smoke: bool = False
    threads: int = 4
    queue_capacity: int = 128
    max_batch: int = 8
    batch_wait_ms: float = 5
    timeout_seconds: float = 60


class ModelRunner:
    def __init__(self, settings):
        args = build_parser().parse_args([])
        for key in ("model", "revision", "device", "smoke", "threads"):
            setattr(args, key, getattr(settings, key))
        torch.set_num_threads(settings.threads)
        torch.set_float32_matmul_precision("highest")
        if settings.device == "cuda":
            torch.cuda.matmul.allow_tf32 = False
        self.device = torch.device(settings.device)
        self.model = load_model(args, self.device)
        self.vocab_size = self.model.config.vocab_size
        self.context_limit = getattr(self.model.config, "max_position_embeddings", 4096)
        self.info = {"model": "random-tiny-gpt2" if settings.smoke else settings.model,
                     "revision": getattr(self.model.config, "_commit_hash", None),
                     "device": settings.device, "smoke_only": settings.smoke, "dtype": "fp32"}

    def __call__(self, rows, new_tokens, cached):
        return generate_trial(self.model, torch.tensor(rows, device=self.device), new_tokens, kv_cache=cached)


def device_metrics(device):
    result = {"gpu_utilization_percent": None, "device_memory_used_bytes": None}
    if device != "cuda":
        return {**result, "unavailable_reason": "NVIDIA CUDA telemetry only"}
    nvml = None
    try:
        import pynvml
        nvml = pynvml
        nvml.nvmlInit()
        uuid = torch.cuda.get_device_properties(torch.device("cuda")).uuid
        handle = nvml.nvmlDeviceGetHandleByUUID(str(uuid))
        return {"gpu_utilization_percent": nvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                "device_memory_used_bytes": nvml.nvmlDeviceGetMemoryInfo(handle).used,
                "scope": "whole device, current NVML sample"}
    except Exception as exc:
        return {**result, "unavailable_reason": str(exc)}
    finally:
        if nvml:
            try:
                nvml.nvmlShutdown()
            except Exception:
                pass


def create_app(settings=None, runner=None):
    settings = settings or Settings()
    if not math.isfinite(settings.timeout_seconds) or settings.timeout_seconds <= 0 or settings.threads < 1:
        raise ValueError("Timeout and threads must be positive")

    @asynccontextmanager
    async def lifespan(app):
        model_runner = runner or await asyncio.to_thread(ModelRunner, settings)
        batcher = Batcher(model_runner, settings.queue_capacity, settings.max_batch, settings.batch_wait_ms)
        app.state.runner, app.state.batcher = model_runner, batcher
        batcher.start()
        try:
            yield
        finally:
            await batcher.close()

    app = FastAPI(title="MiniInfer", lifespan=lifespan)

    async def execute(body, request, cached):
        model_runner = app.state.runner
        if any(type(token) is not int or token < 0 or token >= model_runner.vocab_size for token in body.input_ids):
            raise HTTPException(422, "input_ids contains a token outside this model's vocabulary")
        if len(body.input_ids) + body.new_tokens > model_runner.context_limit:
            raise HTTPException(422, "Prompt plus output exceeds the model context limit")
        batcher = app.state.batcher
        try:
            job = batcher.submit(body.input_ids, body.new_tokens, cached)
        except (Overloaded, Closed) as exc:
            raise HTTPException(503, str(exc)) from exc
        try:
            async with asyncio.timeout(settings.timeout_seconds):
                while not job.future.done():
                    if await request.is_disconnected():
                        raise HTTPException(499, "Client disconnected")
                    await asyncio.wait([job.future], timeout=.05)
                return job.future.result()
        except TimeoutError as exc:
            raise HTTPException(504, "Request timed out") from exc
        except (HTTPException, asyncio.CancelledError):
            raise
        except Closed as exc:
            raise HTTPException(503, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(500, "Inference failed") from exc
        finally:
            batcher.cancel(job)

    @app.post("/baseline")
    async def baseline(body: GenerationRequest, request: Request):
        return await execute(body, request, False)

    @app.post("/optimized")
    async def optimized(body: GenerationRequest, request: Request):
        return await execute(body, request, True)

    @app.post("/benchmark")
    async def benchmark(body: BenchmarkRequest, request: Request):
        return await execute(body, request, body.mode == "optimized")

    @app.get("/metrics")
    async def metrics():
        return {**app.state.batcher.metrics(), "model": app.state.runner.info,
                "settings": vars(settings), "telemetry": await asyncio.to_thread(device_metrics, settings.device)}

    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai-community/gpt2")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--queue-capacity", type=int, default=128)
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--batch-wait-ms", type=float, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=60)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    config = vars(args).copy()
    host, port = config.pop("host"), config.pop("port")
    uvicorn.run(create_app(Settings(**config)), host=host, port=port, workers=1)


if __name__ == "__main__":
    main()
