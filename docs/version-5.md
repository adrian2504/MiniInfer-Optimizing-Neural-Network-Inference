# Version 5 — Serving more than one request

The server adds a queue between HTTP requests and the model. I want to understand when batching helps throughput and how long requests wait for it.

## The flow

```text
HTTP requests → bounded queue → one batch worker → model → responses
```

The model is loaded once at startup. Inference runs in a worker thread so the HTTP event loop can keep accepting requests, checking disconnects, and returning metrics. Only one model batch runs at a time.

The API takes token IDs. This keeps tokenizer work out of the server experiment and makes the workload easy to reproduce. IDs must be valid for the loaded model.

| Route | What it does |
| --- | --- |
| `POST /baseline` | Queued uncached generation, one request per model call |
| `POST /optimized` | Cached generation with dynamic batching |
| `POST /benchmark` | One measured request; body selects `baseline` or `optimized` mode |
| `GET /metrics` | Queue depth, counters, batch sizes, recent latency percentiles, and optional NVIDIA readings |

The optimized route currently means FP32 with caching and batching. It doesn't integrate the Triton RMSNorm kernel, quantization, or compilation.

## Run it locally

```bash
python -m pip install -e '.[dev,server]'
miniinfer-serve --smoke --device cpu
```

In another terminal:

```bash
curl http://127.0.0.1:8000/optimized \
  -H 'Content-Type: application/json' \
  -d '{"input_ids": [3, 4, 5, 6], "new_tokens": 8}'
```

Use `--model openai-community/gpt2 --device cuda` for pretrained NVIDIA execution. Leave out `--smoke`. The default bind address is localhost, and the server uses one process. It is an experimental API without authentication; keep it local or behind your own access controls.

## How batching works

A batch can contain different token sequences, but every request must have the same prompt length, output length, and cache mode. There is no padding. The worker starts from the oldest request and collects compatible queued requests up to `--max-batch`.

The optimized path waits up to the configured collection delay before selecting its batch. This simple fixed window can add latency even when a batch is already full. It isn't continuous batching: requests cannot join a batch already generating tokens.

Defaults are a queue capacity of 128, a maximum batch of 8, and a 5 ms collection wait. Try `--max-batch 1` to isolate caching from batching, then increase the batch size to study the extra effect.

## Overload and cancellation

- A full waiting queue returns HTTP 503. The active batch is separate from that queue limit.
- A request timeout returns HTTP 504. The default is 60 seconds from queue admission.
- Disconnected or cancelled requests leave the waiting queue.
- An active batch finishes even if a client leaves; that client's result is discarded. A Python request cancellation cannot safely interrupt a running GPU kernel.
- Shutdown rejects queued work and waits for the active model call to finish.
- A failed model call fails its batch; the worker stays available for later work.

Bounds limit request token counts and batch size. They don't guarantee that every supported request fits the selected GPU's memory.

## Load testing

Keep the server running, then run:

```bash
miniinfer-load --route optimized --concurrency 1 10 50 100 \
  --requests 200 --new-tokens 8 --output results/load-optimized.json
```

Repeat with `--route baseline` and a different output file. Use the same server settings and inputs. Save dependency versions and the model revision alongside published results.

These are closed-loop clients: each client waits for its response before sending another request. Each concurrency level submits the stated total request count. A separate warmup request happens first; later batch sizes may still have first-use effects.

Reports retain every request's status, latency, output IDs, and server timings. They include p50/p95/p99 successful-request latency, successful requests/sec, output tokens/sec, and periodic server telemetry. Failed requests remain visible and don't count toward successful throughput. Telemetry polling itself adds a small amount of load.

## Read the timings

- `queue_ms`: time from queue admission until batch dispatch, including batch collection wait.
- `model_batch_ms`: generation time for the whole model batch.
- `model_ttft_ms`: model-side first-token latency for that batch.
- `server_ms`: queue admission through result preparation, including thread dispatch and input transfer.
- Client `latency_ms`: the full HTTP request and response, including network and serialization.

Responses aren't streamed. Model TTFT isn't the time at which a client receives a first token. Server latency percentiles cover only the most recent 2048 completed requests. Lifetime server rates include idle time; use the load-test report for rates over a measured test interval.

## What is verified?

Tests cover compatible batching, baseline singleton execution, queue overload, cancellation, shutdown, inference errors, input validation, timeout responses, and real random-model HTTP generation parity. CPU checks establish functionality; GPU throughput and 1–100-client performance claims need repeated runs on the intended hardware.

Start reading [batching.py](../src/miniinfer/batching.py), then [server.py](../src/miniinfer/server.py) and [load_test.py](../src/miniinfer/load_test.py).
