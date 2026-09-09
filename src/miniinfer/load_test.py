"""Closed-loop HTTP load tests; client latency includes the full response."""

import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import time

import httpx

from miniinfer.metrics import percentile


async def measure(client, route, payload, concurrency, requests):
    samples = []
    next_request = 0
    async def worker():
        nonlocal next_request
        while next_request < requests:
            index = next_request
            next_request += 1
            start = time.perf_counter()
            try:
                response = await client.post(route, json=payload)
                data = response.json() if response.is_success else None
                samples.append({"request": index, "latency_ms": (time.perf_counter() - start) * 1000,
                                "status": response.status_code, "response": data})
            except (httpx.HTTPError, ValueError) as exc:
                samples.append({"request": index, "latency_ms": (time.perf_counter() - start) * 1000,
                                "status": "client-error", "error": str(exc), "response": None})
    started = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    elapsed = time.perf_counter() - started
    successful = [row for row in samples if row["status"] == 200]
    latencies = [row["latency_ms"] for row in successful]
    return {"concurrency": concurrency, "requests": requests, "elapsed_seconds": elapsed,
            "status_counts": dict(Counter(str(row["status"]) for row in samples)),
            "successful_requests_per_second": len(successful) / elapsed,
            "output_tokens_per_second": sum(len(row["response"]["output_ids"]) for row in successful) / elapsed,
            "successful_latency_ms": {name: percentile(latencies, q) if latencies else None
                                      for name, q in (("p50", .5), ("p95", .95), ("p99", .99))},
            "samples": sorted(samples, key=lambda row: row["request"])}


async def run(args):
    if args.output.exists():
        raise FileExistsError(f"Output exists: {args.output}")
    if min(args.concurrency) < 1 or args.requests < max(args.concurrency) or args.timeout <= 0:
        raise ValueError("Use positive concurrency/timeout and requests >= maximum concurrency")
    payload = {"input_ids": args.input_ids, "new_tokens": args.new_tokens}
    limits = httpx.Limits(max_connections=max(args.concurrency), max_keepalive_connections=max(args.concurrency))
    async with httpx.AsyncClient(base_url=args.url, timeout=args.timeout, limits=limits) as client:
        initial = await client.get("/metrics")
        initial.raise_for_status()
        # One separate HTTP warmup. Later batch shapes may still have first-use costs.
        warmup = await client.post(f"/{args.route}", json=payload)
        warmup.raise_for_status()
        telemetry = []
        stop = asyncio.Event()
        async def poll():
            while not stop.is_set():
                try:
                    response = await client.get("/metrics")
                    response.raise_for_status()
                    telemetry.append({"monotonic_seconds": time.perf_counter(), "metrics": response.json()})
                except (httpx.HTTPError, ValueError) as exc:
                    telemetry.append({"error": str(exc)})
                try:
                    await asyncio.wait_for(stop.wait(), timeout=1)
                except TimeoutError:
                    pass
        poller = asyncio.create_task(poll())
        try:
            cases = [await measure(client, f"/{args.route}", payload, concurrency, args.requests)
                     for concurrency in args.concurrency]
        finally:
            stop.set()
            await poller
    report = {"schema_version": 1, "experiment": "v5-http-load", "config": {**vars(args), "output": str(args.output)},
              "server": initial.json(), "cases": cases, "telemetry_samples": telemetry,
              "note": "Closed-loop clients; latency is successful full-response HTTP latency, not streamed TTFT. Errors are retained. One separate warmup; telemetry polling adds load."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--route", choices=["baseline", "optimized"], default="optimized")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 10, 50, 100])
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--input-ids", nargs="+", type=int, default=[3, 4, 5, 6])
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = asyncio.run(run(args))
    except (ValueError, OSError, httpx.HTTPError) as exc:
        parser.error(str(exc))
    for case in report["cases"]:
        print(json.dumps({key: value for key, value in case.items() if key != "samples"}))


if __name__ == "__main__":
    main()
