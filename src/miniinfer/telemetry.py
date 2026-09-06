"""Optional, device-wide NVML samples. Unavailable metrics remain null."""

import threading
import time


class GpuSampler:
    def __init__(self, device, interval=0.05):
        self.interval = interval
        self.samples = []
        self.error = None
        self.stop_event = threading.Event()
        self.thread = None
        self.nvml = None
        if device.type != "cuda":
            self.error = "NVML telemetry requires NVIDIA CUDA"
            return
        try:
            import pynvml
            import torch

            self.nvml = pynvml
            pynvml.nvmlInit()
            # UUID respects CUDA_VISIBLE_DEVICES remapping; never assume index 0.
            uuid = torch.cuda.get_device_properties(device).uuid
            self.handle = pynvml.nvmlDeviceGetHandleByUUID(str(uuid))
        except Exception as exc:
            self.error = str(exc)

    def _sample(self):
        while not self.stop_event.is_set():
            try:
                utilization = self.nvml.nvmlDeviceGetUtilizationRates(self.handle)
                memory = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
                self.samples.append({
                    "monotonic_seconds": time.perf_counter(),
                    "gpu_utilization_percent": utilization.gpu,
                    "device_memory_used_bytes": memory.used,
                })
            except Exception as exc:
                self.error = str(exc)
                return
            self.stop_event.wait(self.interval)

    def __enter__(self):
        if self.error is None:
            self.thread = threading.Thread(target=self._sample, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        if self.nvml:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                pass

    def report(self):
        values = [s["gpu_utilization_percent"] for s in self.samples]
        return {
            "scope": "whole device, including other processes; sampled across measured trials",
            "sampling_interval_seconds": self.interval,
            "mean_gpu_utilization_percent": sum(values) / len(values) if values else None,
            "peak_device_memory_used_bytes": max(
                (s["device_memory_used_bytes"] for s in self.samples), default=None
            ),
            "unavailable_reason": self.error,
            "samples": self.samples,
        }
