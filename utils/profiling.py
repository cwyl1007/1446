import os
import resource
import threading
import time
import torch


def configure_torch_cpu_threads(default_cap=32):
    current = max(1, int(torch.get_num_threads()))
    requested = os.environ.get("LP_TORCH_CPU_THREADS")
    if requested is None or not requested.strip():
        target = min(current, max(1, int(default_cap)))
    elif requested.strip().lower() in {"0", "auto", "none", "off"}:
        return current
    else:
        try:
            target = max(1, int(requested))
        except ValueError as exc:
            raise ValueError("LP_TORCH_CPU_THREADS must be a positive integer, 0, or auto.") from exc
    if target != current:
        torch.set_num_threads(target)
    return int(torch.get_num_threads())


def current_cpu_rss_mb():
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        return 0.0
    return 0.0


def peak_cpu_rss_mb():
    peak_kb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if os.uname().sysname == "Darwin":
        return peak_kb / (1024.0 * 1024.0)
    return peak_kb / 1024.0


def _cuda_enabled(device):
    return torch.cuda.is_available() and str(device).startswith("cuda")


class StageProfiler:

    def __init__(self, device, cpu_sample_interval_sec=0.02):
        self.device = device
        self.use_cuda = _cuda_enabled(device)
        self.cpu_sample_interval_sec = max(0.001, float(cpu_sample_interval_sec))
        self.t0 = None
        self.cpu_rss0 = 0.0
        self._cpu_peak_rss_mb = 0.0
        self._cpu_sampler_stop = None
        self._cpu_sampler_thread = None

    def _sample_cpu_rss(self):
        while not self._cpu_sampler_stop.wait(self.cpu_sample_interval_sec):
            self._cpu_peak_rss_mb = max(self._cpu_peak_rss_mb, current_cpu_rss_mb())

    def start(self):
        if self.use_cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self.cpu_rss0 = current_cpu_rss_mb()
        self._cpu_peak_rss_mb = self.cpu_rss0
        self._cpu_sampler_stop = threading.Event()
        self._cpu_sampler_thread = threading.Thread(target=self._sample_cpu_rss, name="stage-profiler-rss", daemon=True)
        self._cpu_sampler_thread.start()
        self.t0 = time.time()

    def stop(self):
        if self.use_cuda:
            torch.cuda.synchronize(self.device)
        elapsed = time.time() - self.t0 if self.t0 is not None else 0.0
        cpu_rss = current_cpu_rss_mb()
        self._cpu_peak_rss_mb = max(self._cpu_peak_rss_mb, cpu_rss)
        if self._cpu_sampler_stop is not None:
            self._cpu_sampler_stop.set()
        if self._cpu_sampler_thread is not None:
            self._cpu_sampler_thread.join()
        info = {
            "sec": elapsed,
            "cpu_rss_mb": cpu_rss,
            "cpu_rss_delta_mb": cpu_rss - self.cpu_rss0,
            "cpu_peak_rss_mb": self._cpu_peak_rss_mb,
            "cuda_allocated_mb": 0.0,
            "cuda_reserved_mb": 0.0,
            "cuda_peak_allocated_mb": 0.0,
            "cuda_peak_reserved_mb": 0.0,
        }
        if self.use_cuda:
            info.update(
                {
                    "cuda_allocated_mb": float(torch.cuda.memory_allocated(self.device)) / (1024.0 * 1024.0),
                    "cuda_reserved_mb": float(torch.cuda.memory_reserved(self.device)) / (1024.0 * 1024.0),
                    "cuda_peak_allocated_mb": float(torch.cuda.max_memory_allocated(self.device)) / (1024.0 * 1024.0),
                    "cuda_peak_reserved_mb": float(torch.cuda.max_memory_reserved(self.device)) / (1024.0 * 1024.0),
                }
            )
        return info


def empty_stage_info():
    return {
        "sec": 0.0,
        "cpu_rss_mb": 0.0,
        "cpu_rss_delta_mb": 0.0,
        "cpu_peak_rss_mb": 0.0,
        "cuda_allocated_mb": 0.0,
        "cuda_reserved_mb": 0.0,
        "cuda_peak_allocated_mb": 0.0,
        "cuda_peak_reserved_mb": 0.0,
    }
