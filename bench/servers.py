"""Start each backend as its own OpenAI-compatible HTTP server.

Running every backend out-of-process means one client measures them all the same
way, and the server's PID gives a clean memory reading per backend.
"""
from __future__ import annotations

import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import psutil

from bench.hwdetect import ROOT
from bench.setup import llama_server_path

# llamacpp_server_cpu is the same C++ server with offload disabled, so C++-vs-Python can be
# compared on equal (CPU) footing even when the Python wheel has no GPU support.
BACKENDS = ("llamacpp_server", "llamacpp_server_cpu", "llamacpp_python", "hf_transformers")
LOG_DIR = ROOT / "results" / "logs"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def build_command(backend: str, model: Path, port: int, hw: dict, ctx: int, parallel: int,
                  hf_id: str | None = None) -> list[str]:
    threads, ngl = str(hw["threads"]), str(hw["gpu_layers"])
    if backend in ("llamacpp_server", "llamacpp_server_cpu"):
        # -ngl 0 alone still uploads weights for GPU prompt processing (op offload);
        # "--device none" hides the GPU entirely so this really is CPU-only.
        cpu_only = ["--device", "none"] if backend == "llamacpp_server_cpu" else []
        if cpu_only:
            ngl = "0"
        return [str(llama_server_path()), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
                "-t", threads, "-ngl", ngl, "-c", str(ctx * parallel), "--parallel", str(parallel),
                "--no-webui", "--metrics", *cpu_only]
    if backend == "llamacpp_python":
        return [sys.executable, "-m", "llama_cpp.server", "--model", str(model), "--host", "127.0.0.1",
                "--port", str(port), "--n_threads", threads, "--n_gpu_layers", ngl, "--n_ctx", str(ctx)]
    if backend == "hf_transformers":
        return [sys.executable, "-m", "backends.hf_server", "--model", hf_id, "--port", str(port),
                "--threads", threads]
    raise ValueError(f"unknown backend {backend}")


class MemorySampler(threading.Thread):
    """Samples RSS of a process tree (and GPU memory where the OS exposes it) until stopped."""

    def __init__(self, pid: int, interval: float = 0.1):
        super().__init__(daemon=True)
        self.pid, self.interval = pid, interval
        self.peak_rss = 0
        self.peak_gpu = 0
        self._halt = threading.Event()   # not `_stop`: that name is taken by Thread internals

    def run(self) -> None:
        last_gpu = 0.0
        while not self._halt.is_set():
            try:
                root = psutil.Process(self.pid)
                procs = [root, *root.children(recursive=True)]
                rss = sum(p.memory_info().rss for p in procs if p.is_running())
                self.peak_rss = max(self.peak_rss, rss)
            except psutil.Error:
                pass
            if time.time() - last_gpu > 1.0:   # GPU queries are slow; sample at 1 Hz
                self.peak_gpu = max(self.peak_gpu, gpu_memory_bytes(self.pid))
                last_gpu = time.time()
            self._halt.wait(self.interval)

    def stop(self) -> dict:
        self._halt.set()
        self.join(timeout=5)
        return {"peak_rss_mb": round(self.peak_rss / 2**20, 1),
                "peak_gpu_mb": round(self.peak_gpu / 2**20, 1) if self.peak_gpu else None}


def gpu_memory_bytes(pid: int) -> int:
    """Per-process GPU memory: nvidia-smi on NVIDIA, the GPU perf counters on Windows."""
    if shutil.which("nvidia-smi"):
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        return sum(int(m) * 2**20 for p, m in (l.split(",") for l in out.strip().splitlines() if l)
                   if int(p) == pid)
    if platform.system() == "Windows":
        # Dedicated + shared usage covers both discrete cards and integrated GPUs.
        cmd = (f"(Get-Counter -ErrorAction SilentlyContinue '\\GPU Process Memory(pid_{pid}_*)\\Dedicated Usage',"
               f"'\\GPU Process Memory(pid_{pid}_*)\\Shared Usage').CounterSamples | "
               f"Measure-Object -Property CookedValue -Sum | ForEach-Object Sum")
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                             capture_output=True, text=True).stdout.strip()
        try:
            return int(float(out))
        except ValueError:
            return 0
    return 0


@dataclass
class Server:
    backend: str
    cmd: list[str]
    port: int
    proc: subprocess.Popen | None = None
    load_seconds: float = 0.0
    log_path: Path = field(default=None)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout: float = 300) -> "Server":
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = LOG_DIR / f"{self.backend}-{self.port}.log"
        log = self.log_path.open("w")
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(self.cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        while time.perf_counter() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"{self.backend} exited early, see {self.log_path}")
            try:
                if httpx.get(f"{self.url}/v1/models", timeout=2).status_code == 200:
                    # llama-server answers /v1/models while the model is still loading.
                    h = httpx.get(f"{self.url}/health", timeout=2)
                    if h.status_code in (200, 404):
                        self.load_seconds = time.perf_counter() - t0
                        return self
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
        self.stop()
        raise TimeoutError(f"{self.backend} not ready after {timeout}s, see {self.log_path}")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            for child in psutil.Process(self.proc.pid).children(recursive=True):
                child.kill()
            self.proc.kill()
            self.proc.wait(timeout=10)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
