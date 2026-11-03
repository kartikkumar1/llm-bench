# llm-bench: LLM inference benchmark and serving stack

Benchmarks **llama.cpp (C++)** against a **Python serving path** on your own hardware, across
quantization levels, measuring latency, throughput, memory and quality. A **FastAPI gateway**
with Prometheus metrics serves the model, and a **React dashboard** shows the results.

```
              ┌──────────────── bench/run.py ────────────────┐
              │  streams requests, samples RSS + GPU memory   │
              ▼                     ▼                         ▼
   llama-server (C++)       llama-cpp-python           HF Transformers     <- each its own
   GPU, or CPU-only              (CPU)                   (optional)           OpenAI-style server
              ▲
              │  /v1/chat/completions (SSE proxy)
       gateway/app.py ── /metrics (Prometheus) ── /api/* ──> dashboard (React + Recharts)
```

## Results

Measured on two of my own machines with the same model and settings. Re-run on your hardware
with `python -m bench.run --perplexity`, then regenerate the tables with `python -m bench.report`.

**Model:** `qwen2.5-0.5b` (Qwen2.5-0.5B-Instruct) · 128 new tokens per request · 3 repeats

| | Laptop | Desktop |
|---|---|---|
| CPU | AMD Ryzen 3 5300U (4C/8T) | AMD Ryzen 3 3200G (4C/4T) |
| RAM | 14.85 GB | 25.72 GB |
| GPU | Radeon integrated (shares system RAM) | Radeon RX 7600, 8 GB GDDR6 (discrete) |
| OS · backend · llama.cpp | Windows 11 · Vulkan · `b11753` | Windows 10 · Vulkan · `b12077` |
| Run | `2026-10-20 10:47 UTC` | `2026-11-03 20:51` and `21:39 UTC` |

### Side by side (q8_0)

| metric | Laptop GPU (iGPU) | Desktop GPU (RX 7600) | Laptop C++ CPU | Desktop C++ CPU |
|---|---:|---:|---:|---:|
| Decode, tok/s (1 request) | 46.9 | **139.9** | 39.4 | 25.7 |
| Time to first token p50, ms (512-token prompt) | 1,024 | **130** | 3,562 | 5,969 |
| Throughput, tok/s (4 concurrent requests) | 77.2 | **165.4** | 44.8 | 24.6 |
| Perplexity vs fp16 | +0.4% | +0.6% | same as GPU | same as GPU |

### What the numbers say

- **Decoding is limited by memory bandwidth, and the discrete GPU proves it.** The laptop's iGPU
  shares the CPU's memory bus, so it decodes no faster than the CPU. The RX 7600 has its own
  GDDR6 memory and decodes q8_0 **3x faster** (140 vs 47 tok/s) and fp16 **5.5x faster**
  (120 vs 22 tok/s).
- **Prompt processing is compute-bound, so a real GPU wins by far more.** Time to first token on a
  512-token prompt drops from ~6 s on the desktop CPU to **130 ms** on the RX 7600, about 46x faster.
- **A 0.5B model is too small to saturate a discrete GPU.** On the RX 7600, fp16 is only ~15%
  slower than q8_0 (120 vs 140 tok/s), even though its weights are twice as large. At this size,
  fixed per-token overhead matters more than bandwidth; a 7B model would show the gap again.
- **Quantization is nearly free down to 8 bits.** q8_0 halves the file for +0.4–0.6% perplexity
  on both machines. On the laptop's full sweep, q4_k_m costs +3.7% and q2_k (+14%) is where
  quality visibly drops.
- **Batching helps on both GPUs.** llama-server batches concurrent requests: q8_0 rises from
  47 to 77 tok/s at 4 requests on the iGPU (1.6x) and from 140 to 165 tok/s on the RX 7600 (1.2x).
- **Python costs memory on the same engine.** llama-cpp-python wraps the same C++ kernels but uses
  about 2x the RAM of llama-server on CPU, on both machines.
- **The desktop CPU is slower than the laptop CPU** (fp16: 10.6 vs 22.5 tok/s). The 3200G has no
  SMT (4 threads vs 8). Because decoding is bandwidth-bound, the RAM setup is also a likely factor:
  single-channel or slow memory would explain the gap, and I have not ruled it out yet.

**Caveats and known issues**

- **llama-cpp-python under concurrency.** At 4 concurrent requests most requests to
  `llama_cpp.server` fail rather than queue, so its throughput figures are computed from the
  requests that succeeded. This is likely its default `interrupt_requests=True`; a fix and re-run
  are in progress. Treat that column as unreliable until then.
- **Noise.** 3 repeats per configuration. On the laptop only ~2–3 GB of RAM was free during the
  run, so neighbouring CPU quants are noisy (e.g. q5_k_m slower than q4_k_m). On the desktop,
  GPU throughput is not monotonic across quants (q5_k_m 191 > fp16 174 > q8_0 165).
- **Different llama.cpp builds** (`b11753` vs `b12077`), so some of the gap may come from the build
  rather than the hardware.
- The llama-cpp-python wheel is CPU-only, which is why `llama.cpp C++ (CPU)` exists as the
  like-for-like comparison. The `[6.1]` GPU memory on the desktop CPU backend is the Vulkan
  context the process still opens, not model weights.

<details>
<summary><b>Full tables: laptop (Ryzen 3 5300U, integrated GPU)</b></summary>

#### Quality (perplexity on wikitext-2, lower is better)
| quant | file MB | perplexity | vs fp16 |
|---|---|---|---|
| fp16 | 1,208 | 16.734 | +0.0% |
| q8_0 | 644 | 16.804 | +0.4% |
| q5_k_m | 498 | 17.089 | +2.1% |
| q4_k_m | 469 | 17.357 | +3.7% |
| q3_k_m | 412 | 17.998 | +7.6% |
| q2_k | 396 | 19.144 | +14.4% |

#### Decode speed, tok/s (prompt 128, 1 request)
| quant | llama.cpp C++ (GPU) | llama.cpp C++ (CPU) | llama-cpp-python (CPU) |
|---|---|---|---|
| fp16 | 21.7 | 22.5 | 21.9 |
| q8_0 | 46.9 | 39.4 | 36.1 |
| q5_k_m | 44.5 | 47.7 | 38.6 |
| q4_k_m | 42.9 | 49.9 | 38.5 |
| q3_k_m | 47.1 | 60.8 | 46.2 |
| q2_k | 46.0 | 63.3 | 47.7 |

#### Time to first token p50, ms (prompt 512, 1 request)
| quant | llama.cpp C++ (GPU) | llama.cpp C++ (CPU) | llama-cpp-python (CPU) |
|---|---|---|---|
| fp16 | 946 | 3,232 | 4,886 |
| q8_0 | 1,024 | 3,562 | 4,842 |
| q5_k_m | 1,083 | 5,667 | 5,901 |
| q4_k_m | 1,091 | 4,407 | 5,267 |
| q3_k_m | 1,030 | 2,057 | 3,513 |
| q2_k | 1,039 | 2,033 | 3,663 |

#### Aggregate throughput, tok/s (prompt 128, 4 concurrent requests)
| quant | llama.cpp C++ (GPU) | llama.cpp C++ (CPU) | llama-cpp-python (CPU) |
|---|---|---|---|
| fp16 | 57.2 | 43.6 | 11.7 |
| q8_0 | 77.2 | 44.8 | 15.1 |
| q5_k_m | 55.5 | 34.8 | 12.0 |
| q4_k_m | 58.0 | 45.2 | 14.1 |
| q3_k_m | 54.7 | 67.8 | 19.4 |
| q2_k | 46.7 | 64.4 | 19.7 |

#### Peak memory, MB (RSS; GPU memory in brackets)
| quant | llama.cpp C++ (GPU) | llama.cpp C++ (CPU) | llama-cpp-python (CPU) |
|---|---|---|---|
| fp16 | 955 [1,409] | 1,338 | 2,384 |
| q8_0 | 456 [840] | 894 | 1,821 |
| q5_k_m | 420 [699] | 724 | 1,674 |
| q4_k_m | 412 [670] | 797 | 1,673 |
| q3_k_m | 397 [635] | 907 | 1,767 |
| q2_k | 395 [611] | 847 | 1,707 |

</details>

<details>
<summary><b>Full tables: desktop (Ryzen 3 3200G + RX 7600)</b> (fp16, q8_0 and q5_k_m only)</summary>

#### Quality (perplexity on wikitext-2, lower is better)
| quant | file MB | perplexity | vs fp16 |
|---|---:|---:|---:|
| fp16 | 1,208 | 16.7316 | +0.0% |
| q8_0 | 644 | 16.8246 | +0.6% |
| q5_k_m | 498 | 17.0900 | +2.1% |

#### Decode speed, tok/s (prompt 128, 1 request)
| quant | llama.cpp C++ (GPU) | llama.cpp C++ (CPU) | llama-cpp-python (CPU) |
|---|---:|---:|---:|
| fp16 | 119.9 | 10.6 | 10.6 |
| q8_0 | 139.9 | 25.7 | 28.5 |
| q5_k_m | 138.1 | 29.5 | 28.6 |

#### Time to first token p50, ms (prompt 512, 1 request)
| quant | llama.cpp C++ (GPU) | llama.cpp C++ (CPU) | llama-cpp-python (CPU) |
|---|---:|---:|---:|
| fp16 | 127 | 6,092 | 10,707 |
| q8_0 | 130 | 5,969 | 8,299 |
| q5_k_m | 133 | 7,390 | 9,942 |

#### Aggregate throughput, tok/s (prompt 128, 4 concurrent requests)
| quant | llama.cpp C++ (GPU) | llama.cpp C++ (CPU) | llama-cpp-python (CPU) |
|---|---:|---:|---:|
| fp16 | 174.3 | 19.6 | 5.7 |
| q8_0 | 165.4 | 24.6 | 9.6 |
| q5_k_m | 191.3 | 23.9 | 8.3 |

#### Peak memory, MB (RSS; GPU memory in brackets)
| quant | llama.cpp C++ (GPU) | llama.cpp C++ (CPU) | llama-cpp-python (CPU) |
|---|---:|---:|---:|
| fp16 | 1,636.6 [1,415.1] | 1,335.6 [6.1] | 2,379.8 |
| q8_0 | 1,077.9 [878.4] | 891.7 [6.1] | 1,816.5 |
| q5_k_m | 998.3 [797.1] | 755.4 [6.1] | 1,669.6 |

</details>

## Hardware is detected automatically

`python -m bench.hwdetect` reads the CPU, RAM and GPU and fills in every `auto` value in
[config.yaml](config.yaml):

| Setting | auto rule |
|---|---|
| `device` | NVIDIA -> `cuda`, Apple Silicon -> `metal`, AMD/Intel GPU with Vulkan -> `vulkan`, else `cpu` |
| `threads` | physical core count |
| `gpu_layers` | 99 (all) on a GPU, 0 on CPU |
| `max_ram_gb` | min(40% of total RAM, 80% of currently free RAM) |
| `max_vram_gb` | 80% of VRAM on a discrete GPU; integrated GPUs share the RAM budget |
| `model` | `profile: minimal` -> 0.5B, `balanced` -> 1.5B, `max` -> largest that fits |
| `quants` | every default quant whose weights + 0.6 GB overhead fit the budget |

Override any of these by replacing `auto` in `config.yaml`. The benchmark, setup script and
gateway all read the same resolved config.

## Ways to run

### 1. First-time setup (native)

```bash
python -m venv .venv
.venv\Scripts\activate                     # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install "llama-cpp-python[server]" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
python -m bench.hwdetect                   # shows detected hardware + resolved config
python -m bench.setup all                  # llama.cpp binaries for your device, GGUFs, wikitext-2
```

`bench.setup` picks the matching llama.cpp release build (CUDA, Vulkan, Metal or CPU). If Hugging
Face downloads stall at 0 bytes, set `HF_HUB_DISABLE_XET=1` and re-run; finished files are skipped.

### 2. Run the benchmark

```bash
python -m bench.run --perplexity                              # full sweep + quality -> results/runs/*.json
python -m bench.run --quick --backends llamacpp_server        # ~1-minute smoke test
python -m bench.run --quants q8_0 q4_k_m                      # subset of quants
python -m bench.run --backends llamacpp_server llamacpp_server_cpu   # GPU vs CPU only
python -m bench.report                                        # Markdown tables for the latest run
```

Optional pure-Python baseline: `pip install torch transformers`, then add `hf_transformers` to
`--backends` (or to `benchmark.backends` in `config.yaml`).

### 3. Serve the model behind the gateway

```bash
python serve.py                    # llama-server (q4_k_m) on :8081 + gateway on :8000
python serve.py --quant q8_0       # pick another quant
```

Or run the pieces separately, pointing the gateway at any OpenAI-compatible server:

```bash
uvicorn gateway.app:app --port 8000          # UPSTREAM_URL env var overrides config.yaml
```

Talk to it:

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"stream\":true}"
```

| Endpoint | What it returns |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible chat, streaming or not |
| `GET /health` | gateway and upstream status |
| `GET /metrics` | Prometheus: requests, TTFT, latency, tok/s histograms, in-flight |
| `GET /api/live` | rolling stats of recent requests (used by the dashboard) |
| `GET /api/results`, `/api/results/{id or latest}` | benchmark runs |
| `GET /api/hardware` | detected hardware and resolved config |

### 4. Dashboard

```bash
cd dashboard && npm install && npm run build     # gateway then serves it at http://localhost:8000
cd dashboard && npm run dev                      # hot reload on :5173, proxies /api to :8000
```

### 5. Docker

```bash
docker compose up --build                                                    # CPU
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up --build   # NVIDIA GPU
```

This starts llama-server (:8081), the gateway and dashboard (:8000) and Prometheus (:9090). It
needs the GGUFs in `models/` first (`python -m bench.setup models`). Copy `.env.example` to `.env`
to pick the model file, threads and memory limit. Docker Desktop cannot pass AMD, Intel or Apple
GPUs into containers, so use the native path (section 3) on those machines.

## What is measured

| Metric | How |
|---|---|
| TTFT p50/p95 | request sent -> first streamed token |
| Inter-token latency | mean gap between streamed tokens |
| Decode tok/s | (completion tokens - 1) / time after first token, per request |
| Throughput | all tokens generated by concurrent requests / wall time |
| Peak RSS | `psutil`, summed over the server's process tree, sampled at 10 Hz |
| GPU memory | `nvidia-smi` per PID, or Windows GPU perf counters (dedicated + shared) |
| Quality | `llama-perplexity` on wikitext-2 (20 chunks by default) |

Methodology notes:

- Each backend runs as its own server, so one client measures all of them the same way and
  memory is attributed to the right process.
- Every request starts with a unique tag, so no server can reuse a cached prompt prefix; without
  it, repeats skip prompt processing and TTFT looks ~10x better than reality.
- llama-server gets `ignore_eos`, so every request generates exactly `max_new_tokens`.
- `llamacpp_server_cpu` runs with `--device none`: `-ngl 0` alone still copies weights to the GPU
  to speed up prompt processing.
- Each run records the hardware, resolved config and llama.cpp version alongside the numbers.

## Layout

```
bench/      hwdetect, setup (downloads), servers (launch + memory), client (timing), run (matrix), report
backends/   hf_server.py: Transformers behind an OpenAI-style API
gateway/    FastAPI proxy, Prometheus metrics, dashboard API
dashboard/  React + Vite + Recharts
ops/        Prometheus config
results/    hardware.json, latest.json, runs/*.json (logs/ is git-ignored)
```
