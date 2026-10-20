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

Measured on a budget laptop. Re-run on your machine with `python -m bench.run --perplexity`,
then regenerate these tables with `python -m bench.report`.

**Hardware:** AMD Ryzen 3 5300U (4C/8T), 14.85 GB RAM, AMD Radeon integrated GPU · device `vulkan` · llama.cpp `b11753`  
**Model:** `qwen2.5-0.5b` (Qwen2.5-0.5B-Instruct) · 128 new tokens per request · 3 repeats · run `2026-10-20 10:47 UTC`

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

### What the numbers say

- **Quantization is nearly free down to 8 bits.** q8_0 halves the file for +0.4% perplexity;
  q4_k_m costs +3.7%; q2_k (+14%) is where quality visibly drops.
- **The integrated GPU wins prompt processing, not decoding.** TTFT on a 512-token prompt is
  ~1 s on Vulkan vs 2–5.7 s on CPU, but decode speed is similar: an iGPU shares the CPU's
  memory bus, and decoding is memory-bandwidth bound.
- **fp16 decodes at ~22 tok/s on every backend.** Bandwidth again: its weights are 2–3x larger.
- **Python costs speed and memory, even on the same engine.** llama-cpp-python wraps the same
  C++ kernels, yet decodes up to ~25% slower and uses ~2x the RAM of llama-server on CPU.
- **Serving architecture matters more than the language under load.** llama-server batches
  concurrent requests (q8_0 on GPU: 42 -> 77 tok/s at 4 requests); `llama_cpp.server` handles
  one request at a time, so aggregate throughput *falls* and queued requests wait.

**Caveats:** 3 repeats per configuration with only ~2–3 GB of RAM free during the run, so
neighbouring CPU quants are noisy (e.g. q5_k_m slower than q4_k_m, RSS not monotonic). The
llama-cpp-python wheel is CPU-only, which is why `llama.cpp C++ (CPU)` exists as the like-for-like
comparison. For publishable numbers, close other apps and raise `benchmark.repeats` to 5+.

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
