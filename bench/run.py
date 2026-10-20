"""Run the benchmark matrix: backends x quants x prompt lengths x concurrency.

    python -m bench.run                         # everything in config.yaml
    python -m bench.run --quants q4_k_m q8_0    # override the sweep
    python -m bench.run --backends llamacpp_server --quick
    python -m bench.run --perplexity            # also score quality with llama-perplexity
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from bench.catalog import MODELS
from bench.client import make_prompt, run_round
from bench.hwdetect import ROOT, load_resolved
from bench.servers import BACKENDS, MemorySampler, Server, build_command, free_port
from bench.setup import BIN_DIR, DATA_DIR, llama_server_path, model_path

RESULTS_DIR = ROOT / "results"


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(results, walls: list[float]) -> dict:
    ok = [r for r in results if r.ok]
    if not ok:
        return {"ok": 0, "errors": len(results), "error": results[0].error if results else ""}
    ttft = [r.ttft_ms for r in ok]
    e2e = [r.e2e_ms for r in ok]
    return {
        "ok": len(ok),
        "errors": len(results) - len(ok),
        "ttft_ms_p50": round(pct(ttft, 50), 1),
        "ttft_ms_p95": round(pct(ttft, 95), 1),
        "e2e_ms_p50": round(pct(e2e, 50), 1),
        "e2e_ms_p95": round(pct(e2e, 95), 1),
        "itl_ms_mean": round(statistics.mean(r.itl_ms for r in ok), 2),
        "decode_tps_mean": round(statistics.mean(r.decode_tps for r in ok), 2),
        # Aggregate: every token generated across concurrent requests / wall time.
        "throughput_tps": round(sum(r.completion_tokens for r in ok) / sum(walls), 2),
        "requests_per_s": round(len(ok) / sum(walls), 3),
        "prompt_tokens": ok[0].prompt_tokens,
        "completion_tokens_mean": round(statistics.mean(r.completion_tokens for r in ok), 1),
    }


def perplexity(model: Path, hw: dict, chunks: int) -> float | None:
    text = DATA_DIR / "wiki.test.raw"
    if not text.exists():
        print("  perplexity skipped: run `python -m bench.setup wikitext` first")
        return None
    exe = llama_server_path().with_name(llama_server_path().name.replace("server", "perplexity"))
    out = subprocess.run([str(exe), "-m", str(model), "-f", str(text), "--chunks", str(chunks),
                          "-t", str(hw["threads"]), "-ngl", str(hw["gpu_layers"]), "-c", "512"],
                         capture_output=True, text=True)
    m = re.search(r"Final estimate: PPL = ([\d.]+)", out.stdout + out.stderr)
    return float(m.group(1)) if m else None


def bench_server(server: Server, bcfg: dict) -> tuple[list[dict], dict]:
    ignore_eos = server.backend.startswith("llamacpp_server")
    rows = []
    sampler = MemorySampler(server.proc.pid)
    sampler.start()
    for plen in bcfg["prompt_tokens"]:
        prompt = make_prompt(plen)
        for _ in range(bcfg["warmup"]):
            asyncio.run(run_round(server.url, prompt, 16, 1, ignore_eos))
        for conc in bcfg["concurrency"]:
            results, walls = [], []
            for _ in range(bcfg["repeats"]):
                rs, wall = asyncio.run(run_round(server.url, prompt, bcfg["max_new_tokens"], conc, ignore_eos))
                results += rs
                walls.append(wall)
            s = summarize(results, walls)
            rows.append({"prompt_tokens_target": plen, "concurrency": conc, **s})
            print(f"    prompt={plen:<5} conc={conc:<3} "
                  + (f"ttft p50={s['ttft_ms_p50']:>7.0f}ms  decode={s['decode_tps_mean']:>6.1f} tok/s  "
                     f"throughput={s['throughput_tps']:>6.1f} tok/s" if s["ok"] else f"FAILED {s['error']}"))
    return rows, sampler.stop()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backends", nargs="+", choices=BACKENDS)
    ap.add_argument("--quants", nargs="+")
    ap.add_argument("--quick", action="store_true", help="1 prompt length, concurrency 1, 1 repeat")
    ap.add_argument("--perplexity", action="store_true")
    ap.add_argument("--ppl-chunks", type=int, default=20)
    args = ap.parse_args()

    hw_info, cfg = load_resolved()
    hw, bcfg, mcfg = cfg["hardware"], dict(cfg["benchmark"]), cfg["model"]
    backends = args.backends or bcfg["backends"]
    quants = args.quants or mcfg["quants"]
    if args.quick:
        bcfg.update(prompt_tokens=bcfg["prompt_tokens"][:1], concurrency=[1], repeats=1)
    parallel = max(bcfg["concurrency"])
    name = mcfg["name"]

    missing = [q for q in quants if not model_path(name, q).exists()]
    if missing and any(b != "hf_transformers" for b in backends):
        raise SystemExit(f"missing GGUFs for {missing}; run `python -m bench.setup models`")

    print(f"device={hw['device']} threads={hw['threads']} ngl={hw['gpu_layers']}  model={name}")
    started = datetime.now(timezone.utc)
    runs = []
    for backend in backends:
        # Transformers loads the original weights, so it gets one run instead of one per quant.
        for quant in (["bf16"] if backend == "hf_transformers" else quants):
            print(f"\n[{backend} / {quant}]")
            gguf = model_path(name, quant) if backend != "hf_transformers" else None
            port = free_port()
            cmd = build_command(backend, gguf, port, hw, mcfg["ctx_size"], parallel,
                                hf_id=MODELS[name]["hf_id"])
            entry = {"backend": backend, "quant": quant,
                     "model_file_mb": round(gguf.stat().st_size / 2**20, 1) if gguf else None}
            try:
                with Server(backend, cmd, port) as srv:
                    print(f"  loaded in {srv.load_seconds:.1f}s")
                    rows, memory = bench_server(srv, bcfg)
                    entry.update(load_seconds=round(srv.load_seconds, 2), memory=memory, rows=rows)
                    print(f"  peak RSS {memory['peak_rss_mb']} MB"
                          + (f", GPU {memory['peak_gpu_mb']} MB" if memory["peak_gpu_mb"] else ""))
            except (RuntimeError, TimeoutError, FileNotFoundError) as e:
                print(f"  skipped: {e}")
                entry["error"] = str(e)
            runs.append(entry)

    if args.perplexity:
        print("\n[perplexity]")
        ppl = {}
        for q in quants:
            ppl[q] = perplexity(model_path(name, q), hw, args.ppl_chunks)
            print(f"  {q}: {ppl[q]}")
        for r in runs:
            r["perplexity"] = ppl.get(r["quant"])

    version_file = BIN_DIR / "VERSION"
    report = {
        "started_at": started.isoformat(),
        "duration_s": round((datetime.now(timezone.utc) - started).total_seconds(), 1),
        "model": name,
        "llamacpp_version": version_file.read_text().strip() if version_file.exists() else None,
        "hardware": hw_info,
        "config": cfg,
        "runs": runs,
    }
    out_dir = RESULTS_DIR / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{started.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2))
    (RESULTS_DIR / "latest.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
