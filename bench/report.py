"""Render a benchmark run as Markdown tables (for the README or a PR).

    python -m bench.report                    # results/latest.json
    python -m bench.report results/runs/20261020-104726.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from bench.catalog import QUANT_ORDER
from bench.hwdetect import ROOT

LABELS = {
    "llamacpp_server": "llama.cpp C++ (GPU)",
    "llamacpp_server_cpu": "llama.cpp C++ (CPU)",
    "llamacpp_python": "llama-cpp-python (CPU)",
    "hf_transformers": "HF Transformers",
}


def _row(run: dict, prompt: int, conc: int) -> dict:
    return next((r for r in run.get("rows", []) if r["prompt_tokens_target"] == prompt
                 and r["concurrency"] == conc and r.get("ok")), {})


def _fmt(v, digits=1) -> str:
    return "–" if v is None else f"{v:,.{digits}f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def render(report: dict) -> str:
    runs = [r for r in report["runs"] if r.get("rows")]
    backends = [b for b in LABELS if any(r["backend"] == b for r in runs)]
    quants = sorted({r["quant"] for r in runs}, key=lambda q: QUANT_ORDER.index(q) if q in QUANT_ORDER else -1)
    by = {(r["backend"], r["quant"]): r for r in runs}
    cfg, hw = report["config"], report["hardware"]
    prompts, concs = cfg["benchmark"]["prompt_tokens"], cfg["benchmark"]["concurrency"]
    lo_p, hi_p, hi_c = min(prompts), max(prompts), max(concs)

    def grid(metric, prompt, conc, digits=1):
        rows = []
        for q in quants:
            cells = []
            for b in backends:
                r = by.get((b, q))
                cells.append(_fmt(_row(r, prompt, conc).get(metric), digits) if r else "–")
            rows.append([q, *cells])
        return _table(["quant", *[LABELS[b] for b in backends]], rows)

    gpu = hw.get("selected_gpu") or {}
    parts = [
        f"**Hardware:** {hw['cpu']['name']} ({hw['cpu']['physical_cores']}C/{hw['cpu']['logical_cores']}T), "
        f"{hw['ram']['total_gb']} GB RAM, {gpu.get('name', 'no GPU')} · device `{cfg['hardware']['device']}` · "
        f"llama.cpp `{(report.get('llamacpp_version') or '?').split()[0]}`  ",
        f"**Model:** `{report['model']}` · {cfg['benchmark']['max_new_tokens']} new tokens per request · "
        f"{cfg['benchmark']['repeats']} repeats · run `{report['started_at'][:16].replace('T', ' ')} UTC`",
        "",
        "#### Quality (perplexity on wikitext-2, lower is better)",
    ]
    ppl = {r["quant"]: r.get("perplexity") for r in runs if r.get("perplexity")}
    if ppl:
        base = ppl.get("fp16")
        parts.append(_table(["quant", "file MB", "perplexity", "vs fp16"], [
            [q, _fmt(by.get(("llamacpp_server", q), {}).get("model_file_mb"), 0), _fmt(ppl[q], 3),
             f"{(ppl[q] / base - 1) * 100:+.1f}%" if base else "–"]
            for q in quants if q in ppl]))
    parts += [
        "", f"#### Decode speed, tok/s (prompt {lo_p}, 1 request)", grid("decode_tps_mean", lo_p, 1),
        "", f"#### Time to first token p50, ms (prompt {hi_p}, 1 request)", grid("ttft_ms_p50", hi_p, 1, 0),
        "", f"#### Aggregate throughput, tok/s (prompt {lo_p}, {hi_c} concurrent requests)",
        grid("throughput_tps", lo_p, hi_c),
        "", "#### Peak memory, MB (RSS; GPU memory in brackets)",
        _table(["quant", *[LABELS[b] for b in backends]], [
            [q, *[(lambda m: _fmt(m.get("peak_rss_mb"), 0)
                   + (f" [{_fmt(m['peak_gpu_mb'], 0)}]" if m.get("peak_gpu_mb") and m["peak_gpu_mb"] > 50 else ""))
                  (by[(b, q)].get("memory") or {}) if (b, q) in by else "–" for b in backends]]
            for q in quants]),
    ]
    return "\n".join(parts)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "latest.json"
    print(render(json.loads(path.read_text())))


if __name__ == "__main__":
    main()
