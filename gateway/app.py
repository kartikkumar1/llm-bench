"""FastAPI gateway: OpenAI-compatible proxy with Prometheus metrics, plus the dashboard's API.

    uvicorn gateway.app:app --port 8000
Environment overrides: UPSTREAM_URL (default from config.yaml gateway.upstream), RESULTS_DIR.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
UPSTREAM = os.environ.get("UPSTREAM_URL", CONFIG["gateway"]["upstream"]).rstrip("/")
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
DASHBOARD_DIST = ROOT / "dashboard" / "dist"

REQUESTS = Counter("gateway_requests_total", "Chat requests", ["status"])
INFLIGHT = Gauge("gateway_inflight_requests", "Requests currently being served")
TOKENS = Counter("gateway_completion_tokens_total", "Completion tokens streamed")
TTFT = Histogram("gateway_ttft_seconds", "Time to first token",
                 buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16))
E2E = Histogram("gateway_request_seconds", "End-to-end request latency",
                buckets=(0.25, 0.5, 1, 2, 4, 8, 16, 32, 64))
TPS = Histogram("gateway_decode_tokens_per_second", "Per-request decode speed",
                buckets=(1, 2, 5, 10, 20, 40, 80, 160))

# Rolling window for the dashboard's live panel (cheaper than parsing Prometheus in the browser).
RECENT: deque[dict] = deque(maxlen=200)

app = FastAPI(title="llm-bench gateway")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
client = httpx.AsyncClient(timeout=httpx.Timeout(600, connect=5))


@app.get("/health")
async def health():
    try:
        r = await client.get(f"{UPSTREAM}/health", timeout=3)
        upstream = "ok" if r.status_code == 200 else f"HTTP {r.status_code}"
    except httpx.HTTPError as e:
        upstream = f"down ({type(e).__name__})"
    return {"gateway": "ok", "upstream": UPSTREAM, "upstream_status": upstream}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/models")
async def models():
    r = await client.get(f"{UPSTREAM}/v1/models")
    return JSONResponse(r.json(), status_code=r.status_code)


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    if not body.get("stream"):
        return await _chat_blocking(body)

    body.setdefault("stream_options", {"include_usage": True})
    t0 = time.perf_counter()
    upstream_req = client.build_request("POST", f"{UPSTREAM}/v1/chat/completions", json=body)
    try:
        upstream = await client.send(upstream_req, stream=True)
    except httpx.HTTPError as e:
        REQUESTS.labels("upstream_error").inc()
        raise HTTPException(502, f"upstream unreachable: {e}") from e
    if upstream.status_code != 200:
        text = await upstream.aread()
        await upstream.aclose()
        REQUESTS.labels(str(upstream.status_code)).inc()
        return Response(text, status_code=upstream.status_code, media_type="application/json")

    async def relay():
        INFLIGHT.inc()
        first = None
        n_tokens = 0
        status = "ok"
        try:
            async for line in upstream.aiter_lines():
                if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]"):
                    try:
                        chunk = json.loads(line[5:])
                    except json.JSONDecodeError:
                        chunk = {}
                    if any((c.get("delta") or {}).get("content") for c in chunk.get("choices") or []):
                        first = first or time.perf_counter()
                        n_tokens += 1
                    if chunk.get("usage", {}).get("completion_tokens"):
                        n_tokens = chunk["usage"]["completion_tokens"]
                yield line + "\n"
        except httpx.HTTPError:
            status = "stream_error"
            raise
        finally:
            await upstream.aclose()
            INFLIGHT.dec()
            _record(t0, first, n_tokens, status)

    return StreamingResponse(relay(), media_type="text/event-stream")


async def _chat_blocking(body: dict):
    t0 = time.perf_counter()
    INFLIGHT.inc()
    try:
        r = await client.post(f"{UPSTREAM}/v1/chat/completions", json=body)
    except httpx.HTTPError as e:
        REQUESTS.labels("upstream_error").inc()
        raise HTTPException(502, f"upstream unreachable: {e}") from e
    finally:
        INFLIGHT.dec()
    data = r.json()
    if r.status_code == 200:
        _record(t0, None, data.get("usage", {}).get("completion_tokens", 0), "ok")
    else:
        REQUESTS.labels(str(r.status_code)).inc()
    return JSONResponse(data, status_code=r.status_code)


def _record(t0: float, first: float | None, n_tokens: int, status: str) -> None:
    end = time.perf_counter()
    REQUESTS.labels(status).inc()
    E2E.observe(end - t0)
    TOKENS.inc(n_tokens)
    entry = {"ts": time.time(), "e2e_ms": (end - t0) * 1000, "tokens": n_tokens, "status": status}
    if first:
        TTFT.observe(first - t0)
        entry["ttft_ms"] = (first - t0) * 1000
        if n_tokens > 1 and end > first:
            entry["decode_tps"] = (n_tokens - 1) / (end - first)
            TPS.observe(entry["decode_tps"])
    RECENT.append(entry)


# ---- dashboard API -------------------------------------------------------------------------

@app.get("/api/live")
def live():
    recent = list(RECENT)
    ok = [r for r in recent if r["status"] == "ok"]

    def avg(key):
        vals = [r[key] for r in ok if key in r]
        return round(sum(vals) / len(vals), 1) if vals else None

    return {"inflight": INFLIGHT._value.get(), "window": len(recent),
            "errors": len(recent) - len(ok), "ttft_ms_avg": avg("ttft_ms"),
            "e2e_ms_avg": avg("e2e_ms"), "decode_tps_avg": avg("decode_tps"),
            "recent": recent[-50:]}


@app.get("/api/hardware")
def hardware():
    path = RESULTS_DIR / "hardware.json"
    if not path.exists():
        raise HTTPException(404, "run `python -m bench.hwdetect` first")
    return json.loads(path.read_text())


@app.get("/api/results")
def results_index():
    runs = sorted((RESULTS_DIR / "runs").glob("*.json"), reverse=True)
    return [{"id": p.stem, "size_kb": round(p.stat().st_size / 1024, 1)} for p in runs]


@app.get("/api/results/{run_id}")
def result(run_id: str):
    path = RESULTS_DIR / "latest.json" if run_id == "latest" else RESULTS_DIR / "runs" / f"{run_id}.json"
    if not path.resolve().is_relative_to(RESULTS_DIR.resolve()) or not path.exists():
        raise HTTPException(404, "no such run")
    return json.loads(path.read_text())


if DASHBOARD_DIST.exists():
    app.mount("/", StaticFiles(directory=DASHBOARD_DIST, html=True), name="dashboard")
