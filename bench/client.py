"""Streaming client that times one chat completion against an OpenAI-compatible server."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import asdict, dataclass

import httpx

# Neutral filler text; repeated to reach the target prompt length (~1.3 tokens per word).
_FILLER = (
    "The history of computing spans mechanical calculators, vacuum tubes, transistors and "
    "integrated circuits. Each generation made machines smaller, faster and cheaper, and "
    "opened new uses that earlier designers had not imagined. "
)


def make_prompt(target_tokens: int) -> str:
    words_needed = int(target_tokens / 1.3)
    words = (_FILLER * (words_needed // len(_FILLER.split()) + 1)).split()[:words_needed]
    return " ".join(words) + "\n\nContinue this text in detail."


@dataclass
class RequestResult:
    ok: bool
    ttft_ms: float = 0.0
    e2e_ms: float = 0.0
    itl_ms: float = 0.0             # mean inter-token latency
    prompt_tokens: int = 0
    completion_tokens: int = 0
    decode_tps: float = 0.0         # completion tokens / time after first token
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


async def chat_once(client: httpx.AsyncClient, url: str, prompt: str, max_tokens: int,
                    ignore_eos: bool) -> RequestResult:
    # Servers reuse the KV cache of a matching prompt prefix, which would make every repeat
    # skip prompt processing and hide the real TTFT. A unique leading tag defeats that on
    # every backend, at the cost of a few extra tokens.
    unique = f"[request {uuid.uuid4().hex[:12]}]\n"
    body = {
        "model": "bench",
        "messages": [{"role": "user", "content": unique + prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:   # llama-server only: always generate max_tokens for a fixed workload
        body["ignore_eos"] = True
        body["cache_prompt"] = False

    t0 = time.perf_counter()
    stamps: list[float] = []
    usage = {}
    try:
        async with client.stream("POST", f"{url}/v1/chat/completions", json=body, timeout=600) as r:
            if r.status_code != 200:
                return RequestResult(ok=False, error=f"HTTP {r.status_code}: {(await r.aread())[:200]!r}")
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    if (choice.get("delta") or {}).get("content"):
                        stamps.append(time.perf_counter())
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        return RequestResult(ok=False, error=repr(e))

    if not stamps:
        return RequestResult(ok=False, error="no tokens streamed")
    end = time.perf_counter()
    completion = usage.get("completion_tokens") or len(stamps)
    decode_s = end - stamps[0]
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    return RequestResult(
        ok=True,
        ttft_ms=(stamps[0] - t0) * 1000,
        e2e_ms=(end - t0) * 1000,
        itl_ms=(sum(gaps) / len(gaps) * 1000) if gaps else 0.0,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=completion,
        decode_tps=(completion - 1) / decode_s if decode_s > 0 and completion > 1 else 0.0,
    )


async def run_round(url: str, prompt: str, max_tokens: int, concurrency: int,
                    ignore_eos: bool) -> tuple[list[RequestResult], float]:
    """Fire `concurrency` requests at once; return their results and the round's wall time."""
    limits = httpx.Limits(max_connections=concurrency + 2)
    async with httpx.AsyncClient(limits=limits) as client:
        t0 = time.perf_counter()
        results = await asyncio.gather(*[chat_once(client, url, prompt, max_tokens, ignore_eos)
                                         for _ in range(concurrency)])
        return list(results), time.perf_counter() - t0
