"""Optional pure-Python baseline: Hugging Face Transformers behind a minimal OpenAI-style API.

Needs: pip install torch transformers  (CPU wheels are fine)
    python -m backends.hf_server --model Qwen/Qwen2.5-0.5B-Instruct --port 8082
"""
from __future__ import annotations

import argparse
import json
import threading
import time

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

app = FastAPI()
state: dict = {}
lock = threading.Lock()   # one generation at a time; transformers has no batching scheduler here


@app.get("/v1/models")
def models():
    return {"data": [{"id": state["name"], "object": "model"}]}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
def chat(body: dict):
    tok, model = state["tok"], state["model"]
    ids = tok.apply_chat_template(body["messages"], add_generation_prompt=True, return_tensors="pt")
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    max_new = int(body.get("max_tokens", 128))

    def generate():
        with lock, torch.inference_mode():
            model.generate(ids, streamer=streamer, max_new_tokens=max_new, do_sample=False)

    def events():
        threading.Thread(target=generate, daemon=True).start()
        n = 0
        for text in streamer:
            if not text:
                continue
            n += 1
            yield f"data: {json.dumps({'choices': [{'index': 0, 'delta': {'content': text}}]})}\n\n"
        usage = {"prompt_tokens": ids.shape[1], "completion_tokens": n}
        yield f"data: {json.dumps({'choices': [], 'usage': usage, 'created': int(time.time())})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    state.update(name=args.model, tok=AutoTokenizer.from_pretrained(args.model),
                 model=AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).eval())
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
