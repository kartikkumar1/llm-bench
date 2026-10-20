"""Start llama-server with one quant and the gateway in front of it.

    python serve.py                 # q4_k_m of the configured model
    python serve.py --quant q8_0
Then open http://localhost:8000 (dashboard) or POST to http://localhost:8000/v1/chat/completions.
"""
import argparse
import os
import subprocess
import sys

from bench.hwdetect import load_resolved
from bench.servers import Server, build_command
from bench.setup import model_path

UPSTREAM_PORT = 8081


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", default="q4_k_m")
    args = ap.parse_args()

    _, cfg = load_resolved()
    gguf = model_path(cfg["model"]["name"], args.quant)
    if not gguf.exists():
        sys.exit(f"{gguf.name} missing; run `python -m bench.setup models`")
    cmd = build_command("llamacpp_server", gguf, UPSTREAM_PORT, cfg["hardware"], cfg["model"]["ctx_size"], 4)
    with Server("llamacpp_server", cmd, UPSTREAM_PORT) as srv:
        print(f"llama-server ready on {srv.url} ({gguf.name}, device={cfg['hardware']['device']})")
        env = {**os.environ, "UPSTREAM_URL": srv.url}
        g = cfg["gateway"]
        try:
            subprocess.run([sys.executable, "-m", "uvicorn", "gateway.app:app",
                            "--host", g["host"], "--port", str(g["port"])], env=env)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
