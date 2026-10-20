"""Fetch everything the benchmark needs, matched to the detected hardware.

    python -m bench.setup llamacpp      # prebuilt llama.cpp binaries for the detected device
    python -m bench.setup models        # GGUF files for the configured model + quants
    python -m bench.setup wikitext      # wikitext-2 test set for perplexity
    python -m bench.setup all
"""
from __future__ import annotations

import argparse
import io
import platform
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

import httpx

from bench.catalog import MODELS
from bench.hwdetect import ROOT, load_resolved

BIN_DIR = ROOT / "llama.cpp"
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"
WIKITEXT_URL = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"


def _asset_patterns(device: str) -> list[str]:
    """Substrings identifying the right release asset, best match first."""
    system, arch = platform.system(), platform.machine().lower()
    if system == "Windows":
        arch = "arm64" if "arm" in arch else "x64"
        return {"cuda": [f"bin-win-cuda-12", f"bin-win-cuda"],
                "vulkan": [f"bin-win-vulkan-{arch}"]}.get(device, []) + [f"bin-win-cpu-{arch}"]
    if system == "Darwin":
        return [f"bin-macos-{'arm64' if 'arm' in arch else 'x64'}"]
    return {"vulkan": ["bin-ubuntu-vulkan-x64"]}.get(device, []) + ["bin-ubuntu-x64"]


def llama_server_path() -> Path:
    exe = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
    hits = sorted(BIN_DIR.rglob(exe))
    if not hits:
        sys.exit(f"{exe} not found under {BIN_DIR}. Run: python -m bench.setup llamacpp")
    return hits[0]


def setup_llamacpp(device: str) -> None:
    # The b<NNNN> binary builds are tagged as pre-releases, so "latest" can point at a release
    # without binaries. Take the newest release that has a matching asset.
    releases = httpx.get(RELEASES_API, timeout=30, follow_redirects=True).json()
    release = asset = None
    for rel in releases:
        for pattern in _asset_patterns(device):
            match = [a for a in rel["assets"] if pattern in a["name"]
                     and a["name"].endswith((".zip", ".tar.gz")) and not a["name"].startswith("cudart")]
            if match:
                release, asset = rel, match[0]
                break
        if asset:
            break
    if not asset:
        sys.exit(f"No llama.cpp release asset for device={device}; build from source instead (see README).")
    assets = release["assets"]

    print(f"llama.cpp {release['tag_name']}: downloading {asset['name']} "
          f"({asset['size'] / 1e6:.0f} MB)")
    data = httpx.get(asset["browser_download_url"], timeout=600, follow_redirects=True).content
    if BIN_DIR.exists():
        shutil.rmtree(BIN_DIR)
    BIN_DIR.mkdir()
    if asset["name"].endswith(".zip"):
        zipfile.ZipFile(io.BytesIO(data)).extractall(BIN_DIR)
    else:
        tarfile.open(fileobj=io.BytesIO(data)).extractall(BIN_DIR, filter="data")
    (BIN_DIR / "VERSION").write_text(f"{release['tag_name']} {asset['name']}\n")

    # CUDA builds on Windows ship the runtime DLLs as a separate asset.
    if device == "cuda" and platform.system() == "Windows":
        cudart = [a for a in assets if a["name"].startswith("cudart") and "win" in a["name"]]
        if cudart:
            print(f"downloading {cudart[0]['name']}")
            blob = httpx.get(cudart[0]["browser_download_url"], timeout=600, follow_redirects=True).content
            zipfile.ZipFile(io.BytesIO(blob)).extractall(llama_server_path().parent)
    print(f"installed to {BIN_DIR.relative_to(ROOT)}")


def model_path(name: str, quant: str) -> Path:
    return MODELS_DIR / MODELS[name]["file"].format(quant=quant)


def setup_models(name: str, quants: list[str]) -> None:
    from huggingface_hub import hf_hub_download

    m = MODELS[name]
    MODELS_DIR.mkdir(exist_ok=True)
    total = sum(m["sizes_gb"][q] for q in quants if not model_path(name, q).exists())
    print(f"{name}: {len(quants)} quants, ~{total:.1f} GB to download")
    for q in quants:
        if model_path(name, q).exists():
            print(f"  {q}: present")
            continue
        print(f"  {q}: downloading")
        hf_hub_download(m["repo"], m["file"].format(quant=q), local_dir=MODELS_DIR)


def setup_wikitext() -> None:
    target = DATA_DIR / "wiki.test.raw"
    if target.exists():
        print("wikitext: present")
        return
    print("wikitext: downloading")
    data = httpx.get(WIKITEXT_URL, timeout=120, follow_redirects=True).content
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        member = next(n for n in z.namelist() if n.endswith("wiki.test.raw"))
        DATA_DIR.mkdir(exist_ok=True)
        target.write_bytes(z.read(member))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["llamacpp", "models", "wikitext", "all"])
    args = ap.parse_args()
    _, cfg = load_resolved()
    if args.what in ("llamacpp", "all"):
        setup_llamacpp(cfg["hardware"]["device"])
    if args.what in ("models", "all"):
        setup_models(cfg["model"]["name"], cfg["model"]["quants"])
    if args.what in ("wikitext", "all"):
        setup_wikitext()


if __name__ == "__main__":
    main()
