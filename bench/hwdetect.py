"""Detect CPU / RAM / GPU and resolve every "auto" value in config.yaml.

    python -m bench.hwdetect            # print detected hardware + resolved config
    python -m bench.hwdetect --json     # machine-readable

Device priority: cuda (NVIDIA) > metal (Apple Silicon) > vulkan (AMD/Intel GPU) > cpu.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

import psutil
import yaml

from bench.catalog import DEFAULT_QUANTS, MODELS, OVERHEAD_GB, PROFILE_MODEL, QUANT_ORDER

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
HARDWARE_JSON = ROOT / "results" / "hardware.json"

RAM_BUDGET_FRACTION = 0.4
VRAM_BUDGET_FRACTION = 0.8


def _run(cmd: list[str], timeout: float = 10) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def detect_cpu() -> dict:
    name = platform.processor()
    if platform.system() == "Windows":
        out = _run(["powershell", "-NoProfile", "-Command",
                    "(Get-CimInstance Win32_Processor).Name"])
        name = out.strip() or name
    elif platform.system() == "Linux":
        m = re.search(r"model name\s*:\s*(.+)", Path("/proc/cpuinfo").read_text(errors="ignore"))
        name = m.group(1).strip() if m else name
    elif platform.system() == "Darwin":
        name = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip() or name
    return {
        "name": name,
        "arch": platform.machine(),
        "physical_cores": psutil.cpu_count(logical=False) or 1,
        "logical_cores": psutil.cpu_count(logical=True) or 1,
    }


def detect_ram() -> dict:
    vm = psutil.virtual_memory()
    return {"total_gb": round(vm.total / 1e9, 2), "available_gb": round(vm.available / 1e9, 2)}


def _nvidia_gpus() -> list[dict]:
    if not shutil.which("nvidia-smi"):
        return []
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    gpus = []
    for line in out.strip().splitlines():
        name, mem = [p.strip() for p in line.split(",")]
        gpus.append({"vendor": "nvidia", "name": name, "vram_gb": round(int(mem) / 1024, 2),
                     "integrated": False})
    return gpus


def _windows_gpus() -> list[dict]:
    # AdapterRAM is a uint32 so it caps at 4 GB; the registry value is exact when present.
    script = (
        "Get-CimInstance Win32_VideoController | ForEach-Object { "
        "$q=(Get-ItemProperty -ErrorAction SilentlyContinue "
        "\"HKLM:\\SYSTEM\\ControlSet001\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}\\0*\" "
        "| Where-Object { $_.DriverDesc -eq $_.Name } | Select-Object -First 1).'HardwareInformation.qwMemorySize'; "
        "[pscustomobject]@{name=$_.Name; ram=$_.AdapterRAM; qw=$q} } | ConvertTo-Json -Compress"
    )
    out = _run(["powershell", "-NoProfile", "-Command", script], timeout=20)
    try:
        rows = json.loads(out) if out.strip() else []
    except json.JSONDecodeError:
        return []
    rows = rows if isinstance(rows, list) else [rows]
    gpus = []
    for r in rows:
        name = r.get("name") or ""
        if "Basic" in name or "Virtual" in name:
            continue
        mem = r.get("qw") or r.get("ram") or 0
        gpus.append({"vendor": _vendor(name), "name": name, "vram_gb": round(int(mem) / 2**30, 2)})
    return gpus


def _linux_gpus() -> list[dict]:
    gpus = []
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]")):
        dev = card / "device"
        try:
            vendor_id = (dev / "vendor").read_text().strip()
        except OSError:
            continue
        vendor = {"0x1002": "amd", "0x8086": "intel", "0x10de": "nvidia"}.get(vendor_id)
        if not vendor or vendor == "nvidia":   # NVIDIA handled via nvidia-smi
            continue
        vram = dev / "mem_info_vram_total"
        mem = int(vram.read_text()) if vram.exists() else 0
        gpus.append({"vendor": vendor, "name": f"{vendor} GPU ({card.name})",
                     "vram_gb": round(mem / 2**30, 2)})
    return gpus


def _vendor(name: str) -> str:
    n = name.lower()
    if "nvidia" in n or "geforce" in n or "quadro" in n:
        return "nvidia"
    if "amd" in n or "radeon" in n:
        return "amd"
    if "intel" in n:
        return "intel"
    if "apple" in n:
        return "apple"
    return "unknown"


def _is_integrated(gpu: dict) -> bool:
    n = gpu["name"].lower()
    if gpu["vendor"] == "intel" and "arc" not in n:
        return True
    if gpu["vendor"] == "amd" and re.search(r"radeon\(tm\) graphics|radeon graphics|vega \d+ graphics", n):
        return True
    # Integrated GPUs report a small carve-out and share system RAM.
    return gpu["vram_gb"] < 2


def _has_vulkan() -> bool:
    system = platform.system()
    if system == "Windows":
        return Path(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "vulkan-1.dll").exists()
    if system == "Linux":
        return bool(shutil.which("vulkaninfo")) or any(
            Path(p).exists() for p in ("/usr/lib/x86_64-linux-gnu/libvulkan.so.1", "/usr/lib/libvulkan.so.1"))
    return False


def detect_gpus() -> list[dict]:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return [{"vendor": "apple", "name": "Apple Silicon GPU", "vram_gb": 0.0, "integrated": True}]
    gpus = _nvidia_gpus()
    seen_nvidia = bool(gpus)
    others = _windows_gpus() if platform.system() == "Windows" else _linux_gpus()
    for g in others:
        if seen_nvidia and g["vendor"] == "nvidia":
            continue
        g.setdefault("integrated", _is_integrated(g))
        gpus.append(g)
    return gpus


def pick_device(gpus: list[dict]) -> tuple[str, dict | None]:
    for g in gpus:
        if g["vendor"] == "nvidia":
            return "cuda", g
    for g in gpus:
        if g["vendor"] == "apple":
            return "metal", g
    if _has_vulkan():
        # Prefer a discrete card over an integrated one.
        for g in sorted(gpus, key=lambda g: g["integrated"]):
            if g["vendor"] in ("amd", "intel"):
                return "vulkan", g
    return "cpu", None


def detect() -> dict:
    gpus = detect_gpus()
    device, gpu = pick_device(gpus)
    return {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": detect_cpu(),
        "ram": detect_ram(),
        "gpus": gpus,
        "recommended_device": device,
        "selected_gpu": gpu,
        "vulkan_available": _has_vulkan(),
    }


def _auto(v) -> bool:
    return v is None or v == "auto"


def resolve_config(hw: dict, cfg: dict) -> dict:
    """Return a copy of cfg with every "auto" replaced by a hardware-derived value."""
    cfg = json.loads(json.dumps(cfg))   # deep copy
    hwc, mem, model = cfg["hardware"], cfg["memory"], cfg["model"]

    if _auto(hwc.get("device")):
        hwc["device"] = hw["recommended_device"]
    if _auto(hwc.get("threads")):
        hwc["threads"] = hw["cpu"]["physical_cores"]
    if _auto(hwc.get("gpu_layers")):
        hwc["gpu_layers"] = 0 if hwc["device"] == "cpu" else 99

    gpu = hw["selected_gpu"]
    discrete = gpu is not None and not gpu.get("integrated", True)
    if _auto(mem.get("max_ram_gb")):
        # Stay within what is actually free right now, not just what is installed.
        mem["max_ram_gb"] = round(min(hw["ram"]["total_gb"] * RAM_BUDGET_FRACTION,
                                      hw["ram"]["available_gb"] * 0.8), 1)
    if _auto(mem.get("max_vram_gb")):
        # Integrated GPUs share system RAM, so their budget is the RAM budget.
        mem["max_vram_gb"] = round(gpu["vram_gb"] * VRAM_BUDGET_FRACTION, 1) if discrete else None

    # On a discrete GPU with full offload the weights live in VRAM; otherwise in RAM.
    budget = mem["max_vram_gb"] if discrete and hwc["gpu_layers"] else mem["max_ram_gb"]

    if _auto(model.get("name")):
        profile = model.get("profile", "minimal")
        if profile in PROFILE_MODEL:
            model["name"] = PROFILE_MODEL[profile]
        else:   # "max": biggest model whose q4_k_m fits
            fitting = [k for k, m in MODELS.items() if m["sizes_gb"]["q4_k_m"] + OVERHEAD_GB <= budget]
            model["name"] = max(fitting, key=lambda k: MODELS[k]["params_b"]) if fitting else "qwen2.5-0.5b"

    sizes = MODELS[model["name"]]["sizes_gb"]
    if _auto(model.get("quants")):
        model["quants"] = [q for q in DEFAULT_QUANTS if sizes[q] + OVERHEAD_GB <= budget]
    skipped = [q for q in model["quants"] if sizes.get(q, 0) + OVERHEAD_GB > budget]
    model["quants"] = sorted((q for q in model["quants"] if q not in skipped), key=QUANT_ORDER.index)

    cfg["_resolved"] = {"memory_budget_gb": budget, "skipped_quants_over_budget": skipped}
    return cfg


def load_config(path: Path = CONFIG_PATH) -> dict:
    return yaml.safe_load(path.read_text())


def load_resolved() -> tuple[dict, dict]:
    hw = detect()
    return hw, resolve_config(hw, load_config())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    hw, cfg = load_resolved()
    HARDWARE_JSON.parent.mkdir(exist_ok=True)
    HARDWARE_JSON.write_text(json.dumps({"hardware": hw, "config": cfg}, indent=2))

    if args.json:
        print(json.dumps({"hardware": hw, "config": cfg}, indent=2))
        return
    c = hw["cpu"]
    print(f"OS      : {hw['os']}")
    print(f"CPU     : {c['name']}  ({c['physical_cores']}C/{c['logical_cores']}T, {c['arch']})")
    print(f"RAM     : {hw['ram']['total_gb']} GB total, {hw['ram']['available_gb']} GB free")
    for g in hw["gpus"] or [{"name": "none", "vram_gb": 0, "integrated": False}]:
        kind = "integrated, shares RAM" if g.get("integrated") else f"{g['vram_gb']} GB VRAM"
        print(f"GPU     : {g['name']}  ({kind})")
    print(f"Vulkan  : {'yes' if hw['vulkan_available'] else 'no'}")
    print()
    h, m, mo = cfg["hardware"], cfg["memory"], cfg["model"]
    print(f"device={h['device']}  threads={h['threads']}  gpu_layers={h['gpu_layers']}")
    print(f"memory budget: {cfg['_resolved']['memory_budget_gb']} GB "
          f"(max_ram_gb={m['max_ram_gb']}, max_vram_gb={m['max_vram_gb']})")
    print(f"model={mo['name']}  quants={mo['quants']}")
    if cfg["_resolved"]["skipped_quants_over_budget"]:
        print(f"skipped (over budget): {cfg['_resolved']['skipped_quants_over_budget']}")
    print(f"\nwrote {HARDWARE_JSON.relative_to(ROOT)}  -- edit config.yaml to override any value")


if __name__ == "__main__":
    main()
