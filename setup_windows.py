"""
setup_windows.py — Download llama.cpp binaries for Windows
============================================================
Downloads the Vulkan build (for Intel Arc / AMD GPUs) or CUDA build
(for NVIDIA GPUs) from the latest llama.cpp GitHub release.

Run once:
    python setup_windows.py               # auto-detects GPU
    python setup_windows.py --vulkan      # force Vulkan (Intel Arc / AMD)
    python setup_windows.py --cuda        # force CUDA (NVIDIA)
    python setup_windows.py --cpu         # CPU only
"""

import os
import sys
import json
import zipfile
import argparse
import platform
import subprocess
import urllib.request
from pathlib import Path

ROOT_DIR     = Path(__file__).parent
LLAMA_CPP    = ROOT_DIR / "llama.cpp"
LLAMA_SRC    = ROOT_DIR / "llama-src"

# ─── Helpers ──────────────────────────────────────────────────────────────────

def header(msg):  print(f"\n[->] {msg}")
def ok(msg):      print(f"[OK] {msg}")
def warn(msg):    print(f"[!]  {msg}")
def err(msg):     print(f"[ERR] {msg}")


def download(url: str, dest: Path, label: str = ""):
    label = label or dest.name
    def _progress(count, block, total):
        pct = min(count * block / total * 100, 100) if total > 0 else 0
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        mb_done = count * block / 1e6
        mb_total = total / 1e6
        print(f"\r  [{bar}] {pct:.0f}%  {mb_done:.0f}/{mb_total:.0f} MB", end="", flush=True)
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


# ─── GPU Detection ────────────────────────────────────────────────────────────

def detect_gpu_vendor() -> str:
    """Returns 'nvidia', 'intel', 'amd', or 'cpu'"""
    try:
        # NVIDIA
        r = subprocess.run("nvidia-smi", capture_output=True, text=True)
        if r.returncode == 0:
            return "nvidia"
    except FileNotFoundError:
        pass

    try:
        # Check DXDiag / WMIC for GPU vendor
        r = subprocess.run(
            ["wmic", "path", "win32_videocontroller", "get", "Name"],
            capture_output=True, text=True
        )
        output = r.stdout.lower()
        if "nvidia" in output: return "nvidia"
        if "intel arc" in output or "intel(r) arc" in output: return "intel"
        if "radeon" in output or "amd" in output: return "amd"
    except Exception:
        pass

    return "cpu"


# ─── GitHub Release ───────────────────────────────────────────────────────────

def get_latest_release() -> tuple[str, dict]:
    """Fetch latest llama.cpp release and return (tag, assets_dict)"""
    header("Fetching latest llama.cpp release...")
    url = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"
    req = urllib.request.Request(url, headers={"User-Agent": "quant-kit/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.load(resp)

    tag    = data["tag_name"]
    assets = {a["name"]: a["browser_download_url"] for a in data["assets"]}
    ok(f"Latest release: {tag}")
    return tag, assets


# ─── Download & Extract ───────────────────────────────────────────────────────

def download_binaries(build_type: str):
    """Download and extract the right Windows llama.cpp zip."""

    LLAMA_CPP.mkdir(exist_ok=True)

    # Check if already installed
    existing = LLAMA_CPP / "llama-quantize.exe"
    if existing.exists():
        ok(f"llama.cpp binaries already installed — skipping download")
        ok(f"  Delete {LLAMA_CPP} and re-run to force update")
        return

    tag, assets = get_latest_release()

    # Pick the right zip based on build type
    build_map = {
        "vulkan": f"llama-{tag}-bin-win-vulkan-x64.zip",
        "cuda":   f"llama-{tag}-bin-win-cuda-cu12.4.0-x64.zip",
        "cpu":    f"llama-{tag}-bin-win-x64.zip",
    }
    zip_name = build_map.get(build_type, build_map["vulkan"])

    if zip_name not in assets:
        # Try alternate CUDA version names
        for name in assets:
            if "win" in name and "cuda" in name and "x64" in name and build_type == "cuda":
                zip_name = name
                break
        else:
            err(f"Could not find {zip_name} in release assets")
            err("Available Windows zips:")
            for name in sorted(assets):
                if "win" in name: err(f"  {name}")
            sys.exit(1)

    url = assets[zip_name]
    zip_path = ROOT_DIR / zip_name

    header(f"Downloading: {zip_name}")
    download(url, zip_path)
    ok(f"Downloaded: {zip_path.stat().st_size / 1e6:.1f} MB")

    header("Extracting binaries...")
    with zipfile.ZipFile(zip_path, "r") as z:
        members = z.namelist()
        # Flatten — all files go to LLAMA_CPP/ regardless of zip subfolder
        for member in members:
            filename = Path(member).name
            if not filename:
                continue
            dest = LLAMA_CPP / filename
            with z.open(member) as src, open(dest, "wb") as dst:
                dst.write(src.read())

    zip_path.unlink()  # delete zip

    # List what we got
    exes = sorted(LLAMA_CPP.glob("*.exe"))
    ok(f"Extracted {len(exes)} executables to {LLAMA_CPP}")
    for exe in exes:
        print(f"  {exe.name}")


# ─── Python Packages ──────────────────────────────────────────────────────────

def install_deps():
    header("Installing Python packages...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r",
                    str(ROOT_DIR / "requirements.txt"), "-q"], check=True)
    ok("Python packages installed")


# ─── Conversion Scripts ───────────────────────────────────────────────────────

def setup_conversion_scripts():
    header("Setting up conversion scripts (llama-src)...")

    if (LLAMA_SRC / "convert_hf_to_gguf.py").exists():
        ok("Conversion scripts already set up — skipping")
        return

    LLAMA_SRC.mkdir(exist_ok=True)
    llama_cpp_py = "https://github.com/ggerganov/llama.cpp"

    # Sparse clone — only conversion scripts
    subprocess.run(
        f'git clone --depth 1 --filter=blob:none --no-checkout {llama_cpp_py} "{LLAMA_SRC}"',
        shell=True, check=True
    )
    subprocess.run(
        f'git -C "{LLAMA_SRC}" sparse-checkout set convert_hf_to_gguf.py gguf-py',
        shell=True, check=True
    )
    subprocess.run(f'git -C "{LLAMA_SRC}" checkout', shell=True, check=True)
    ok("Conversion scripts ready")


# ─── Verify ───────────────────────────────────────────────────────────────────

def verify():
    from config import LLAMA_QUANTIZE, LLAMA_BENCH, CONVERT_SCRIPT
    checks = [
        ("llama-quantize.exe", LLAMA_QUANTIZE),
        ("llama-bench.exe",    LLAMA_BENCH),
        ("convert script",     CONVERT_SCRIPT),
    ]
    all_ok = True
    print()
    for name, path in checks:
        if path.exists():
            ok(f"{name:22s} {path}")
        else:
            err(f"{name:22s} NOT FOUND at {path}")
            all_ok = False
    return all_ok


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vulkan", action="store_true", help="Force Vulkan build (Intel Arc / AMD)")
    parser.add_argument("--cuda",   action="store_true", help="Force CUDA build (NVIDIA)")
    parser.add_argument("--cpu",    action="store_true", help="CPU only build")
    args = parser.parse_args()

    print()
    print("=" * 58)
    print("  quant-kit — Windows Setup")
    print("=" * 58)
    print(f"[OK] OS: {platform.system()} {platform.machine()}")

    # Determine build type
    if args.cuda:
        build_type = "cuda"
    elif args.vulkan:
        build_type = "vulkan"
    elif args.cpu:
        build_type = "cpu"
    else:
        vendor = detect_gpu_vendor()
        build_map = {"nvidia": "cuda", "intel": "vulkan", "amd": "vulkan", "cpu": "cpu"}
        build_type = build_map[vendor]
        ok(f"Detected GPU vendor: {vendor} → using {build_type.upper()} build")

    install_deps()
    download_binaries(build_type)
    setup_conversion_scripts()

    all_ok = verify()
    print()
    if all_ok:
        print("=" * 58)
        print("  [OK] Setup complete!")
        print()
        print("  Next steps:")
        print("    1. Put your GGUFs in: output\\<model-name>\\")
        print("    2. python benchmark.py --model <model-name> --ngl 999")
        print("    3. python model_card.py --model <model-name> ...")
        print("    4. python upload.py --model <model-name> ...")
        print("=" * 58)
    else:
        print("=" * 58)
        err("Setup incomplete — fix errors above")
        print("=" * 58)
        sys.exit(1)


if __name__ == "__main__":
    main()
