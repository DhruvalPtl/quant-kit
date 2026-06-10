"""
setup_sd_cpp.py — Download stable-diffusion.cpp for your OS
=============================================================
One-time setup: downloads the stable-diffusion.cpp binary (sd / sd.exe)
into the stable-diffusion.cpp/ folder inside quant-kit.

Windows: Downloads Vulkan pre-built binary (works on NVIDIA, AMD, Intel Arc)
Linux:   Downloads CUDA pre-built binary (or falls back to Vulkan)

Usage:
    python setup_sd_cpp.py
    python setup_sd_cpp.py --backend vulkan   (force Vulkan — good for Intel Arc)
    python setup_sd_cpp.py --backend cuda     (force CUDA)
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import os
import platform
import zipfile
import tarfile
import argparse
import subprocess
import urllib.request
from pathlib import Path

from config import SD_CPP_DIR, SD_BINARY, IS_WINDOWS
from utils import print_step

# stable-diffusion.cpp GitHub release URL pattern
SD_CPP_RELEASES = "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/latest"


def get_latest_release_assets() -> list[dict]:
    """Fetch latest release asset list from GitHub API."""
    import urllib.request, json
    req = urllib.request.Request(
        SD_CPP_RELEASES,
        headers={"User-Agent": "quant-kit/1.0", "Accept": "application/vnd.github+json"}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    return data.get("assets", []), data.get("tag_name", "unknown")


def pick_asset(assets: list[dict], backend: str) -> dict | None:
    """Pick the right binary for this OS + backend."""
    os_name = "win" if IS_WINDOWS else "linux"
    arch    = "x64"  # 64-bit assumed

    # Priority order for backend keywords in filename
    if backend == "cuda":
        keywords = ["cuda12", "cuda11", "cuda"]
    elif backend == "vulkan":
        keywords = ["vulkan"]
    elif backend == "metal":
        keywords = ["metal"]  # macOS
    else:
        # Auto: prefer CUDA on Linux, Vulkan on Windows (covers all GPU vendors incl. Arc)
        if IS_WINDOWS:
            keywords = ["vulkan", "cuda12", "cuda"]
        else:
            keywords = ["cuda12", "cuda11", "cuda", "vulkan"]

    ext = ".zip" if IS_WINDOWS else ".tar.gz"

    for kw in keywords:
        for asset in assets:
            name = asset["name"].lower()
            if os_name in name and kw in name and arch in name and name.endswith(ext):
                return asset

    # Final fallback: any matching OS + ext
    for asset in assets:
        name = asset["name"].lower()
        if os_name in name and arch in name and name.endswith(ext):
            return asset

    return None


def download_with_progress(url: str, dest: Path):
    """Download a file with a simple progress display."""
    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb  = downloaded / 1e6
            print(f"\r  Downloading... {mb:.1f} MB ({pct}%)", end="", flush=True)
    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print()  # newline after progress


def extract_archive(archive_path: Path, dest_dir: Path):
    """Extract zip or tar.gz into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest_dir)
    elif archive_path.name.endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(dest_dir)
    else:
        print_step("err", f"Unknown archive format: {archive_path.name}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Setup stable-diffusion.cpp for diffusion model quantization")
    parser.add_argument("--backend", choices=["cuda", "vulkan", "metal", "auto"], default="auto",
                        help="GPU backend. 'auto' = Vulkan on Windows (works on Intel Arc), CUDA on Linux")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  🎨 stable-diffusion.cpp Setup")
    print("=" * 60)
    print(f"  Install dir : {SD_CPP_DIR}")
    print(f"  Platform    : {'Windows' if IS_WINDOWS else 'Linux'}")
    print(f"  Backend     : {args.backend}")
    print("=" * 60 + "\n")

    # Already installed?
    if SD_BINARY.exists():
        print_step("ok", f"stable-diffusion.cpp already installed: {SD_BINARY}")
        r = subprocess.run([str(SD_BINARY), "--help"], capture_output=True, text=True)
        if r.returncode in (0, 1):  # sd.exe --help returns 1 on some builds
            print_step("ok", "Binary is functional")
            print_step("info", "To reinstall, delete stable-diffusion.cpp/ and re-run")
            return
        else:
            print_step("warn", "Binary exists but may be broken — re-downloading")

    print_step("info", "Fetching latest release from GitHub...")
    try:
        assets, version = get_latest_release_assets()
        print_step("ok", f"Latest version: {version}")
    except Exception as e:
        print_step("err", f"GitHub API failed: {e}")
        print_step("info", "Download manually from: https://github.com/leejet/stable-diffusion.cpp/releases")
        sys.exit(1)

    asset = pick_asset(assets, args.backend)
    if not asset:
        print_step("err", f"No matching binary found for OS={platform.system()} backend={args.backend}")
        print_step("info", "Available assets:")
        for a in assets:
            print(f"    {a['name']}")
        print_step("info", "Download the right one manually and place contents in stable-diffusion.cpp/")
        sys.exit(1)

    print_step("info", f"Selected: {asset['name']} ({asset['size']/1e6:.1f} MB)")

    # Download
    tmp_path = SD_CPP_DIR.parent / asset["name"]
    SD_CPP_DIR.mkdir(parents=True, exist_ok=True)

    print_step("info", "Downloading...")
    download_with_progress(asset["browser_download_url"], tmp_path)

    # Extract
    print_step("info", "Extracting...")
    extract_archive(tmp_path, SD_CPP_DIR)
    tmp_path.unlink()

    # Flatten if extracted into a subdirectory
    # Many releases extract as sd.cpp-v1.x.x-xxx/sd.exe
    for sub in SD_CPP_DIR.iterdir():
        if sub.is_dir():
            for f in sub.iterdir():
                target = SD_CPP_DIR / f.name
                if not target.exists():
                    f.rename(target)
            try:
                sub.rmdir()
            except Exception:
                pass

    # Check binary
    if SD_BINARY.exists():
        if not IS_WINDOWS:
            os.chmod(SD_BINARY, 0o755)
        print_step("ok", f"Binary installed: {SD_BINARY}")
    else:
        # Try to find sd binary with different name
        found = list(SD_CPP_DIR.glob("sd*"))
        if found:
            found[0].rename(SD_BINARY)
            print_step("ok", f"Binary installed: {SD_BINARY}")
        else:
            print_step("err", "Could not find sd binary after extraction")
            print_step("info", f"Contents of {SD_CPP_DIR}:")
            for f in SD_CPP_DIR.iterdir():
                print(f"    {f.name}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("  ✅ stable-diffusion.cpp is ready!")
    print("=" * 60)
    print(f"  Backend : {args.backend}")
    print(f"  Binary  : {SD_BINARY}")
    print()
    print("  Now quantize diffusion models:")
    print("  python quantize_diffusion.py --model black-forest-labs/FLUX.1-schnell --quant q4_k")
    print("  python quant.py --model stabilityai/stable-diffusion-xl-base-1.0 --quant q8_0")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
