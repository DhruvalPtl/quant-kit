"""
setup_whisper_cpp.py — Download whisper.cpp for your OS
========================================================
One-time setup: downloads whisper.cpp binaries (whisper / quantize)
and the convert-h5-to-ggml.py script into the whisper.cpp/ folder.

Usage:
    python setup_whisper_cpp.py
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import os
import zipfile
import tarfile
import json
import subprocess
import urllib.request
import argparse
from pathlib import Path

from config import (
    WHISPER_CPP_DIR, WHISPER_BINARY, WHISPER_QUANTIZE,
    WHISPER_CONVERT_SCRIPT, IS_WINDOWS
)
from utils import print_step

WHISPER_RELEASES = "https://api.github.com/repos/ggerganov/whisper.cpp/releases/latest"


def get_latest_release(url: str) -> tuple[list[dict], str]:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "quant-kit/1.0", "Accept": "application/vnd.github+json"}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    return data.get("assets", []), data.get("tag_name", "unknown"), data.get("tarball_url", "")


def pick_whisper_asset(assets: list[dict]) -> dict | None:
    """Pick the whisper.cpp binary for this OS."""
    os_kw  = "win" if IS_WINDOWS else "ubuntu"
    ext    = ".zip" if IS_WINDOWS else ".tar.gz"

    for asset in assets:
        name = asset["name"].lower()
        if os_kw in name and name.endswith(ext):
            return asset

    # Fallback: any zip/tar
    for asset in assets:
        name = asset["name"].lower()
        if name.endswith(ext) and "whisper" in name:
            return asset

    return None


def download_with_progress(url: str, dest: Path):
    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb  = downloaded / 1e6
            print(f"\r  Downloading... {mb:.1f} MB ({pct}%)", end="", flush=True)
    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print()


def extract_archive(archive_path: Path, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest_dir)
    elif archive_path.name.endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(dest_dir)


def clone_for_convert_script(version: str):
    """
    Clone whisper.cpp (sparse) to get the convert-h5-to-ggml.py script.
    This is only needed for the Python conversion script.
    """
    models_dir = WHISPER_CPP_DIR / "models"
    convert_script = models_dir / "convert-h5-to-ggml.py"

    if convert_script.exists():
        print_step("ok", f"Convert script already exists: {convert_script}")
        return

    models_dir.mkdir(parents=True, exist_ok=True)

    # Download just the convert script from GitHub raw
    script_url = (
        "https://raw.githubusercontent.com/ggerganov/whisper.cpp/"
        f"{version}/models/convert-h5-to-ggml.py"
    )
    print_step("info", f"Downloading convert-h5-to-ggml.py from GitHub ({version})...")
    try:
        urllib.request.urlretrieve(script_url, str(convert_script))
        print_step("ok", f"Convert script downloaded: {convert_script}")
    except Exception as e:
        print_step("warn", f"Failed to download convert script: {e}")
        print_step("info", "You can download it manually from:")
        print_step("info", f"  https://github.com/ggerganov/whisper.cpp/blob/{version}/models/convert-h5-to-ggml.py")


def install_convert_requirements():
    """Install Python packages required by convert-h5-to-ggml.py."""
    print_step("info", "Installing Python requirements for converter...")
    pkgs = ["torch", "transformers", "numpy"]
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q"] + pkgs,
        check=False
    )
    print_step("ok", "Python requirements installed")


def main():
    parser = argparse.ArgumentParser(description="Setup whisper.cpp for Whisper ASR quantization")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  🎙️  whisper.cpp Setup")
    print("=" * 60)
    print(f"  Install dir : {WHISPER_CPP_DIR}")
    print(f"  Platform    : {'Windows' if IS_WINDOWS else 'Linux'}")
    print("=" * 60 + "\n")

    # Already installed?
    if WHISPER_BINARY.exists() and WHISPER_QUANTIZE.exists():
        print_step("ok", "whisper.cpp binaries already installed")
        if WHISPER_CONVERT_SCRIPT.exists():
            print_step("ok", "Convert script already present")
            print_step("info", "To reinstall, delete whisper.cpp/ and re-run")
            return
        else:
            print_step("info", "Binaries found but convert script missing — fetching script only")

    print_step("info", "Fetching latest whisper.cpp release from GitHub...")
    try:
        assets, version, tarball_url = get_latest_release(WHISPER_RELEASES)
        print_step("ok", f"Latest version: {version}")
    except Exception as e:
        print_step("err", f"GitHub API failed: {e}")
        print_step("info", "Download manually from: https://github.com/ggerganov/whisper.cpp/releases")
        sys.exit(1)

    if not (WHISPER_BINARY.exists() and WHISPER_QUANTIZE.exists()):
        asset = pick_whisper_asset(assets)
        if not asset:
            print_step("warn", "No pre-built binary found for your OS")
            print_step("info", "Available assets:")
            for a in assets:
                print(f"    {a['name']}")
            print_step("info", "Download manually and extract to whisper.cpp/")
            print_step("info", "Then re-run this script to get the convert script")
        else:
            print_step("info", f"Selected: {asset['name']} ({asset['size']/1e6:.1f} MB)")
            tmp_path = WHISPER_CPP_DIR.parent / asset["name"]
            WHISPER_CPP_DIR.mkdir(parents=True, exist_ok=True)

            print_step("info", "Downloading...")
            download_with_progress(asset["browser_download_url"], tmp_path)

            print_step("info", "Extracting...")
            extract_archive(tmp_path, WHISPER_CPP_DIR)
            tmp_path.unlink()

            # Flatten nested directory if needed
            for sub in WHISPER_CPP_DIR.iterdir():
                if sub.is_dir():
                    for f in sub.iterdir():
                        target = WHISPER_CPP_DIR / f.name
                        if not target.exists():
                            f.rename(target)
                    try:
                        sub.rmdir()
                    except Exception:
                        pass

            if not IS_WINDOWS:
                for binary in [WHISPER_BINARY, WHISPER_QUANTIZE]:
                    if binary.exists():
                        os.chmod(binary, 0o755)

            if WHISPER_BINARY.exists():
                print_step("ok", f"whisper binary: {WHISPER_BINARY}")
            else:
                print_step("warn", f"whisper binary not found at expected path: {WHISPER_BINARY}")
                print_step("info", f"Contents of {WHISPER_CPP_DIR}:")
                for f in WHISPER_CPP_DIR.iterdir():
                    print(f"    {f.name}")

            if WHISPER_QUANTIZE.exists():
                print_step("ok", f"quantize binary: {WHISPER_QUANTIZE}")

    # Always get the convert script (tiny Python file)
    clone_for_convert_script(version)
    install_convert_requirements()

    print("\n" + "=" * 60)
    print("  ✅ whisper.cpp is ready!")
    print("=" * 60)
    print(f"  Binary      : {WHISPER_BINARY}")
    print(f"  Quantize    : {WHISPER_QUANTIZE}")
    print(f"  Convert     : {WHISPER_CONVERT_SCRIPT}")
    print()
    print("  Now quantize Whisper models:")
    print("  python quantize_whisper.py --model openai/whisper-large-v3-turbo")
    print("  python quant.py --model openai/whisper-medium --quant q8_0")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
