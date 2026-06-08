import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
check_setup.py — Verify your environment is ready
====================================================
Run this before anything else to check all tools are installed correctly.

Usage:
    python check_setup.py
"""

import subprocess
from pathlib import Path
from config import LLAMA_QUANTIZE, LLAMA_CLI, CONVERT_SCRIPT, HF_TOKEN, LLAMA_CPP_DIR

OK   = "\033[92m[OK]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

def check(label: str, passed: bool, fix: str = ""):
    status = OK if passed else FAIL
    print(f"  {status} {label}")
    if not passed and fix:
        print(f"      {INFO} Fix: {fix}")
    return passed


def main():
    print("\n" + "="*55)
    print("  quant-kit — Environment Check")
    print("="*55 + "\n")

    all_ok = True

    # ── Python version ─────────────────────────────────────
    print("[ Python ]")
    ver = sys.version_info
    ok = ver >= (3, 10)
    all_ok &= check(
        f"Python {ver.major}.{ver.minor}.{ver.micro}",
        ok,
        "Install Python 3.10 or newer from https://python.org"
    )
    print()

    # ── Python packages ────────────────────────────────────
    print("[ Python Packages ]")
    packages = {
        "huggingface_hub": "pip install huggingface_hub",
        "transformers":    "pip install transformers",
        "torch":           "pip install torch",
        "psutil":          "pip install psutil",
        "jinja2":          "pip install jinja2",
        "tqdm":            "pip install tqdm",
    }
    for pkg, fix in packages.items():
        try:
            mod = __import__(pkg)
            ver_str = getattr(mod, "__version__", "installed")
            all_ok &= check(f"{pkg} ({ver_str})", True)
        except ImportError:
            all_ok &= check(pkg, False, fix)
    print()

    # ── llama.cpp binaries ──────────────────────────────────────────
    print("[ llama.cpp Binaries ]")

    release_url = "https://github.com/ggerganov/llama.cpp/releases/latest"

    all_ok &= check(
        f"llama-quantize.exe  ({LLAMA_QUANTIZE})",
        LLAMA_QUANTIZE.exists(),
        f"Download llama-b*-bin-win-vulkan-x64.zip from {release_url}"
    )
    all_ok &= check(
        f"llama-cli.exe       ({LLAMA_CLI})",
        LLAMA_CLI.exists(),
        f"Same zip from {release_url}"
    )
    all_ok &= check(
        f"convert_hf_to_gguf.py ({CONVERT_SCRIPT})",
        CONVERT_SCRIPT.exists(),
        "Download from: https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py"
    )
    print()

    # ── HuggingFace token ──────────────────────────────────────────
    print("[ HuggingFace ]")
    if HF_TOKEN:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=HF_TOKEN)
            user = api.whoami()
            all_ok &= check(f"Token valid — logged in as: {user['name']}", True)
        except Exception:
            all_ok &= check("HuggingFace token", False, 'Check hf_token in .env — token may be invalid')
    else:
        all_ok &= check("HuggingFace token in .env", False, 'Add hf_token = "hf_xxxx" to your .env file')
    print()

    # ── Disk space ─────────────────────────────────────────
    print("[ Disk Space ]")
    import shutil
    total, used, free = shutil.disk_usage(Path(__file__).parent)
    free_gb = free / (1024 ** 3)
    ok = free_gb >= 20
    check(
        f"Free disk space: {free_gb:.1f} GB {'(OK)' if ok else '(LOW — need at least 20 GB)'}",
        ok,
        "Free up disk space before downloading large models"
    )
    print()

    # ── Summary ────────────────────────────────────────────
    print("="*55)
    if all_ok:
        print(f"  {OK} All checks passed! You are ready to go.")
        print()
        print("  Start with:")
        print("    python quantize.py --model Qwen/Qwen2.5-1.5B-Instruct")
    else:
        print(f"  {FAIL} Some checks failed. Fix the issues above first.")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
