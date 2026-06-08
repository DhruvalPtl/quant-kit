"""
quantize.py — Core quantization script
========================================
Downloads a HuggingFace model, converts it to GGUF format,
and quantizes it into multiple bit-width variants using llama.cpp.

Usage:
    python quantize.py --model Qwen/Qwen2.5-1.5B-Instruct
    python quantize.py --model Qwen/Qwen2.5-7B-Instruct --quants Q4_K_M Q8_0
    python quantize.py --model meta-llama/Llama-3.2-3B-Instruct --quants Q4_K_M Q5_K_M Q8_0
"""

import os
import sys
import shutil
import subprocess
import argparse
import time
from pathlib import Path
from huggingface_hub import snapshot_download
from tqdm import tqdm

from config import LLAMA_CPP_DIR, MODELS_DIR, OUTPUT_DIR, DEFAULT_QUANTS, LLAMA_QUANTIZE, CONVERT_SCRIPT

# ─── Helpers ───────────────────────────────────────────────────────────────────

def print_step(step: str, msg: str):
    colors = {"info": "\033[94m", "ok": "\033[92m", "warn": "\033[93m", "err": "\033[91m"}
    reset = "\033[0m"
    icons = {"info": "→", "ok": "✓", "warn": "⚠", "err": "✗"}
    color = colors.get(step, "")
    icon = icons.get(step, "•")
    print(f"{color}{icon} {msg}{reset}")


def check_llama_cpp():
    """Make sure llama.cpp binaries exist."""
    if not LLAMA_QUANTIZE.exists():
        print_step("err", f"llama-quantize.exe not found at: {LLAMA_QUANTIZE}")
        print_step("info", "Download llama.cpp from: https://github.com/ggerganov/llama.cpp/releases/latest")
        print_step("info", "Get the file: llama-b*-bin-win-vulkan-x64.zip")
        print_step("info", f"Extract it to: {LLAMA_CPP_DIR}")
        sys.exit(1)

    if not CONVERT_SCRIPT.exists():
        print_step("err", f"convert_hf_to_gguf.py not found at: {CONVERT_SCRIPT}")
        print_step("info", "Download from: https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py")
        sys.exit(1)

    print_step("ok", "llama.cpp binaries found")
    return LLAMA_QUANTIZE, CONVERT_SCRIPT


def download_model(model_id: str) -> Path:
    """Download model from HuggingFace Hub."""
    # Convert "Qwen/Qwen2.5-1.5B-Instruct" → "Qwen2.5-1.5B-Instruct"
    model_name = model_id.split("/")[-1]
    local_dir = MODELS_DIR / model_name

    if local_dir.exists() and any(local_dir.iterdir()):
        print_step("ok", f"Model already downloaded: {local_dir}")
        return local_dir

    print_step("info", f"Downloading {model_id} from HuggingFace...")
    print_step("info", "This may take a while depending on model size and internet speed")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            ignore_patterns=["*.bin", "*.pt"],  # prefer .safetensors
        )
        print_step("ok", f"Downloaded to: {local_dir}")
        return local_dir

    except Exception as e:
        print_step("err", f"Download failed: {e}")
        print_step("info", "Make sure you are logged in: huggingface-cli login")
        sys.exit(1)


def convert_to_fp16_gguf(model_dir: Path, convert_script: Path) -> Path:
    """Convert HuggingFace model to FP16 GGUF (lossless base format)."""
    model_name = model_dir.name
    output_path = OUTPUT_DIR / model_name / f"{model_name}-F16.gguf"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print_step("ok", f"F16 GGUF already exists: {output_path.name}")
        return output_path

    print_step("info", f"Converting to FP16 GGUF...")

    cmd = [
        sys.executable,
        str(convert_script),
        str(model_dir),
        "--outfile", str(output_path),
        "--outtype", "f16",
    ]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print_step("err", "Conversion failed!")
        sys.exit(1)

    print_step("ok", f"FP16 GGUF created: {output_path.name} ({get_size(output_path)})")
    return output_path


def quantize_gguf(fp16_path: Path, quant_type: str, quantize_bin: Path) -> Path:
    """Quantize an FP16 GGUF to a specific quant type."""
    model_name = fp16_path.parent.name
    output_path = fp16_path.parent / f"{model_name}-{quant_type}.gguf"

    if output_path.exists():
        print_step("ok", f"Already exists: {output_path.name} ({get_size(output_path)})")
        return output_path

    print_step("info", f"Quantizing to {quant_type}...")
    start = time.time()

    cmd = [
        str(quantize_bin),
        str(fp16_path),
        str(output_path),
        quant_type,
    ]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print_step("err", f"Quantization to {quant_type} failed!")
        return None

    elapsed = time.time() - start
    print_step("ok", f"{quant_type} done in {elapsed:.1f}s → {output_path.name} ({get_size(output_path)})")
    return output_path


def get_size(path: Path) -> str:
    """Return human-readable file size."""
    size = path.stat().st_size
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Quantize a HuggingFace model to GGUF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quantize.py --model Qwen/Qwen2.5-1.5B-Instruct
  python quantize.py --model Qwen/Qwen2.5-7B-Instruct --quants Q4_K_M Q8_0
  python quantize.py --model meta-llama/Llama-3.2-3B-Instruct --quants Q4_K_M Q5_K_M Q8_0
        """
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="HuggingFace model ID (e.g. Qwen/Qwen2.5-1.5B-Instruct)"
    )
    parser.add_argument(
        "--quants", "-q",
        nargs="+",
        default=DEFAULT_QUANTS,
        help=f"Quant types to generate (default: {' '.join(DEFAULT_QUANTS)})"
    )
    parser.add_argument(
        "--keep-fp16",
        action="store_true",
        help="Keep the FP16 GGUF file after quantization (deleted by default to save disk)"
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("  quant-kit — GGUF Quantization Tool")
    print("="*60)
    print(f"  Model  : {args.model}")
    print(f"  Quants : {', '.join(args.quants)}")
    print("="*60 + "\n")

    # Step 1: Check tools
    quantize_bin, convert_script = check_llama_cpp()

    # Step 2: Download model
    model_dir = download_model(args.model)

    # Step 3: Convert to FP16 GGUF
    fp16_path = convert_to_fp16_gguf(model_dir, convert_script)

    # Step 4: Quantize to each target format
    created_quants = []
    print()
    for quant_type in args.quants:
        result = quantize_gguf(fp16_path, quant_type, quantize_bin)
        if result:
            created_quants.append(result)

    # Step 5: Clean up FP16 (large intermediate file)
    if not args.keep_fp16 and fp16_path.exists():
        print()
        print_step("info", f"Removing FP16 base file to save disk space...")
        fp16_path.unlink()
        print_step("ok", "FP16 file removed")

    # Summary
    print("\n" + "="*60)
    print("  Done! Created quants:")
    for q in created_quants:
        print(f"    • {q.name} ({get_size(q)})")
    print()
    print("  Next steps:")
    print("    1. Run benchmark.py to measure performance")
    print("    2. Run model_card.py to generate README")
    print("    3. Run upload.py to publish to HuggingFace")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
