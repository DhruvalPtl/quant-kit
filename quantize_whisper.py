"""
quantize_whisper.py — Whisper ASR Model Quantization via whisper.cpp
=====================================================================
Converts OpenAI Whisper models from HuggingFace to GGUF format
and quantizes them using whisper.cpp.

Supported models:
  openai/whisper-tiny           openai/whisper-tiny.en
  openai/whisper-base           openai/whisper-base.en
  openai/whisper-small          openai/whisper-small.en
  openai/whisper-medium         openai/whisper-medium.en
  openai/whisper-large-v1       openai/whisper-large-v2
  openai/whisper-large-v3       openai/whisper-large-v3-turbo
  distil-whisper/distil-large-v3  (and other distil-whisper variants)

Quant types (whisper.cpp uses different names from llama.cpp):
  q5_0  q5_1  q8_0  (recommended)

Setup (ONE-TIME):
  python setup_whisper_cpp.py   ← downloads whisper.cpp for your OS

Usage:
    python quantize_whisper.py --model openai/whisper-large-v3-turbo
    python quantize_whisper.py --model openai/whisper-medium --quant q8_0
    python quantize_whisper.py --model distil-whisper/distil-large-v3 --quant q5_0
"""

import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import shutil
import subprocess
import argparse
import time
from pathlib import Path
from huggingface_hub import snapshot_download

from config import IS_WINDOWS, MODELS_DIR, OUTPUT_DIR, WHISPER_CPP_DIR, WHISPER_BINARY, WHISPER_QUANTIZE, WHISPER_CONVERT_SCRIPT
from utils import print_step, get_size

# ── Whisper quant types ────────────────────────────────────────────────────────
WHISPER_QUANT_TYPES = {
    "q8_0": "8-bit quantization (best quality, recommended)",
    "q5_1": "5-bit quantization (good balance, smaller)",
    "q5_0": "5-bit quantization (smaller)",
    "q4_1": "4-bit quantization (compact)",
    "q4_0": "4-bit quantization (most compact)",
}

# Map q-type to whisper-quantize integer code
QUANT_CODE = {
    "q4_0": 2,
    "q4_1": 3,
    "q5_0": 8,
    "q5_1": 9,
    "q8_0": 7,
}


def check_whisper_cpp():
    """Verify whisper.cpp binaries exist."""
    missing = []
    for name, path in [("whisper binary", WHISPER_BINARY),
                        ("whisper-quantize", WHISPER_QUANTIZE),
                        ("convert script", WHISPER_CONVERT_SCRIPT)]:
        if not path.exists():
            missing.append(f"  {name}: {path}")

    if missing:
        print_step("err", "Missing whisper.cpp files:")
        for m in missing:
            print(m)
        print_step("info", "Run: python setup_whisper_cpp.py")
        print_step("info", "Or download from: https://github.com/ggerganov/whisper.cpp/releases")
        sys.exit(1)
    print_step("ok", "whisper.cpp binaries found")


def download_model(model_id: str) -> Path:
    """Download Whisper model from HuggingFace."""
    model_name = model_id.split("/")[-1]
    local_dir = MODELS_DIR / model_name

    if local_dir.exists() and any(local_dir.iterdir()):
        print_step("ok", f"Model already downloaded: {local_dir}")
        return local_dir

    print_step("info", f"Downloading {model_id}...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            ignore_patterns=["*.bin", "flax_model*", "tf_model*", "rust_model*"],
        )
        print_step("ok", f"Downloaded to: {local_dir}")
        return local_dir
    except Exception as e:
        print_step("err", f"Download failed: {e}")
        sys.exit(1)


def convert_to_ggml(model_dir: Path) -> Path:
    """
    Convert HuggingFace Whisper model to GGML/GGUF format
    using whisper.cpp's convert-h5-to-ggml.py script.
    Output: <model_name>-ggml-model-f16.gguf
    """
    model_name = model_dir.name
    out_dir    = OUTPUT_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path   = out_dir / f"{model_name}-ggml-f16.gguf"

    if out_path.exists() and out_path.stat().st_size > 50 * 1024 * 1024:
        print_step("ok", f"F16 GGUF already exists: {out_path.name}")
        return out_path

    print_step("info", "Converting Whisper → GGML F16 GGUF...")

    # whisper.cpp's convert script needs: model_dir output_dir [use_f16]
    cmd = [
        sys.executable,
        str(WHISPER_CONVERT_SCRIPT),
        str(model_dir),       # path to HF model directory
        str(out_dir),         # output directory
        "1",                  # 1 = use f16 (standard)
    ]

    result = subprocess.run(cmd, text=True, cwd=str(WHISPER_CPP_DIR))
    if result.returncode != 0:
        print_step("err", "GGML conversion failed!")
        sys.exit(1)

    # whisper.cpp names the file ggml-model-f16.bin or .gguf depending on version
    for candidate in ["ggml-model-f16.gguf", "ggml-model-f16.bin"]:
        raw_out = out_dir / candidate
        if raw_out.exists():
            if raw_out != out_path:
                raw_out.rename(out_path)
            break

    if not out_path.exists():
        # Try to find any f16 file created
        for p in out_dir.glob("*f16*"):
            p.rename(out_path)
            break

    if not out_path.exists():
        print_step("err", "Could not find converted F16 file in output directory")
        sys.exit(1)

    print_step("ok", f"F16 GGUF: {out_path.name} ({get_size(out_path)})")
    return out_path


def quantize_whisper(f16_path: Path, quant_type: str) -> Path | None:
    """Quantize a Whisper F16 GGUF using whisper-quantize binary."""
    model_name = f16_path.parent.name
    out_path = f16_path.parent / f"{model_name}-{quant_type}.gguf"

    if out_path.exists() and out_path.stat().st_size > 10 * 1024 * 1024:
        print_step("ok", f"Already exists: {out_path.name} ({get_size(out_path)})")
        return out_path

    quant_code = QUANT_CODE.get(quant_type)
    if quant_code is None:
        print_step("err", f"Unknown quant type: {quant_type}")
        return None

    print_step("info", f"Quantizing → {quant_type} (type code: {quant_code})...")
    t0 = time.time()

    cmd = [str(WHISPER_QUANTIZE), str(f16_path), str(out_path), str(quant_code)]
    result = subprocess.run(cmd, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print_step("err", f"whisper-quantize failed! (exit {result.returncode})")
        return None

    print_step("ok", f"Done in {elapsed:.1f}s → {out_path.name} ({get_size(out_path)})")
    return out_path


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Quantize Whisper ASR models to GGUF via whisper.cpp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quantize_whisper.py --model openai/whisper-large-v3-turbo
  python quantize_whisper.py --model openai/whisper-medium --quant q5_0
  python quantize_whisper.py --model distil-whisper/distil-large-v3 --quant q8_0
        """,
    )
    parser.add_argument("--model",      "-m", required=True, help="HuggingFace Whisper model ID")
    parser.add_argument("--quant",      "-q", default="q8_0",
                        choices=list(WHISPER_QUANT_TYPES.keys()),
                        help="Quantization type (default: q8_0)")
    parser.add_argument("--keep-fp16",  action="store_true", help="Keep F16 GGUF after quantization")
    parser.add_argument("--delete-src", action="store_true", help="Delete downloaded model after conversion")
    args = parser.parse_args()

    model_name = args.model.split("/")[-1]

    print("\n" + "=" * 60)
    print("  quant-kit — Whisper ASR Quantizer")
    print("=" * 60)
    print(f"  Model  : {args.model}")
    print(f"  Quant  : {args.quant}  ({WHISPER_QUANT_TYPES[args.quant]})")
    print(f"  Output : {OUTPUT_DIR / model_name}/")
    print("=" * 60 + "\n")

    check_whisper_cpp()
    model_dir = download_model(args.model)
    print()

    f16_path = convert_to_ggml(model_dir)
    print()

    if args.delete_src:
        print_step("info", "--delete-src: removing source model files...")
        freed = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
        shutil.rmtree(str(model_dir))
        print_step("ok", f"Freed {freed/(1024**3):.1f} GB")
        print()

    result = quantize_whisper(f16_path, args.quant)

    if not args.keep_fp16 and f16_path.exists() and result:
        print_step("info", "Removing F16 base file...")
        f16_path.unlink()
        print_step("ok", "F16 removed")

    if result:
        print("\n" + "=" * 60)
        print("  ✅ Whisper Quantization Complete!")
        print("=" * 60)
        print(f"  📦 {result.name}  ({get_size(result)})")
        print()
        print("  Run transcription:")
        print(f"     whisper -m {result} -f audio.wav")
        print()
        print("  Next steps:")
        print(f"     python model_card.py --model {model_name} --original {args.model}")
        print(f"     python upload.py     --model {model_name}")
        print("=" * 60 + "\n")
    else:
        print_step("err", "Quantization failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
