"""
quantize_diffusion.py — Stable Diffusion / FLUX.1 Quantization via stable-diffusion.cpp
==========================================================================================
Quantizes image generation models to GGUF format using stable-diffusion.cpp.

Supported models:
  • Stable Diffusion 1.x  (sd-v1-4, sd-v1-5, ...)
  • Stable Diffusion 2.x  (stabilityai/stable-diffusion-2-1)
  • Stable Diffusion XL   (stabilityai/stable-diffusion-xl-base-1.0)
  • Stable Diffusion 3 / 3.5  (stabilityai/stable-diffusion-3.5-large)
  • FLUX.1-schnell         (black-forest-labs/FLUX.1-schnell)  ← open, no gating
  • FLUX.1-dev             (black-forest-labs/FLUX.1-dev)      ← requires HF login

Output: output/<model-name>/<model-name>-<QUANT>.gguf (single file, includes VAE)

Setup (ONE-TIME):
  python setup_sd_cpp.py   ← downloads stable-diffusion.cpp binary for your OS

Usage:
    python quantize_diffusion.py --model black-forest-labs/FLUX.1-schnell --quant q4_k
    python quantize_diffusion.py --model stabilityai/stable-diffusion-xl-base-1.0 --quant q8_0
    python quantize_diffusion.py --model stabilityai/stable-diffusion-3.5-large --quant q5_1
"""

import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import json
import shutil
import subprocess
import argparse
import time
from pathlib import Path
from huggingface_hub import snapshot_download

from config import IS_WINDOWS, MODELS_DIR, OUTPUT_DIR, SD_CPP_DIR, SD_BINARY
from utils import print_step, get_size

# ── Diffusion quant types (stable-diffusion.cpp naming) ───────────────────────
SD_QUANT_TYPES = {
    "f32":  "Full precision (32-bit, very large)",
    "f16":  "Half precision (16-bit)",
    "q8_0": "8-bit quantization (best quality)",
    "q5_1": "5-bit quantization (good balance)",
    "q5_0": "5-bit quantization",
    "q4_1": "4-bit quantization (good balance)",
    "q4_0": "4-bit quantization",
    "q4_k": "4-bit K-quant (best for FLUX)",
    "q3_k": "3-bit K-quant (aggressive)",
    "q2_k": "2-bit K-quant (extreme compression)",
}

# ── Model type detection heuristics ───────────────────────────────────────────
def detect_diffusion_type(model_dir: Path, model_id: str) -> str:
    """Detect SD model type from repo structure / model ID."""
    mid = model_id.lower()
    # Check by ID first
    if "flux" in mid:
        return "flux"
    if "sd3" in mid or "stable-diffusion-3" in mid:
        return "sd3"
    if "xl" in mid or "sdxl" in mid:
        return "sdxl"
    if "2-1" in mid or "2.1" in mid or "stable-diffusion-2" in mid:
        return "sd2"

    # Fall back to checking model_index.json
    idx_path = model_dir / "model_index.json"
    if idx_path.exists():
        try:
            with open(idx_path) as f:
                data = json.load(f)
            pipe = data.get("_class_name", "")
            if "Flux" in pipe:
                return "flux"
            if "StableDiffusionXL" in pipe:
                return "sdxl"
            if "StableDiffusion3" in pipe:
                return "sd3"
        except Exception:
            pass

    return "sd1"  # safest default


def check_sd_cpp():
    """Verify stable-diffusion.cpp binary exists."""
    if not SD_BINARY.exists():
        print_step("err", f"stable-diffusion binary not found: {SD_BINARY}")
        print_step("info", "Run: python setup_sd_cpp.py")
        print_step("info", "Or download from: https://github.com/leejet/stable-diffusion.cpp/releases")
        sys.exit(1)
    print_step("ok", f"stable-diffusion.cpp found: {SD_BINARY.name}")


def download_model(model_id: str) -> Path:
    """Download diffusion model from HuggingFace Hub."""
    model_name = model_id.split("/")[-1]
    local_dir = MODELS_DIR / model_name

    if local_dir.exists() and any(local_dir.iterdir()):
        print_step("ok", f"Model already downloaded: {local_dir}")
        return local_dir

    print_step("info", f"Downloading {model_id}...")
    free_gb = shutil.disk_usage(str(MODELS_DIR.parent)).free / (1024 ** 3)
    print_step("info", f"Free disk: {free_gb:.1f} GB")

    # For FLUX / SD models we want safetensors, not diffusers pytorch_model.bin shards
    ignore = ["*.bin", "*.pt", "*.msgpack", "flax_model*", "rust_model*", "tf_model*"]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(repo_id=model_id, local_dir=str(local_dir), ignore_patterns=ignore)
        print_step("ok", f"Downloaded to: {local_dir}")
        return local_dir
    except Exception as e:
        print_step("err", f"Download failed: {e}")
        print_step("info", "If this is a gated model (e.g. FLUX.1-dev), ensure HF_TOKEN is set in .env")
        sys.exit(1)


def find_model_file(model_dir: Path, model_type: str) -> Path | None:
    """
    Find the primary safetensors checkpoint file.
    Returns None if not found (user needs to provide --model-file).
    """
    # Priority order per model type
    if model_type == "flux":
        candidates = [
            "flux1-schnell.safetensors",
            "flux1-dev.safetensors",
            "flux1-dev-fp8.safetensors",
        ]
    elif model_type in ("sd3", "sdxl"):
        candidates = [
            "sd3.5_large.safetensors",
            "sd3.5_medium.safetensors",
            "sd3_medium_incl_clips.safetensors",
            "sd_xl_base_1.0.safetensors",
        ]
    else:
        candidates = []

    for name in candidates:
        p = model_dir / name
        if p.exists():
            return p

    # Fallback: find any .safetensors file in root that looks like a checkpoint
    for p in sorted(model_dir.glob("*.safetensors")):
        # Skip CLIP / VAE / scheduler small files
        if p.stat().st_size > 500 * 1024 * 1024:  # >500MB = likely model file
            return p

    return None


def quantize_sd(model_dir: Path, model_id: str, quant_type: str,
                model_file: Path | None = None, vae_file: Path | None = None) -> Path | None:
    """
    Run stable-diffusion.cpp convert + quantize.
    stable-diffusion.cpp converts and quantizes in a single command.
    """
    model_name = model_id.split("/")[-1]
    out_dir    = OUTPUT_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path   = out_dir / f"{model_name}-{quant_type}.gguf"

    if out_path.exists() and out_path.stat().st_size > 100 * 1024 * 1024:
        print_step("ok", f"Already exists: {out_path.name} ({get_size(out_path)})")
        return out_path

    model_type = detect_diffusion_type(model_dir, model_id)
    print_step("info", f"Model type: {model_type.upper()}")

    # Locate the main model file
    src_file = model_file or find_model_file(model_dir, model_type)
    if src_file is None:
        print_step("err", "Could not find main safetensors file!")
        print_step("info", "Use --model-file path/to/model.safetensors to specify it manually")
        return None

    print_step("info", f"Source: {src_file.name} ({get_size(src_file)})")
    print_step("info", f"Quantizing → {quant_type} (this takes 5-30 min for large models)...")

    t0 = time.time()
    cmd = [
        str(SD_BINARY),
        "--mode", "convert",
        "-m", str(src_file),
        "-o", str(out_path),
        "--type", quant_type,
    ]

    # Add VAE if separately provided (SDXL, some SD3 configs)
    if vae_file and vae_file.exists():
        cmd += ["--vae", str(vae_file)]
        print_step("info", f"VAE: {vae_file.name}")

    result = subprocess.run(cmd, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print_step("err", f"stable-diffusion.cpp conversion failed! (exit {result.returncode})")
        return None

    if out_path.exists():
        print_step("ok", f"Done in {elapsed:.0f}s → {out_path.name} ({get_size(out_path)})")
        return out_path
    else:
        print_step("err", "Output file not created — check stable-diffusion.cpp output above")
        return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Quantize Stable Diffusion / FLUX models to GGUF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported quant types:
  q4_k  (best for FLUX, K-quant)   q8_0  (best quality, large)
  q5_1  (good balance)              f16   (half precision)
  q4_0  (4-bit, smaller)           q4_1  (4-bit with better accuracy)

Examples:
  python quantize_diffusion.py --model black-forest-labs/FLUX.1-schnell --quant q4_k
  python quantize_diffusion.py --model stabilityai/stable-diffusion-xl-base-1.0 --quant q8_0
  python quantize_diffusion.py --model stabilityai/stable-diffusion-3.5-large --quant q5_1
        """,
    )
    parser.add_argument("--model",      "-m", required=True, help="HuggingFace model ID")
    parser.add_argument("--quant",      "-q", default="q8_0",
                        choices=list(SD_QUANT_TYPES.keys()),
                        help="Quantization type (default: q8_0)")
    parser.add_argument("--model-file", help="Path to specific .safetensors file (auto-detected if omitted)")
    parser.add_argument("--vae",        help="Path to separate VAE .safetensors (optional)")
    parser.add_argument("--delete-src", action="store_true", help="Delete downloaded model after quantization")
    args = parser.parse_args()

    model_name = args.model.split("/")[-1]

    print("\n" + "=" * 60)
    print("  quant-kit — Diffusion Model Quantizer")
    print("=" * 60)
    print(f"  Model  : {args.model}")
    print(f"  Quant  : {args.quant}  ({SD_QUANT_TYPES[args.quant]})")
    print(f"  Output : {OUTPUT_DIR / model_name}/")
    print("=" * 60 + "\n")

    check_sd_cpp()
    model_dir = download_model(args.model)

    model_file = Path(args.model_file) if args.model_file else None
    vae_file   = Path(args.vae)        if args.vae        else None

    result = quantize_sd(model_dir, args.model, args.quant, model_file, vae_file)

    if args.delete_src and result:
        print_step("info", "--delete-src: removing downloaded model files...")
        freed = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
        shutil.rmtree(str(model_dir))
        print_step("ok", f"Freed {freed/(1024**3):.1f} GB")

    if result:
        print("\n" + "=" * 60)
        print("  ✅ Diffusion Quantization Complete!")
        print("=" * 60)
        print(f"  📦 {result.name}  ({get_size(result)})")
        print()
        print("  Run inference:")
        print(f"     # stable-diffusion.cpp:")
        print(f"     sd -m {result} -p \"a photo of a cat\" -o output.png")
        print()
        print("  Or load in ComfyUI / AUTOMATIC1111 (with GGUF plugin).")
        print()
        print("  Next steps:")
        print(f"     python model_card.py --model {model_name} --original {args.model}")
        print(f"     python upload.py     --model {model_name}")
        print("=" * 60 + "\n")
    else:
        print_step("err", "Quantization failed. See output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
