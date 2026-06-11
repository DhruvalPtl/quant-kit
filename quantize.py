"""
quantize.py — Core quantization script
========================================
Downloads a HuggingFace model, converts it to GGUF format,
and quantizes it into multiple bit-width variants using llama.cpp.

Usage:
    python quantize.py --model Qwen/Qwen2.5-1.5B-Instruct
    python quantize.py --model google/gemma-4-12b-it --preset full
    python quantize.py --model Qwen/Qwen2.5-7B-Instruct --quants Q4_K_M Q8_0
"""

import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import shutil
import subprocess
import argparse
import time
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download

from config import (
    IS_WINDOWS, LLAMA_CPP_DIR, LLAMA_SRC_DIR,
    MODELS_DIR, OUTPUT_DIR, DEFAULT_QUANTS, QUANT_PRESETS,
    LLAMA_QUANTIZE, CONVERT_SCRIPT, LLAMA_IMATRIX
)
from utils import print_step, get_size

# --- Helpers -------------------------------------------------------------------

def check_llama_cpp(requires_imatrix: bool = False):
    """Make sure llama.cpp binaries exist."""
    if not LLAMA_QUANTIZE.exists():
        print_step("err", f"llama-quantize not found at: {LLAMA_QUANTIZE}")
        if IS_WINDOWS:
            print_step("info", "Download llama-b*-bin-win-vulkan-x64.zip from:")
            print_step("info", "https://github.com/ggerganov/llama.cpp/releases/latest")
        else:
            print_step("info", "Run: python setup_linux.py")
        sys.exit(1)

    if not CONVERT_SCRIPT.exists():
        print_step("err", f"convert_hf_to_gguf.py not found at: {CONVERT_SCRIPT}")
        sys.exit(1)

    if requires_imatrix and not LLAMA_IMATRIX.exists():
        print_step("err", f"llama-imatrix not found at: {LLAMA_IMATRIX}")
        sys.exit(1)

    print_step("ok", "llama.cpp binaries found")


def get_supported_architectures() -> set[str]:
    """Dynamically read supported architectures from your local convert_hf_to_gguf.py."""
    try:
        import re
        src = CONVERT_SCRIPT.read_text(encoding="utf-8", errors="ignore")
        # The converter lists them as class definitions with gguf_writer or _model_writers
        # Most reliable: grep all class names that end with standard patterns
        class_names = re.findall(r'^class (\w+(?:ForCausalLM|ForConditionalGeneration|Model|ForMaskedLM|ForSequenceClassification|ForTokenClassification))\b',
                                 src, re.MULTILINE)
        return set(class_names)
    except Exception:
        return set()  # If we can't read it, allow all (fail-open)


def preflight_check(model_id: str) -> None:
    """
    Download ONLY config.json (~2KB) and verify architecture is supported
    by your local llama.cpp BEFORE triggering a potentially huge download.
    Exits with a clear error if the model cannot be converted.
    """
    from huggingface_hub import hf_hub_download
    import json

    print_step("info", f"Pre-flight check: verifying architecture support...")
    try:
        config_path = hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            local_dir=str(MODELS_DIR / "_preflight_cache"),
        )
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        print_step("warn", f"Could not fetch config.json: {e} — proceeding anyway")
        return

    archs = config.get("architectures", [])
    if not archs:
        print_step("warn", "No 'architectures' key in config.json — proceeding")
        return

    supported = get_supported_architectures()
    unsupported = [a for a in archs if a not in supported]

    if unsupported and supported:  # Only block if we successfully read the supported list
        print()
        print("  " + "=" * 58)
        print("  🚫 UNSUPPORTED ARCHITECTURE — Download Blocked")
        print("  " + "=" * 58)
        print(f"  Model      : {model_id}")
        print(f"  Architecture: {unsupported[0]}")
        print()
        print("  This architecture is NOT in your local convert_hf_to_gguf.py.")
        print("  Downloading would waste disk space — the conversion will fail.")
        print()
        print("  Options:")
        print("  1. Update llama-src: git -C llama-src pull")
        print("     (New architectures are added often — check again after update)")
        print("  2. Use bitsandbytes instead (4-bit via HuggingFace Transformers)")
        print("  3. Force-proceed anyway: add --skip-preflight to your command")
        print("  " + "=" * 58)
        print()
        sys.exit(1)
    else:
        print_step("ok", f"Architecture supported: {archs[0]}")


def download_model(model_id: str, skip_preflight: bool = False) -> Path:
    """Download model from HuggingFace Hub."""
    model_name = model_id.split("/")[-1]
    local_dir = MODELS_DIR / model_name

    # 🛡️ Pre-flight: verify architecture BEFORE downloading gigabytes
    if not skip_preflight:
        preflight_check(model_id)

    print_step("info", f"Downloading {model_id} from HuggingFace...")
    import shutil as _shutil
    free_gb = _shutil.disk_usage(str(MODELS_DIR.parent)).free / (1024**3)
    print_step("info", f"Free disk space: {free_gb:.1f} GB")

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
        print_step("info", "Make sure HF_TOKEN is set in your .env file")
        sys.exit(1)


def convert_to_fp16_gguf(model_dir: Path) -> Path:
    """Convert HuggingFace model to FP16 GGUF (lossless base format)."""
    model_name = model_dir.name
    output_path = OUTPUT_DIR / model_name / f"{model_name}-F16.gguf"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        # Check if it's a corrupted partial file from a canceled run (must be > 1GB)
        if output_path.stat().st_size > 1024**3:
            print_step("ok", f"F16 GGUF already exists: {output_path.name}")
            return output_path
        else:
            print_step("warn", f"Found incomplete F16 file ({get_size(output_path)}). Rebuilding...")
            output_path.unlink()

    print_step("info", f"Converting to FP16 GGUF...")

    cmd = [
        sys.executable,
        str(CONVERT_SCRIPT),
        str(model_dir),
        "--outfile", str(output_path),
        "--outtype", "f16",
    ]

    result = subprocess.run(cmd, capture_output=False, text=True, cwd=str(LLAMA_SRC_DIR))

    if result.returncode != 0:
        print_step("err", "Conversion failed!")
        sys.exit(1)

    print_step("ok", f"FP16 GGUF created: {output_path.name} ({get_size(output_path)})")
    return output_path


def generate_imatrix(fp16_path: Path, calibration_file: Path) -> Path:
    """Generate importance matrix for better low-bit quantization."""
    output_path = fp16_path.parent / "imatrix.dat"
    if output_path.exists():
        print_step("ok", f"imatrix already exists: {output_path.name}")
        return output_path

    print_step("info", f"Generating imatrix using {calibration_file.name}...")
    start = time.time()

    cmd = [
        str(LLAMA_IMATRIX),
        "-m", str(fp16_path),
        "-f", str(calibration_file),
        "-o", str(output_path),
    ]

    env = os.environ.copy()
    lib_dir = str(LLAMA_IMATRIX.parent)
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else lib_dir

    result = subprocess.run(cmd, capture_output=False, text=True, env=env)

    if result.returncode != 0:
        print_step("err", f"imatrix generation failed! (exit code {result.returncode})")
        sys.exit(1)

    elapsed = time.time() - start
    print_step("ok", f"imatrix created in {elapsed:.1f}s -> {output_path.name}")
    return output_path


def quantize_gguf(fp16_path: Path, quant_type: str, imatrix_path: Path = None) -> Path:
    """Quantize an FP16 GGUF to a specific quant type."""
    model_name = fp16_path.parent.name
    output_path = fp16_path.parent / f"{model_name}-{quant_type}.gguf"

    if output_path.exists():
        # A real quantized GGUF for a 12B model must be at least 100MB
        if output_path.stat().st_size > 100 * 1024 * 1024:
            print_step("ok", f"Already exists: {output_path.name} ({get_size(output_path)})")
            return output_path
        else:
            print_step("warn", f"Found corrupted quant ({get_size(output_path)}), deleting and re-quantizing...")
            output_path.unlink()

    print_step("info", f"Quantizing to {quant_type}...")
    start = time.time()

    cmd = [
        str(LLAMA_QUANTIZE),
        str(fp16_path),
        str(output_path),
        quant_type,
    ]
    
    if imatrix_path and imatrix_path.exists() and quant_type.startswith("IQ"):
        cmd.extend(["--imatrix", str(imatrix_path)])

    env = os.environ.copy()
    lib_dir = str(LLAMA_QUANTIZE.parent)
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else lib_dir

    result = subprocess.run(cmd, capture_output=False, text=True, env=env)

    if result.returncode != 0:
        print_step("err", f"Quantization to {quant_type} failed! (exit code {result.returncode})")
        if output_path.exists():
            output_path.unlink()  # Delete the corrupted partial file
        sys.exit(1)

    elapsed = time.time() - start
    print_step("ok", f"{quant_type} done in {elapsed:.1f}s → {output_path.name} ({get_size(output_path)})")
    return output_path


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Quantize a HuggingFace model to GGUF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", "-m", required=True, help="HuggingFace model ID (e.g. Qwen/Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--quants", "-q", nargs="+", help="Explicit quant types to generate")
    parser.add_argument("--preset", "-p", choices=list(QUANT_PRESETS.keys()), default="standard", help="Quantization preset (standard, full, imatrix, all)")
    parser.add_argument("--calibration", "-c", help="Path to text file for imatrix calibration")
    parser.add_argument("--keep-fp16", action="store_true", help="Keep the FP16 GGUF file after quantization")
    parser.add_argument("--delete-src", action="store_true", help="Delete downloaded model files after FP16 conversion to free disk space")
    parser.add_argument("--batch", "-b", type=int, default=0,
        help="Process N quants per run (re-run to continue). Already-completed quants are skipped. Example: --batch 2")
    parser.add_argument("--skip-preflight", action="store_true",
        help="Skip architecture support check (use if you know the model works but preflight blocks it)")

    args = parser.parse_args()

    all_quants = args.quants if args.quants else QUANT_PRESETS[args.preset]
    requires_imatrix = any(q.startswith("IQ") for q in all_quants)

    # ── Batch mode: figure out which quants still need to be done ──────────────
    model_name = args.model.split("/")[-1]
    output_dir = OUTPUT_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    def is_done(quant_type: str) -> bool:
        """Return True if this quant already exists and is valid (> 100MB)."""
        p = output_dir / f"{model_name}-{quant_type}.gguf"
        return p.exists() and p.stat().st_size > 100 * 1024 * 1024

    pending   = [q for q in all_quants if not is_done(q)]
    completed = [q for q in all_quants if is_done(q)]

    # If --batch N, only take the next N pending quants this run
    if args.batch > 0:
        quants_this_run = pending[:args.batch]
    else:
        quants_this_run = pending

    remaining_after = [q for q in pending if q not in quants_this_run]

    print("\n" + "="*60)
    print("  quant-kit — GGUF Quantization Tool")
    print("="*60)
    print(f"  Model      : {args.model}")
    print(f"  Total      : {len(all_quants)} quants ({', '.join(all_quants)})")
    print(f"  Done       : {len(completed)} ({', '.join(completed) if completed else 'none'})")
    print(f"  This run   : {len(quants_this_run)} ({', '.join(quants_this_run) if quants_this_run else 'none'})")
    if remaining_after:
        print(f"  After this : {len(remaining_after)} remaining — re-run the same command to continue")
    print(f"  iMatrix    : {'Required' if requires_imatrix else 'Not needed'}")
    print("="*60 + "\n")

    if not quants_this_run:
        print_step("ok", "All quants are already complete! Nothing to do.")
        print_step("info", "Run: python benchmark.py --model " + model_name)
        return

    check_llama_cpp(requires_imatrix=requires_imatrix)
    model_dir = download_model(args.model, skip_preflight=args.skip_preflight)
    fp16_path = convert_to_fp16_gguf(model_dir)

    if args.delete_src:
        print()
        print_step("info", "--delete-src: removing source model files to free disk...")
        freed = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file()) if model_dir.exists() else 0
        shutil.rmtree(str(model_dir))
        print_step("ok", f"Source files deleted — freed {freed / (1024**3):.1f} GB")

    imatrix_path = None
    if requires_imatrix:
        calib_file = Path(args.calibration) if args.calibration else Path("calibration_data.txt")
        if not calib_file.exists():
            print_step("warn", f"Calibration file {calib_file} not found. Skipping imatrix generation.")
            print_step("info", "IQ quants will be generated without an imatrix (suboptimal).")
        else:
            imatrix_path = generate_imatrix(fp16_path, calib_file)

    created_quants = []
    print()
    for i, quant_type in enumerate(quants_this_run, 1):
        print_step("info", f"--- [{i}/{len(quants_this_run)}] Starting {quant_type} ---")
        result = quantize_gguf(fp16_path, quant_type, imatrix_path)
        if result:
            created_quants.append(result)

    # Only delete FP16 if ALL quants across ALL batches are now done
    all_done_now = all(is_done(q) for q in all_quants)
    if all_done_now and not args.keep_fp16 and fp16_path.exists():
        print()
        print_step("info", "All quants complete — removing FP16 base file to free disk...")
        fp16_path.unlink()
        print_step("ok", f"FP16 file removed")
    elif not all_done_now and fp16_path.exists():
        print()
        print_step("info", f"FP16 kept — {len(remaining_after)} quant(s) still pending. Re-run to continue.")

    print("\n" + "="*60)
    print("  Done! Created quants:")
    for q in created_quants:
        print(f"    • {q.name} ({get_size(q)})")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
