"""
quantize_vlm.py — VLM (Vision-Language Model) Quantization
===========================================================
Produces TWO GGUF files per VLM:
  1. <model>-<QUANT>.gguf       — quantized text backbone (e.g. Q4_K_M)
  2. <model>-mmproj-f16.gguf   — vision encoder, always F16

Supported architectures (detected automatically):
  LlavaForConditionalGeneration, Qwen2VLForConditionalGeneration,
  Qwen2_5_VLForConditionalGeneration, Gemma3ForConditionalGeneration,
  Gemma4ForConditionalGeneration, Llama4ForConditionalGeneration,
  InternVisionModel, MiniCPMV4_6ForConditionalGeneration,
  Idefics3ForConditionalGeneration, SmolVLMForConditionalGeneration,
  Mistral3ForConditionalGeneration, Phi4ForCausalLMV, and many more.

Usage:
    python quantize_vlm.py --model Qwen/Qwen2.5-VL-7B-Instruct
    python quantize_vlm.py --model meta-llama/Llama-3.2-11B-Vision-Instruct --preset standard
    python quantize_vlm.py --model google/paligemma2-3b-pt-224 --quants Q4_K_M Q8_0
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

from config import (
    IS_WINDOWS, LLAMA_CPP_DIR, LLAMA_SRC_DIR,
    MODELS_DIR, OUTPUT_DIR, DEFAULT_QUANTS, QUANT_PRESETS,
    LLAMA_QUANTIZE, CONVERT_SCRIPT,
)
from utils import print_step, get_size

# ── VLM architecture registry ──────────────────────────────────────────────────
# Architectures that support --mmproj in convert_hf_to_gguf.py
# (generated from --print-supported-models MMPROJ section)
MMPROJ_ARCHITECTURES = {
    "AudioFlamingo3ForConditionalGeneration",
    "CogVLMForCausalLM",
    "Exaone4_5_ForConditionalGeneration",
    "Gemma3ForConditionalGeneration",
    "Gemma3nForConditionalGeneration",
    "Gemma4ForConditionalGeneration",
    "Gemma4UnifiedForConditionalGeneration",
    "Glm4vForConditionalGeneration",
    "Glm4vMoeForConditionalGeneration",
    "GlmOcrForConditionalGeneration",
    "HunYuanVLForConditionalGeneration",
    "Idefics3ForConditionalGeneration",
    "InternVisionModel",
    "JanusForConditionalGeneration",
    "KimiK25ForConditionalGeneration",
    "KimiVLForConditionalGeneration",
    "Llama4ForConditionalGeneration",
    "LlavaForConditionalGeneration",
    "MiMoV2ForCausalLM",
    "MiniCPMV4_6ForConditionalGeneration",
    "Mistral3ForConditionalGeneration",
    "Phi4ForCausalLMV",
    "Qwen2AudioForConditionalGeneration",
    "Qwen2VLForConditionalGeneration",
    "Qwen2VLModel",
    "Qwen2_5OmniModel",
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLMoeForConditionalGeneration",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
    "SmolVLMForConditionalGeneration",
    "StepVLForConditionalGeneration",
    "UltravoxModel",
    "YoutuVLForConditionalGeneration",
}


def detect_architecture(model_dir: Path) -> tuple[str | None, bool]:
    """
    Read config.json and return (architecture_name, is_mmproj_supported).
    Returns (None, False) if config.json doesn't exist or has no architectures.
    """
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return None, False

    with open(config_path) as f:
        config = json.load(f)

    archs = config.get("architectures", [])
    if not archs:
        return None, False

    arch = archs[0]
    is_vlm = arch in MMPROJ_ARCHITECTURES

    # Also check auto_map — custom architectures use this
    auto_map = config.get("auto_map", {})
    if auto_map and not is_vlm:
        model_type = config.get("model_type", "unknown")
        return arch, False  # custom arch — not supported

    return arch, is_vlm


def check_llama_cpp():
    """Verify required llama.cpp binaries are present."""
    missing = []
    if not LLAMA_QUANTIZE.exists():
        missing.append(str(LLAMA_QUANTIZE))
    if not CONVERT_SCRIPT.exists():
        missing.append(str(CONVERT_SCRIPT))
    if missing:
        print_step("err", "Missing llama.cpp binaries:")
        for m in missing:
            print_step("err", f"  {m}")
        if IS_WINDOWS:
            print_step("info", "Download from: https://github.com/ggerganov/llama.cpp/releases/latest")
        else:
            print_step("info", "Run: python setup_linux.py")
        sys.exit(1)
    print_step("ok", "llama.cpp binaries found")


def download_model(model_id: str) -> Path:
    """Download model from HuggingFace Hub."""
    model_name = model_id.split("/")[-1]
    local_dir = MODELS_DIR / model_name

    if local_dir.exists() and any(local_dir.iterdir()):
        print_step("ok", f"Model already downloaded: {local_dir}")
        return local_dir

    print_step("info", f"Downloading {model_id}...")
    free_gb = shutil.disk_usage(str(MODELS_DIR.parent)).free / (1024 ** 3)
    print_step("info", f"Free disk space: {free_gb:.1f} GB")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            ignore_patterns=["*.bin", "*.pt", "*.msgpack", "flax_model*"],
        )
        print_step("ok", f"Downloaded to: {local_dir}")
        return local_dir
    except Exception as e:
        print_step("err", f"Download failed: {e}")
        sys.exit(1)


def convert_text_backbone(model_dir: Path, no_mtp: bool = False) -> Path:
    """Convert text backbone to F16 GGUF."""
    model_name = model_dir.name
    out_dir = OUTPUT_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}-F16.gguf"

    if out_path.exists() and out_path.stat().st_size > 1024 ** 3:
        print_step("ok", f"F16 GGUF already exists: {out_path.name}")
        return out_path

    if out_path.exists():
        print_step("warn", f"Found incomplete F16 ({get_size(out_path)}). Rebuilding...")
        out_path.unlink()

    print_step("info", "Converting text backbone → F16 GGUF...")
    cmd = [
        sys.executable, str(CONVERT_SCRIPT),
        str(model_dir),
        "--outfile", str(out_path),
        "--outtype", "f16",
    ]
    if no_mtp:
        cmd.append("--no-mtp")
    result = subprocess.run(cmd, text=True, cwd=str(LLAMA_SRC_DIR))
    if result.returncode != 0:
        print_step("err", "Text backbone conversion failed!")
        sys.exit(1)

    print_step("ok", f"Text backbone F16 GGUF: {out_path.name} ({get_size(out_path)})")
    return out_path


def convert_mmproj(model_dir: Path) -> Path:
    """Convert VLM vision encoder to F16 GGUF (the mmproj file)."""
    model_name = model_dir.name
    out_dir = OUTPUT_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # llama.cpp names it mmproj-<model>-f16.gguf (with mmproj- prefix)
    # We rename it to <model>-mmproj-f16.gguf for consistency
    raw_out   = out_dir / f"mmproj-{model_name}-f16.gguf"
    final_out = out_dir / f"{model_name}-mmproj-f16.gguf"

    if final_out.exists() and final_out.stat().st_size > 10 * 1024 * 1024:
        print_step("ok", f"mmproj already exists: {final_out.name}")
        return final_out

    print_step("info", "Converting vision encoder → mmproj F16 GGUF...")
    cmd = [
        sys.executable, str(CONVERT_SCRIPT),
        str(model_dir),
        "--outfile", str(raw_out),
        "--outtype", "f16",
        "--mmproj",
    ]
    result = subprocess.run(cmd, text=True, cwd=str(LLAMA_SRC_DIR))
    if result.returncode != 0:
        print_step("err", "Vision encoder (mmproj) conversion failed!")
        sys.exit(1)

    # Rename to consistent naming convention
    if raw_out.exists() and raw_out != final_out:
        raw_out.rename(final_out)

    print_step("ok", f"mmproj F16 GGUF: {final_out.name} ({get_size(final_out)})")
    return final_out


def quantize_gguf(fp16_path: Path, quant_type: str) -> Path | None:
    """Quantize the text backbone F16 GGUF to a specific quant type."""
    model_name = fp16_path.parent.name
    out_path = fp16_path.parent / f"{model_name}-{quant_type}.gguf"

    if out_path.exists() and out_path.stat().st_size > 100 * 1024 * 1024:
        print_step("ok", f"Already exists: {out_path.name} ({get_size(out_path)})")
        return out_path
    if out_path.exists():
        print_step("warn", f"Incomplete quant found, rebuilding: {out_path.name}")
        out_path.unlink()

    print_step("info", f"Quantizing text backbone → {quant_type}...")
    t0 = time.time()
    cmd = [str(LLAMA_QUANTIZE), str(fp16_path), str(out_path), quant_type]

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(LLAMA_QUANTIZE.parent)

    result = subprocess.run(cmd, text=True, env=env)
    if result.returncode != 0:
        print_step("err", f"Quantization to {quant_type} failed!")
        return None

    elapsed = time.time() - t0
    print_step("ok", f"{quant_type} done in {elapsed:.1f}s → {out_path.name} ({get_size(out_path)})")
    return out_path


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Quantize a VLM to GGUF (2-file output: text + mmproj)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quantize_vlm.py --model Qwen/Qwen2.5-VL-7B-Instruct
  python quantize_vlm.py --model meta-llama/Llama-3.2-11B-Vision-Instruct --preset standard
  python quantize_vlm.py --model google/gemma-3-4b-it --quants Q4_K_M Q8_0
        """,
    )
    parser.add_argument("--model",     "-m", required=True, help="HuggingFace model ID")
    parser.add_argument("--quants",    "-q", nargs="+",     help="Explicit quant types for text backbone")
    parser.add_argument("--preset",    "-p", choices=list(QUANT_PRESETS.keys()), default="standard")
    parser.add_argument("--keep-fp16", action="store_true", help="Keep F16 GGUF after quantization")
    parser.add_argument("--delete-src",action="store_true", help="Delete downloaded model after F16 conversion")
    parser.add_argument("--batch",     "-b", type=int, default=0, help="Process N quants per run")
    parser.add_argument("--skip-preflight", action="store_true",
        help="Skip architecture check before downloading (use if preflight incorrectly blocks)")
    parser.add_argument("--no-mtp", action="store_true", help="Exclude the multi-token prediction (MTP) head from GGUF")
    args = parser.parse_args()

    model_name  = args.model.split("/")[-1]
    all_quants  = args.quants or QUANT_PRESETS[args.preset]
    output_dir  = OUTPUT_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    def is_done(qt): 
        p = output_dir / f"{model_name}-{qt}.gguf"
        return p.exists() and p.stat().st_size > 100 * 1024 * 1024

    pending   = [q for q in all_quants if not is_done(q)]
    completed = [q for q in all_quants if is_done(q)]
    quants_this_run = pending[:args.batch] if args.batch > 0 else pending
    remaining_after = [q for q in pending if q not in quants_this_run]

    print("\n" + "=" * 60)
    print("  quant-kit — VLM GGUF Quantizer (2-file output)")
    print("=" * 60)
    print(f"  Model      : {args.model}")
    print(f"  Output     : {output_dir}")
    print(f"  Quants     : {', '.join(all_quants)}")
    print(f"  Done       : {len(completed)} ({', '.join(completed) or 'none'})")
    print(f"  This run   : {len(quants_this_run)} ({', '.join(quants_this_run) or 'none'})")
    print(f"  mmproj     : Always F16 (vision encoder is never quantized)")
    if remaining_after:
        print(f"  After this : {len(remaining_after)} remaining — re-run to continue")
    print("=" * 60 + "\n")

    if not quants_this_run:
        print_step("ok", "All quants already complete!")
        return

    check_llama_cpp()

    # 🛡️ Pre-flight: check architecture from config.json BEFORE downloading gigabytes
    if not args.skip_preflight:
        from huggingface_hub import hf_hub_download
        import json as _json
        print_step("info", "Pre-flight: checking architecture before download...")
        try:
            _cfg_path = hf_hub_download(
                repo_id=args.model,
                filename="config.json",
                local_dir=str(MODELS_DIR / "_preflight_cache"),
            )
            with open(_cfg_path, encoding="utf-8") as _f:
                _cfg = _json.load(_f)
            _archs = _cfg.get("architectures", [])
            if _archs:
                _arch = _archs[0]
                if _arch in MMPROJ_ARCHITECTURES:
                    print_step("ok", f"Architecture supported (VLM): {_arch}")
                else:
                    # Check if it's at least a text LLM (llama.cpp TEXT models)
                    print()
                    print("  " + "=" * 58)
                    print("  🚫 NOT A SUPPORTED VLM — Download Blocked")
                    print("  " + "=" * 58)
                    print(f"  Model       : {args.model}")
                    print(f"  Architecture: {_arch}")
                    print()
                    print("  This architecture is NOT in quantize_vlm.py's VLM list.")
                    print("  If it's a plain text LLM, use: python quantize.py instead.")
                    print("  If it's a new VLM, update llama-src: git -C llama-src pull")
                    print("  Force download anyway: add --skip-preflight")
                    print("  " + "=" * 58)
                    print()
                    sys.exit(1)
        except SystemExit:
            raise
        except Exception as e:
            print_step("warn", f"Pre-flight config fetch failed: {e} — proceeding")

    # Download
    model_dir = download_model(args.model)

    # Post-download architecture check (informational only now — pre-flight already blocked bad ones)
    arch, is_vlm = detect_architecture(model_dir)
    print()
    if arch:
        print_step("info", f"Detected architecture: {arch}")
    if is_vlm:
        print_step("ok", "Architecture supports --mmproj (VLM confirmed)")
    print()

    # Step 1: Convert text backbone to F16
    fp16_path = convert_text_backbone(model_dir, no_mtp=args.no_mtp)
    print()

    # Step 2: Convert vision encoder (mmproj) — always F16, done once
    mmproj_path = convert_mmproj(model_dir)
    print()

    # Optional: delete source to free disk
    if args.delete_src:
        print_step("info", "--delete-src: removing source model files...")
        freed = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
        shutil.rmtree(str(model_dir))
        print_step("ok", f"Freed {freed/(1024**3):.1f} GB")
        print()

    # Step 3: Quantize text backbone
    created_quants = []
    for i, quant_type in enumerate(quants_this_run, 1):
        print_step("info", f"--- [{i}/{len(quants_this_run)}] {quant_type} ---")
        result = quantize_gguf(fp16_path, quant_type)
        if result:
            created_quants.append(result)
        print()

    # Clean up F16 if all quants across all batches are done
    all_done_now = all(is_done(q) for q in all_quants)
    if all_done_now and not args.keep_fp16 and fp16_path.exists():
        print_step("info", "All quants complete — removing F16 base file...")
        fp16_path.unlink()
        print_step("ok", "F16 removed")
    elif not all_done_now and fp16_path.exists():
        print_step("info", f"F16 kept — {len(remaining_after)} quant(s) still pending. Re-run to continue.")

    print("\n" + "=" * 60)
    print("  ✅ VLM Quantization Complete!")
    print("=" * 60)
    print(f"  📁 Output: {output_dir}")
    print()
    print(f"  📦 Text backbone quants ({len(created_quants)}):")
    for q in created_quants:
        print(f"     • {q.name} ({get_size(q)})")
    print()
    print(f"  🖼️  Vision encoder (mmproj):")
    print(f"     • {mmproj_path.name} ({get_size(mmproj_path)})")
    print()
    print("  ⚠️  Both files are needed to run this VLM in llama.cpp:")
    print(f"     llama-llava-cli -m {model_name}-Q4_K_M.gguf --mmproj {model_name}-mmproj-f16.gguf")
    print()
    print("  Next steps:")
    print(f"     python model_card.py --model {model_name} --original {args.model}")
    print(f"     python upload.py     --model {model_name}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
