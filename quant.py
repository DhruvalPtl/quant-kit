"""
quant.py — Unified Multi-Modal Quantization Entry Point
=========================================================
One command for all model types. Auto-detects what kind of model
you're quantizing and dispatches to the right script.

Usage:
    # Text LLM
    python quant.py --model google/gemma-3-4b-it

    # VLM (Vision-Language Model)
    python quant.py --model Qwen/Qwen2.5-VL-7B-Instruct

    # Diffusion (Stable Diffusion, FLUX)
    python quant.py --model black-forest-labs/FLUX.1-schnell

    # Whisper ASR
    python quant.py --model openai/whisper-large-v3-turbo

    # Force a specific type (if auto-detection is wrong)
    python quant.py --model some/model --type vlm
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import json
import argparse
import subprocess
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, model_info
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from config import HF_TOKEN

# ── VLM architecture sets (from quantize_vlm.py) ──────────────────────────────
MMPROJ_ARCHITECTURES = {
    "AudioFlamingo3ForConditionalGeneration", "CogVLMForCausalLM",
    "Exaone4_5_ForConditionalGeneration", "Gemma3ForConditionalGeneration",
    "Gemma3nForConditionalGeneration", "Gemma4ForConditionalGeneration",
    "Gemma4UnifiedForConditionalGeneration", "Glm4vForConditionalGeneration",
    "Glm4vMoeForConditionalGeneration", "GlmOcrForConditionalGeneration",
    "HunYuanVLForConditionalGeneration", "Idefics3ForConditionalGeneration",
    "InternVisionModel", "JanusForConditionalGeneration",
    "KimiK25ForConditionalGeneration", "KimiVLForConditionalGeneration",
    "Llama4ForConditionalGeneration", "LlavaForConditionalGeneration",
    "MiMoV2ForCausalLM", "MiniCPMV4_6ForConditionalGeneration",
    "Mistral3ForConditionalGeneration", "Phi4ForCausalLMV",
    "Qwen2AudioForConditionalGeneration", "Qwen2VLForConditionalGeneration",
    "Qwen2VLModel", "Qwen2_5OmniModel", "Qwen2_5_VLForConditionalGeneration",
    "Qwen3VLForConditionalGeneration", "Qwen3VLMoeForConditionalGeneration",
    "Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration",
    "SmolVLMForConditionalGeneration", "StepVLForConditionalGeneration",
    "UltravoxModel", "YoutuVLForConditionalGeneration",
}

# Diffusion pipeline class names
DIFFUSION_PIPELINES = {
    "FluxPipeline", "FluxImg2ImgPipeline",
    "StableDiffusionPipeline", "StableDiffusionXLPipeline",
    "StableDiffusion3Pipeline", "StableDiffusion3Img2ImgPipeline",
    "StableCascadePriorPipeline", "StableCascadeDecoderPipeline",
}

# Whisper architecture names
WHISPER_ARCHITECTURES = {
    "WhisperForConditionalGeneration", "WhisperModel",
}


def detect_model_type(model_id: str) -> tuple[str, str]:
    """
    Detect model type from HuggingFace metadata.
    Returns (type, reason) where type is one of:
      'llm', 'vlm', 'diffusion', 'whisper', 'unknown'
    """
    # Quick heuristic from model ID
    mid = model_id.lower()
    if any(x in mid for x in ["whisper", "distil-whisper"]):
        return "whisper", "model ID contains 'whisper'"
    if any(x in mid for x in ["flux", "stable-diffusion", "sdxl", "/sd-"]):
        return "diffusion", "model ID matches diffusion pattern"

    if not HF_AVAILABLE:
        return "unknown", "huggingface_hub not available for metadata lookup"

    # Fetch config.json from HF
    try:
        config_path = hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            token=HF_TOKEN,
        )
        with open(config_path) as f:
            config = json.load(f)

        archs = config.get("architectures", [])
        pipeline_tag = config.get("pipeline_tag", "")
        model_type = config.get("model_type", "")

        # Check architectures
        for arch in archs:
            if arch in WHISPER_ARCHITECTURES:
                return "whisper", f"architecture: {arch}"
            if arch in MMPROJ_ARCHITECTURES:
                return "vlm", f"architecture: {arch}"

        # Check pipeline_tag from model card
        if "automatic-speech-recognition" in pipeline_tag:
            return "whisper", f"pipeline_tag: {pipeline_tag}"
        if "text-to-image" in pipeline_tag or "image-to-image" in pipeline_tag:
            return "diffusion", f"pipeline_tag: {pipeline_tag}"
        if pipeline_tag in ("text-generation", "text2text-generation"):
            return "llm", f"pipeline_tag: {pipeline_tag}"
        if pipeline_tag == "image-text-to-text":
            return "vlm", f"pipeline_tag: {pipeline_tag}"

        # Check model_index.json for diffusion
        try:
            idx_path = hf_hub_download(repo_id=model_id, filename="model_index.json", token=HF_TOKEN)
            with open(idx_path) as f:
                idx = json.load(f)
            class_name = idx.get("_class_name", "")
            if class_name in DIFFUSION_PIPELINES or "Stable" in class_name or "Flux" in class_name:
                return "diffusion", f"model_index._class_name: {class_name}"
        except Exception:
            pass

        # Has architectures but not in any known set
        if archs:
            return "llm", f"text architecture (assumed): {archs[0]}"

    except Exception as e:
        return "unknown", f"metadata fetch failed: {e}"

    return "unknown", "could not determine type"


MODEL_TYPE_INFO = {
    "llm": {
        "label": "Text LLM",
        "script": "quantize.py",
        "desc": "Standard transformer language model → quantize.py",
    },
    "vlm": {
        "label": "Vision-Language Model (VLM)",
        "script": "quantize_vlm.py",
        "desc": "Multimodal model → quantize_vlm.py (outputs text GGUF + mmproj GGUF)",
    },
    "diffusion": {
        "label": "Diffusion / Flow-Matching Model",
        "script": "quantize_diffusion.py",
        "desc": "Image generation model → quantize_diffusion.py (uses stable-diffusion.cpp)",
    },
    "whisper": {
        "label": "Whisper ASR",
        "script": "quantize_whisper.py",
        "desc": "Audio transcription model → quantize_whisper.py (uses whisper.cpp)",
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="quant-kit: Unified model quantization — auto-detects model type",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quant.py --model google/gemma-3-4b-it
  python quant.py --model Qwen/Qwen2.5-VL-7B-Instruct
  python quant.py --model black-forest-labs/FLUX.1-schnell --quant q4_k
  python quant.py --model openai/whisper-large-v3-turbo
  python quant.py --model some/model --type vlm --preset standard
        """,
    )
    parser.add_argument("--model", "-m",   required=True, help="HuggingFace model ID")
    parser.add_argument("--type",          choices=["llm", "vlm", "diffusion", "whisper"],
                        help="Force model type (skip auto-detection)")
    parser.add_argument("--detect-only",   action="store_true",
                        help="Only detect type, don't run quantization")
    # Pass-through arguments to the underlying script
    parser.add_argument("--preset",  "-p", help="Quant preset (for LLM/VLM: standard/full/imatrix)")
    parser.add_argument("--quants",  "-q", nargs="+", help="Explicit quant types (for LLM/VLM)")
    parser.add_argument("--quant",         help="Single quant type (for diffusion/whisper)")
    parser.add_argument("--batch",   "-b", type=int, help="Batch size (LLM/VLM only)")
    parser.add_argument("--keep-fp16",     action="store_true")
    parser.add_argument("--delete-src",    action="store_true")
    parser.add_argument("--private",       action="store_true", help="Make HF repo private (passed to upload.py)")

    # Diffusion-specific
    parser.add_argument("--model-file", help="Path to .safetensors file (diffusion only)")
    parser.add_argument("--vae",        help="Path to VAE file (diffusion only)")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  🧰 quant-kit — Unified Multi-Modal Quantizer")
    print("=" * 60)
    print(f"  Model: {args.model}")

    # Detect or use forced type
    if args.type:
        model_type = args.type
        reason = "forced via --type"
    else:
        print("  Detecting model type from HuggingFace metadata...")
        model_type, reason = detect_model_type(args.model)

    if model_type == "unknown":
        print(f"\n  ❓ Could not determine model type: {reason}")
        print()
        print("  Use --type to force the type:")
        for k, v in MODEL_TYPE_INFO.items():
            print(f"    --type {k:<12} {v['label']}")
        print()
        print("  If it's a completely custom architecture (like LocateAnything-3B),")
        print("  it is not supported by any llama.cpp / whisper.cpp / sd.cpp backend.")
        sys.exit(1)

    info = MODEL_TYPE_INFO[model_type]
    print(f"  Type:  {info['label']}  [{reason}]")
    print(f"  Tool:  {info['script']}")
    print("=" * 60)

    if args.detect_only:
        print(f"\n  --detect-only: stopping here. Run:")
        print(f"     python {info['script']} --model {args.model}")
        return

    # Build the command for the sub-script
    script = Path(__file__).parent / info["script"]
    cmd = [sys.executable, str(script), "--model", args.model]

    if model_type in ("llm", "vlm"):
        if args.preset:  cmd += ["--preset", args.preset]
        if args.quants:  cmd += ["--quants"] + args.quants
        if args.batch:   cmd += ["--batch", str(args.batch)]
        if args.keep_fp16:  cmd.append("--keep-fp16")
        if args.delete_src: cmd.append("--delete-src")

    elif model_type == "diffusion":
        if args.quant:      cmd += ["--quant", args.quant]
        if args.model_file: cmd += ["--model-file", args.model_file]
        if args.vae:        cmd += ["--vae", args.vae]
        if args.delete_src: cmd.append("--delete-src")

    elif model_type == "whisper":
        if args.quant:      cmd += ["--quant", args.quant]
        if args.keep_fp16:  cmd.append("--keep-fp16")
        if args.delete_src: cmd.append("--delete-src")

    print(f"\n  Launching: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
