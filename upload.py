"""
upload.py — Push quantized models to HuggingFace Hub
======================================================
Supports all model types: LLM, VLM (text GGUF + mmproj), Diffusion, Whisper.
Auto-detects model type from the output folder.

Usage:
    python upload.py --model Qwen2.5-1.5B-Instruct
    python upload.py --model Qwen2.5-VL-3B-Instruct           # auto-detects VLM
    python upload.py --model gemma-4-12b-it --private
    python upload.py --model my-model --repo-name custom-name-GGUF
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo, CommitOperationAdd
from config import OUTPUT_DIR, HF_TOKEN, verify_token
from utils import print_step, format_size


def detect_model_type(model_dir: Path) -> str:
    """Detect model type from output folder contents."""
    names = [f.name for f in model_dir.glob("*.gguf")]
    if any("mmproj" in n for n in names):
        return "vlm"
    if any("ggml" in n.lower() for n in names):
        return "whisper"
    return "llm"


def get_files_to_upload(model_dir: Path, model_type: str, include_fp16: bool = False) -> list[Path]:
    """Return ordered list of files to upload — README first, then GGUFs."""
    files: list[Path] = []

    # Always include README and benchmark JSONs first
    for meta_file in ["README.md", "benchmark.json", "quality_benchmark.json",
                       "kaggle_results_Q4_K_M.json", "vastai_results_Q4_K_M.json"]:
        p = model_dir / meta_file
        if p.exists():
            files.append(p)

    all_guufs = sorted(model_dir.glob("*.gguf"))

    if model_type == "vlm":
        # VLM: always include mmproj (F16, required), plus quantized text backbone
        mmproj = [f for f in all_guufs if "mmproj" in f.name]
        text   = [f for f in all_guufs if "mmproj" not in f.name]

        if not mmproj:
            print_step("warn", "No mmproj file found! VLM users need it to run the model.")

        if include_fp16:
            files.extend(text)
        else:
            quant_text = [f for f in text if "F16" not in f.name]
            if not quant_text:
                print_step("err", "No quantized text GGUFs found (use --include-fp16 to upload F16)")
                sys.exit(1)
            files.extend(quant_text)

        # mmproj always uploaded (it's always F16 but it's small and required)
        files.extend(mmproj)

    else:
        # LLM / Diffusion / Whisper: standard logic
        if include_fp16:
            files.extend(all_guufs)
        else:
            quant_files = [f for f in all_guufs if "F16" not in f.name]
            if not quant_files:
                print_step("err", "No quantized GGUF files found (use --include-fp16 to upload F16)")
                sys.exit(1)
            files.extend(quant_files)

    return files


def main():
    parser = argparse.ArgumentParser(
        description="Upload GGUF quants to HuggingFace Hub (supports LLM, VLM, Diffusion, Whisper)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model",       "-m", required=True, help="Model folder name in output/")
    parser.add_argument("--author",      "-a", help="HuggingFace username (auto-detected if omitted)")
    parser.add_argument("--private",     action="store_true", help="Make repo private")
    parser.add_argument("--repo-name",         help="Custom repo name (default: <model>-GGUF)")
    parser.add_argument("--include-fp16",action="store_true",
                        help="Also upload F16 GGUF (lossless, ~3x larger)")
    parser.add_argument("--type",        choices=["llm", "vlm", "diffusion", "whisper"],
                        help="Force model type (default: auto-detect)")
    args = parser.parse_args()

    model_dir = OUTPUT_DIR / args.model
    if not model_dir.exists():
        print_step("err", f"Output folder not found: {model_dir}")
        sys.exit(1)

    if not verify_token():
        print_step("err", "No HuggingFace token found in .env file!")
        sys.exit(1)

    api = HfApi(token=HF_TOKEN)
    try:
        user   = api.whoami()
        author = args.author or user["name"]
        print_step("ok", f"Logged in as: {user['name']} (uploading to {author})")
    except Exception:
        print_step("err", "Invalid HuggingFace token!")
        sys.exit(1)

    model_type = args.type or detect_model_type(model_dir)
    print_step("info", f"Model type: {model_type.upper()}")

    repo_name = args.repo_name or f"{args.model}-GGUF"
    repo_id   = f"{author}/{repo_name}"

    print("\n" + "=" * 60)
    print("  📤 Uploading to HuggingFace Hub")
    print("=" * 60)
    print(f"  Model type : {model_type.upper()}")
    print(f"  Repo       : {repo_id}")
    print(f"  Private    : {args.private}")
    print("=" * 60 + "\n")

    # Create or verify repo
    try:
        create_repo(repo_id=repo_id, repo_type="model", private=args.private,
                    exist_ok=True, token=HF_TOKEN)
        print_step("ok", f"Repo ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print_step("err", f"Failed to create repo: {e}")
        sys.exit(1)

    files = get_files_to_upload(model_dir, model_type, include_fp16=args.include_fp16)

    print()
    print_step("info", f"Files to upload ({len(files)} total):")
    total_bytes = 0
    for f in files:
        size = f.stat().st_size
        total_bytes += size
        tag = " 🖼️ mmproj (required)" if "mmproj" in f.name else ""
        print(f"    • {f.name} ({format_size(size)}){tag}")
    print(f"\n  Total: {format_size(total_bytes)}")
    print()

    if model_type == "vlm":
        mmproj = [f for f in files if "mmproj" in f.name]
        if mmproj:
            print_step("info", "VLM upload: both text backbone(s) AND mmproj will be uploaded.")
            print_step("info", "Users need BOTH to run this model with llama.cpp / LM Studio.")
            print()

    print_step("info", "Uploading (this may take a while for large files)...")

    # Upload in batches of 5 to avoid timeouts on large uploads
    BATCH_SIZE = 5
    all_ops = [CommitOperationAdd(path_in_repo=f.name, path_or_fileobj=str(f)) for f in files]

    for i in range(0, len(all_ops), BATCH_SIZE):
        batch = all_ops[i:i+BATCH_SIZE]
        batch_files = files[i:i+BATCH_SIZE]
        print_step("info", f"Batch {i//BATCH_SIZE + 1}/{(len(all_ops)+BATCH_SIZE-1)//BATCH_SIZE}: "
                   f"{', '.join(f.name for f in batch_files)}")
        try:
            api.create_commit(
                repo_id=repo_id,
                repo_type="model",
                operations=batch,
                commit_message=f"Upload {model_type.upper()} GGUF quants via quant-kit (batch {i//BATCH_SIZE+1})",
                token=HF_TOKEN,
            )
            print_step("ok", f"Batch uploaded successfully")
        except Exception as e:
            print_step("err", f"Upload failed: {e}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("  ✅ Upload Complete!")
    print("=" * 60)
    print(f"  🔗 View your model: https://huggingface.co/{repo_id}")
    if model_type == "vlm":
        print()
        print("  VLM usage instructions for your users:")
        text_q4 = next((f.name for f in files if "Q4_K_M" in f.name and "mmproj" not in f.name), "model-Q4_K_M.gguf")
        mmproj  = next((f.name for f in files if "mmproj" in f.name), "model-mmproj-f16.gguf")
        print(f"  huggingface-cli download {repo_id} {text_q4} {mmproj}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
