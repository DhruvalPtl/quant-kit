"""
upload.py — Push quantized models to HuggingFace Hub
======================================================
Uploads all GGUF files + README.md from your output folder
to your HuggingFace account as a new model repository.

Usage:
    python upload.py --model Qwen2.5-1.5B-Instruct
    python upload.py --model Qwen2.5-7B-Instruct --private
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo, CommitOperationAdd
from config import OUTPUT_DIR, HF_TOKEN, verify_token
from utils import print_step, format_size

def get_files_to_upload(model_dir: Path, include_fp16: bool = False) -> list[Path]:
    files = []
    
    for meta_file in ["README.md", "benchmark.json", "quality_benchmark.json"]:
        p = model_dir / meta_file
        if p.exists():
            files.append(p)

    gguf_files = sorted(model_dir.glob("*.gguf"))

    if include_fp16:
        # Include all GGUFs including F16
        files.extend(gguf_files)
        f16 = [f for f in gguf_files if "F16" in f.name]
        if f16:
            print_step("info", f"F16 file will be uploaded: {f16[0].name} ({format_size(f16[0].stat().st_size)})")
    else:
        # Default: skip F16 (too large for most users to download)
        quant_files = [f for f in gguf_files if "F16" not in f.name]
        if not quant_files:
            print_step("err", "No quantized GGUF files found (use --include-fp16 to upload F16)")
            sys.exit(1)
        files.extend(quant_files)

    return files

def main():
    parser = argparse.ArgumentParser(description="Upload GGUF quants to HuggingFace")
    parser.add_argument("--model", "-m", required=True, help="Model folder name in output/")
    parser.add_argument("--author", "-a", help="HuggingFace username (auto-detected if omitted)")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    parser.add_argument("--repo-name", help="Custom repo name (default: <model>-GGUF)")
    parser.add_argument("--include-fp16", action="store_true",
        help="Also upload the F16 GGUF (lossless quality, ~24GB — for power users)")

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
        user = api.whoami()
        author = args.author or user['name']
        print_step("ok", f"Logged in as: {user['name']} (uploading to {author})")
    except Exception:
        print_step("err", "Invalid HuggingFace token!")
        sys.exit(1)

    repo_name = args.repo_name or f"{args.model}-GGUF"
    repo_id = f"{author}/{repo_name}"

    print("\n" + "="*60)
    print(f"  Uploading to HuggingFace Hub")
    print(f"  Repo   : {repo_id}")
    print("="*60 + "\n")

    try:
        create_repo(repo_id=repo_id, repo_type="model", private=args.private, exist_ok=True, token=HF_TOKEN)
        print_step("ok", f"Repo ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print_step("err", f"Failed to create repo: {e}")
        sys.exit(1)

    files = get_files_to_upload(model_dir, include_fp16=args.include_fp16)

    print()
    print_step("info", f"Files to upload ({len(files)} total):")
    for f in files:
        print(f"    • {f.name} ({format_size(f.stat().st_size)})")
    print()

    print_step("info", "Uploading files (this may take a while)...")
    operations = [
        CommitOperationAdd(path_in_repo=f.name, path_or_fileobj=str(f))
        for f in files
    ]

    try:
        api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            operations=operations,
            commit_message=f"Upload {len(files)} files via quant-kit",
            token=HF_TOKEN
        )
        print_step("ok", "Upload complete!")
    except Exception as e:
        print_step("err", f"Failed to upload files: {e}")
        sys.exit(1)

    print("\n" + "="*60)
    print(f"  View your model: https://huggingface.co/{repo_id}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
