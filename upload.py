"""
upload.py — Push quantized models to HuggingFace Hub
======================================================
Uploads all GGUF files + README.md from your output folder
to your HuggingFace account as a new model repository.

Usage:
    python upload.py --model Qwen2.5-1.5B-Instruct --author your-hf-username
    python upload.py --model Qwen2.5-7B-Instruct --author your-hf-username --private

Prerequisites:
    huggingface-cli login    ← run this once to authenticate
"""

import sys
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file
from config import OUTPUT_DIR, HF_TOKEN, verify_token


def print_step(step: str, msg: str):
    colors = {"info": "\033[94m", "ok": "\033[92m", "warn": "\033[93m", "err": "\033[91m"}
    reset = "\033[0m"
    icons = {"info": "→", "ok": "✓", "warn": "⚠", "err": "✗"}
    print(f"{colors.get(step, '')}{icons.get(step, '•')} {msg}{reset}")


def get_files_to_upload(model_dir: Path) -> list[Path]:
    """Get all files to upload: GGUF quants + README."""
    files = []

    # README / model card
    readme = model_dir / "README.md"
    if readme.exists():
        files.append(readme)
    else:
        print_step("warn", "No README.md found — upload will not include model card")
        print_step("info", "Run model_card.py first to generate a model card")

    # GGUF files (skip F16 — too large, not useful for most users)
    gguf_files = sorted(model_dir.glob("*.gguf"))
    gguf_files = [f for f in gguf_files if "F16" not in f.name]

    if not gguf_files:
        print_step("err", "No quantized GGUF files found (F16 excluded)")
        sys.exit(1)

    files.extend(gguf_files)
    return files


def format_size(path: Path) -> str:
    size = path.stat().st_size
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def main():
    parser = argparse.ArgumentParser(description="Upload GGUF quants to HuggingFace Hub")
    parser.add_argument("--model", "-m", required=True, help="Model folder name in output/ (e.g. Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--author", "-a", required=True, help="Your HuggingFace username")
    parser.add_argument("--private", action="store_true", help="Make the repo private (default: public)")
    parser.add_argument("--repo-name", help="Custom repo name (default: <model>-GGUF)")

    args = parser.parse_args()

    model_dir = OUTPUT_DIR / args.model
    if not model_dir.exists():
        print_step("err", f"Output folder not found: {model_dir}")
        print_step("info", "Run quantize.py first")
        sys.exit(1)

    repo_name = args.repo_name or f"{args.model}-GGUF"
    repo_id = f"{args.author}/{repo_name}"

    print("\n" + "="*60)
    print(f"  Uploading to HuggingFace Hub")
    print(f"  Repo   : {repo_id}")
    print(f"  Private: {args.private}")
    print("="*60 + "\n")

    # Check token
    if not verify_token():
        print_step("err", "No HuggingFace token found in .env file!")
        print_step("info", 'Add to .env: hf_token = "hf_xxxx"')
        sys.exit(1)

    api = HfApi(token=HF_TOKEN)

    # Check login
    try:
        user = api.whoami()
        print_step("ok", f"Logged in as: {user['name']}")
    except Exception:
        print_step("err", "Invalid HuggingFace token!")
        print_step("info", "Check your hf_token in .env")
        sys.exit(1)

    # Create repo if it doesn't exist
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
            token=HF_TOKEN,
        )
        print_step("ok", f"Repo ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print_step("err", f"Failed to create repo: {e}")
        sys.exit(1)

    # Gather files
    files = get_files_to_upload(model_dir)

    print()
    print_step("info", f"Files to upload ({len(files)} total):")
    for f in files:
        print(f"    • {f.name} ({format_size(f)})")
    print()

    # Upload each file
    success = []
    failed = []

    for file_path in files:
        print_step("info", f"Uploading {file_path.name}...")
        try:
            upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=file_path.name,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Add {file_path.name}",
                token=HF_TOKEN,
            )
            print_step("ok", f"  {file_path.name} uploaded ({format_size(file_path)})")
            success.append(file_path.name)
        except Exception as e:
            print_step("err", f"  Failed to upload {file_path.name}: {e}")
            failed.append(file_path.name)

    # Summary
    print("\n" + "="*60)
    print(f"  Upload complete!")
    print(f"  ✓ {len(success)} files uploaded")
    if failed:
        print(f"  ✗ {len(failed)} files failed: {', '.join(failed)}")
    print()
    print(f"  View your model: https://huggingface.co/{repo_id}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
