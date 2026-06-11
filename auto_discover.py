"""
auto_discover.py — Find trending HuggingFace models that are quantizable and have no GGUF yet
==============================================================================================
Queries the HuggingFace API for trending models, filters by:
  - Architecture supported by your local llama.cpp
  - Small enough to fit on a laptop (configurable max size)
  - No complete GGUF already published by major quantizers

Usage (standalone):
    python auto_discover.py
    python auto_discover.py --max-gb 10 --min-downloads 5000 --count 5
    python auto_discover.py --task image-text-to-text   (VLMs)
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import json
import re
import time
import urllib.request
import urllib.parse
import argparse
from pathlib import Path
from typing import Any

from config import CONVERT_SCRIPT, MODELS_DIR, HF_TOKEN
from utils import print_step

# ─── Known GGUF publishers to check against ───────────────────────────────────
# If ANY of these have already published a GGUF → model is saturated
MAJOR_QUANTIZERS = {
    "bartowski", "mradermacher", "unsloth", "QuantFactory",
    "TheBloke", "lmstudio-community", "mlx-community",
    "second-state", "MaziyarPanahi",
}

# ─── MMPROJ architectures (VLM) ───────────────────────────────────────────────
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

# ─── Helpers ──────────────────────────────────────────────────────────────────

def hf_api(url: str, token: str | None = None) -> Any:
    """Fallback REST API fetcher"""
    headers = {"User-Agent": "quant-kit/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    import urllib.request
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            import json
            return json.loads(resp.read())
    except Exception:
        return None


def get_supported_architectures() -> set[str]:
    """Dynamically read supported architectures by calling convert_hf_to_gguf.py."""
    if not CONVERT_SCRIPT.exists():
        return set()
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, str(CONVERT_SCRIPT), "--print-supported-models"],
            capture_output=True, text=True
        )
        # Some output goes to stdout, some to stderr
        out = result.stdout + result.stderr
        # Match lines like "  - ModelNameForCausalLM"
        names = re.findall(r"-\s+(\w+)", out)
        return set(names)
    except Exception as e:
        print_step("warn", f"Could not extract architectures: {e}")
        return set()


def get_model_size_gb(model_info: dict) -> float | None:
    """
    Estimate download size in GB from safetensors metadata in the API response.
    Returns None if unknown.
    """
    st = model_info.get("safetensors")
    if st:
        total_params = st.get("total", 0)
        # BF16 = 2 bytes/param → rough download size
        # Most models are BF16 or FP16
        size_bytes = total_params * 2
        return size_bytes / (1024 ** 3)

    # Fallback: count .safetensors siblings
    siblings = model_info.get("siblings", [])
    total = sum(
        s.get("size", 0) for s in siblings
        if s.get("rfilename", "").endswith(".safetensors")
    )
    if total > 0:
        return total / (1024 ** 3)
    return None


def check_gguf_exists(model_name: str, token: str | None = None) -> tuple[bool, list[str]]:
    """
    Search HuggingFace for existing GGUF repos for this model.
    Returns (is_saturated, [existing_repo_ids])
    """
    # Search by model name + GGUF
    encoded = urllib.parse.quote(f"{model_name} GGUF")
    url = f"https://huggingface.co/api/models?search={encoded}&library=gguf&limit=10"
    results = hf_api(url, token)

    if not results:
        return False, []

    existing = [r.get("id", "") for r in results if r.get("id")]
    # Check if any major quantizer has done it
    saturated = any(
        any(q.lower() in repo.lower() for q in MAJOR_QUANTIZERS)
        for repo in existing
    )
    return saturated, existing


def score_model(model_info: dict, gguf_count: int) -> float:
    """
    Score a candidate model. Higher = better opportunity.
    Formula: downloads + (likes * 100) - (gguf_repos * 5000)
    """
    downloads = model_info.get("downloads", 0) or 0
    likes     = model_info.get("likes", 0) or 0
    score     = downloads + (likes * 100) - (gguf_count * 5000)
    return score


def discover(
    task: str = "text-generation",
    max_gb: float = 15.0,
    min_downloads: int = 500,
    count: int = 5,
    token: str | None = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Main discovery function. Returns list of candidate dicts, best first.

    Each dict has:
      model_id, model_name, architecture, model_type, size_gb,
      downloads, likes, gguf_repos, score
    """
    supported_archs = get_supported_architectures()
    if not supported_archs:
        print_step("warn", "Could not read supported architectures from convert_hf_to_gguf.py")

    if verbose:
        print_step("info", f"Querying HuggingFace trending models (task={task})...")

    from huggingface_hub import HfApi
    api = HfApi(token=token)

    try:
        # Fetch top trending models
        models = list(api.list_models(
            sort="trendingScore", 
            pipeline_tag=task,
            limit=200
        ))
    except Exception as e:
        print_step("err", f"Failed to fetch trending models: {e}")
        return []

    if verbose:
        print_step("ok", f"Got {len(models)} trending models, filtering...")

    candidates = []
    checked = 0

    for m in models:
        model_id   = m.id
        model_name = model_id.split("/")[-1]
        downloads  = getattr(m, "downloads", 0) or 0
        likes      = getattr(m, "likes", 0) or 0

        # ── Filter 1: minimum download threshold ─────────────────────────────
        if downloads < min_downloads:
            continue

        # ── Filter 2: skip if already GGUF (repo name contains GGUF) ─────────
        if "gguf" in model_name.lower() or "gguf" in model_id.lower():
            continue

        # ── Filter 3: check if GGUF already well-covered ──────────────────────
        checked += 1
        saturated, existing_repos = check_gguf_exists(model_name, token)
        gguf_count = len(existing_repos)

        if saturated:
            continue

        # ── Filter 4: Fetch full info to get size and architecture ─────────────
        if verbose:
            print(f"    🔍 [{checked}] {model_id} — inspecting details...")
            
        try:
            info = api.model_info(model_id)
        except Exception:
            continue

        # Size check
        size_gb = 0.0
        if info.safetensors and info.safetensors.get("total"):
            size_gb = (info.safetensors["total"] * 2) / (1024 ** 3)
        elif info.siblings:
            total_bytes = sum(s.size or 0 for s in info.siblings if s.rfilename.endswith(".safetensors"))
            size_gb = total_bytes / (1024 ** 3)
            
        if size_gb > max_gb:
            if verbose:
                print(f"    ⏭  {model_id} ({size_gb:.1f} GB) — too large")
            continue

        # Architecture check
        archs = getattr(info, "config", {}).get("architectures", []) if getattr(info, "config", None) else []
        if not archs:
            if verbose:
                print(f"    ⏭  {model_id} — no architecture info")
            continue

        arch = archs[0]

        if arch in MMPROJ_ARCHITECTURES:
            model_type = "vlm"
        elif supported_archs and arch in supported_archs:
            model_type = "llm"
        elif not supported_archs:
            model_type = "llm"
        else:
            if verbose:
                print(f"    ❌ {model_id} — unsupported arch: {arch}")
            continue

        # ── Passed all filters — compute score ────────────────────────────────
        s = score_model({"downloads": downloads, "likes": likes}, gguf_count)
        candidates.append({
            "model_id":   model_id,
            "model_name": model_name,
            "architecture": arch,
            "model_type": model_type,
            "size_gb":    round(size_gb, 2),
            "downloads":  downloads,
            "likes":      likes,
            "gguf_repos": gguf_count,
            "gguf_existing": existing_repos,
            "score":      s,
        })

        if verbose:
            gap = "🟢 ZERO GGUF" if gguf_count == 0 else f"🟡 {gguf_count} partial"
            print(f"         ✅ CANDIDATE! {gap} | {downloads:,} downloads | {likes} likes | score={s:,.0f}")

        if len(candidates) >= count:
            break

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Find trending HuggingFace models that are quantizable with no GGUF yet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python auto_discover.py
  python auto_discover.py --max-gb 8 --count 10
  python auto_discover.py --task image-text-to-text --max-gb 12
  python auto_discover.py --min-downloads 10000 --count 3
        """,
    )
    parser.add_argument("--task",          default="text-generation",
                        help="HuggingFace pipeline_tag to search (default: text-generation)")
    parser.add_argument("--max-gb",        type=float, default=15.0,
                        help="Max model download size in GB (default: 15.0)")
    parser.add_argument("--min-downloads", type=int,   default=500,
                        help="Minimum downloads to consider (default: 500)")
    parser.add_argument("--count",         type=int,   default=5,
                        help="How many candidates to return (default: 5)")
    parser.add_argument("--json",          action="store_true",
                        help="Output results as JSON (for autopilot.py)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  🔍 quant-kit Auto-Discover")
    print("=" * 60)
    print(f"  Task:          {args.task}")
    print(f"  Max size:      {args.max_gb} GB")
    print(f"  Min downloads: {args.min_downloads:,}")
    print(f"  Want:          {args.count} candidates")
    print("=" * 60 + "\n")

    candidates = discover(
        task=args.task,
        max_gb=args.max_gb,
        min_downloads=args.min_downloads,
        count=args.count,
        token=HF_TOKEN,
        verbose=not args.json,
    )

    if args.json:
        print(json.dumps(candidates, indent=2))
        return

    if not candidates:
        print_step("warn", "No suitable candidates found. Try --max-gb 20 or --min-downloads 100")
        return

    print("\n" + "=" * 60)
    print(f"  🏆 Top {len(candidates)} Candidates")
    print("=" * 60)
    for i, c in enumerate(candidates, 1):
        gap = "🟢 ZERO GGUF" if c["gguf_repos"] == 0 else f"🟡 {c['gguf_repos']} partial"
        tag = "VLM" if c["model_type"] == "vlm" else "LLM"
        print(f"  {i}. [{tag}] {c['model_id']}")
        print(f"     Size: {c['size_gb']:.1f} GB | Downloads: {c['downloads']:,} | Likes: {c['likes']}")
        print(f"     Arch: {c['architecture']} | GGUF: {gap}")
        if c["gguf_existing"]:
            print(f"     Existing: {', '.join(c['gguf_existing'][:2])}")
        print()

    print("  Run autopilot to quantize all of these automatically:")
    print("  python autopilot.py --count 3")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
