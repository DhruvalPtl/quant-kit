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
import sys
import time
import argparse
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download

from config import CONVERT_SCRIPT, MODELS_DIR, HF_TOKEN
from utils import print_step

# ─── Known GGUF publishers to check against ───────────────────────────────────
# If ANY of these have already published a GGUF → model is saturated
MAJOR_QUANTIZERS = {
    "bartowski", "mradermacher", "unsloth", "QuantFactory",
    "TheBloke", "lmstudio-community", "mlx-community",
    "second-state", "MaziyarPanahi", "ggml-org",
}

# ── Orgs that only publish test/derivative models (never original weights) ────
SKIP_ORGS = {
    "trl-internal-testing", "peft-internal-testing",
    "hf-internal-testing", "Narsil", "fxmarty",
    "unsloth",          # derivative (fine-tunes + quants, not originals)
    "mlx-community",    # MLX-converted, not original weights
    "casperhansen",     # AWQ variants only
    "hugging-quants",   # quantized variants only
    "lmstudio-community",  # derivative
    "second-state",     # GGUF variants
}

# ── Keywords in model NAME that indicate it's NOT original weights ─────────────
SKIP_NAME_KEYWORDS = {
    # Quantization formats (not original FP16/BF16 weights)
    "awq", "gptq", "fp8", "fp4", "int4", "int8",
    "bnb", "4bit", "8bit", "nvfp4", "mxfp4", "w4a16",
    # Derivative types
    "mlx", "gguf", "ggml", "exl2", "exllamav2",
    # Test/random models
    "tiny", "random", "test", "dummy", "mock", "example",
    # Fine-tune indicators that pollute results
    "abliterated", "uncensored",
}

# ── Specific model IDs that are known to fail conversion ──────────────────
SKIP_MODEL_IDS = {
    # Broken tokenizers / bad configs
    "openai-community/gpt2",
    "openai-community/gpt2-medium",
    "openai-community/gpt2-large",
    "openai-community/gpt2-xl",
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "openai/gpt-oss-20b", # uses mxfp4 which is disabled
    # Oversaturated classics (TheBloke has 100+ variants)
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "tiiuae/falcon-7b",
    "bigscience/bloom-560m",
    "bigscience/bloom",
    "EleutherAI/gpt-j-6b",
    "EleutherAI/pythia-2.8b",
    # Gated models requiring Meta license approval
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-8B",
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

def _hf_headers(token: str | None = None) -> dict:
    h = {"User-Agent": "quant-kit/1.0"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def hf_get(params: dict, token: str | None = None) -> list | None:
    """GET https://huggingface.co/api/models with given params. Returns list or None."""
    try:
        resp = requests.get(
            "https://huggingface.co/api/models",
            params=params,
            headers=_hf_headers(token),
            timeout=20,
        )
        if resp.ok:
            return resp.json()
        return None
    except Exception:
        return None


def get_supported_architectures() -> set[str]:
    """
    Get supported architectures by running the converter's --print-supported-models.
    This is the authoritative source — no regex parsing needed.
    """
    if not CONVERT_SCRIPT.exists():
        return set()
    try:
        import subprocess as _sp
        result = _sp.run(
            [sys.executable, str(CONVERT_SCRIPT), "--print-supported-models"],
            capture_output=True, text=True, timeout=30,
        )
        # Parse lines like "  - Qwen3ForCausalLM"
        combined = result.stdout + result.stderr
        names = re.findall(r"^\s+-\s+(\w+)$", combined, re.MULTILINE)
        return set(names)
    except Exception:
        return set()


def fetch_architecture(model_id: str, token: str | None = None) -> str | None:
    """
    Download only config.json (~2KB) to get the architecture.
    Returns None if can't access, 'GATED' if repo is restricted (403).
    """
    try:
        cfg_path = hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            local_dir=str(MODELS_DIR / "_preflight_cache"),
            token=token,
        )
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        archs = cfg.get("architectures", [])
        return archs[0] if archs else None
    except Exception as e:
        err = str(e)
        if "403" in err or "gated" in err.lower() or "restricted" in err.lower():
            return "GATED"
        return None


def check_gguf_coverage(
    model_name: str,
    my_username: str,
    token: str | None = None,
) -> dict:
    """
    Check GGUF coverage for a model.
    Returns dict with:
      - already_mine: bool   (you already published this — skip it)
      - gguf_count: int      (how many external GGUF repos exist)
      - existing: list[str]  (repo IDs found)
      - has_major: bool      (bartowski/TheBloke etc. have it)
    """
    results = hf_get(
        {"search": f"{model_name} GGUF", "limit": 20, "full": "false"},
        token=token,
    )
    if not results:
        return {"already_mine": False, "gguf_count": 0,
                "existing": [], "has_major": False}

    model_lower = model_name.lower()
    exact_matches = [
        r.get("id", "") for r in results
        if model_lower in r.get("id", "").lower()
    ]

    # Check if YOU already have it published (Direct API lookup bypassing search index delay)
    my_repo = f"{my_username}/{model_name}-GGUF".lower()
    already_mine = any(r.lower() == my_repo for r in exact_matches)
    
    if not already_mine:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            api.model_info(f"{my_username}/{model_name}-GGUF")
            already_mine = True
        except Exception:
            pass

    # Check if major quantizers have it
    has_major = any(
        any(q.lower() in repo.lower() for q in MAJOR_QUANTIZERS)
        for repo in exact_matches
    )

    return {
        "already_mine": already_mine,
        "gguf_count":   len(exact_matches),
        "existing":     exact_matches,
        "has_major":    has_major,
    }


def score_model(model_info: dict, coverage: dict) -> float:
    """
    Score a candidate. Higher = better opportunity for YOUR repo.

    Tiers:
      ZERO GGUF anywhere     → massive bonus  (first mover!)
      Has GGUFs but no major → good opportunity
      Has major quantizer    → still worth doing for repo growth
    """
    downloads = model_info.get("downloads", 0) or 0
    likes     = model_info.get("likes", 0) or 0
    n         = coverage["gguf_count"]
    has_major = coverage["has_major"]

    base  = downloads + (likes * 100)
    # Penalty per existing GGUF repo (diminishing opportunity)
    penalty = n * 2000
    # Bonus for first-mover
    bonus = 50_000 if n == 0 else (10_000 if not has_major else 0)
    return base + bonus - penalty


def discover(
    task: str = "text-generation",
    max_gb: float = 15.0,
    min_downloads: int = 500,
    min_likes: int = 5,
    my_username: str = "Dhptl",
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
    else:
        print_step("ok", f"Loaded {len(supported_archs)} supported architectures from llama.cpp")

    if verbose:
        print_step("info", f"Querying HuggingFace models (task={task}, sort=downloads+likes)...")

    # Sort by likes — much more reliable than downloads.
    # Downloads are inflated by CI/CD bots and pip install tutorials.
    # Likes = real human interest in the model.
    models = hf_get({
        "sort":     "likes",
        "limit":    200,
        "pipeline_tag": task,
        "full":     "true",
    }, token=token)

    if not models:
        print_step("err", "Failed to fetch models from HuggingFace API")
        return []

    if verbose:
        print_step("ok", f"Got {len(models)} models, filtering for laptop-friendly + no GGUF...")

    candidates = []
    checked = 0

    for model_info in models:
        model_id   = model_info.get("id", "")
        model_name = model_id.split("/")[-1]
        downloads  = model_info.get("downloads", 0) or 0
        likes      = model_info.get("likes", 0) or 0

        # ── Filter 1: minimum download threshold ────────────────────────────
        if downloads < min_downloads:
            continue

        # ── Filter 2: known-bad specific model IDs ───────────────────────────
        if model_id in SKIP_MODEL_IDS:
            continue

        # ── Filter 3: skip bad orgs (test repos, derivative-only publishers) ─
        org = model_id.split("/")[0] if "/" in model_id else ""
        if org in SKIP_ORGS:
            continue

        # ── Filter 4: skip quantized/derivative model names ──────────────────
        name_lower = model_name.lower()
        if any(kw in name_lower for kw in SKIP_NAME_KEYWORDS):
            continue

        # ── Filter 5: skip if already GGUF in repo name ──────────────────────
        if "gguf" in name_lower or "gguf" in model_id.lower():
            continue

        # ── Filter 6: require minimum likes (weeds out test/junk repos) ──────
        if likes < min_likes:
            continue

        # ── Filter 6: size must fit on laptop ─────────────────────────────────
        st = model_info.get("safetensors") or {}
        total_params = st.get("total", 0) if isinstance(st, dict) else 0
        size_gb = (total_params * 2) / (1024 ** 3) if total_params else 0.0

        if size_gb > 0 and max_gb and size_gb > max_gb:
            continue  # too large

        # ── Filter 4: fetch architecture (config.json, ~2KB only) ───────────
        checked += 1
        tag_prefix = f"[{checked:>3}]"
        if verbose:
            sz_label = f"{size_gb:.1f} GB" if size_gb else "? GB"
            print(f"    {tag_prefix} {model_id} ({sz_label}, {downloads:,} dl)", end=" ", flush=True)

        arch = fetch_architecture(model_id, token)
        if arch == "GATED":
            if verbose: print("-> gated/restricted repo, skip")
            continue
        if not arch:
            if verbose: print("-> no arch in config, skip")
            continue

        # Also skip if API explicitly marks model as gated
        if model_info.get("gated") and model_info.get("gated") != False:
            if verbose: print("-> gated model, skip")
            continue

        # Determine model type
        if arch in MMPROJ_ARCHITECTURES:
            model_type = "vlm"
        elif supported_archs and arch in supported_archs:
            model_type = "llm"
        elif not supported_archs:
            model_type = "llm"  # unknown supported list — allow (fail-open)
        else:
            if verbose: print(f"-> unsupported arch ({arch}), skip")
            continue

        # ── Verify actual size if missing from safetensors dict (e.g. DeepSeek-R1) ──
        if size_gb == 0.0:
            if verbose: print("-> checking real size...", end=" ", flush=True)
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=token)
                info = api.model_info(model_id, files_metadata=True)
                total_bytes = sum(f.size for f in info.siblings if f.size is not None and (f.rfilename.endswith('.safetensors') or f.rfilename.endswith('.bin')))
                size_gb = total_bytes / (1024 ** 3)
                if max_gb and size_gb > max_gb:
                    if verbose: print(f"too large ({size_gb:.1f} GB > {max_gb} GB), skip")
                    continue
                if verbose: print(f"({size_gb:.1f} GB)", end=" ")
            except Exception:
                pass

        # ── Check GGUF coverage — only skip if YOU already published it ─────
        if verbose: print(f"-> {model_type.upper()} [{arch}] checking GGUF...", end=" ", flush=True)

        coverage = check_gguf_coverage(model_name, my_username, token)

        # Only real skip: you already have this model published
        if coverage["already_mine"]:
            if verbose: print(f"ALREADY IN YOUR REPO -> skip")
            continue

        # ── Passed all filters — compute score ───────────────────────────────
        s = score_model(model_info, coverage)
        gguf_count = coverage["gguf_count"]

        # Determine opportunity tier for display
        if gguf_count == 0:
            tier = "[ZERO GGUF] FIRST MOVER!"
        elif not coverage["has_major"]:
            tier = f"[{gguf_count} small repos] OPEN GAP"
        else:
            major_names = [r.split('/')[0] for r in coverage['existing']
                           if any(q.lower() in r.lower() for q in MAJOR_QUANTIZERS)][:2]
            tier = f"[{gguf_count} repos incl. {', '.join(major_names)}] BUILD YOUR REPO"

        candidates.append({
            "model_id":      model_id,
            "model_name":    model_name,
            "architecture":  arch,
            "model_type":    model_type,
            "size_gb":       round(size_gb, 2),
            "downloads":     downloads,
            "likes":         likes,
            "gguf_repos":    gguf_count,
            "gguf_existing": coverage["existing"],
            "has_major":     coverage["has_major"],
            "score":         s,
        })

        if verbose:
            print(f"{tier} -> score={s:,.0f}")

        time.sleep(0.4)

    # Sort by score descending
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:count]


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
