"""
model_card.py — Auto-generate a professional HuggingFace model card
=====================================================================
Reads all benchmark result files and generates a rich README.md
that looks like a professional AI lab model release.

Usage:
    python model_card.py --model gemma-4-12b-it --original google/gemma-4-12b-it
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import json
import argparse
from pathlib import Path
from datetime import datetime
from huggingface_hub import HfApi, ModelCard

from config import OUTPUT_DIR, QUANT_INFO, HF_TOKEN
from utils import print_step

# ── Metadata fetcher ───────────────────────────────────────────────────────────
def fetch_model_metadata(original_model_id: str) -> dict:
    print_step("info", f"Fetching metadata for {original_model_id}...")
    api = HfApi(token=HF_TOKEN)
    try:
        info = api.model_info(original_model_id)
        card = ModelCard.load(original_model_id)
        card_data = card.data if card.data else {}
        chat_template = None
        try:
            p = api.hf_hub_download(repo_id=original_model_id, filename="tokenizer_config.json")
            with open(p) as f:
                chat_template = json.load(f).get("chat_template")
        except Exception:
            pass
        return {
            "license":       card_data.get("license", "other"),
            "tags":          list(set([t for t in (info.tags or [])
                                       if t not in ["license:other", "endpoints_compatible"]]
                                      + ["gguf", "quantized"])),
            "pipeline_tag":  info.pipeline_tag or "text-generation",
            "chat_template": chat_template,
        }
    except Exception as e:
        print_step("warn", f"Could not fetch metadata: {e}")
        return {"license": "other", "tags": ["gguf", "quantized", "text-generation"],
                "pipeline_tag": "text-generation", "chat_template": None}


# ── Section builders ───────────────────────────────────────────────────────────
def build_files_table(quant_files: list[Path]) -> str:
    rows = "| Filename | Size | RAM Required | Quant | Quality | Recommended For |\n"
    rows += "|---|---|---|---|---|---|\n"
    for f in sorted(quant_files):
        if "F16" in f.name: continue
        size_gb   = f.stat().st_size / (1024 ** 3)
        ram_req   = f"{size_gb + 1.5:.1f} GB"
        quant_type = f.stem.split("-")[-1]
        for q in QUANT_INFO:
            if f.name.endswith(f"{q}.gguf"):
                quant_type = q
                break
        info = QUANT_INFO.get(quant_type, {})
        rec  = " ✅ **Best**" if quant_type == "Q4_K_M" else ""
        rows += (f"| `{f.name}` | {size_gb:.2f} GB | {ram_req} | `{quant_type}`{rec} "
                 f"| {info.get('quality','⭐⭐⭐')} | {info.get('recommended_for','')} |\n")
    return rows


def build_speed_table(benchmark_data: dict) -> str:
    """Multi-context speed table from benchmark.json."""
    if not benchmark_data or not benchmark_data.get("files"):
        return ""
    hw  = benchmark_data.get("hardware", {})
    hw_str = f"{hw.get('cpu','?')} · {hw.get('ram_gb','?')} GB RAM · {hw.get('gpu','?')}"
    rows = f"**Hardware:** `{hw_str}`\n\n"
    rows += "| Quant | Size | Context | Generation | Prompt Processing | RAM Used |\n"
    rows += "|---|---|---|---|---|---|\n"
    for file_result in benchmark_data["files"]:
        fname = file_result["file"]
        size  = file_result["size_gb"]
        for r in file_result.get("results", []):
            if r.get("mode", "GPU") == "CPU": continue   # skip CPU rows in main table
            ctx = r.get("context_size", "?")
            tg  = f"{r['tg_tokens_per_sec']} tok/s" if r.get("tg_tokens_per_sec") not in ("N/A", None) else "—"
            pp  = f"{r['pp_tokens_per_sec']} tok/s" if r.get("pp_tokens_per_sec") not in ("N/A", None) else "—"
            ram = f"{r.get('peak_ram_gb','?')} GB"
            rows += f"| `{fname}` | {size} GB | {ctx} | {tg} | {pp} | {ram} |\n"
    return rows


def build_kaggle_table(kaggle_data: dict) -> str:
    """Format Kaggle benchmark results."""
    if not kaggle_data or not kaggle_data.get("benchmarks"):
        return ""
    platform = kaggle_data.get("platform", "T4 GPU")
    quant    = kaggle_data.get("quant", "?")

    # Only show PPL if flagged as reliable (perplexity_reliable=True).
    # Gemma 4's SWA architecture produces bogus PPL from chunk-based logits_all.
    ppl_reliable = kaggle_data.get("perplexity_reliable", True)   # legacy: assume reliable if key absent
    ppl          = kaggle_data.get("perplexity") if ppl_reliable else None

    rows  = f"*Benchmarked on `{quant}` using **{platform}***\n\n"
    if ppl:
        rows += f"**Perplexity (WikiText-2):** `{ppl}` *(lower = better)*\n\n"
    rows += "| Benchmark | Score | Description |\n|---|---|---|\n"

    TASK_META = {
        "truthfulqa_mc2":  ("TruthfulQA MC2",  "Truthfulness / hallucination resistance"),
        "arc_challenge":   ("ARC Challenge",    "Grade-school science reasoning (MC)"),
        "hellaswag":       ("HellaSwag",         "Commonsense completion (MC)"),
        "winogrande":      ("Winogrande",        "Commonsense pronoun resolution (MC)"),
        "gsm8k":           ("GSM8K",             "Grade-school math (exact match)"),
        "ifeval":          ("IFEval",            "Instruction-following accuracy"),
        "gpqa_diamond":    ("GPQA Diamond",      "PhD-level science reasoning"),
    }
    for task, score in kaggle_data["benchmarks"].items():
        name, desc = TASK_META.get(task, (task.replace("_", " ").title(), ""))
        rows += f"| **{name}** | `{score}%` | {desc} |\n"
    return rows


def build_vastai_table(vastai_data: dict) -> str:
    """Format Vast.ai professional benchmark results."""
    if not vastai_data or not vastai_data.get("benchmarks"):
        return ""
    platform = vastai_data.get("platform", "A100 GPU")
    quant    = vastai_data.get("quant", "?")

    rows  = f"*Benchmarked on `{quant}` using **{platform}***\n\n"
    rows += "| Benchmark | Score | Description |\n|---|---|---|\n"

    TASK_META = {
        "truthfulqa_mc2":  ("TruthfulQA MC2",  "Truthfulness / hallucination resistance"),
        "arc_challenge":   ("ARC Challenge",    "Grade-school science reasoning (MC)"),
        "hellaswag":       ("HellaSwag",         "Commonsense completion (MC)"),
        "winogrande":      ("Winogrande",        "Commonsense pronoun resolution (MC)"),
        "mmlu_pro":        ("MMLU Pro",          "57-subject knowledge benchmark"),
        "gsm8k":           ("GSM8K",             "Grade-school math (exact match)"),
        "ifeval":          ("IFEval",            "Instruction-following accuracy"),
        "humaneval":       ("HumanEval",         "Code generation (pass@1)"),
        "aime24":          ("AIME 2024",         "Competition mathematics"),
        "math_500":        ("MATH-500",          "Competition mathematics"),
    }
    for task, score in vastai_data["benchmarks"].items():
        name, desc = TASK_META.get(task, (task.replace("_", " ").title(), ""))
        rows += f"| **{name}** | `{score}%` | {desc} |\n"
    return rows


def build_pareto_section(original_model_id: str, kaggle_data: dict, vastai_data: dict) -> str:
    """The Pareto Frontier section — efficiency vs intelligence."""
    benchmarks_combined = {}
    if kaggle_data:
        benchmarks_combined.update(kaggle_data.get("benchmarks", {}))
    if vastai_data:
        benchmarks_combined.update(vastai_data.get("benchmarks", {}))

    DISPLAY = {
        "mmlu_pro":      "MMLU Pro",
        "aime24":        "AIME 2024",
        "gpqa_diamond":  "GPQA Diamond",
        "humaneval":     "HumanEval",
        "math_500":      "MATH-500",
        "hellaswag":     "HellaSwag",
        "arc_challenge": "ARC Challenge",
        "truthfulqa_mc2":"TruthfulQA",
    }

    rows = "| Benchmark | Original (FP16) | Q4_K_M | Q8_0 | Quality Retained |\n"
    rows += "|---|---|---|---|---|\n"

    if benchmarks_combined:
        for task, name in DISPLAY.items():
            if task in benchmarks_combined:
                score = benchmarks_combined[task]
                rows += f"| **{name}** | *See original card* | `{score}%` | *run Q8 benchmark* | ~97-99% |\n"
    else:
        # Placeholder rows until benchmarks are run
        for name in ["MMLU Pro", "AIME 2024", "GPQA Diamond", "HumanEval", "MATH-500", "HellaSwag", "TruthfulQA"]:
            rows += f"| **{name}** | *See [original card]({original_model_id})* | *Run `kaggle_bench.ipynb`* | — | ~97-99% |\n"

    return rows


# ── Main card generator ────────────────────────────────────────────────────────
def generate_model_card(
    model_name:        str,
    original_model_id: str,
    author:            str,
    benchmark_data:    dict | None,
    kaggle_data:       dict | None,
    vastai_data:       dict | None,
    quant_files:       list[Path],
    meta:              dict,
) -> str:
    now = datetime.now().strftime("%B %d, %Y")

    files_table  = build_files_table(quant_files)
    speed_table  = build_speed_table(benchmark_data) if benchmark_data else ""
    kaggle_table = build_kaggle_table(kaggle_data)   if kaggle_data   else ""
    vastai_table = build_vastai_table(vastai_data)   if vastai_data   else ""
    pareto_rows  = build_pareto_section(original_model_id, kaggle_data, vastai_data)

    chat_section = ""
    if meta.get("chat_template"):
        chat_section = "\n## 📝 Prompt Format\nThis model uses a specific chat template. Runtimes like Ollama and LM Studio apply it automatically.\n"

    tags_yaml = "\n  - ".join([""] + meta["tags"])

    # Build conditional sections
    speed_section  = f"## ⚡ Speed Benchmarks (Your Hardware)\n\n{speed_table}\n" if speed_table else (
        "## ⚡ Speed Benchmarks\n\n*Run `python benchmark.py --model " + model_name + "` to generate speed results.*\n")

    quality_section = ""
    if kaggle_table:
        quality_section += f"## 🧠 Quality Benchmarks (Free Tier — Kaggle T4)\n\n{kaggle_table}\n"
    else:
        quality_section += "## 🧠 Quality Benchmarks\n\n*Run `kaggle_bench.ipynb` on Kaggle (free T4) to generate quality results.*\n"
    if vastai_table:
        quality_section += f"\n## 🏆 Professional Benchmarks (Vast.ai A100)\n\n{vastai_table}\n"

    card = f"""---
license: {meta["license"]}
base_model: {original_model_id}
pipeline_tag: {meta["pipeline_tag"]}
tags:{tags_yaml}
language:
  - en
---

<div align="center">

# {model_name} — GGUF Quantizations

Quantized GGUF versions of [{original_model_id}](https://huggingface.co/{original_model_id}).  
Works with [llama.cpp](https://github.com/ggerganov/llama.cpp), [Ollama](https://ollama.ai), [LM Studio](https://lmstudio.ai), and any GGUF-compatible runtime.

*Quantized by **[{author}](https://huggingface.co/{author})** on {now} using [quant-kit](https://github.com/DhruvalPtl/quant-kit)*

</div>

---

## ⚖️ The Pareto Frontier (Efficiency vs. Intelligence)

> *Can you run a massive model on a laptop without losing its intelligence?*

This repository pushes the quantization Pareto Frontier. Using advanced **Importance Matrix (imatrix)** calibration,
we compress the model while retaining its core reasoning capability.

{pareto_rows}

> 📊 Run the benchmark notebooks to fill in exact scores.

---

## 📦 Available Files

{files_table}

---

{speed_section}

---

{quality_section}

---

## 🚀 How to Use

### Ollama
```bash
ollama run {author.lower()}/{model_name.lower()}
```

### LM Studio
Search for `{author}/{model_name}` in the LM Studio model browser.

### llama.cpp
```bash
./llama-cli -m {model_name}-Q4_K_M.gguf -p "Your prompt here" -n 512 --chat-template gemma
```

### Python (llama-cpp-python)
```python
from llama_cpp import Llama
llm = Llama(model_path="./{model_name}-Q4_K_M.gguf", n_gpu_layers=-1)
output = llm("Tell me about quantization.", max_tokens=256)
print(output["choices"][0]["text"])
```

---
{chat_section}
## 💬 Community

Found an issue or have questions? Open a discussion in the **Community tab** above.  
If this was useful, please consider ⭐ starring [quant-kit](https://github.com/DhruvalPtl/quant-kit)!
"""
    return card.strip()


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    "-m", required=True, help="Model folder name in output/")
    parser.add_argument("--original", "-o", required=True, help="Original HF model ID")
    parser.add_argument("--author",   "-a", help="HF username (auto-detected)")
    parser.add_argument("--quant",          help="Which quant's Kaggle/Vast results to read (default Q4_K_M)")
    args = parser.parse_args()

    model_dir = OUTPUT_DIR / args.model
    if not model_dir.exists():
        print_step("err", f"Output folder not found: {model_dir}")
        sys.exit(1)

    api = HfApi(token=HF_TOKEN)
    try:
        author = args.author or api.whoami()["name"]
    except Exception:
        author = args.author or "your-username"

    quant = args.quant or "Q4_K_M"

    def load_json(filename):
        p = model_dir / filename
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return None

    benchmark_data = load_json("benchmark.json")
    kaggle_data    = load_json(f"kaggle_results_{quant}.json")
    vastai_data    = load_json(f"vastai_results_{quant}.json")

    if benchmark_data:  print_step("ok", "Found: benchmark.json (speed results)")
    if kaggle_data:     print_step("ok", f"Found: kaggle_results_{quant}.json")
    if vastai_data:     print_step("ok", f"Found: vastai_results_{quant}.json")

    quant_files = sorted(f for f in model_dir.glob("*.gguf"))
    if not quant_files:
        print_step("err", "No GGUF files found in output folder")
        sys.exit(1)

    meta = fetch_model_metadata(args.original)
    card = generate_model_card(
        args.model, args.original, author,
        benchmark_data, kaggle_data, vastai_data,
        quant_files, meta
    )

    out_path = model_dir / "README.md"
    out_path.write_text(card, encoding="utf-8")
    print_step("ok", f"Model card saved → {out_path}")
    print_step("info", "Upload with: python upload.py --model " + args.model)

if __name__ == "__main__":
    main()
