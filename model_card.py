"""
model_card.py — Auto-generate a professional HuggingFace model card
=====================================================================
Reads benchmark results and generates a rich README.md for your
HuggingFace model repo. Follows community standards (like bartowski).

Usage:
    python model_card.py --model Qwen2.5-1.5B-Instruct --original Qwen/Qwen2.5-1.5B-Instruct
    python model_card.py --model Qwen2.5-7B-Instruct --original Qwen/Qwen2.5-7B-Instruct --author yourname
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import json
import argparse
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path(__file__).parent / "output"

# ─── Quant descriptions (what to tell users about each type) ──────────────────

QUANT_INFO = {
    "Q4_K_M": {
        "recommended_for": "Best balance of size and quality. Recommended for most users.",
        "use_case": "General use, everyday inference",
        "quality": "⭐⭐⭐⭐",
    },
    "Q5_K_M": {
        "recommended_for": "Better quality than Q4, slightly larger. Great if you have the RAM.",
        "use_case": "When you want a bit more accuracy",
        "quality": "⭐⭐⭐⭐½",
    },
    "Q8_0": {
        "recommended_for": "Closest to original quality. Use when RAM is not a concern.",
        "use_case": "High-quality inference, evaluation",
        "quality": "⭐⭐⭐⭐⭐",
    },
    "IQ4_XS": {
        "recommended_for": "Smallest file with good quality. Great for low-RAM devices.",
        "use_case": "Minimal RAM usage",
        "quality": "⭐⭐⭐½",
    },
    "Q3_K_M": {
        "recommended_for": "Very small file. Quality drop noticeable. Use only if you need extreme compression.",
        "use_case": "Very limited RAM",
        "quality": "⭐⭐⭐",
    },
}

# ─── Original model benchmark data (from official model cards) ─────────────────

ORIGINAL_BENCHMARKS = {
    "google/gemma-4-12B": {
        "label": "Gemma 4 12B (Base)",
        "source": "https://huggingface.co/google/gemma-4-12B",
        "note": "Results reported by Google on the base model.",
        "rows": [
            ("MMLU Pro",            "Text",         "77.2%"),
            ("GPQA Diamond",        "Science",      "78.8%"),
            ("AIME 2026 (no tools)","Math",         "77.5%"),
            ("LiveCodeBench v6",    "Coding",       "72.0%"),
            ("BigBench Extra Hard", "Reasoning",    "53.0%"),
            ("MMMLU",               "Multilingual", "83.4%"),
            ("MMMU Pro",            "Vision",       "69.1%"),
            ("MRCR v2 8-needle 128k", "Long Context", "43.4%"),
        ]
    },
    "google/gemma-4-12B-it": {
        "label": "Gemma 4 12B Unified (IT)",
        "source": "https://huggingface.co/google/gemma-4-12B-it",
        "note": "Results reported by Google for the instruction-tuned model.",
        "rows": [
            ("MMLU Pro",            "Text",         "77.2%"),
            ("GPQA Diamond",        "Science",      "78.8%"),
            ("AIME 2026 (no tools)","Math",         "77.5%"),
            ("LiveCodeBench v6",    "Coding",       "72.0%"),
            ("BigBench Extra Hard", "Reasoning",    "53.0%"),
            ("MMMLU",               "Multilingual", "83.4%"),
            ("MMMU Pro",            "Vision",       "69.1%"),
            ("MRCR v2 8-needle 128k", "Long Context", "43.4%"),
        ]
    },
}

def generate_model_card(
    model_name: str,
    original_model_id: str,
    author: str,
    benchmark_data: dict | None,
    quant_files: list[Path],
) -> str:
    """Generate a full model card README.md string."""

    now = datetime.now().strftime("%B %d, %Y")
    hf_username = author

    # Build the quant table
    quant_rows = []
    for f in sorted(quant_files):
        if "F16" in f.name:
            continue
        size_gb = f.stat().st_size / (1024 ** 3)
        # Extract quant type from filename e.g. "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf" → "Q4_K_M"
        quant_type = f.stem.split("-")[-1]
        info = QUANT_INFO.get(quant_type, {})
        recommended = "✅ **Recommended**" if quant_type == "Q4_K_M" else ""
        quant_rows.append({
            "file": f.name,
            "size": f"{size_gb:.2f} GB",
            "quant": quant_type,
            "recommended": recommended,
            "use_case": info.get("use_case", ""),
        })

    quant_table = "| Filename | Size | Quant | Use Case |\n"
    quant_table += "|---|---|---|---|\n"
    for row in quant_rows:
        rec = f" {row['recommended']}" if row["recommended"] else ""
        quant_table += f"| `{row['file']}` | {row['size']} | `{row['quant']}`{rec} | {row['use_case']} |\n"

    # Build benchmark table if available
    benchmark_section = ""
    if benchmark_data and benchmark_data.get("results"):
        hw = benchmark_data.get("hardware", {})
        hw_str = f"{hw.get('cpu', 'Unknown')} | {hw.get('ram_gb', '?')}GB RAM"
        if hw.get("gpu"):
            hw_str += f" | {hw['gpu']}"

        bench_rows = "| Model | Size | Generation | Prompt Processing |\n|---|---|---|---|\n"
        for r in benchmark_data["results"]:
            if r.get("error"):
                continue
            tg  = f"{r['tg_tokens_per_sec']} tok/s" if r.get('tg_tokens_per_sec') not in ("N/A", None) else "—"
            pp  = f"{r['pp_tokens_per_sec']} tok/s" if r.get('pp_tokens_per_sec') not in ("N/A", None) else "—"
            bench_rows += f"| `{r['file']}` | {r['size_gb']} GB | {tg} | {pp} |\n"

        benchmark_section = f"""
## 📊 Speed Benchmarks

Tested on: `{hw_str}`

{bench_rows}
> **Generation speed** = how fast the model outputs tokens (higher = better).
> **Prompt processing** = how fast it reads your input (higher = better).
> Results vary by hardware and system load.
"""

    # Build original model quality benchmarks section
    quality_section = ""
    orig_bench = ORIGINAL_BENCHMARKS.get(original_model_id)
    if orig_bench:
        rows = "| Benchmark | Category | Score |\n|---|---|---|\n"
        for name, cat, score in orig_bench["rows"]:
            rows += f"| {name} | {cat} | {score} |\n"
        quality_section = f"""
## 🧠 Original Model Quality Benchmarks

> Results from **[{orig_bench['label']}]({orig_bench['source']})** — reported by Google.
> {orig_bench['note']}
> These benchmarks apply to the original BF16 model. GGUF quantization preserves
> ~98–99% of quality for Q5/Q8 and ~96–97% for Q4 variants.

{rows}
"""

    # Choose quant to recommend in usage example
    example_file = quant_rows[0]["file"] if quant_rows else f"{model_name}-Q4_K_M.gguf"
    for row in quant_rows:
        if "Q4_K_M" in row["file"]:
            example_file = row["file"]
            break

    card = f"""---
license: gemma
base_model: {original_model_id}
tags:
  - llm
  - gguf
  - quantized
  - llama-cpp
  - ollama
  - gemma
language:
  - en
---

# {model_name} — GGUF Quantizations

Quantized GGUF versions of [{original_model_id}](https://huggingface.co/{original_model_id}).

These files work with [llama.cpp](https://github.com/ggerganov/llama.cpp), [Ollama](https://ollama.ai),
[LM Studio](https://lmstudio.ai), [Jan](https://jan.ai), and any other GGUF-compatible runtime.

> Quantized by **[{hf_username}](https://huggingface.co/{hf_username})** on {now}

---

## 📦 Available Files

{quant_table}

### Which file should I download?

| If you have... | Download this |
|---|---|
| 8 GB RAM | `IQ4_XS` — Smallest, runs on 8GB |
| 10 GB RAM | `Q4_K_M` — Best choice ✅ |
| 12 GB RAM | `Q5_K_M` — Better quality |
| 16 GB+ RAM | `Q8_0` — Near-original quality |

{quality_section}
{benchmark_section}

---

## 🚀 How to Use

### With Ollama
```bash
ollama run {hf_username}/{model_name.lower()}
```

### With llama.cpp
```bash
./llama-cli -m {example_file} -p "Your prompt here" -n 512
```

### With LM Studio
1. Open LM Studio
2. Search for `{hf_username}/{model_name}`
3. Download your preferred quant
4. Load and chat

### With Python (llama-cpp-python)
```python
from llama_cpp import Llama

llm = Llama(
    model_path="./{example_file}",
    n_ctx=4096,
    n_gpu_layers=-1,  # -1 = offload all layers to GPU
)

output = llm("Explain quantum computing in simple terms:", max_tokens=256)
print(output["choices"][0]["text"])
```

---

## 🔧 Quantization Details

| Format | Bits | Description |
|---|---|---|
| `Q4_K_M` | 4-bit | K-quantization, medium — Best size/quality balance |
| `Q5_K_M` | 5-bit | K-quantization, medium — Higher quality |
| `Q8_0`   | 8-bit | Near-lossless — Largest GGUF file |
| `IQ4_XS` | ~4-bit | Importance-matrix quant — Smallest with good quality |

Quantization was done using [llama.cpp](https://github.com/ggerganov/llama.cpp).

---

## ℹ️ About the Original Model

- **Original Model**: [{original_model_id}](https://huggingface.co/{original_model_id})
- **Architecture**: Gemma 4 Unified (multimodal — text + vision capable)
- **Parameters**: ~12 Billion
- **Context Length**: 128K tokens
- **License**: [Gemma Terms of Use](https://ai.google.dev/gemma/terms)

---

## 💬 Feedback

If you find issues or have questions, open a [discussion](https://huggingface.co/{hf_username}/{model_name}-GGUF/discussions).

If these quants are useful to you, please ⭐ the repo!
"""

    return card.strip()


def main():
    parser = argparse.ArgumentParser(description="Generate a HuggingFace model card for your GGUF quants")
    parser.add_argument("--model", "-m", required=True, help="Model folder name in output/ (e.g. Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--original", "-o", required=True, help="Original HuggingFace model ID (e.g. Qwen/Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--author", "-a", default="your-hf-username", help="Your HuggingFace username")

    args = parser.parse_args()

    model_dir = OUTPUT_DIR / args.model
    if not model_dir.exists():
        print(f"✗ Output folder not found: {model_dir}")
        print("  Run quantize.py first")
        sys.exit(1)

    # Load benchmark data if it exists
    benchmark_data = None
    bench_file = model_dir / "benchmark.json"
    if bench_file.exists():
        with open(bench_file, encoding="utf-8") as f:
            benchmark_data = json.load(f)
        print(f"[OK] Loaded benchmark data from {bench_file.name}")
    else:
        print("[!] No benchmark.json found -- model card will skip benchmark section")
        print("  Run benchmark.py first to include performance data")

    # Find all GGUF files
    quant_files = sorted(model_dir.glob("*.gguf"))
    quant_files = [f for f in quant_files if "F16" not in f.name]

    if not quant_files:
        print("[ERR] No quantized GGUF files found in output folder")
        sys.exit(1)

    print(f"[OK] Found {len(quant_files)} GGUF files")

    # Generate model card
    card = generate_model_card(
        model_name=args.model,
        original_model_id=args.original,
        author=args.author,
        benchmark_data=benchmark_data,
        quant_files=quant_files,
    )

    # Save
    out_path = model_dir / "README.md"
    out_path.write_text(card, encoding="utf-8")
    print(f"[OK] Model card saved -> {out_path}")
    print()
    print("  Next step: python upload.py --model", args.model, "--author", args.author)


if __name__ == "__main__":
    main()
