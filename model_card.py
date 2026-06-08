"""
model_card.py — Auto-generate a professional HuggingFace model card
=====================================================================
Reads benchmark results and generates a rich README.md for your
HuggingFace model repo. Follows community standards (like bartowski).

Usage:
    python model_card.py --model Qwen2.5-1.5B-Instruct --original Qwen/Qwen2.5-1.5B-Instruct
    python model_card.py --model Qwen2.5-7B-Instruct --original Qwen/Qwen2.5-7B-Instruct --author yourname
"""

import json
import argparse
import sys
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

        bench_rows = "| Model | Size | Speed | RAM Usage |\n|---|---|---|---|\n"
        for r in benchmark_data["results"]:
            if r.get("error"):
                continue
            tps = f"{r['tokens_per_sec']} tok/s" if r.get("tokens_per_sec") != "N/A" else "—"
            ram = f"~{r['ram_used_mb']:.0f} MB" if r.get("ram_used_mb") else "—"
            bench_rows += f"| `{r['file']}` | {r['size_gb']} GB | {tps} | {ram} |\n"

        benchmark_section = f"""
## 📊 Benchmark Results

Tested on: `{hw_str}`

{bench_rows}

> Benchmarks use a fixed prompt with {benchmark_data['results'][0].get('n_tokens_generated', 128)} tokens generated.
> Results vary by hardware and system load.
"""

    # Choose quant to recommend in usage example
    example_file = quant_rows[0]["file"] if quant_rows else f"{model_name}-Q4_K_M.gguf"
    for row in quant_rows:
        if "Q4_K_M" in row["file"]:
            example_file = row["file"]
            break

    card = f"""---
license: apache-2.0
base_model: {original_model_id}
tags:
  - llm
  - gguf
  - quantized
  - llama-cpp
  - ollama
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
| 8 GB RAM | `Q4_K_M` — Best choice |
| 12 GB RAM | `Q5_K_M` — Better quality |
| 16 GB+ RAM | `Q8_0` — Near-original quality |
| Less than 6 GB RAM | `IQ4_XS` or `Q3_K_M` |

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

| Format | Description |
|---|---|
| `Q4_K_M` | 4-bit, K-quantization, medium — Best size/quality balance |
| `Q5_K_M` | 5-bit, K-quantization, medium — Higher quality |
| `Q8_0` | 8-bit — Near-lossless, largest file |

Quantization was done using [llama.cpp](https://github.com/ggerganov/llama.cpp).

---

## ℹ️ About the Original Model

- **Original Model**: [{original_model_id}](https://huggingface.co/{original_model_id})
- **Architecture**: Transformer (decoder-only)
- **License**: Check the [original model page](https://huggingface.co/{original_model_id})

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
        with open(bench_file) as f:
            benchmark_data = json.load(f)
        print(f"✓ Loaded benchmark data from {bench_file.name}")
    else:
        print("⚠ No benchmark.json found — model card will skip benchmark section")
        print("  Run benchmark.py first to include performance data")

    # Find all GGUF files
    quant_files = sorted(model_dir.glob("*.gguf"))
    quant_files = [f for f in quant_files if "F16" not in f.name]

    if not quant_files:
        print("✗ No quantized GGUF files found in output folder")
        sys.exit(1)

    print(f"✓ Found {len(quant_files)} GGUF files")

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
    print(f"✓ Model card saved → {out_path}")
    print()
    print("  Next step: python upload.py --model", args.model, "--author", args.author)


if __name__ == "__main__":
    main()
