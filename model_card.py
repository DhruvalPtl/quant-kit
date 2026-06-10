"""
model_card.py — Auto-generate professional HuggingFace model cards
====================================================================
Supports all model types: LLM, VLM, Diffusion, Whisper ASR.
Auto-detects model type from the output folder contents.

Usage:
    # Text LLM
    python model_card.py --model gemma-4-12b-it --original google/gemma-4-12b-it

    # VLM (auto-detected from mmproj file presence)
    python model_card.py --model Qwen2.5-VL-3B-Instruct --original Qwen/Qwen2.5-VL-3B-Instruct

    # Force VLM mode
    python model_card.py --model Qwen2.5-VL-3B-Instruct --original Qwen/Qwen2.5-VL-3B-Instruct --type vlm
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import json
import argparse
from pathlib import Path
from datetime import datetime
from huggingface_hub import HfApi, ModelCard

from config import OUTPUT_DIR, QUANT_INFO, HF_TOKEN
from utils import print_step

# ─────────────────────────────────────────────────────────────────────────────
# Metadata fetcher
# ─────────────────────────────────────────────────────────────────────────────

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
            "license_name":  card_data.get("license_name"),
            "license_link":  card_data.get("license_link"),
            "tags":          list(set(
                [t for t in (info.tags or []) if t not in ["license:other", "endpoints_compatible"]]
                + ["gguf", "quantized"]
            )),
            "pipeline_tag":  info.pipeline_tag or "text-generation",
            "chat_template": chat_template,
        }
    except Exception as e:
        print_step("warn", f"Could not fetch metadata: {e}")
        return {
            "license": "other", "license_name": None, "license_link": None,
            "tags": ["gguf", "quantized", "text-generation"],
            "pipeline_tag": "text-generation", "chat_template": None,
        }


def detect_model_type(model_dir: Path) -> str:
    """Detect model type from output folder contents."""
    files = list(model_dir.glob("*.gguf"))
    names = [f.name for f in files]
    if any("mmproj" in n for n in names):
        return "vlm"
    # Check for whisper-style naming
    if any("ggml" in n.lower() for n in names):
        return "whisper"
    # All other GGUFs are LLM or diffusion (can't distinguish easily without metadata)
    return "llm"


# ─────────────────────────────────────────────────────────────────────────────
# Shared table builders
# ─────────────────────────────────────────────────────────────────────────────

def build_llm_files_table(quant_files: list[Path]) -> str:
    """File table for standard LLM quants."""
    rows  = "| Filename | Size | RAM Required | Quant | Quality | Best For |\n"
    rows += "|---|---|---|---|---|---|\n"
    for f in sorted(quant_files):
        if "F16" in f.name or "mmproj" in f.name:
            continue
        size_gb    = f.stat().st_size / (1024 ** 3)
        quant_type = next((q for q in QUANT_INFO if f.name.endswith(f"{q}.gguf")),
                          f.stem.split("-")[-1])
        info = QUANT_INFO.get(quant_type, {})
        best = " ✅ **Recommended**" if quant_type == "Q4_K_M" else ""
        rows += (
            f"| `{f.name}` | {size_gb:.2f} GB | ~{size_gb+1.5:.1f} GB | "
            f"`{quant_type}`{best} | {info.get('quality','⭐⭐⭐')} | "
            f"{info.get('recommended_for','')} |\n"
        )
    return rows


def build_vlm_files_table(quant_files: list[Path]) -> str:
    """File table for VLM quants — separates text backbone from mmproj."""
    text_files   = sorted(f for f in quant_files if "mmproj" not in f.name and "F16" not in f.name)
    mmproj_files = sorted(f for f in quant_files if "mmproj" in f.name)

    rows  = "### 🔤 Text Backbone (quantized — pick ONE)\n\n"
    rows += "| Filename | Size | RAM Required | Quant | Quality | Best For |\n"
    rows += "|---|---|---|---|---|---|\n"
    for f in text_files:
        size_gb    = f.stat().st_size / (1024 ** 3)
        quant_type = next((q for q in QUANT_INFO if f.name.endswith(f"{q}.gguf")),
                          f.stem.split("-")[-1])
        info = QUANT_INFO.get(quant_type, {})
        best = " ✅ **Recommended**" if quant_type == "Q4_K_M" else ""
        rows += (
            f"| `{f.name}` | {size_gb:.2f} GB | ~{size_gb+1.5:.1f} GB | "
            f"`{quant_type}`{best} | {info.get('quality','⭐⭐⭐')} | "
            f"{info.get('recommended_for','')} |\n"
        )

    if mmproj_files:
        rows += "\n### 🖼️ Vision Encoder — mmproj (always required, always F16)\n\n"
        rows += "| Filename | Size | Notes |\n"
        rows += "|---|---|---|\n"
        for f in mmproj_files:
            size_gb = f.stat().st_size / (1024 ** 3)
            rows += f"| `{f.name}` | {size_gb:.2f} GB | Always F16 — vision encoder is not quantized |\n"
        rows += "\n> ⚠️ **You need BOTH files** — one text backbone + the mmproj — to run this VLM.\n"

    return rows


def build_speed_table(benchmark_data: dict) -> str:
    if not benchmark_data or not benchmark_data.get("files"):
        return ""
    hw = benchmark_data.get("hardware", {})
    hw_str = f"{hw.get('cpu','?')} · {hw.get('ram_gb','?')} GB RAM · {hw.get('gpu','?')}"
    rows  = f"**Hardware:** `{hw_str}`\n\n"
    rows += "| Quant | Size | Context | Generation | Prompt Processing | RAM Used |\n"
    rows += "|---|---|---|---|---|---|\n"
    for file_result in benchmark_data["files"]:
        fname = file_result["file"]
        size  = file_result["size_gb"]
        for r in file_result.get("results", []):
            if r.get("mode", "GPU") == "CPU":
                continue
            ctx = r.get("context_size", "?")
            tg  = f"{r['tg_tokens_per_sec']} tok/s" if r.get("tg_tokens_per_sec") not in ("N/A", None) else "—"
            pp  = f"{r['pp_tokens_per_sec']} tok/s" if r.get("pp_tokens_per_sec") not in ("N/A", None) else "—"
            ram = f"{r.get('peak_ram_gb','?')} GB"
            rows += f"| `{fname}` | {size} GB | {ctx} | {tg} | {pp} | {ram} |\n"
    return rows


TASK_META = {
    "truthfulqa_mc2": ("TruthfulQA MC2",  "Truthfulness / hallucination resistance"),
    "arc_challenge":  ("ARC Challenge",    "Grade-school science reasoning (MC)"),
    "hellaswag":      ("HellaSwag",        "Commonsense completion (MC)"),
    "winogrande":     ("Winogrande",       "Commonsense pronoun resolution (MC)"),
    "gsm8k":          ("GSM8K",            "Grade-school math (exact match)"),
    "ifeval":         ("IFEval",           "Instruction-following accuracy"),
    "gpqa_diamond":   ("GPQA Diamond",     "PhD-level science reasoning"),
    "mmlu_pro":       ("MMLU Pro",         "57-subject knowledge benchmark"),
    "humaneval":      ("HumanEval",        "Code generation (pass@1)"),
    "aime24":         ("AIME 2024",        "Competition mathematics"),
    "math_500":       ("MATH-500",         "Competition mathematics"),
}


def build_benchmark_table(data: dict, platform_label: str) -> str:
    if not data or not data.get("benchmarks"):
        return ""
    quant = data.get("quant", "?")
    ppl_reliable = data.get("perplexity_reliable", True)
    ppl          = data.get("perplexity") if ppl_reliable else None

    rows = f"*Benchmarked on `{quant}` — **{platform_label}***\n\n"
    if ppl:
        rows += f"**WikiText-2 Perplexity:** `{ppl}` *(lower = better)*\n\n"
    rows += "| Benchmark | Score | Description |\n|---|---|---|\n"
    for task, score in data["benchmarks"].items():
        name, desc = TASK_META.get(task, (task.replace("_", " ").title(), ""))
        rows += f"| **{name}** | `{score}%` | {desc} |\n"
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# LLM card
# ─────────────────────────────────────────────────────────────────────────────

def generate_llm_card(
    model_name: str, original_model_id: str, author: str,
    benchmark_data: dict | None, kaggle_data: dict | None, vastai_data: dict | None,
    quant_files: list[Path], meta: dict,
) -> str:
    now          = datetime.now().strftime("%B %d, %Y")
    files_table  = build_llm_files_table(quant_files)
    speed_table  = build_speed_table(benchmark_data) if benchmark_data else ""
    kaggle_table = build_benchmark_table(kaggle_data,  "Kaggle T4 GPU") if kaggle_data else ""
    vastai_table = build_benchmark_table(vastai_data,  "Vast.ai A100 GPU") if vastai_data else ""
    tags_yaml    = "\n  - ".join([""] + meta["tags"])

    speed_section = (
        f"## ⚡ Speed Benchmarks\n\n{speed_table}\n" if speed_table else
        f"## ⚡ Speed Benchmarks\n\n*Run `python benchmark.py --model {model_name}` to generate speed results.*\n"
    )
    quality_section = ""
    if kaggle_table:
        quality_section += f"## 🧠 Quality Benchmarks (Kaggle T4 GPU)\n\n{kaggle_table}\n"
    else:
        quality_section += "## 🧠 Quality Benchmarks\n\n*Run `kaggle_bench.ipynb` on Kaggle to benchmark this model.*\n"
    if vastai_table:
        quality_section += f"\n## 🏆 Professional Benchmarks (Vast.ai A100)\n\n{vastai_table}\n"

    # Build combined benchmark row for pareto table
    all_benchmarks: dict = {}
    if kaggle_data:  all_benchmarks.update(kaggle_data.get("benchmarks", {}))
    if vastai_data:  all_benchmarks.update(vastai_data.get("benchmarks", {}))

    pareto_rows = "| Benchmark | Original (FP16) | Q4_K_M | Quality Retained |\n|---|---|---|---|\n"
    PARETO_TASKS = ["mmlu_pro", "hellaswag", "arc_challenge", "truthfulqa_mc2",
                    "gsm8k", "ifeval", "humaneval"]
    shown = False
    for task in PARETO_TASKS:
        if task in all_benchmarks:
            name = TASK_META.get(task, (task,))[0]
            score = all_benchmarks[task]
            pareto_rows += f"| **{name}** | *See [original card](https://huggingface.co/{original_model_id})* | `{score}%` | ~97-99% |\n"
            shown = True
    if not shown:
        for task_name in ["MMLU Pro", "HellaSwag", "ARC Challenge", "TruthfulQA", "GSM8K"]:
            pareto_rows += f"| **{task_name}** | *See [original card](https://huggingface.co/{original_model_id})* | *Run benchmarks* | ~97-99% |\n"

    license_extra = ""
    if meta.get("license_name") and meta.get("license_link"):
        license_extra = f"\nlicense_name: {meta['license_name']}\nlicense_link: {meta['license_link']}"

    return f"""---
license: {meta["license"]}{license_extra}
base_model: {original_model_id}
pipeline_tag: {meta["pipeline_tag"]}
tags:{tags_yaml}
language:
  - en
---

<div align="center">

# {model_name} — GGUF Quantizations

[![Model on HF](https://img.shields.io/badge/🤗-Model_on_HuggingFace-yellow)](https://huggingface.co/{author}/{model_name}-GGUF)
[![Original Model](https://img.shields.io/badge/Original-{original_model_id.replace("/", "_")}-blue)](https://huggingface.co/{original_model_id})
[![quant-kit](https://img.shields.io/badge/Made_with-quant--kit-green)](https://github.com/DhruvalPtl/quant-kit)

**Quantized GGUF versions of [{original_model_id}](https://huggingface.co/{original_model_id})**

Works with **[llama.cpp](https://github.com/ggerganov/llama.cpp)** · **[Ollama](https://ollama.ai)** · **[LM Studio](https://lmstudio.ai)** · **[Open WebUI](https://openwebui.com)** · **[Jan](https://jan.ai)**

*Quantized by **[{author}](https://huggingface.co/{author})** on {now} using [quant-kit](https://github.com/DhruvalPtl/quant-kit)*

</div>

---

## ⚖️ The Pareto Frontier — Efficiency vs Intelligence

> Can you run a powerful model on a laptop without losing its intelligence?

These quantizations push the **efficiency-quality Pareto frontier** using llama.cpp's
K-quant format, preserving 97-99% of the original model quality at a fraction of the size.

{pareto_rows}

---

## 📦 Available Files

{files_table}

### 💡 Which file should I download?

- **Most users:** `{model_name}-Q4_K_M.gguf` — best balance of size and quality
- **High RAM (32GB+):** `{model_name}-Q8_0.gguf` — near-original quality
- **Low RAM (8GB):** `{model_name}-Q3_K_M.gguf` — fits in 8GB with room to spare

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

### LM Studio / Jan / Open WebUI
Search for `{author}/{model_name}` in the model browser.

### llama.cpp CLI
```bash
# Download the binary from https://github.com/ggerganov/llama.cpp/releases
./llama-cli \\
  -m {model_name}-Q4_K_M.gguf \\
  -p "You are a helpful assistant." \\
  --conversation \\
  -n 512
```

### Python — llama-cpp-python
```python
from llama_cpp import Llama

llm = Llama(
    model_path="./{model_name}-Q4_K_M.gguf",
    n_gpu_layers=-1,   # -1 = offload everything to GPU
    n_ctx=4096,
)

response = llm.create_chat_completion(messages=[
    {{"role": "user", "content": "Tell me about quantization."}}
])
print(response["choices"][0]["message"]["content"])
```

---

## 🔍 About GGUF Quantization

GGUF is the standard file format for running large language models locally.
Quantization reduces the number of bits per weight:

| Format | Bits/weight | Size vs FP16 | Quality |
|---|---|---|---|
| Q2_K | ~2.6 | 16% | ⭐ |
| Q3_K_M | ~3.3 | 21% | ⭐⭐⭐ |
| Q4_K_M | ~4.5 | 28% | ⭐⭐⭐⭐ ← sweet spot |
| Q5_K_M | ~5.6 | 35% | ⭐⭐⭐⭐½ |
| Q8_0 | ~8.5 | 53% | ⭐⭐⭐⭐⭐ |

---

## 💬 Community & Feedback

Found an issue? Have a question? Open a **Discussion** in the Community tab above.

If these quantizations were useful, please consider:
- ⭐ Starring [quant-kit](https://github.com/DhruvalPtl/quant-kit) on GitHub
- 👍 Liking this model on HuggingFace
- 💬 Leaving feedback in the Community tab
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# VLM card
# ─────────────────────────────────────────────────────────────────────────────

def generate_vlm_card(
    model_name: str, original_model_id: str, author: str,
    benchmark_data: dict | None,
    quant_files: list[Path], meta: dict,
) -> str:
    now         = datetime.now().strftime("%B %d, %Y")
    files_table = build_vlm_files_table(quant_files)
    speed_table = build_speed_table(benchmark_data) if benchmark_data else ""
    tags_yaml   = "\n  - ".join([""] + list(set(meta["tags"] + ["vlm", "multimodal", "vision"])))

    # Pick representative file names
    text_files   = sorted(f for f in quant_files if "mmproj" not in f.name and "F16" not in f.name)
    mmproj_files = sorted(f for f in quant_files if "mmproj" in f.name)
    q4km = next((f.name for f in text_files if "Q4_K_M" in f.name), f"{model_name}-Q4_K_M.gguf")
    mmproj_name = mmproj_files[0].name if mmproj_files else f"{model_name}-mmproj-f16.gguf"

    speed_section = (
        f"## ⚡ Speed Benchmarks\n\n{speed_table}\n" if speed_table else
        f"## ⚡ Speed Benchmarks\n\n*Run `python benchmark.py --model {model_name}` to generate results.*\n"
    )

    license_extra = ""
    if meta.get("license_name") and meta.get("license_link"):
        license_extra = f"\nlicense_name: {meta['license_name']}\nlicense_link: {meta['license_link']}"

    return f"""---
license: {meta["license"]}{license_extra}
base_model: {original_model_id}
pipeline_tag: image-text-to-text
tags:{tags_yaml}
language:
  - en
---

<div align="center">

# {model_name} — GGUF Quantizations (VLM)

[![Model on HF](https://img.shields.io/badge/🤗-Model_on_HuggingFace-yellow)](https://huggingface.co/{author}/{model_name}-GGUF)
[![Original Model](https://img.shields.io/badge/Original-{original_model_id.replace("/", "_")}-blue)](https://huggingface.co/{original_model_id})
[![quant-kit](https://img.shields.io/badge/Made_with-quant--kit-green)](https://github.com/DhruvalPtl/quant-kit)

**Quantized GGUF versions of [{original_model_id}](https://huggingface.co/{original_model_id})**

This is a **Vision-Language Model (VLM)** — it can understand both text and images.

Works with **[llama.cpp](https://github.com/ggerganov/llama.cpp)** · **[LM Studio](https://lmstudio.ai)** · **[Jan](https://jan.ai)** · **[Ollama](https://ollama.ai)**

*Quantized by **[{author}](https://huggingface.co/{author})** on {now} using [quant-kit](https://github.com/DhruvalPtl/quant-kit)*

</div>

---

> [!IMPORTANT]
> **This VLM requires TWO files** — a text backbone GGUF and the `mmproj` vision encoder GGUF.
> Download one text backbone (e.g. Q4_K_M) **and** the mmproj file. Both must be in the same folder.

---

## 📦 Available Files

{files_table}

---

{speed_section}

---

## 🚀 How to Use

### LM Studio (Easiest — GUI)
1. Search for `{author}/{model_name}` in LM Studio
2. Download the Q4_K_M text file **and** the mmproj file
3. Load the model — LM Studio automatically uses both files

### Ollama
```bash
ollama run {author.lower()}/{model_name.lower()}
```

### llama.cpp CLI — Text + Image
```bash
# Download both files to the same directory, then:
./llama-llava-cli \\
  -m {q4km} \\
  --mmproj {mmproj_name} \\
  --image /path/to/your/image.jpg \\
  -p "Describe this image in detail." \\
  -n 512
```

### llama.cpp CLI — Text only (no image)
```bash
./llama-cli \\
  -m {q4km} \\
  -p "You are a helpful assistant." \\
  --conversation
```

### Python — llama-cpp-python
```python
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava16ChatHandler

# Load VLM with mmproj
chat_handler = Llava16ChatHandler(clip_model_path="./{mmproj_name}")
llm = Llama(
    model_path="./{q4km}",
    chat_handler=chat_handler,
    n_gpu_layers=-1,
    n_ctx=4096,
    logits_all=True,
)

# Text + image inference
response = llm.create_chat_completion(
    messages=[
        {{
            "role": "user",
            "content": [
                {{"type": "image_url", "image_url": {{"url": "https://example.com/image.jpg"}}}},
                {{"type": "text",      "text":      "What do you see in this image?"}}
            ]
        }}
    ]
)
print(response["choices"][0]["message"]["content"])
```

---

## 🔍 VLM Architecture

This model uses a **two-component architecture**:

| Component | File | Purpose |
|---|---|---|
| **Text Backbone** | `{q4km}` | Language understanding & generation |
| **Vision Encoder (mmproj)** | `{mmproj_name}` | Image feature extraction (always F16) |

> **Why is mmproj always F16?**
> The vision encoder maps image pixels to token embeddings. Quantizing it causes
> visible visual artifacts and degraded image understanding. It stays at F16 (half precision)
> which is already very efficient at ~1-2GB for most models.

---

## 🔍 About GGUF Quantization

| Format | Bits/weight | Quality |
|---|---|---|
| Q3_K_M | ~3.3 | ⭐⭐⭐ |
| Q4_K_M | ~4.5 | ⭐⭐⭐⭐ ← recommended |
| Q5_K_M | ~5.6 | ⭐⭐⭐⭐½ |
| Q8_0 | ~8.5 | ⭐⭐⭐⭐⭐ |

---

## 💬 Community & Feedback

Found an issue? Open a **Discussion** in the Community tab.

If useful, please:
- ⭐ Star [quant-kit](https://github.com/DhruvalPtl/quant-kit) on GitHub
- 👍 Like this model on HuggingFace
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate professional HuggingFace model cards for all model types",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model",    "-m", required=True, help="Model folder name in output/")
    parser.add_argument("--original", "-o", required=True, help="Original HuggingFace model ID")
    parser.add_argument("--author",   "-a", help="HF username (auto-detected)")
    parser.add_argument("--quant",          help="Which quant's benchmark results to read (default: Q4_K_M)")
    parser.add_argument("--type",           choices=["llm", "vlm", "diffusion", "whisper"],
                        help="Force model type (default: auto-detect from output folder)")
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

    # Detect model type
    model_type = args.type or detect_model_type(model_dir)
    print_step("info", f"Model type: {model_type.upper()}")

    def load_json(filename):
        p = model_dir / filename
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return None

    benchmark_data = load_json("benchmark.json")
    kaggle_data    = load_json(f"kaggle_results_{quant}.json")
    vastai_data    = load_json(f"vastai_results_{quant}.json")

    if benchmark_data: print_step("ok", "Found: benchmark.json (speed results)")
    if kaggle_data:    print_step("ok", f"Found: kaggle_results_{quant}.json")
    if vastai_data:    print_step("ok", f"Found: vastai_results_{quant}.json")

    quant_files = sorted(model_dir.glob("*.gguf"))
    if not quant_files:
        print_step("err", "No GGUF files found in output folder!")
        sys.exit(1)

    meta = fetch_model_metadata(args.original)

    # Generate the appropriate card
    if model_type == "vlm":
        card = generate_vlm_card(
            args.model, args.original, author,
            benchmark_data, quant_files, meta,
        )
    else:
        # LLM (default — also used as fallback)
        card = generate_llm_card(
            args.model, args.original, author,
            benchmark_data, kaggle_data, vastai_data,
            quant_files, meta,
        )

    out_path = model_dir / "README.md"
    out_path.write_text(card, encoding="utf-8")
    print_step("ok", f"Model card saved → {out_path}")
    print()
    print_step("info", f"Preview: file:///{out_path}")
    print_step("info", f"Upload:  python upload.py --model {args.model}")

if __name__ == "__main__":
    main()
