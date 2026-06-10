<div align="center">

# 🧰 quant-kit

**The complete multi-modal GGUF quantization pipeline.**  
Download → Quantize → Benchmark → Upload. For **Text LLMs, VLMs, Diffusion Models, and Whisper ASR.**

[![License](https://img.shields.io/github/license/DhruvalPtl/quant-kit?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![GitHub stars](https://img.shields.io/github/stars/DhruvalPtl/quant-kit?style=for-the-badge)](https://github.com/DhruvalPtl/quant-kit/stargazers)
[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-Models-yellow?style=for-the-badge)](https://huggingface.co/Dhptl)

</div>

---

**quant-kit** is a professional toolkit for quantizing and publishing AI models in GGUF format. It automates downloading from HuggingFace, converting to GGUF via `llama.cpp` / `stable-diffusion.cpp` / `whisper.cpp`, benchmarking quality and speed, and generating polished model cards for the HuggingFace Hub.

## ✨ Features

- **🧠 Universal — 4 Model Types**: Text LLMs, Vision-Language Models (VLMs), Stable Diffusion / FLUX, and Whisper ASR.
- **🚀 1-Command Auto-Detection**: `python quant.py --model org/model` — detects model type and runs the right pipeline automatically.
- **🖼️ VLM Support**: Full 2-GGUF workflow (text backbone + mmproj vision encoder) for Qwen2-VL, LLaVA, InternVL2, Gemma3, Llama-3.2-Vision, PaliGemma, and 30+ more.
- **🎨 Diffusion & FLUX**: Quantize SD 1.x / 2.x / XL / 3 / 3.5 and FLUX.1-schnell/dev via `stable-diffusion.cpp`.
- **🎙️ Whisper ASR**: Quantize Whisper tiny through large-v3-turbo via `whisper.cpp`.
- **📊 Quality Benchmarking**: Kaggle (T4 free) and Vast.ai (A100 professional) benchmark notebooks included.
- **⚡ Speed Benchmarking**: Auto-detects your hardware and measures token generation throughput.
- **📄 Smart Model Cards**: Auto-generated professional `README.md` with badges, usage examples, and file tables — VLM-aware (explains 2-file requirement).
- **🧹 Disk Management**: `--delete-src` removes source weights after conversion; `--batch N` splits large jobs across sessions.

---

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/DhruvalPtl/quant-kit
cd quant-kit
pip install -e .
```

### 2. Set up llama.cpp

**Windows:** Download the latest `llama-b*-bin-win-vulkan-x64.zip` from [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases/latest) and extract into a `llama.cpp/` folder.

**Linux:** Run `python setup_linux.py`

### 3. Configure

```bash
cp .env.example .env
# Edit .env and add: hf_token = "hf_your_token_here"
```

### 4. Quantize!

```bash
# Auto-detects model type — works for LLM, VLM, Diffusion, Whisper
python quant.py --model Qwen/Qwen2.5-7B-Instruct        # Text LLM
python quant.py --model Qwen/Qwen2.5-VL-3B-Instruct     # VLM
python quant.py --model black-forest-labs/FLUX.1-schnell # Diffusion (needs setup_sd_cpp.py first)
python quant.py --model openai/whisper-large-v3-turbo    # Whisper (needs setup_whisper_cpp.py first)
```

---

## 📖 Complete Workflow

### Step 1 — Check your setup
```bash
python check_setup.py
```

### Step 2 — (Optional) Set up extra backends

```bash
# For Stable Diffusion / FLUX quantization:
python setup_sd_cpp.py

# For Whisper ASR quantization:
python setup_whisper_cpp.py
```

### Step 3 — Quantize

```bash
# Unified entry point (recommended — auto-detects everything):
python quant.py --model google/gemma-3-4b-it

# Or use model-specific scripts directly:
python quantize.py     --model Qwen/Qwen2.5-7B-Instruct --preset full      # LLM
python quantize_vlm.py --model Qwen/Qwen2.5-VL-7B-Instruct --preset standard  # VLM
python quantize_diffusion.py --model stabilityai/stable-diffusion-xl-base-1.0 --quant q8_0
python quantize_whisper.py   --model openai/whisper-large-v3-turbo
```

### Step 4 — Benchmark (Optional)

```bash
python benchmark.py --model gemma-3-4b-it    # Speed benchmark (local)
# Run kaggle_bench.ipynb on Kaggle for free T4 quality benchmarks
# Run vastai_bench.ipynb on Vast.ai for professional A100 benchmarks
```

### Step 5 — Generate Model Card & Upload

```bash
python model_card.py --model Qwen2.5-VL-3B-Instruct --original Qwen/Qwen2.5-VL-3B-Instruct
python upload.py     --model Qwen2.5-VL-3B-Instruct
```

---

## 🗂️ Script Reference

| Script | Purpose |
|---|---|
| `quant.py` | **Unified entry point** — auto-detects model type and dispatches |
| `quantize.py` | Text LLM → GGUF (Q2_K through Q8_0, IQ quants) |
| `quantize_vlm.py` | VLM → 2 GGUFs (text backbone + mmproj vision encoder) |
| `quantize_diffusion.py` | SD / SDXL / SD3 / FLUX → GGUF via stable-diffusion.cpp |
| `quantize_whisper.py` | Whisper ASR → GGUF via whisper.cpp |
| `model_card.py` | Generate professional HuggingFace README.md |
| `upload.py` | Upload all files to HuggingFace Hub |
| `benchmark.py` | Local speed benchmark (tokens/sec) |
| `check_setup.py` | Verify all binaries and configuration |
| `setup_linux.py` | Build llama.cpp from source on Linux/Vast.ai |
| `setup_sd_cpp.py` | Download stable-diffusion.cpp binary |
| `setup_whisper_cpp.py` | Download whisper.cpp binary |

---

## 🖼️ Supported Model Types

### Text LLMs (via llama.cpp)
Standard transformer models: Qwen, Gemma, Llama, Mistral, Phi, Falcon, DeepSeek, and [200+ more](https://github.com/ggerganov/llama.cpp).

### VLMs (via llama.cpp — 2-file output)
| Model Family | Architectures |
|---|---|
| Qwen2-VL / Qwen2.5-VL / Qwen3-VL | `Qwen2VLForConditionalGeneration`, `Qwen2_5_VLForConditionalGeneration` |
| LLaVA 1.5 / 1.6 | `LlavaForConditionalGeneration` |
| Gemma3 / Gemma4 | `Gemma3ForConditionalGeneration` |
| Llama-3.2 Vision | `Llama4ForConditionalGeneration` |
| InternVL2 | `InternVisionModel` |
| SmolVLM | `SmolVLMForConditionalGeneration` |
| PaliGemma2 | `Gemma3ForConditionalGeneration` |
| Mistral3 | `Mistral3ForConditionalGeneration` |
| + 25 more | See `quantize_vlm.py` for full list |

### Diffusion / Flow Matching (via stable-diffusion.cpp)
SD 1.x · SD 2.x · SDXL · SD 3 / 3.5 · FLUX.1-schnell · FLUX.1-dev

### Whisper ASR (via whisper.cpp)
tiny · base · small · medium · large-v1 · large-v2 · large-v3 · large-v3-turbo · distil-large-v3

---

## 📊 LLM Quantization Types

| Quant | Bits | Quality | Best For |
|---|---|---|---|
| Q8_0 | 8-bit | ⭐⭐⭐⭐⭐ | Near-original quality |
| Q6_K | 6-bit | ⭐⭐⭐⭐⭐ | High quality, large RAM |
| Q5_K_M | 5-bit | ⭐⭐⭐⭐½ | High accuracy |
| **Q4_K_M** | **4-bit** | **⭐⭐⭐⭐** | **← Sweet spot, recommended** |
| Q3_K_M | 3-bit | ⭐⭐⭐ | Low RAM |
| Q2_K | 2-bit | ⭐ | Extreme compression |
| IQ4_XS | 4-bit | ⭐⭐⭐⭐ | imatrix — best small size |
| IQ3_M | 3-bit | ⭐⭐⭐½ | imatrix — low RAM |

---

## ☁️ Benchmarking Notebooks

| Notebook | Platform | GPU | Use |
|---|---|---|---|
| `kaggle_bench.ipynb` | Kaggle | T4 16GB (free) | Quick quality check |
| `vastai_bench.ipynb` | Vast.ai | A100 / RTX 4090 | Professional benchmarks |

---

## 🤝 Contributing

Pull requests welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions.

## 📄 License

[MIT License](LICENSE)
