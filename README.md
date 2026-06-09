<div align="center">
  
# 🧰 quant-kit

**Download → Quantize → Benchmark → Upload. The complete GGUF quantization pipeline.**

[![PyPI version](https://img.shields.io/pypi/v/quant-kit?style=for-the-badge)](https://pypi.org/project/quant-kit/)
[![License](https://img.shields.io/github/license/DhruvalPtl/quant-kit?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![GitHub stars](https://img.shields.io/github/stars/DhruvalPtl/quant-kit?style=for-the-badge)](https://github.com/DhruvalPtl/quant-kit/stargazers)
[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-Models-yellow?style=for-the-badge)](https://huggingface.co/Dhptl)

</div>

---

**quant-kit** is the ultimate toolkit for building, evaluating, and sharing high-quality local LLMs. It automates the complex process of downloading HuggingFace models, compiling them to GGUF using `llama.cpp`, generating importance matrices (imatrix), running quality and speed benchmarks, and generating professional model cards for the HuggingFace Hub.

## ✨ Features

- **🚀 1-Command Pipeline**: Go from HuggingFace ID to 13+ GGUF quants instantly.
- **🧠 Advanced Quantization**: Full support for standard K-quants and cutting-edge IQ-quants via imatrix.
- **📊 Quality Benchmarking**: Built-in scripts to measure Perplexity, KL Divergence, and downstream metrics.
- **⚡ Speed Benchmarking**: Auto-detects your GPU and measures token generation speed.
- **📄 Professional Model Cards**: Automatically generates rich `README.md` files for HuggingFace, detecting chat templates, model architecture, and calculating RAM requirements.
- **🧹 Disk Space Management**: Built-in `--delete-src` logic safely removes 30GB+ source models once converted to save space.

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/DhruvalPtl/quant-kit
cd quant-kit
pip install -e .
```

You must also have `llama.cpp` built or downloaded. Place the binaries (`llama-quantize`, `llama-bench`, etc.) in a `llama.cpp/` folder inside this directory. Run `python check_setup.py` to verify your environment.

### 2. Configure
Copy `.env.example` to `.env` and add your HuggingFace token:
```env
hf_token = "hf_your_token_here"
```

### 3. Quantize!
```bash
python quantize.py --model Qwen/Qwen2.5-1.5B-Instruct --preset full
```

## 📖 Usage Guide

### 1. Check your setup
Always ensure your environment is ready:
```bash
python check_setup.py
```

### 2. Run the Quantizer
You can select specific quants or use presets (`standard`, `full`, `imatrix`, `all`):
```bash
# Standard 3 quants
python quantize.py --model google/gemma-4-12b-it --preset standard

# Specific quants + free disk space as you go
python quantize.py --model google/gemma-4-12b-it --quants Q4_K_M Q8_0 --delete-src

# Advanced: Use imatrix for extreme low-bit IQ quants
python quantize.py --model google/gemma-4-12b-it --preset imatrix --calibration wiki.train.raw
```

### 3. Benchmark Quality & Speed
Test how fast your model runs on your hardware, and check quality loss (Perplexity):
```bash
python benchmark.py --model gemma-4-12b-it
python quality_bench.py --model gemma-4-12b-it --data wiki.test.raw
```

### 4. Upload to HuggingFace
Generates a beautiful model card and uploads all files in one atomic commit:
```bash
# Generates model card auto-detecting the original Qwen config
python model_card.py --model Qwen2.5-1.5B-Instruct --original Qwen/Qwen2.5-1.5B-Instruct

# Upload to your profile
python upload.py --model Qwen2.5-1.5B-Instruct
```

---

## 📊 Supported Quantization Types

We currently support generating **13 different quantization types**:

| Quant | Bits | Quality | Recommended For |
|---|---|---|---|
| **Q8_0** | 8-bit | ⭐⭐⭐⭐⭐ | Closest to original quality. Use when RAM is not a concern. |
| **Q6_K** | 6-bit | ⭐⭐⭐⭐⭐ | Near-perfect quality, very large. |
| **Q5_K_M** | 5-bit | ⭐⭐⭐⭐½ | Better quality than Q4, slightly larger. |
| **Q5_K_S** | 5-bit | ⭐⭐⭐⭐ | Large but accurate. |
| **Q4_K_M** | 4-bit | ⭐⭐⭐⭐ | **Best balance of size and quality. Recommended.** |
| **Q4_K_S** | 4-bit | ⭐⭐⭐½ | Good speed/size balance. |
| **IQ4_NL** | 4-bit | ⭐⭐⭐⭐ | Non-linear quant. Highly accurate for the size. |
| **IQ4_XS** | 4-bit | ⭐⭐⭐⭐ | Smallest file with good quality. |
| **Q3_K_L** | 3-bit | ⭐⭐⭐ | Slightly better than Q3_K_M. |
| **Q3_K_M** | 3-bit | ⭐⭐⭐ | Very small file. Quality drop noticeable. |
| **Q3_K_S** | 3-bit | ⭐⭐ | Very high compression, high quality loss. |
| **IQ3_M** | 3-bit | ⭐⭐⭐½ | imatrix quant. Excellent quality for the size. |
| **Q2_K** | 2-bit | ⭐ | Desperation / very low RAM. |

---

## 🤝 Contributing

We welcome pull requests! See [CONTRIBUTING.md](CONTRIBUTING.md) for how to set up the dev environment and submit your changes.

## 📄 License

This project is licensed under the [MIT License](LICENSE).
