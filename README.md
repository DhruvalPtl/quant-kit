# quant-kit 🛠️

A clean, beginner-friendly GGUF quantization toolkit.

Download any HuggingFace model → Quantize to GGUF → Benchmark → Upload to HuggingFace.

---

## Workflow

```
python quantize.py   →   python benchmark.py   →   python model_card.py   →   python upload.py
```

---

## Setup

### 1. Clone & create virtual environment
```bash
git clone https://github.com/your-username/quant-kit
cd quant-kit
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / Mac
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Download llama.cpp (Windows)
1. Go to https://github.com/ggerganov/llama.cpp/releases/latest
2. Download `llama-b*-bin-win-vulkan-x64.zip`
3. Extract to `quant-kit/llama.cpp/`
4. Also download `convert_hf_to_gguf.py` from the llama.cpp repo and place it in `quant-kit/llama.cpp/`

Your folder should look like:
```
quant-kit/
└── llama.cpp/
    ├── llama-quantize.exe
    ├── llama-cli.exe
    └── convert_hf_to_gguf.py
```

### 3. Login to HuggingFace
```bash
huggingface-cli login
```

---

## Usage

### Step 1 — Quantize a model
```bash
# Activate venv first
.venv\Scripts\activate

# Quantize with default quants (Q4_K_M, Q5_K_M, Q8_0)
python quantize.py --model Qwen/Qwen2.5-1.5B-Instruct

# Custom quants
python quantize.py --model Qwen/Qwen2.5-7B-Instruct --quants Q4_K_M Q8_0
```

### Step 2 — Benchmark (optional but recommended)
```bash
python benchmark.py --model Qwen2.5-1.5B-Instruct
```

### Step 3 — Generate model card
```bash
python model_card.py --model Qwen2.5-1.5B-Instruct --original Qwen/Qwen2.5-1.5B-Instruct --author your-hf-username
```

### Step 4 — Upload to HuggingFace
```bash
python upload.py --model Qwen2.5-1.5B-Instruct --author your-hf-username
```

---

## Supported Quant Types

| Type | Size | Quality | Best For |
|---|---|---|---|
| `Q4_K_M` | ~40% of original | ⭐⭐⭐⭐ | Most users — best balance |
| `Q5_K_M` | ~50% of original | ⭐⭐⭐⭐½ | Better quality |
| `Q8_0` | ~80% of original | ⭐⭐⭐⭐⭐ | Near-original quality |
| `IQ4_XS` | ~37% of original | ⭐⭐⭐½ | Low RAM devices |

---

## Hardware Support

GGUF works on **any hardware** via llama.cpp:

| Hardware | Backend | Notes |
|---|---|---|
| NVIDIA GPU | CUDA | Best performance |
| AMD GPU | ROCm | Linux recommended |
| Intel Arc | SYCL / Vulkan | Windows supported |
| Apple M-series | Metal | Great performance |
| CPU only | CPU | Slow but works everywhere |

The Vulkan binary works on Intel Arc, AMD, and NVIDIA out of the box on Windows.

---

## Project Structure

```
quant-kit/
├── quantize.py      ← Download + convert + quantize
├── benchmark.py     ← Measure tokens/sec and RAM
├── model_card.py    ← Generate HuggingFace README
├── upload.py        ← Push to HuggingFace Hub
├── requirements.txt
│
├── llama.cpp/       ← llama.cpp binaries (you download this)
├── models/          ← Downloaded HF models (auto-created)
└── output/          ← Your GGUF files go here (auto-created)
    └── Qwen2.5-1.5B-Instruct/
        ├── Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
        ├── Qwen2.5-1.5B-Instruct-Q5_K_M.gguf
        ├── Qwen2.5-1.5B-Instruct-Q8_0.gguf
        ├── benchmark.json
        └── README.md
```

---

## Start Small

First model to try: `Qwen/Qwen2.5-1.5B-Instruct`
- Only ~3 GB to download
- Fast to quantize even on CPU
- Well-known model the community cares about
