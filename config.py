"""
config.py — Shared configuration loader
=========================================
Loads settings from .env file. All other scripts import from here.

.env file format:
    hf_token = "hf_xxxxxxxxxxxxxxxxxxxx"
"""

import os
import platform
from pathlib import Path
from dotenv import load_dotenv

# ── OS Detection ───────────────────────────────────────────────────────────────
IS_WINDOWS = platform.system() == "Windows"
EXE        = ".exe" if IS_WINDOWS else ""    # binaries have .exe on Windows only

# Always load .env from the project root (same folder as this file)
ENV_FILE = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=ENV_FILE)

# ── HuggingFace Token ──────────────────────────────────────────────────────────
# Reads hf_token from .env — strips quotes if present
HF_TOKEN: str | None = os.getenv("hf_token") or os.getenv("HF_TOKEN")

if HF_TOKEN:
    # Set it as env var so huggingface_hub picks it up automatically
    os.environ["HF_TOKEN"] = HF_TOKEN

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).parent
LLAMA_CPP_DIR = ROOT_DIR / "llama.cpp"
LLAMA_SRC_DIR = ROOT_DIR / "llama-src"     # llama.cpp Python source (convert scripts)
MODELS_DIR    = ROOT_DIR / "models"
OUTPUT_DIR    = ROOT_DIR / "output"

# stable-diffusion.cpp — for SD / SDXL / SD3 / FLUX quantization
SD_CPP_DIR    = ROOT_DIR / "stable-diffusion.cpp"
SD_BINARY     = SD_CPP_DIR / f"sd{EXE}"

# whisper.cpp — for Whisper ASR quantization
WHISPER_CPP_DIR      = ROOT_DIR / "whisper.cpp"
WHISPER_BINARY       = WHISPER_CPP_DIR / f"main{EXE}"
WHISPER_QUANTIZE     = WHISPER_CPP_DIR / f"quantize{EXE}"
WHISPER_CONVERT_SCRIPT = WHISPER_CPP_DIR / "models" / "convert-h5-to-ggml.py"

# ── llama.cpp binaries ─────────────────────────────────────────────────────────
# Works on both Windows (.exe) and Linux (no extension)
LLAMA_QUANTIZE   = LLAMA_CPP_DIR / f"llama-quantize{EXE}"
LLAMA_CLI        = LLAMA_CPP_DIR / f"llama-cli{EXE}"
LLAMA_BENCH      = LLAMA_CPP_DIR / f"llama-bench{EXE}"
LLAMA_IMATRIX    = LLAMA_CPP_DIR / f"llama-imatrix{EXE}"
LLAMA_PERPLEXITY = LLAMA_CPP_DIR / f"llama-perplexity{EXE}"
CONVERT_SCRIPT   = LLAMA_SRC_DIR / "convert_hf_to_gguf.py"   # run from LLAMA_SRC_DIR

# ── Quant types ────────────────────────────────────────────────────────────────
QUANT_PRESETS = {
    "standard": ["Q4_K_M", "Q5_K_M", "Q8_0"],
    "full":     ["Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0"],
    "imatrix":  ["IQ3_M", "IQ4_XS", "IQ4_NL"],
}
# Combine for 'all'
QUANT_PRESETS["all"] = QUANT_PRESETS["full"] + QUANT_PRESETS["imatrix"]
DEFAULT_QUANTS = QUANT_PRESETS["full"]

# Metadata for each quant type used by model_card.py
QUANT_INFO = {
    "Q2_K": {"bits": "2-bit", "recommended_for": "Extreme compression, significant quality loss.", "use_case": "Desperation / very low RAM", "quality": "⭐"},
    "Q3_K_S": {"bits": "3-bit", "recommended_for": "Very high compression, high quality loss.", "use_case": "Low RAM", "quality": "⭐⭐"},
    "Q3_K_M": {"bits": "3-bit", "recommended_for": "Very small file. Quality drop noticeable.", "use_case": "Very limited RAM", "quality": "⭐⭐⭐"},
    "Q3_K_L": {"bits": "3-bit", "recommended_for": "Slightly better than Q3_K_M, still a compromise.", "use_case": "Limited RAM", "quality": "⭐⭐⭐"},
    "Q4_K_S": {"bits": "4-bit", "recommended_for": "Good speed/size balance, slight quality loss.", "use_case": "General use", "quality": "⭐⭐⭐½"},
    "Q4_K_M": {"bits": "4-bit", "recommended_for": "Best balance of size and quality. Recommended for most users.", "use_case": "General use, everyday inference", "quality": "⭐⭐⭐⭐"},
    "Q5_K_S": {"bits": "5-bit", "recommended_for": "Large but accurate.", "use_case": "High accuracy needs", "quality": "⭐⭐⭐⭐"},
    "Q5_K_M": {"bits": "5-bit", "recommended_for": "Better quality than Q4, slightly larger. Great if you have the RAM.", "use_case": "When you want a bit more accuracy", "quality": "⭐⭐⭐⭐½"},
    "Q6_K": {"bits": "6-bit", "recommended_for": "Near-perfect quality, very large.", "use_case": "High accuracy, evaluation", "quality": "⭐⭐⭐⭐⭐"},
    "Q8_0": {"bits": "8-bit", "recommended_for": "Closest to original quality. Use when RAM is not a concern.", "use_case": "High-quality inference, evaluation", "quality": "⭐⭐⭐⭐⭐"},
    "IQ3_M": {"bits": "3-bit", "recommended_for": "Importance matrix quant. Excellent quality for the size.", "use_case": "Low RAM, imatrix optimized", "quality": "⭐⭐⭐½"},
    "IQ4_XS": {"bits": "4-bit", "recommended_for": "Smallest file with good quality. Great for low-RAM devices.", "use_case": "Minimal RAM usage, imatrix optimized", "quality": "⭐⭐⭐⭐"},
    "IQ4_NL": {"bits": "4-bit", "recommended_for": "Non-linear quant. Highly accurate for the size.", "use_case": "General use, imatrix optimized", "quality": "⭐⭐⭐⭐"},
}

def verify_token() -> bool:
    """Return True if HF token is loaded."""
    return bool(HF_TOKEN)

def verify_llama_cpp() -> dict:
    """Check which llama.cpp binaries are present."""
    return {
        "quantize":   LLAMA_QUANTIZE.exists(),
        "cli":        LLAMA_CLI.exists(),
        "bench":      LLAMA_BENCH.exists(),
        "imatrix":    LLAMA_IMATRIX.exists(),
        "perplexity": LLAMA_PERPLEXITY.exists(),
        "convert":    CONVERT_SCRIPT.exists(),
    }
