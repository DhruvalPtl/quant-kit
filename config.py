"""
config.py — Shared configuration loader
=========================================
Loads settings from .env file. All other scripts import from here.

.env file format:
    hf_token = "hf_xxxxxxxxxxxxxxxxxxxx"
"""

import os
from pathlib import Path
from dotenv import load_dotenv

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
MODELS_DIR    = ROOT_DIR / "models"
OUTPUT_DIR    = ROOT_DIR / "output"

# ── llama.cpp binaries ─────────────────────────────────────────────────────────
LLAMA_QUANTIZE  = LLAMA_CPP_DIR / "llama-quantize.exe"
LLAMA_CLI       = LLAMA_CPP_DIR / "llama-cli.exe"
LLAMA_IMATRIX   = LLAMA_CPP_DIR / "llama-imatrix.exe"
CONVERT_SCRIPT  = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"

# ── Default quant types ────────────────────────────────────────────────────────
DEFAULT_QUANTS = ["Q4_K_M", "Q5_K_M", "Q8_0"]


def verify_token() -> bool:
    """Return True if HF token is loaded."""
    return bool(HF_TOKEN)


def verify_llama_cpp() -> dict:
    """Check which llama.cpp binaries are present."""
    return {
        "quantize": LLAMA_QUANTIZE.exists(),
        "cli":      LLAMA_CLI.exists(),
        "convert":  CONVERT_SCRIPT.exists(),
    }
