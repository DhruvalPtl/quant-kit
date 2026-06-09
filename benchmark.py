"""
benchmark.py — Measure GGUF model performance using GPU
=========================================================
Uses llama-cpp-python (CUDA build) to measure real GPU inference speed.
Falls back to CPU if CUDA is not available.

Metrics:
  - TG speed  : token generation (tokens/sec) — most important for users
  - PP speed  : prompt processing (tokens/sec)

Usage:
    python benchmark.py --model gemma-4-12B
    python benchmark.py --model gemma-4-12B --ngl 0    # CPU only
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import json
import time
import platform
import argparse
import psutil
from pathlib import Path
from config import OUTPUT_DIR, LLAMA_CPP_DIR

# ─── Helpers ──────────────────────────────────────────────────────────────────

def print_step(step: str, msg: str):
    colors = {"info": "\033[94m", "ok": "\033[92m", "warn": "\033[93m", "err": "\033[91m"}
    reset  = "\033[0m"
    icons  = {"info": "->", "ok": "[OK]", "warn": "[!]", "err": "[ERR]"}
    print(f"{colors.get(step, '')}{icons.get(step, '-')} {msg}{reset}", flush=True)


def get_size_gb(path: Path) -> float:
    return round(path.stat().st_size / (1024 ** 3), 2)


def detect_gpu() -> str:
    """Return GPU name or 'CPU only'."""
    try:
        import subprocess
        r = subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader",
                           shell=True, capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return "CPU only"


# ─── GPU Benchmark (llama-cpp-python) ─────────────────────────────────────────

def benchmark_gguf_gpu(gguf_path: Path, ngl: int = -1) -> dict:
    """
    Benchmark using llama-cpp-python (CUDA-enabled).
    ngl=-1 means offload ALL layers to GPU automatically.
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        print_step("err", "llama-cpp-python not installed — re-run Cell 2 (setup)")
        return {"file": gguf_path.name, "error": "llama-cpp-python not found"}

    size_gb = get_size_gb(gguf_path)
    print_step("info", f"Benchmarking: {gguf_path.name}  ({size_gb:.1f} GB)")
    print(f"   GPU layers : {ngl if ngl >= 0 else 'ALL (auto)'}", flush=True)
    print(f"   Loading model into GPU...", flush=True)

    ram_before = psutil.virtual_memory().used / (1024 ** 2)
    t_start    = time.time()

    try:
        llm = Llama(
            model_path   = str(gguf_path),
            n_gpu_layers = ngl,      # -1 = offload all layers
            n_ctx        = 640,      # 512 prompt + 128 generation headroom
            verbose      = False,
        )
    except Exception as e:
        print_step("err", f"Failed to load model: {e}")
        return {"file": gguf_path.name, "error": str(e)}

    load_elapsed = time.time() - t_start
    print(f"   Model loaded in {load_elapsed:.1f}s", flush=True)

    # ── Prompt Processing (PP) speed ──────────────────────────────────────────
    # Tokenize a 512-token prompt and eval it
    pp_prompt  = ("The quick brown fox jumps over the lazy dog. " * 30)[:1000]
    pp_tokens  = llm.tokenize(pp_prompt.encode())[:512]

    print(f"   Running PP benchmark ({len(pp_tokens)} tokens)...", flush=True)
    t0 = time.time()
    llm.eval(pp_tokens)
    pp_elapsed = time.time() - t0
    pp_tps = round(len(pp_tokens) / pp_elapsed, 1) if pp_elapsed > 0 else 0

    # ── Token Generation (TG) speed ───────────────────────────────────────────
    # Generate 128 tokens from a short seed
    seed_tokens = pp_tokens[:10]
    print(f"   Running TG benchmark (128 tokens)...", flush=True)
    t0 = time.time()
    count = 0
    for _ in llm.generate(seed_tokens, temp=0.0, top_k=1, reset=True):
        count += 1
        if count >= 128:
            break
    tg_elapsed = time.time() - t0
    tg_tps = round(count / tg_elapsed, 1) if tg_elapsed > 0 else 0

    ram_after   = psutil.virtual_memory().used / (1024 ** 2)
    ram_used_mb = round(max(0, ram_after - ram_before), 0)
    total_sec   = round(time.time() - t_start, 1)

    print_step("ok", f"TG: {tg_tps} tok/s  |  PP: {pp_tps} tok/s  |  RAM: {ram_used_mb:.0f} MB  |  Total: {total_sec}s")

    # Clean up to free VRAM before next model
    del llm

    return {
        "file"             : gguf_path.name,
        "size_gb"          : size_gb,
        "pp_tokens_per_sec": pp_tps,
        "tg_tokens_per_sec": tg_tps,
        "ram_used_mb"      : ram_used_mb,
        "gpu_layers"       : ngl,
        "elapsed_sec"      : total_sec,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark GGUF quantized models with GPU")
    parser.add_argument("--model", "-m", required=True,
                        help="Model folder name in output/ (e.g. gemma-4-12B)")
    parser.add_argument("--ngl", type=int, default=-1,
                        help="GPU layers to offload. -1=all (default), 0=CPU only")
    args = parser.parse_args()

    model_output_dir = OUTPUT_DIR / args.model
    if not model_output_dir.exists():
        print_step("err", f"Output folder not found: {model_output_dir}")
        print_step("info", "Run Cell 7 to restore GGUFs from Drive first")
        sys.exit(1)

    gguf_files = sorted(
        f for f in model_output_dir.glob("*.gguf") if "F16" not in f.name
    )
    if not gguf_files:
        print_step("err", "No quantized GGUF files found in output folder")
        sys.exit(1)

    gpu_name = detect_gpu()
    ngl_label = "ALL layers on GPU" if args.ngl < 0 else (
                 "CPU only" if args.ngl == 0 else f"{args.ngl} layers on GPU")

    print("\n" + "=" * 62)
    print(f"  Benchmarking : {args.model}")
    print(f"  Files        : {len(gguf_files)} GGUFs")
    print(f"  GPU          : {gpu_name}")
    print(f"  Mode         : {ngl_label}")
    print("=" * 62 + "\n")

    results = []
    for gguf in gguf_files:
        data = benchmark_gguf_gpu(gguf, ngl=args.ngl)
        results.append(data)
        print()

    # ── Save results ──────────────────────────────────────────────────────────
    out_file = model_output_dir / "benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "model"   : args.model,
            "hardware": {
                "os"    : platform.system(),
                "cpu"   : platform.processor() or "Unknown",
                "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 1),
                "gpu"   : gpu_name,
            },
            "results" : results,
        }, f, indent=2)

    print_step("ok", f"Results saved → {out_file}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n  Summary:")
    print(f"  {'File':<45} {'Size':>6}  {'TG tok/s':>10}  {'PP tok/s':>10}")
    print("  " + "─" * 76)
    for r in results:
        if "error" in r:
            print(f"  {r['file']:<45}  {'ERROR':>10}")
        else:
            tg = f"{r['tg_tokens_per_sec']}" if r.get('tg_tokens_per_sec') else "N/A"
            pp = f"{r['pp_tokens_per_sec']}" if r.get('pp_tokens_per_sec') else "N/A"
            print(f"  {r['file']:<45} {r['size_gb']:>5.1f}G  {tg:>10}  {pp:>10}")
    print()


if __name__ == "__main__":
    main()
