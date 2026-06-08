"""
benchmark.py — Measure GGUF model performance
================================================
Runs a quick inference test on each GGUF file and reports:
  - Tokens per second (generation speed)
  - RAM usage (system memory)
  - Model load time

Usage:
    python benchmark.py --model Qwen2.5-1.5B-Instruct
    python benchmark.py --model Qwen2.5-7B-Instruct --prompt "Explain quantum computing"

Results are saved to output/<model>/benchmark.json (used by model_card.py)
"""

import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import json
import time
import subprocess
import argparse
import psutil
from pathlib import Path
from config import OUTPUT_DIR, LLAMA_CLI

# Short prompt for consistent benchmarking across all models
DEFAULT_PROMPT = "Explain the difference between machine learning and deep learning in simple terms."
DEFAULT_N_TOKENS = 128  # Generate 128 tokens for the speed test


def print_step(step: str, msg: str):
    colors = {"info": "\033[94m", "ok": "\033[92m", "warn": "\033[93m", "err": "\033[91m"}
    reset = "\033[0m"
    icons = {"info": "->", "ok": "[OK]", "warn": "[!]", "err": "[ERR]"}
    print(f"{colors.get(step, '')}{icons.get(step, '-')} {msg}{reset}")


def get_ram_usage_mb() -> float:
    """Returns current process + children RAM usage in MB."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss
    return mem / (1024 * 1024)


def get_file_size_gb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 3)


def benchmark_gguf(gguf_path: Path, prompt: str, n_tokens: int) -> dict:
    """
    Run llama-cli on a GGUF file and parse the performance stats.
    llama.cpp prints timing info at the end like:
      llama_print_timings: eval time = ... ms / N tokens
    """
    if not LLAMA_CLI.exists():
        print_step("warn", f"llama-cli.exe not found — skipping inference benchmark")
        return {"error": "llama-cli not found"}

    print_step("info", f"Benchmarking {gguf_path.name}...")

    # Get RAM before loading model
    ram_before = psutil.virtual_memory().used / (1024 ** 2)
    load_start = time.time()

    cmd = [
        str(LLAMA_CLI),
        "-m", str(gguf_path),
        "-p", prompt,
        "-n", str(n_tokens),
        "--n-gpu-layers", "99",   # offload all layers to GPU if available
        "--threads", "4",
        "--log-disable",           # suppress verbose logs
        "-s", "42",                # fixed seed for reproducibility
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 min max
        )
    except subprocess.TimeoutExpired:
        print_step("warn", "Benchmark timed out after 5 minutes")
        return {"error": "timeout"}

    load_time = time.time() - load_start
    ram_after = psutil.virtual_memory().used / (1024 ** 2)
    ram_used_mb = max(0, ram_after - ram_before)

    # Parse llama.cpp timing output
    tokens_per_sec = None
    for line in result.stderr.splitlines() + result.stdout.splitlines():
        if "eval time" in line and "tokens per second" in line:
            # Example: llama_print_timings: eval time = 2345.67 ms / 128 runs ( 18.33 ms per token, 54.55 tokens per second)
            try:
                tps_part = line.split("tokens per second")[0].strip()
                tokens_per_sec = float(tps_part.split(",")[-1].strip().split()[-1])
            except Exception:
                pass

        # Alternative parse: look for tps directly
        if "tok/s" in line:
            try:
                tokens_per_sec = float(line.split("tok/s")[0].strip().split()[-1])
            except Exception:
                pass

    result_data = {
        "file": gguf_path.name,
        "size_gb": round(get_file_size_gb(gguf_path), 2),
        "load_time_sec": round(load_time, 1),
        "ram_used_mb": round(ram_used_mb, 0),
        "tokens_per_sec": round(tokens_per_sec, 1) if tokens_per_sec else "N/A",
        "n_tokens_generated": n_tokens,
        "prompt": prompt[:80] + "..." if len(prompt) > 80 else prompt,
    }

    if tokens_per_sec:
        print_step("ok", f"  Speed: {tokens_per_sec:.1f} tok/s | RAM: {ram_used_mb:.0f} MB | Load: {load_time:.1f}s")
    else:
        print_step("warn", f"  Could not parse speed — Load: {load_time:.1f}s | RAM: {ram_used_mb:.0f} MB")

    return result_data


def main():
    parser = argparse.ArgumentParser(description="Benchmark GGUF quantized models")
    parser.add_argument("--model", "-m", required=True, help="Model folder name inside output/ (e.g. Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--prompt", "-p", default=DEFAULT_PROMPT, help="Prompt to use for benchmarking")
    parser.add_argument("--n-tokens", "-n", type=int, default=DEFAULT_N_TOKENS, help="Number of tokens to generate")

    args = parser.parse_args()

    model_output_dir = OUTPUT_DIR / args.model
    if not model_output_dir.exists():
        print_step("err", f"Output folder not found: {model_output_dir}")
        print_step("info", "Run quantize.py first to create GGUF files")
        sys.exit(1)

    gguf_files = sorted(model_output_dir.glob("*.gguf"))
    # Skip the F16 base file — too large, not useful for benchmarking
    gguf_files = [f for f in gguf_files if "F16" not in f.name]

    if not gguf_files:
        print_step("err", "No quantized GGUF files found (F16 excluded)")
        sys.exit(1)

    print("\n" + "="*60)
    print(f"  Benchmarking: {args.model}")
    print(f"  Files found:  {len(gguf_files)}")
    print(f"  Tokens:       {args.n_tokens}")
    print("="*60 + "\n")

    results = []
    for gguf in gguf_files:
        data = benchmark_gguf(gguf, args.prompt, args.n_tokens)
        results.append(data)
        print()

    # Save results to JSON for model_card.py to use
    out_file = model_output_dir / "benchmark.json"
    with open(out_file, "w") as f:
        json.dump({
            "model": args.model,
            "hardware": get_hardware_info(),
            "results": results,
        }, f, indent=2)

    print_step("ok", f"Benchmark results saved → {out_file}")

    # Print summary table
    print("\n  Summary:")
    print(f"  {'File':<45} {'Size':>6} {'Speed':>12} {'RAM':>10}")
    print("  " + "-" * 75)
    for r in results:
        tps = f"{r['tokens_per_sec']} tok/s" if r.get('tokens_per_sec') != "N/A" else "N/A"
        ram = f"{r['ram_used_mb']:.0f} MB" if r.get('ram_used_mb') else "N/A"
        print(f"  {r['file']:<45} {r['size_gb']:>5.1f}G {tps:>12} {ram:>10}")
    print()


def get_hardware_info() -> dict:
    """Collect basic hardware info for model card."""
    import platform
    info = {
        "os": platform.system(),
        "cpu": platform.processor() or "Unknown CPU",
        "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 1),
    }
    # Try to detect GPU
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info["gpu"] = result.stdout.strip()
    except Exception:
        info["gpu"] = "No NVIDIA GPU / Intel Arc / AMD"
    return info


if __name__ == "__main__":
    main()
