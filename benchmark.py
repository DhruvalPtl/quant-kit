"""
benchmark.py — Measure GGUF model performance
================================================
Uses llama-bench to measure prompt processing and token generation speeds.

Usage:
    python benchmark.py --model Qwen2.5-1.5B-Instruct
    python benchmark.py --model Qwen2.5-7B-Instruct --ngl 0   # CPU only
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
import time
import subprocess
import argparse
from pathlib import Path

from config import OUTPUT_DIR, LLAMA_BENCH
from utils import print_step, get_hardware_info

def get_file_size_gb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 3)

def benchmark_gguf(gguf_path: Path, ngl: int = 99, threads: int = 4, reps: int = 3) -> dict:
    print_step("info", f"Benchmarking {gguf_path.name}...")
    start = time.time()

    cmd = [
        str(LLAMA_BENCH),
        "-m", str(gguf_path),
        "-ngl", str(ngl),
        "-t", str(threads),
        "-p", "512",
        "-n", "128",
        "-r", str(reps),
        "--output", "json",
    ]

    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600, text=True)
    except subprocess.TimeoutExpired:
        print_step("warn", "Benchmark timed out after 10 minutes — skipping")
        return {"file": gguf_path.name, "error": "timeout"}
    except Exception as e:
        print_step("err", f"Benchmark failed: {e}")
        return {"file": gguf_path.name, "error": str(e)}

    elapsed = time.time() - start
    pp_tps = None
    tg_tps = None

    if result.returncode == 0 and result.stdout.strip():
        try:
            data = json.loads(result.stdout)
            for entry in data:
                if isinstance(entry, dict):
                    if entry.get("n_prompt", 0) > 0 and entry.get("n_gen", 0) == 0:
                        pp_tps = round(float(entry.get("avg_ts", 0)), 1)
                    elif entry.get("n_gen", 0) > 0 and entry.get("n_prompt", 0) == 0:
                        tg_tps = round(float(entry.get("avg_ts", 0)), 1)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    result_data = {
        "file":             gguf_path.name,
        "size_gb":          round(get_file_size_gb(gguf_path), 2),
        "elapsed_sec":      round(elapsed, 1),
        "pp_tokens_per_sec": pp_tps or "N/A",
        "tg_tokens_per_sec": tg_tps or "N/A",
        "gpu_layers":       ngl,
    }

    if tg_tps:
        print_step("ok", f"  Generation: {tg_tps} tok/s | Prompt: {pp_tps} tok/s")
    else:
        print_step("warn", f"  Could not parse speed. Raw stdout:")
        print(result.stdout[:500] if result.stdout else "(empty)")

    return result_data

def main():
    parser = argparse.ArgumentParser(description="Benchmark GGUF quantized models")
    parser.add_argument("--model", "-m", required=True, help="Model folder name in output/")
    parser.add_argument("--ngl", type=int, default=99, help="GPU layers to offload")
    parser.add_argument("--threads", "-t", type=int, default=4, help="CPU threads")
    parser.add_argument("--reps", "-r", type=int, default=1, help="Repetitions (default 1)")
    args = parser.parse_args()

    if not LLAMA_BENCH.exists():
        print_step("err", f"llama-bench not found at {LLAMA_BENCH}")
        sys.exit(1)

    model_dir = OUTPUT_DIR / args.model
    if not model_dir.exists():
        print_step("err", f"Output folder not found: {model_dir}")
        sys.exit(1)

    gguf_files = sorted(f for f in model_dir.glob("*.gguf") if "F16" not in f.name)
    if not gguf_files:
        print_step("err", "No GGUF files found")
        sys.exit(1)

    print("\n" + "="*60)
    print(f"  Benchmarking: {args.model}")
    print(f"  GPU layers:   {args.ngl}")
    print("="*60 + "\n")

    results = []
    for gguf in gguf_files:
        results.append(benchmark_gguf(gguf, args.ngl, args.threads, args.reps))
        print()

    out_file = model_dir / "benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "hardware": get_hardware_info(),
            "results": results,
        }, f, indent=2)

    print_step("ok", f"Results saved -> {out_file}")

if __name__ == "__main__":
    main()
