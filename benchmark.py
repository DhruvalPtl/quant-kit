"""
benchmark.py — Measure GGUF model performance
================================================
Uses llama-bench.exe (non-interactive) to measure:
  - Prompt processing speed (pp512 tokens/sec)
  - Token generation speed (tg128 tokens/sec)
  - RAM usage

Usage:
    python benchmark.py --model Qwen2.5-1.5B-Instruct
    python benchmark.py --model Qwen2.5-7B-Instruct --ngl 0   # CPU only

Results are saved to output/<model>/benchmark.json (used by model_card.py)
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import json
import time
import subprocess
import argparse
import psutil
import platform
from pathlib import Path
from config import OUTPUT_DIR, LLAMA_CPP_DIR

LLAMA_BENCH = LLAMA_CPP_DIR / "llama-bench.exe"

# ─── Helpers ──────────────────────────────────────────────────────────────────

def print_step(step: str, msg: str):
    colors = {"info": "\033[94m", "ok": "\033[92m", "warn": "\033[93m", "err": "\033[91m"}
    reset = "\033[0m"
    icons = {"info": "->", "ok": "[OK]", "warn": "[!]", "err": "[ERR]"}
    print(f"{colors.get(step, '')}{icons.get(step, '-')} {msg}{reset}")


def get_file_size_gb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 3)


def benchmark_gguf(gguf_path: Path, ngl: int = 99) -> dict:
    """
    Run llama-bench.exe on a GGUF file.
    llama-bench outputs a markdown table with pp (prompt processing)
    and tg (token generation) speeds in tokens/sec.

    Example output:
      | model | size | params | backend | ngl | test  | t/s        |
      | ...   | ...  | ...    | Vulkan  | 99  | pp512 | 245.6±1.2  |
      | ...   | ...  | ...    | Vulkan  | 99  | tg128 |  18.3±0.1  |
    """
    print_step("info", f"Benchmarking {gguf_path.name}...")

    ram_before = psutil.virtual_memory().used / (1024 ** 2)
    start = time.time()

    cmd = [
        str(LLAMA_BENCH),
        "-m", str(gguf_path),
        "-ngl", str(ngl),       # GPU layers to offload (99 = all)
        "-t", "4",              # CPU threads
        "-p", "512",            # prompt tokens for pp test
        "-n", "128",            # output tokens for tg test
        "-r", "1",              # 1 repetition (faster)
        "--output", "json",     # clean JSON output — easy to parse
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,     # discard noisy stderr logs
            timeout=300,
            text=True,
        )
    except subprocess.TimeoutExpired:
        print_step("warn", "Benchmark timed out after 5 minutes — skipping")
        return {"file": gguf_path.name, "error": "timeout"}
    except Exception as e:
        print_step("err", f"Benchmark failed: {e}")
        return {"file": gguf_path.name, "error": str(e)}

    elapsed = time.time() - start
    ram_after = psutil.virtual_memory().used / (1024 ** 2)
    ram_used_mb = max(0, ram_after - ram_before)

    pp_tps = None   # prompt processing tokens/sec
    tg_tps = None   # token generation tokens/sec

    # Parse JSON output from llama-bench
    if result.returncode == 0 and result.stdout.strip():
        try:
            data = json.loads(result.stdout)
            # llama-bench JSON: list of {test_name, t_s, ...}
            for entry in data:
                test = entry.get("n_prompt", 0)
                if isinstance(entry, dict):
                    if entry.get("n_prompt", 0) > 0 and entry.get("n_gen", 0) == 0:
                        pp_tps = round(float(entry.get("avg_ts", 0)), 1)
                    elif entry.get("n_gen", 0) > 0 and entry.get("n_prompt", 0) == 0:
                        tg_tps = round(float(entry.get("avg_ts", 0)), 1)
        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback: parse the markdown table from stdout
            for line in result.stdout.splitlines():
                if "|" in line and "pp" in line.lower():
                    parts = [p.strip() for p in line.split("|")]
                    for p in parts:
                        try:
                            val = float(p.split("±")[0].strip())
                            if val > 0:
                                pp_tps = round(val, 1)
                        except (ValueError, IndexError):
                            pass
                if "|" in line and "tg" in line.lower():
                    parts = [p.strip() for p in line.split("|")]
                    for p in parts:
                        try:
                            val = float(p.split("±")[0].strip())
                            if val > 0:
                                tg_tps = round(val, 1)
                        except (ValueError, IndexError):
                            pass

    result_data = {
        "file":             gguf_path.name,
        "size_gb":          round(get_file_size_gb(gguf_path), 2),
        "elapsed_sec":      round(elapsed, 1),
        "ram_used_mb":      round(ram_used_mb, 0),
        "pp_tokens_per_sec": pp_tps or "N/A",   # prompt processing
        "tg_tokens_per_sec": tg_tps or "N/A",   # text generation (most important)
        "gpu_layers":       ngl,
    }

    if tg_tps:
        print_step("ok", f"  Generation: {tg_tps} tok/s | Prompt: {pp_tps} tok/s | RAM: {ram_used_mb:.0f} MB")
    else:
        print_step("warn", f"  Could not parse speed from output. Raw stdout below:")
        print(result.stdout[:500] if result.stdout else "(empty)")
        if result.returncode != 0:
            print_step("err", f"  llama-bench exited with code {result.returncode}")
            print(result.stderr[:300] if result.stderr else "")

    return result_data


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark GGUF quantized models using llama-bench")
    parser.add_argument("--model", "-m", required=True,
                        help="Model folder name in output/ (e.g. Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--ngl", type=int, default=99,
                        help="GPU layers to offload. Use 0 for CPU-only. Default: 99 (all layers)")
    args = parser.parse_args()

    if not LLAMA_BENCH.exists():
        print_step("err", f"llama-bench.exe not found at {LLAMA_BENCH}")
        print_step("info", "It should be inside your llama.cpp/ folder from the Vulkan zip")
        sys.exit(1)

    model_output_dir = OUTPUT_DIR / args.model
    if not model_output_dir.exists():
        print_step("err", f"Output folder not found: {model_output_dir}")
        print_step("info", "Run quantize.py first")
        sys.exit(1)

    gguf_files = sorted(f for f in model_output_dir.glob("*.gguf") if "F16" not in f.name)
    if not gguf_files:
        print_step("err", "No quantized GGUF files found")
        sys.exit(1)

    print("\n" + "="*60)
    print(f"  Benchmarking: {args.model}")
    print(f"  Files:        {len(gguf_files)}")
    print(f"  GPU layers:   {args.ngl} ({'CPU only' if args.ngl == 0 else 'GPU offload'})")
    print("="*60 + "\n")

    results = []
    for gguf in gguf_files:
        data = benchmark_gguf(gguf, ngl=args.ngl)
        results.append(data)
        print()

    # Save results
    out_file = model_output_dir / "benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "model":    args.model,
            "hardware": get_hardware_info(),
            "results":  results,
        }, f, indent=2)

    print_step("ok", f"Results saved -> {out_file}")

    # Summary table
    print("\n  Summary:")
    print(f"  {'File':<45} {'Size':>6} {'TG Speed':>12} {'PP Speed':>12}")
    print("  " + "-"*78)
    for r in results:
        tg  = f"{r['tg_tokens_per_sec']} tok/s" if r.get('tg_tokens_per_sec') not in ("N/A", None) else "N/A"
        pp  = f"{r['pp_tokens_per_sec']} tok/s" if r.get('pp_tokens_per_sec') not in ("N/A", None) else "N/A"
        print(f"  {r['file']:<45} {r['size_gb']:>5.1f}G {tg:>12} {pp:>12}")
    print()


def get_hardware_info() -> dict:
    info = {
        "os":     platform.system(),
        "cpu":    platform.processor() or "Unknown CPU",
        "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 1),
        "gpu":    "Intel Arc 140V (Vulkan)",
    }
    return info


if __name__ == "__main__":
    main()
