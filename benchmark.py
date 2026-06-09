"""
benchmark.py — Complete speed benchmark for GGUF models
=========================================================
Measures: prompt processing speed, token generation speed,
multi-context performance, CPU vs GPU comparison, and RAM usage.

Usage:
    python benchmark.py --model gemma-4-12b-it
    python benchmark.py --model gemma-4-12b-it --ngl 0   # CPU only
    python benchmark.py --model gemma-4-12b-it --quick   # single quant test
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
import time
import threading
import subprocess
import argparse
from pathlib import Path

import psutil

from config import OUTPUT_DIR, LLAMA_BENCH
from utils import print_step, get_hardware_info, format_size

# ── RAM sampler ────────────────────────────────────────────────────────────────
class RAMSampler:
    """Background thread that records peak RAM usage during a benchmark run."""
    def __init__(self):
        self.peak_mb = 0
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        return round(self.peak_mb / 1024, 2)   # return GB

    def _sample(self):
        while self._running:
            used = psutil.virtual_memory().used / (1024 ** 2)
            if used > self.peak_mb:
                self.peak_mb = used
            time.sleep(0.25)

# ── Core benchmark function ────────────────────────────────────────────────────
def run_llama_bench(gguf_path: Path, ngl: int, threads: int, pp: int, n: int, reps: int) -> dict:
    """Run llama-bench and return parsed PP/TG speeds."""
    cmd = [
        str(LLAMA_BENCH),
        "-m",    str(gguf_path),
        "-ngl",  str(ngl),
        "-t",    str(threads),
        "-p",    str(pp),
        "-n",    str(n),
        "-r",    str(reps),
        "--output", "json",
    ]

    ram = RAMSampler()
    ram.start()

    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=900, text=True
        )
    except subprocess.TimeoutExpired:
        ram.stop()
        return {"error": "timeout"}
    except Exception as e:
        ram.stop()
        return {"error": str(e)}

    peak_ram_gb = ram.stop()

    pp_tps, tg_tps = None, None
    if result.returncode == 0 and result.stdout.strip():
        try:
            data = json.loads(result.stdout)
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                if entry.get("n_prompt", 0) > 0 and entry.get("n_gen", 0) == 0:
                    pp_tps = round(float(entry.get("avg_ts", 0)), 2)
                elif entry.get("n_gen", 0) > 0 and entry.get("n_prompt", 0) == 0:
                    tg_tps = round(float(entry.get("avg_ts", 0)), 2)
        except Exception:
            pass

    return {
        "pp_tokens_per_sec": pp_tps or "N/A",
        "tg_tokens_per_sec": tg_tps or "N/A",
        "peak_ram_gb":       peak_ram_gb,
        "context_size":      pp,
    }

# ── Per-file benchmark (multi-context + CPU vs GPU) ───────────────────────────
def benchmark_file(gguf_path: Path, ngl: int, threads: int, reps: int, quick: bool) -> dict:
    size_gb = round(gguf_path.stat().st_size / (1024 ** 3), 2)
    print_step("info", f"{'─'*50}")
    print_step("info", f"File : {gguf_path.name}  ({size_gb} GB)")

    contexts = [512] if quick else [128, 512, 2048]
    context_results = []

    # ── GPU run (or whatever ngl is) ──
    print_step("info", f"  GPU layers: {ngl}")
    for ctx in contexts:
        print_step("info", f"  Context {ctx}...", )
        r = run_llama_bench(gguf_path, ngl, threads, ctx, 128, reps)
        r["mode"] = "GPU" if ngl > 0 else "CPU"
        context_results.append(r)
        if r.get("tg_tokens_per_sec") != "N/A":
            print_step("ok", f"    ctx={ctx}: TG={r['tg_tokens_per_sec']} tok/s  PP={r['pp_tokens_per_sec']} tok/s  RAM={r['peak_ram_gb']} GB")
        else:
            print_step("warn", f"    ctx={ctx}: Could not parse results")

    # ── CPU-only comparison (skip if already CPU) ──
    if ngl > 0 and not quick:
        print_step("info", f"  CPU-only comparison (ngl=0)...")
        r_cpu = run_llama_bench(gguf_path, 0, threads, 512, 128, 1)
        r_cpu["mode"] = "CPU"
        context_results.append(r_cpu)
        if r_cpu.get("tg_tokens_per_sec") != "N/A":
            print_step("ok", f"    CPU: TG={r_cpu['tg_tokens_per_sec']} tok/s  RAM={r_cpu['peak_ram_gb']} GB")

    return {
        "file":       gguf_path.name,
        "size_gb":    size_gb,
        "results":    context_results,
    }

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Benchmark GGUF models — speed, context, CPU vs GPU, RAM")
    parser.add_argument("--model",   "-m", required=True, help="Model folder name in output/")
    parser.add_argument("--ngl",           type=int, default=99,   help="GPU layers (default 99 = all)")
    parser.add_argument("--threads", "-t", type=int, default=4,    help="CPU threads")
    parser.add_argument("--reps",    "-r", type=int, default=3,    help="Repetitions per run (default 3)")
    parser.add_argument("--quant",         help="Benchmark only one quant type (e.g. Q4_K_M)")
    parser.add_argument("--quick",         action="store_true", help="Quick mode: 1 context, 1 rep")
    args = parser.parse_args()

    if not LLAMA_BENCH.exists():
        print_step("err", f"llama-bench not found at {LLAMA_BENCH}")
        sys.exit(1)

    model_dir = OUTPUT_DIR / args.model
    if not model_dir.exists():
        print_step("err", f"Output folder not found: {model_dir}")
        sys.exit(1)

    all_ggufs = sorted(f for f in model_dir.glob("*.gguf") if "F16" not in f.name)
    if args.quant:
        all_ggufs = [f for f in all_ggufs if args.quant in f.name]
    if not all_ggufs:
        print_step("err", "No GGUF files found")
        sys.exit(1)

    hw = get_hardware_info()
    print("\n" + "="*60)
    print(f"  quant-kit — Speed Benchmark")
    print("="*60)
    print(f"  Model   : {args.model}")
    print(f"  CPU     : {hw['cpu']}")
    print(f"  GPU     : {hw['gpu']}")
    print(f"  RAM     : {hw['ram_gb']} GB")
    print(f"  Threads : {args.threads}  |  GPU layers: {args.ngl}  |  Reps: {args.reps}")
    print("="*60 + "\n")

    file_results = []
    for gguf in all_ggufs:
        res = benchmark_file(gguf, args.ngl, args.threads, args.reps, args.quick)
        file_results.append(res)
        print()

    out = {
        "model":    args.model,
        "hardware": hw,
        "files":    file_results,
    }
    out_file = model_dir / "benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print_step("ok", f"Results saved → {out_file}")
    print_step("info", "Run model_card.py to embed these results in your HuggingFace README.")

if __name__ == "__main__":
    main()
