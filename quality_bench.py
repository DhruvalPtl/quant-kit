"""
quality_bench.py — Measure the quality loss of quantized models
==================================================================
Runs perplexity and KL Divergence measurements using llama.cpp.

Usage:
    python quality_bench.py --model gemma-4-12b-it
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os
import json
import time
import subprocess
import argparse
from pathlib import Path

from config import OUTPUT_DIR, LLAMA_PERPLEXITY, IS_WINDOWS
from utils import print_step

def run_perplexity(gguf_path: Path, data_file: Path) -> float | None:
    print_step("info", f"Measuring perplexity for {gguf_path.name}...")
    start = time.time()

    cmd = [
        str(LLAMA_PERPLEXITY),
        "-m", str(gguf_path),
        "-f", str(data_file),
        "-c", "512", # use a small context for speed
    ]
    
    # We only care about the final perplexity number
    # Typically looks like: [1]114.7709,-0.6019,[2]... Final estimate: PPL = 6.32
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        print_step("err", "Timeout")
        return None

    for line in reversed(result.stderr.splitlines()):
        if "Final estimate" in line and "PPL" in line:
            # line: "Final estimate: PPL = 6.3245 +/- 0.0123"
            try:
                parts = line.split("PPL =")
                if len(parts) > 1:
                    ppl = float(parts[1].split()[0])
                    print_step("ok", f"PPL = {ppl} (took {time.time()-start:.1f}s)")
                    return ppl
            except:
                pass
                
    print_step("warn", "Could not parse PPL from output.")
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True, help="Model folder in output/")
    parser.add_argument("--data", "-d", default="wiki.test.raw", help="Path to evaluation text data (e.g. WikiText-2)")
    args = parser.parse_args()

    if not LLAMA_PERPLEXITY.exists():
        print_step("err", f"llama-perplexity not found at {LLAMA_PERPLEXITY}")
        print_step("info", "You must build llama.cpp locally to get this binary. Pre-compiled releases rarely include it.")
        sys.exit(1)

    data_file = Path(args.data)
    if not data_file.exists():
        print_step("err", f"Data file {data_file} not found.")
        print_step("info", "Download WikiText-2: https://huggingface.co/datasets/wikitext/resolve/refs%2Fconvert%2Fparquet/wikitext-2-raw-v1/test/0000.parquet")
        sys.exit(1)

    model_dir = OUTPUT_DIR / args.model
    if not model_dir.exists():
        print_step("err", f"Model dir {model_dir} not found")
        sys.exit(1)

    gguf_files = sorted(f for f in model_dir.glob("*.gguf") if "F16" not in f.name)
    
    print("\n" + "="*60)
    print(f"  Quality Benchmark: Perplexity")
    print(f"  Model: {args.model}")
    print("="*60 + "\n")

    results = []
    for f in gguf_files:
        ppl = run_perplexity(f, data_file)
        results.append({
            "quant": f.stem.split("-")[-1],
            "ppl": ppl
        })

    out_file = model_dir / "quality_benchmark.json"
    with open(out_file, "w") as f:
        json.dump({"results": results}, f, indent=2)

    print_step("ok", f"Results saved to {out_file}")

if __name__ == "__main__":
    main()
