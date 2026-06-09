"""
benchmark.py — Measure GGUF model performance
================================================
Uses llama-bench (pre-built binary) to measure inference speed.

Metrics per quant:
  - TG speed  : token generation (tokens/sec) — what users care about
  - PP speed  : prompt processing (tokens/sec)

Usage:
    python benchmark.py --model gemma-4-12B
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os, json, time, threading, subprocess, argparse, platform, psutil
from pathlib import Path
from config import OUTPUT_DIR, LLAMA_CPP_DIR, LLAMA_BENCH

# ─── Helpers ──────────────────────────────────────────────────────────────────

def print_step(tag: str, msg: str):
    colors = {"info": "\033[94m", "ok": "\033[92m", "warn": "\033[93m", "err": "\033[91m"}
    icons  = {"info": "->", "ok": "[OK]", "warn": "[!]", "err": "[ERR]"}
    print(f"{colors.get(tag,'')} {icons.get(tag,'')} {msg}\033[0m", flush=True)


def get_size_gb(path: Path) -> float:
    return round(path.stat().st_size / (1024 ** 3), 2)


def detect_gpu() -> str:
    r = subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader",
                       shell=True, capture_output=True, text=True)
    return r.stdout.strip().split("\n")[0] if r.returncode == 0 else "CPU only"


# ─── Benchmark one GGUF ───────────────────────────────────────────────────────

def benchmark_gguf(gguf_path: Path, ngl: int) -> dict:
    size_gb = get_size_gb(gguf_path)
    print_step("info", f"Benchmarking: {gguf_path.name}  ({size_gb:.1f} GB)")
    print(f"   Status: running llama-bench... (silent for a few minutes)", flush=True)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(LLAMA_CPP_DIR) + ":" + env.get("LD_LIBRARY_PATH", "")

    cmd = [
        str(LLAMA_BENCH),
        "-m",  str(gguf_path),
        "-ngl", str(ngl),
        "-t",  "4",
        "-p",  "512",
        "-n",  "128",
        "-r",  "1",
        "--output", "json",
    ]

    # Run in background thread + show live timer
    stdout_buf, stderr_buf, proc_ref = [], [], [None]

    def _run():
        proc_ref[0] = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, env=env)
        out, err = proc_ref[0].communicate(timeout=600)
        stdout_buf.append(out)
        stderr_buf.append(err)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    start = time.time()
    tick = 0
    spinner = ["|", "/", "-", "\\"]
    while t.is_alive():
        print(f"\r   {spinner[tick%4]}  {time.time()-start:.0f}s elapsed", end="", flush=True)
        tick += 1
        t.join(timeout=15)
    print()

    if not stdout_buf:
        print_step("err", "No output from llama-bench")
        return {"file": gguf_path.name, "error": "no output"}

    rc = proc_ref[0].returncode if proc_ref[0] else -1
    if rc != 0:
        print_step("err", f"llama-bench exited {rc}")
        print((stderr_buf[0] or "")[:300])
        return {"file": gguf_path.name, "error": f"exit {rc}"}

    # Parse JSON output
    pp_tps = tg_tps = None
    try:
        data = json.loads(stdout_buf[0])
        for entry in data:
            if isinstance(entry, dict):
                if entry.get("n_prompt", 0) > 0 and entry.get("n_gen", 0) == 0:
                    pp_tps = round(float(entry.get("avg_ts", 0)), 1)
                elif entry.get("n_gen", 0) > 0 and entry.get("n_prompt", 0) == 0:
                    tg_tps = round(float(entry.get("avg_ts", 0)), 1)
    except Exception:
        # Markdown table fallback
        for line in stdout_buf[0].splitlines():
            for tag, attr in [("pp", "pp_tps"), ("tg", "tg_tps")]:
                if "|" in line and tag in line.lower():
                    for p in line.split("|"):
                        try:
                            val = float(p.strip().split("±")[0])
                            if val > 0:
                                if attr == "pp_tps": pp_tps = round(val, 1)
                                else: tg_tps = round(val, 1)
                        except ValueError:
                            pass

    elapsed = round(time.time() - start, 1)
    if tg_tps:
        print_step("ok", f"TG: {tg_tps} tok/s  |  PP: {pp_tps} tok/s  |  {elapsed}s")
    else:
        print_step("warn", "Could not parse speeds — raw output:")
        print(stdout_buf[0][:400])

    return {
        "file":              gguf_path.name,
        "size_gb":           size_gb,
        "pp_tokens_per_sec": pp_tps or "N/A",
        "tg_tokens_per_sec": tg_tps or "N/A",
        "gpu_layers":        ngl,
        "elapsed_sec":       elapsed,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--ngl", type=int, default=99,
                        help="GPU layers. 99=all on GPU, 0=CPU only")
    args = parser.parse_args()

    if not LLAMA_BENCH.exists():
        print_step("err", f"llama-bench not found at: {LLAMA_BENCH}")
        print_step("info", "Re-run Cell 2 (setup)")
        sys.exit(1)

    model_dir = OUTPUT_DIR / args.model
    gguf_files = sorted(f for f in model_dir.glob("*.gguf") if "F16" not in f.name)
    if not gguf_files:
        print_step("err", f"No GGUFs in {model_dir}")
        print_step("info", "Run Cell 7 to restore from Drive first")
        sys.exit(1)

    gpu = detect_gpu()
    print(f"\n{'='*62}")
    print(f"  Model   : {args.model}")
    print(f"  Files   : {len(gguf_files)} GGUFs")
    print(f"  GPU     : {gpu}")
    print(f"  ngl     : {args.ngl}  (99=all GPU layers, 0=CPU)")
    print(f"{'='*62}\n")

    results = []
    for gguf in gguf_files:
        results.append(benchmark_gguf(gguf, args.ngl))
        print()

    # Save
    out = model_dir / "benchmark.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "model":    args.model,
            "hardware": {
                "gpu":    gpu,
                "cpu":    platform.processor() or "Unknown",
                "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
            },
            "results":  results,
        }, f, indent=2)
    print_step("ok", f"Saved → {out}")

    # Summary
    print(f"\n  {'File':<45} {'Size':>6}  {'TG tok/s':>10}  {'PP tok/s':>10}")
    print("  " + "─"*76)
    for r in results:
        if "error" in r:
            print(f"  {r['file']:<45}  ERROR")
        else:
            print(f"  {r['file']:<45} {r['size_gb']:>5.1f}G"
                  f"  {str(r['tg_tokens_per_sec']):>10}  {str(r['pp_tokens_per_sec']):>10}")
    print()


if __name__ == "__main__":
    main()
