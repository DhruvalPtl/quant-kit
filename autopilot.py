"""
autopilot.py — Fully Automated Quantization Pipeline
======================================================
Discovers trending models → quantizes → generates model card → uploads to HuggingFace.
Loops until it has processed the requested number of models.

Every step is shown in the terminal. Storage is monitored before each step.
A full log is saved to autopilot_log.json.

Usage:
    python autopilot.py --count 3
    python autopilot.py --count 5 --max-gb 8 --preset full
    python autopilot.py --count 2 --task image-text-to-text   (VLMs)
    python autopilot.py --count 3 --min-downloads 10000       (high-traffic only)
    python autopilot.py --dry-run                              (discover only, no quant)
"""

import sys
import io
# Only wrap stdout if running in a real terminal (not redirected)
# The wrapper was causing double-output when piped through PowerShell
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import json
import shutil
import subprocess
import argparse
import time
import os
from datetime import datetime
from pathlib import Path

from config import OUTPUT_DIR, MODELS_DIR, HF_TOKEN
from utils import print_step, format_size
from auto_discover import discover

LOG_FILE = Path(__file__).parent / "autopilot_log.json"

# ─── Storage helpers ──────────────────────────────────────────────────────────

def free_gb() -> float:
    return shutil.disk_usage(str(Path(__file__).parent)).free / (1024 ** 3)

def used_gb_dir(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 ** 3)

def print_storage_bar(label: str = "Disk"):
    """Print a visual storage bar to terminal."""
    total = shutil.disk_usage(str(Path(__file__).parent)).total / (1024 ** 3)
    used  = total - free_gb()
    pct   = used / total
    bar_len = 40
    filled  = int(bar_len * pct)
    bar     = "█" * filled + "░" * (bar_len - filled)

    color = ""
    if pct > 0.9:
        color = "🔴"
    elif pct > 0.75:
        color = "🟡"
    else:
        color = "🟢"

    print(f"  {color} {label}: [{bar}] {used:.1f}/{total:.1f} GB ({pct*100:.0f}%)")
    print(f"     Free: {free_gb():.1f} GB")

def storage_alert(required_gb: float, label: str = "next step") -> bool:
    """
    Warn if free space is dangerously low.
    Returns False if there's not enough space to proceed safely.
    """
    free = free_gb()
    print()
    print_storage_bar(f"Before {label}")

    if free < required_gb:
        print()
        print(f"  🔴 STORAGE ALERT: Only {free:.1f} GB free, need {required_gb:.1f} GB for {label}!")
        print(f"  💡 Tip: Run 'python check_corrupted.py' to find and remove incomplete files.")
        print(f"  💡 Or: delete models/ cache: Remove-Item models\\ -Recurse -Force")
        print()
        return False

    if free < required_gb + 5:
        print()
        print(f"  🟡 STORAGE WARNING: Only {free:.1f} GB free. Cutting it close for {label}.")
        print()

    return True


# ─── Log helpers ──────────────────────────────────────────────────────────────

def load_log() -> dict:
    if LOG_FILE.exists():
        try:
            return json.loads(LOG_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"runs": [], "completed": [], "failed": [], "skipped": []}

def save_log(log: dict):
    LOG_FILE.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

def log_event(log: dict, event_type: str, model_id: str, details: dict = None):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "model_id": model_id,
        **(details or {}),
    }
    log.setdefault(event_type, []).append(entry)
    save_log(log)


# ─── Step runner ──────────────────────────────────────────────────────────────

def run_step(label: str, cmd: list[str], cwd: str = None) -> bool:
    """Run a subprocess step with full terminal output. Returns True if success."""
    print()
    print("  " + "─" * 56)
    print(f"  ▶  {label}")
    print(f"     {' '.join(str(c) for c in cmd)}")
    print("  " + "─" * 56)
    t0 = time.time()

    result = subprocess.run(cmd, cwd=cwd)
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"  ✅ {label} — done in {elapsed:.0f}s")
        return True
    else:
        print(f"  ❌ {label} — FAILED (exit {result.returncode}) after {elapsed:.0f}s")
        return False


# ─── Per-model pipeline ───────────────────────────────────────────────────────

def process_model(candidate: dict, preset: str, log: dict, dry_run: bool = False) -> bool:
    """
    Run the full pipeline for one model:
      1. Quantize (quantize.py or quantize_vlm.py via quant.py)
      2. Generate model card
      3. Upload to HuggingFace
    Returns True if all steps succeeded.
    """
    model_id   = candidate["model_id"]
    model_name = candidate["model_name"]
    model_type = candidate["model_type"]
    size_gb    = candidate["size_gb"] or 5.0  # fallback if unknown

    py = sys.executable
    root = str(Path(__file__).parent)

    print()
    print("  ╔" + "═" * 58 + "╗")
    print(f"  ║  🚀 PROCESSING: {model_id:<40} ║")
    print(f"  ║  Type: {model_type.upper():<6} | Size: {size_gb:.1f} GB | Arch: {candidate['architecture'][:30]:<30} ║")
    print("  ╚" + "═" * 58 + "╝")

    if dry_run:
        print_step("info", "  --dry-run: skipping actual quantization")
        return True

    # ── Storage check before download ────────────────────────────────────────
    # Need: download size + F16 size (~2x) + quant outputs (~1.5x per quant × 10)
    # Conservative: 5x download size
    required = max(size_gb * 5, 10.0)
    if not storage_alert(required, f"download+quant of {model_name}"):
        log_event(log, "skipped", model_id, {"reason": "insufficient disk space",
                                               "free_gb": free_gb(), "required_gb": required})
        return False

    # ── Step 1: Quantize ─────────────────────────────────────────────────────
    quant_script = "quant.py"
    quant_cmd = [py, quant_script, "--model", model_id, "--preset", preset, "--delete-src"]

    if model_type == "vlm":
        quant_cmd = [py, "quantize_vlm.py", "--model", model_id,
                     "--preset", preset, "--delete-src"]

    ok = run_step(f"Quantize {model_id} ({preset} preset)", quant_cmd, cwd=root)
    if not ok:
        log_event(log, "failed", model_id, {"step": "quantize"})
        return False

    # Check output actually exists — and that it has real QUANTS (not just F16)
    output_dir = OUTPUT_DIR / model_name
    all_gguf   = list(output_dir.glob("*.gguf")) if output_dir.exists() else []
    gguf_files = [f for f in all_gguf if "-F16.gguf" not in f.name and "-mmproj-" not in f.name]

    if not all_gguf:
        print_step("err", f"No GGUF files found in {output_dir} — quantization failed")
        log_event(log, "failed", model_id, {"step": "quantize", "reason": "no output files"})
        return False

    if not gguf_files:
        print_step("err", f"Only F16 found — model may be too small or tokenizer incompatible")
        print_step("info", f"Cleaning up...")
        shutil.rmtree(str(output_dir), ignore_errors=True)
        log_event(log, "failed", model_id, {"step": "quantize", "reason": "no quants, only F16"})
        return False

    # Validate quant sizes — flag any suspiciously small file (disk-full truncation)
    bad_files = [f for f in gguf_files if f.stat().st_size < 1024 * 1024]  # < 1 MB
    if bad_files:
        print()
        print_step("warn", f"{len(bad_files)} quant file(s) look corrupt (< 1MB, disk was likely full):")
        for f in bad_files:
            print(f"     ⚠️  {f.name} ({f.stat().st_size / 1024:.0f} KB) — REMOVING")
            f.unlink()  # delete corrupt file before upload
        gguf_files = [f for f in gguf_files if f.stat().st_size >= 1024 * 1024]
        if not gguf_files:
            print_step("err", "All quants were corrupt — aborting")
            shutil.rmtree(str(output_dir), ignore_errors=True)
            log_event(log, "failed", model_id, {"step": "quantize", "reason": "all quants corrupt"})
            return False

    print()
    print(f"  📦 Quants created ({len(gguf_files)}):")
    for f in sorted(gguf_files):
        size = f.stat().st_size / (1024 ** 3)
        print(f"     • {f.name} ({size:.2f} GB)")
    print_storage_bar("After quantize")

    # ── Step 2: Model card ────────────────────────────────────────────────────
    ok = run_step(
        f"Generate model card for {model_name}",
        [py, "model_card.py", "--model", model_name, "--original", model_id],
        cwd=root,
    )
    if not ok:
        print_step("warn", "Model card failed — uploading without README")

    # ── Step 3: Upload ────────────────────────────────────────────────────────
    # Check storage before upload (uploads read from disk, so no extra space needed)
    print_storage_bar("Before upload")

    ok = run_step(
        f"Upload {model_name} to HuggingFace",
        [py, "upload.py", "--model", model_name],
        cwd=root,
    )
    if not ok:
        log_event(log, "failed", model_id, {"step": "upload"})
        return False

    # ── Auto-cleanup output folder to free space ──────────────────────────────
    output_size = used_gb_dir(output_dir)
    print()
    print(f"  🧹 Auto-cleaning output folder ({output_size:.1f} GB) to free disk space...")
    shutil.rmtree(str(output_dir), ignore_errors=True)
    freed = free_gb()
    print(f"     ✅ Freed {output_size:.1f} GB — disk now has {freed:.1f} GB free")

    print_storage_bar("After cleanup")

    log_event(log, "completed", model_id, {
        "model_type": model_type,
        "preset": preset,
        "quant_count": len(gguf_files),
        "output_gb": output_size,
    })
    return True


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Autopilot: Discover → Quantize → Upload trending models automatically",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python autopilot.py --count 3
  python autopilot.py --count 5 --max-gb 8 --preset full
  python autopilot.py --count 2 --task image-text-to-text
  python autopilot.py --count 3 --min-downloads 10000
  python autopilot.py --dry-run --count 10      (preview only)
        """,
    )
    parser.add_argument("--count",         "-n", type=int, default=3,
                        help="How many models to quantize (default: 3)")
    parser.add_argument("--model",         type=str,
                        help="Skip discovery and run autopilot on a specific model ID (e.g. Qwen/Qwen3-4B-Instruct-2507)")
    parser.add_argument("--max-gb",        type=float, default=15.0,
                        help="Max model download size in GB (default: 15.0)")
    parser.add_argument("--min-downloads", type=int,   default=500,
                        help="Min downloads to consider (default: 500)")
    parser.add_argument("--preset",        default="full",
                        choices=["standard", "full", "imatrix", "all"],
                        help="Quantization preset (default: full)")
    parser.add_argument("--task",          default="text-generation",
                        help="HuggingFace pipeline_tag (default: text-generation)")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Discover and plan only — no actual quantization or upload")
    parser.add_argument("--stop-gb",       type=float, default=10.0,
                        help="Stop if free disk drops below this GB (default: 10.0)")
    parser.add_argument("--min-likes",     type=int,   default=5,
                        help="Minimum likes on HF to consider (default: 5, filters test repos)")
    args = parser.parse_args()

    log = load_log()
    already_done = set(e["model_id"] for e in log.get("completed", []))
    already_failed = set(e["model_id"] for e in log.get("failed", []))

    start_time = datetime.now()

    # ─── Banner ───────────────────────────────────────────────────────────────
    print()
    print("  " + "═" * 60)
    print("  🤖 quant-kit AUTOPILOT")
    print("  " + "═" * 60)
    print(f"  Mode:          {'DRY RUN (no changes)' if args.dry_run else 'LIVE — will quantize & upload'}")
    print(f"  Models target: {args.count}")
    print(f"  Max size:      {args.max_gb} GB per model")
    print(f"  Min downloads: {args.min_downloads:,}")
    print(f"  Preset:        {args.preset}")
    print(f"  Task:          {args.task}")
    print(f"  Stop if free:  < {args.stop_gb} GB")
    print(f"  Log:           {LOG_FILE}")
    print(f"  Previously done: {len(already_done)} models")
    print()
    print_storage_bar("Current disk")
    print("  " + "═" * 60)

    # ─── Storage gate ─────────────────────────────────────────────────────────
    if free_gb() < args.stop_gb and not args.dry_run:
        print()
        print(f"  🔴 STOP: Only {free_gb():.1f} GB free. Need at least {args.stop_gb} GB to start.")
        print(f"  Free up space first, then re-run.")
        sys.exit(1)

    # ─── Discovery ────────────────────────────────────────────────────────────
    print()
    print("  " + "─" * 60)
    
    if args.model:
        print(f"  🎯 TARGET MODE — skipping discovery to process: {args.model}")
        print("  " + "─" * 60)
        candidates = [{
            "model_id": args.model,
            "size_gb": 5.0, # fallback
            "downloads": 0,
            "likes": 0,
            "model_type": "llm",
            "architecture": "unknown"
        }]
    else:
        print("  🔍 STEP 1 — Discovering candidates...")
        print("  " + "─" * 60)
        print()

        # Fetch 5x more than needed — stricter filters mean fewer pass through
        fetch_count = args.count * 5
        candidates = discover(
            task=args.task,
            max_gb=args.max_gb,
            min_downloads=args.min_downloads,
            min_likes=args.min_likes,
            my_username="Dhptl",
            count=fetch_count,
            token=HF_TOKEN,
            verbose=True,
        )

    # Filter out already-done models
    new_candidates = [c for c in candidates if c["model_id"] not in already_done]
    if len(new_candidates) < len(candidates):
        skipped = len(candidates) - len(new_candidates)
        print_step("info", f"Skipping {skipped} already-completed models from previous runs")

    candidates = new_candidates[:args.count]

    if not candidates:
        print()
        print_step("warn", "No new candidates found!")
        print_step("info", "Try: --max-gb 20 or --min-downloads 100 or --task image-text-to-text")
        sys.exit(0)

    # ─── Plan ─────────────────────────────────────────────────────────────────
    print()
    print("  " + "═" * 60)
    print(f"  📋 PLAN — {len(candidates)} models to process")
    print("  " + "═" * 60)
    total_size = sum(c["size_gb"] for c in candidates)
    for i, c in enumerate(candidates, 1):
        n = c["gguf_repos"]
        if n == 0:
            gap = "🟢 ZERO GGUF — first mover!"
        elif not c.get("has_major"):
            gap = f"🟡 {n} small repos — open gap"
        else:
            gap = f"🟠 {n} repos incl. majors — add to your collection"
        tag  = "VLM" if c["model_type"] == "vlm" else "LLM"
        print(f"  {i}. [{tag}] {c['model_id']}")
        print(f"     {c['size_gb']:.1f} GB | {c['downloads']:,} downloads | {c['likes']} likes")
        print(f"     {gap}")
    print(f"\n  Total download: ~{total_size:.1f} GB")
    print(f"  Estimated time: ~{len(candidates) * 30:.0f}-{len(candidates) * 90:.0f} min")

    if args.dry_run:
        print()
        print("  🔍 DRY RUN — no changes made. Remove --dry-run to start.")
        print("  " + "═" * 60)
        return

    # ─── Confirm ──────────────────────────────────────────────────────────────
    print()
    print("  Starting in 5 seconds... (Ctrl+C to cancel)")
    try:
        for i in range(5, 0, -1):
            print(f"  {i}...", end="\r", flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Cancelled.")
        sys.exit(0)
    print()

    log["runs"].append({
        "started_at": start_time.isoformat(),
        "count": args.count,
        "preset": args.preset,
        "task": args.task,
        "candidates": [c["model_id"] for c in candidates],
    })
    save_log(log)

    # ─── Process each model ───────────────────────────────────────────────────
    results = {"ok": [], "failed": []}

    for i, candidate in enumerate(candidates, 1):
        model_id = candidate["model_id"]

        print()
        print()
        print("  " + "█" * 60)
        print(f"  █  MODEL {i}/{len(candidates)}: {model_id}")
        print("  " + "█" * 60)

        # Storage gate before each model
        if free_gb() < args.stop_gb:
            print()
            print(f"  🔴 STORAGE ALERT: {free_gb():.1f} GB free — below stop threshold ({args.stop_gb} GB)")
            print(f"  Stopping autopilot to protect your disk.")
            print(f"  Free up space and re-run. Already-completed models will be skipped.")
            break

        success = process_model(candidate, args.preset, log)

        if success:
            results["ok"].append(model_id)
            print()
            print(f"  ✅ {model_id} — complete!")
        else:
            results["failed"].append(model_id)
            print()
            print(f"  ❌ {model_id} — failed, moving to next...")

        # Cooldown between models
        if i < len(candidates):
            print()
            print_step("info", "Cooling down 10s before next model...")
            time.sleep(10)

    # ─── Final summary ────────────────────────────────────────────────────────
    elapsed = (datetime.now() - start_time).total_seconds()
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print()
    print()
    print("  " + "═" * 60)
    print("  🏁 AUTOPILOT COMPLETE")
    print("  " + "═" * 60)
    print(f"  Total time: {minutes}m {seconds}s")
    print(f"  Processed:  {len(results['ok'])} succeeded, {len(results['failed'])} failed")
    print()

    if results["ok"]:
        print("  ✅ Successfully published:")
        for m in results["ok"]:
            name = m.split("/")[-1]
            print(f"     🔗 huggingface.co/Dhptl/{name}-GGUF")

    if results["failed"]:
        print()
        print("  ❌ Failed (check log for details):")
        for m in results["failed"]:
            print(f"     • {m}")

    print()
    print_storage_bar("Final disk")
    print()
    print(f"  📄 Full log: {LOG_FILE}")
    print("  " + "═" * 60)


if __name__ == "__main__":
    main()
