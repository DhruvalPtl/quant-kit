#!/usr/bin/env python3
"""
patch_tokenizer.py — Emergency hotfix for unrecognized BPE tokenizers.

Run this if you see:
  NotImplementedError: BPE pre-tokenizer was not recognized

Usage:
  python patch_tokenizer.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent

# Map of chkhsh -> (pre-tokenizer name, model reference)
PATCHES = {
    "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945": ("qwen2", "Qwen/Qwen3.6-27B"),
    # Add more hashes here in future as new models come out
}

TARGET = ROOT / "llama-src" / "conversion" / "base.py"
ANCHOR = "        if res is None:"  # Line just before the error is raised

def patch():
    if not TARGET.exists():
        print(f"[ERR] base.py not found at {TARGET}")
        print("      -> Run !python setup_linux.py first to download llama.cpp")
        sys.exit(1)

    content = TARGET.read_text(encoding="utf-8")
    applied = 0

    for chkhsh, (res_name, model_ref) in PATCHES.items():
        if chkhsh in content:
            print(f"[OK ] Already patched: {model_ref} ({res_name})")
            continue

        snippet = (
            f'\n        if chkhsh == "{chkhsh}":\n'
            f'            # ref: {model_ref}\n'
            f'            res = "{res_name}"\n'
            f'\n'
            f'{ANCHOR}'
        )

        if ANCHOR not in content:
            print(f"[ERR] Could not find anchor string in base.py — llama.cpp may have changed its structure.")
            print(f"      -> Please open an issue on the quant-kit GitHub.")
            sys.exit(1)

        content = content.replace(ANCHOR, snippet, 1)
        applied += 1
        print(f"[OK ] Patched: {model_ref} → res = '{res_name}'")

    if applied > 0:
        TARGET.write_text(content, encoding="utf-8")
        print(f"\n✅ Wrote {applied} patch(es) to {TARGET}")
    else:
        print("\n✅ Nothing to patch — all tokenizers already recognized.")

if __name__ == "__main__":
    patch()
