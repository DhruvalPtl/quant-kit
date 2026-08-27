"""
setup_linux.py — Set up llama.cpp on Linux (Google Colab / Ubuntu)
====================================================================
Run this ONCE at the start of each Colab session. It:
  1. Installs Python dependencies
  2. Downloads pre-built llama.cpp Linux binaries (CUDA if GPU available)
  3. Sparse-clones Python conversion scripts from llama.cpp repo

Usage:
    python setup_linux.py
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import json
import tarfile
import platform
import subprocess
import urllib.request
import zipfile
from pathlib import Path

ROOT_DIR      = Path(__file__).parent
LLAMA_CPP_DIR = ROOT_DIR / "llama.cpp"
LLAMA_SRC_DIR = ROOT_DIR / "llama-src"


def header(msg: str):
    print(f"\n[->] {msg}")

def ok(msg: str):
    print(f"[OK] {msg}")

def warn(msg: str):
    print(f"[!]  {msg}")

def err(msg: str):
    print(f"[ERR] {msg}")

def run(cmd: str, cwd=None, check=True):
    """Run a shell command and stream output."""
    result = subprocess.run(cmd, shell=True, cwd=str(cwd) if cwd else None)
    if check and result.returncode != 0:
        err(f"Command failed (exit {result.returncode}): {cmd}")
        sys.exit(1)
    return result


# ─── Step 1: Check OS ─────────────────────────────────────────────────────────

def check_os():
    if platform.system() == "Windows":
        err("This script is for Linux/Colab only.")
        print("  On Windows: download llama-b*-bin-win-vulkan-x64.zip manually")
        print("  See: https://github.com/ggerganov/llama.cpp/releases/latest")
        sys.exit(1)
    ok(f"OS: {platform.system()} {platform.machine()}")


# ─── Step 2: Python dependencies ──────────────────────────────────────────────

def install_python_deps():
    header("Installing Python packages...")
    run(f"{sys.executable} -m pip install -r requirements.txt -q", cwd=ROOT_DIR)
    ok("Python packages installed")


# ─── Step 3: Detect GPU ───────────────────────────────────────────────────────

def check_cuda() -> bool:
    result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        # Extract GPU name
        lines = [l for l in result.stdout.splitlines() if "%" in l or "MiB" in l]
        gpu_line = result.stdout.split("\n")[8] if len(result.stdout.split("\n")) > 8 else ""
        ok(f"CUDA GPU detected")
        return True
    warn("No CUDA GPU detected — using CPU build (slower quantization)")
    return False


# ─── Step 4: Download llama.cpp binaries ──────────────────────────────────────

def get_release_info() -> tuple[str, dict]:
    """Fetch latest llama.cpp release asset URLs from GitHub API."""
    header("Fetching latest llama.cpp release info...")
    url = "https://api.github.com/repos/ggerganov/llama.cpp/releases"
    req = urllib.request.Request(url, headers={"User-Agent": "quant-kit/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        releases = json.loads(r.read())

    # Find the latest release that has actual binary assets
    release = None
    for rel in releases:
        if rel.get("assets") and len(rel["assets"]) > 1:
            release = rel
            break

    if not release:
        # Fallback to the first release if none matched
        release = releases[0] if releases else {}

    tag    = release.get("tag_name", "unknown")
    assets = {a["name"]: a["browser_download_url"] for a in release.get("assets", [])}

    ok(f"Latest release: {tag}")

    # Show available Linux assets for transparency
    linux_assets = [n for n in assets if "ubuntu" in n or "linux" in n]
    print(f"  Available Linux builds:")
    for n in sorted(linux_assets):
        print(f"    {n}")

    return tag, assets



def download_binaries(has_cuda: bool):
    LLAMA_CPP_DIR.mkdir(exist_ok=True)

    # Check if already installed
    if (LLAMA_CPP_DIR / "llama-quantize").exists():
        ok("llama.cpp binaries already installed — skipping download")
        return

    tag, assets = get_release_info()

    # Priority order for Linux binary selection:
    # 1. ubuntu-x64 (standard, works on Colab T4)
    # 2. ubuntu-vulkan-x64 (Vulkan compute — for AMD/Intel, not T4)
    # 3. Any ubuntu x64 build
    # NOTE: Quantization (llama-quantize) runs on CPU — no GPU needed here.
    #       GPU is only used in benchmark.py for inference speed testing.
    
    target = None
    label  = ""

    # Try patterns in priority order
    patterns = [
        lambda n: "ubuntu" in n and "x64" in n and all(x not in n for x in ["vulkan", "rocm", "openvino", "arm", "sycl"]),
        lambda n: "ubuntu" in n and "x64" in n and all(x not in n for x in ["rocm", "openvino", "arm", "sycl"]),
        lambda n: "ubuntu" in n and "x64" in n and "sycl" not in n,
        lambda n: "linux" in n and "x64" in n,
        lambda n: "ubuntu" in n and "x86" in n,
    ]

    for pattern in patterns:
        for name, url in assets.items():
            if pattern(name) and (name.endswith(".tar.gz") or name.endswith(".zip")):
                target = url
                label  = name
                break
        if target:
            break

    if not target:
        err("Could not find a suitable Linux binary.")
        err(f"Available assets: {list(assets.keys())}")
        sys.exit(1)

    header(f"Downloading: {label}")
    archive_path = ROOT_DIR / label

    def progress(block, block_size, total):
        downloaded = block * block_size
        if total > 0:
            pct = min(downloaded / total * 100, 100)
            mb  = downloaded / (1024**2)
            print(f"\r  {pct:.0f}% — {mb:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(target, archive_path, reporthook=progress)
    print()  # newline after progress

    header("Extracting binaries...")
    if label.endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:gz") as t:
            t.extractall(LLAMA_CPP_DIR)   # no filter= so symlinks are preserved
    else:
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(LLAMA_CPP_DIR)
    archive_path.unlink()

    # Flatten: tar.gz extracts into a subdirectory like llama-b9561-bin-ubuntu-x64/
    # Move ALL entries (files AND symlinks) up to LLAMA_CPP_DIR root
    import shutil as _shutil
    subdirs = [d for d in LLAMA_CPP_DIR.iterdir() if d.is_dir()]
    moved = 0
    for subdir in subdirs:
        for f in list(subdir.iterdir()):   # one level only — symlinks live here
            dest = LLAMA_CPP_DIR / f.name
            if not dest.exists():
                if f.is_symlink():
                    # Recreate symlink at root pointing to same target name
                    link_target = os.readlink(f)
                    os.symlink(link_target, dest)
                else:
                    _shutil.move(str(f), str(dest))
                moved += 1

    if moved:
        print(f"  Flattened {moved} entries to {LLAMA_CPP_DIR}")

    # Remove now-empty subdirectories
    for subdir in subdirs:
        try:
            _shutil.rmtree(str(subdir))
        except Exception:
            pass

    # Safety net: create missing versioned symlinks
    # Handles two patterns:
    #   libfoo.so -> libfoo.so.0        (simple, 1-level)
    #   libfoo.so.0.0.9562 -> libfoo.so.0  (3-part version, e.g. llama.cpp b9562+)
    import re as _re
    from collections import defaultdict as _dd

    # Pattern 1: libXXX.so (real file) missing libXXX.so.0
    for f in list(LLAMA_CPP_DIR.glob("*.so")):
        if not f.is_symlink():
            versioned = LLAMA_CPP_DIR / (f.name + ".0")
            if not versioned.exists():
                os.symlink(f.name, str(versioned))
                print(f"  Symlink: {versioned.name} -> {f.name}")

    # Pattern 2: libXXX.so.A.B.C (real file) missing libXXX.so.A
    three_part = _re.compile(r'^(lib.+\.so)\.(\d+)\.\d+\.\d+$')
    by_base = _dd(list)
    for f in LLAMA_CPP_DIR.glob("lib*.so.*"):
        m = three_part.match(f.name)
        if m and not f.is_symlink():
            by_base[m.group(1)].append((int(m.group(2)), f.name))

    for base_so, versions in by_base.items():
        versions.sort(reverse=True)          # pick newest build
        latest, major = versions[0][1], versions[0][0]
        so_major = LLAMA_CPP_DIR / f"{base_so}.{major}"
        if not so_major.exists():
            os.symlink(latest, str(so_major))
            print(f"  Symlink: {so_major.name} -> {latest}")

    # Make executables runnable
    for f in LLAMA_CPP_DIR.glob("llama-*"):
        if f.is_file() and not f.is_symlink():
            try:
                f.chmod(0o755)
            except Exception:
                pass

    ok(f"llama.cpp binaries ready in {LLAMA_CPP_DIR}")

    # Show key binaries
    bins = sorted(f.name for f in LLAMA_CPP_DIR.glob("llama-*") if f.is_file() and not f.suffix)
    print(f"  Binaries: {', '.join(bins[:8])}{'...' if len(bins) > 8 else ''}")



# ─── Step 5: Sparse-clone Python conversion scripts ───────────────────────────

def setup_conversion_scripts():
    if LLAMA_SRC_DIR.exists() and (LLAMA_SRC_DIR / "convert_hf_to_gguf.py").exists():
        ok("Conversion scripts already set up — skipping")
        
        # -------------------------------------------------------------------------
        # HOTFIX: Patch llama.cpp conversion scripts to support Qwen 3.6 tokenizer
        # -------------------------------------------------------------------------
        base_py = LLAMA_SRC_DIR / "conversion" / "base.py"
        if base_py.exists():
            content = base_py.read_text(encoding="utf-8")
            if "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945" not in content:
                patch = '''
        if chkhsh == "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945":
            # ref: Qwen/Qwen3.6-27B
            res = "qwen2"

        if res is None:'''
                content = content.replace("        if res is None:", patch)
                base_py.write_text(content, encoding="utf-8")
                print("[OK] Auto-patched llama.cpp to support Qwen 3.6 tokenizer")

        # -------------------------------------------------------------------------
        # HOTFIX: Patch llama.cpp qwen.py to remove transformers bytes_to_unicode dependency
        # -------------------------------------------------------------------------
        qwen_py = LLAMA_SRC_DIR / "conversion" / "qwen.py"
        if qwen_py.exists():
            content = qwen_py.read_text(encoding="utf-8")
            if "from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode" in content:
                target_str = """        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode  # ty: ignore[unresolved-import]
        byte_encoder = bytes_to_unicode()"""
                replacement = """        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b_val in range(256):
            if b_val not in bs:
                bs.append(b_val)
                cs.append(256 + n)
                n += 1
        byte_encoder = dict(zip(bs, [chr(n) for n in cs]))"""
                content = content.replace(target_str, replacement)
                qwen_py.write_text(content, encoding="utf-8")
                print("[OK] Auto-patched llama.cpp to fix bytes_to_unicode transformers ImportError")
                
        return

    header("Downloading llama.cpp Python conversion scripts...")

    # Sparse clone — only gets the Python files we need, not the full C++ source
    run(
        "git clone --depth 1 --filter=blob:none --sparse "
        "https://github.com/ggerganov/llama.cpp.git llama-src",
        cwd=ROOT_DIR
    )
    run("git sparse-checkout add conversion/ gguf-py/", cwd=LLAMA_SRC_DIR)
    print("[OK] Conversion scripts ready in " + str(LLAMA_SRC_DIR))
    
    # -------------------------------------------------------------------------
    # HOTFIX: Patch llama.cpp conversion scripts to support Qwen 3.6 tokenizer
    # -------------------------------------------------------------------------
    base_py = LLAMA_SRC_DIR / "conversion" / "base.py"
    if base_py.exists():
        content = base_py.read_text(encoding="utf-8")
        if "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945" not in content:
            patch = '''
        if chkhsh == "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945":
            # ref: Qwen/Qwen3.6-27B
            res = "qwen2"

        if res is None:'''
            content = content.replace("        if res is None:", patch)
            base_py.write_text(content, encoding="utf-8")
            print("[OK] Auto-patched llama.cpp to support Qwen 3.6 tokenizer")

    # -------------------------------------------------------------------------
    # HOTFIX: Patch llama.cpp qwen.py to remove transformers bytes_to_unicode dependency
    # -------------------------------------------------------------------------
    qwen_py = LLAMA_SRC_DIR / "conversion" / "qwen.py"
    if qwen_py.exists():
        content = qwen_py.read_text(encoding="utf-8")
        if "from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode" in content:
            target_str = """        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode  # ty: ignore[unresolved-import]
        byte_encoder = bytes_to_unicode()"""
            replacement = """        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b_val in range(256):
            if b_val not in bs:
                bs.append(b_val)
                cs.append(256 + n)
                n += 1
            byte_encoder = dict(zip(bs, [chr(n) for n in cs]))"""
            content = content.replace(target_str, replacement)
            qwen_py.write_text(content, encoding="utf-8")
            print("[OK] Auto-patched llama.cpp to fix bytes_to_unicode transformers ImportError")

    print("\n")
    print_step("info", "Verifying setup...")


# ─── Step 6: Verify everything ────────────────────────────────────────────────

def verify():
    header("Verifying setup...")

    # Must import after setup so paths are fresh
    from config import LLAMA_QUANTIZE, LLAMA_CLI, LLAMA_BENCH, CONVERT_SCRIPT

    checks = [
        ("llama-quantize", LLAMA_QUANTIZE),
        ("llama-cli",      LLAMA_CLI),
        ("llama-bench",    LLAMA_BENCH),
        ("convert script", CONVERT_SCRIPT),
    ]

    all_ok = True
    for name, path in checks:
        if path.exists():
            ok(f"{name:20s} {path}")
        else:
            err(f"{name:20s} NOT FOUND at {path}")
            all_ok = False

    return all_ok


# ─── Disk space check ─────────────────────────────────────────────────────────

def check_disk():
    import shutil
    total, used, free = shutil.disk_usage("/")
    free_gb  = free  / (1024**3)
    total_gb = total / (1024**3)
    ok(f"Disk: {free_gb:.1f} GB free of {total_gb:.1f} GB total")
    if free_gb < 40:
        warn("Less than 40GB free — use --delete-src flag when running quantize.py")
    return free_gb


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print()
    print("=" * 58)
    print("  quant-kit — Linux / Colab Setup")
    print("=" * 58)

    check_os()
    check_disk()
    print()

    install_python_deps()
    print()

    has_cuda = check_cuda()
    print()

    download_binaries(has_cuda)
    print()

    setup_conversion_scripts()
    print()

    all_ok = verify()

    print()
    if all_ok:
        print("=" * 58)
        print("  [OK] Setup complete! Run the next cell to quantize.")
        print("=" * 58)
    else:
        print("=" * 58)
        err("Setup incomplete. Fix the errors above before continuing.")
        print("=" * 58)
        sys.exit(1)


if __name__ == "__main__":
    main()
