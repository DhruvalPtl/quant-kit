import sys
import psutil
import platform
import subprocess
from pathlib import Path

def print_step(step: str, msg: str):
    """Print a styled message to the console."""
    colors = {"info": "\033[94m", "ok": "\033[92m", "warn": "\033[93m", "err": "\033[91m"}
    reset = "\033[0m"
    icons = {"info": "->", "ok": "[OK]", "warn": "[!]", "err": "[ERR]"}
    color = colors.get(step, "")
    icon = icons.get(step, "-")
    print(f"{color}{icon} {msg}{reset}")

def format_size(size_bytes: int) -> str:
    """Return human-readable file size."""
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"

def get_size(path: Path) -> str:
    """Return human-readable file size for a path."""
    if not path.exists():
        return "0 B"
    return format_size(path.stat().st_size)

def get_hardware_info() -> dict:
    """Detect and return system hardware info."""
    cpu = platform.processor() or "Unknown CPU"
    if platform.system() == "Windows":
        try:
            r = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance -ClassName CIM_Processor).Name"],
                capture_output=True, text=True, timeout=8
            )
            name = r.stdout.strip()
            if name and len(name) > 5:
                cpu = name
        except Exception:
            pass

    info = {
        "os":     platform.system(),
        "cpu":    cpu,
        "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 1),
    }
    
    # Very basic GPU detection heuristic
    gpu = "Unknown GPU"
    if platform.system() == "Windows":
        try:
            r = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance -ClassName CIM_VideoController).Name"],
                capture_output=True, text=True, timeout=8
            )
            names = [n.strip() for n in r.stdout.splitlines() if n.strip()]
            if names:
                gpu = " / ".join(names)
        except Exception:
            pass
    info["gpu"] = gpu

    return info
