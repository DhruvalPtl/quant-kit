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

def global_log(script: str, model_id: str, details: dict = None):
    """
    Log an event to a central HuggingFace Dataset (`quant-kit-logs`).
    This aggregates logs across Laptop, Colab, Kaggle, etc.
    """
    import json, time, os, socket, tempfile
    from huggingface_hub import HfApi
    from config import HF_TOKEN

    if not HF_TOKEN:
        return

    try:
        api = HfApi(token=HF_TOKEN)
        user = api.whoami()["name"]
        repo_id = f"{user}/quant-kit-logs"
        
        # Check if repo exists, if not create it
        try:
            api.dataset_info(repo_id)
        except Exception:
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=True)
            
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        
        # Detect environment
        if os.environ.get("COLAB_GPU"):
            env = "Google Colab"
        elif os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
            env = "Kaggle"
        else:
            env = socket.gethostname()

        hw_info = get_hardware_info()
        
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "environment": env,
            "hardware": hw_info,
            "script": script,
            "model_id": model_id,
            "details": details or {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(log_entry, f, indent=2)
            tmp_path = f.name
            
        safe_model_id = model_id.replace('/', '_')
        file_path = f"logs/{timestamp_str}_{safe_model_id}.json"
        
        # Upload fire-and-forget
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=file_path,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Log: {script} -> {model_id}"
        )
        os.unlink(tmp_path)
    except Exception as e:
        # Silently fail so we don't crash the main pipeline if logging fails
        print(f"\n  [WARN] Could not upload to global log: {e}\n")
