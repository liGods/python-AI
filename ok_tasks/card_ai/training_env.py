from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any


GIB = 1024**3


def _linux_memory() -> tuple[float | None, float | None]:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None, None
    values: dict[str, int] = {}
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.strip().split()[0]) * 1024
    return values.get("MemTotal", 0) / GIB, values.get("SwapTotal", 0) / GIB


def _gpu_status() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False, "reason": "nvidia-smi_not_found"}
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": False, "reason": type(error).__name__}
    devices = []
    for line in result.stdout.splitlines():
        parts = [value.strip() for value in line.split(",")]
        if len(parts) == 3:
            devices.append({"name": parts[0], "memory_mib": int(parts[1]), "driver": parts[2]})
    return {"available": bool(devices), "devices": devices}


def _torch_status() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"installed": False, "cuda_available": False}
    return {
        "installed": True,
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device_count": int(torch.cuda.device_count()),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def training_environment_report(project_root: str | Path = ".") -> dict[str, Any]:
    root = Path(project_root).resolve()
    release = platform.release().lower()
    is_wsl = bool(os.environ.get("WSL_DISTRO_NAME")) or "microsoft" in release
    memory_gib, swap_gib = _linux_memory()
    disk = shutil.disk_usage(root)
    gpu = _gpu_status()
    torch_status = _torch_status()
    checks = {
        "running_in_wsl2": is_wsl and "microsoft" in release,
        "logical_processors_at_least_12": (os.cpu_count() or 0) >= 12,
        "memory_at_least_24_gib": memory_gib is not None and memory_gib >= 23.5,
        "swap_at_least_8_gib": swap_gib is not None and swap_gib >= 7.5,
        "free_disk_at_least_150_gib": disk.free >= 150 * GIB,
        "nvidia_gpu_visible": bool(gpu.get("available")),
        "pytorch_cuda_ready": bool(torch_status.get("cuda_available")),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    return {
        "ready": not blockers,
        "platform": platform.platform(),
        "project_root": str(root),
        "logical_processors": os.cpu_count(),
        "memory_gib": round(memory_gib, 2) if memory_gib is not None else None,
        "swap_gib": round(swap_gib, 2) if swap_gib is not None else None,
        "free_disk_gib": round(disk.free / GIB, 2),
        "gpu": gpu,
        "torch": torch_status,
        "checks": checks,
        "blockers": blockers,
    }
