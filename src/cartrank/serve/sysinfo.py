"""Process memory and platform facts for the app's debug panel (no extra dependencies)."""

import os
import platform
import subprocess


def rss_mb() -> float:
    """Resident memory of this process in MB (Linux /proc, otherwise `ps`)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True
    )
    return int(out.stdout.strip()) / 1024


def info() -> dict:
    return {
        "platform": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "rss_mb": round(rss_mb()),
    }
