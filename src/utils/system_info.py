import subprocess
import logging

logger = logging.getLogger(__name__)


def get_system_info():
    info = {}

    # lspci | grep -i vga
    try:
        result = subprocess.run(["lspci"], capture_output=True, text=True, check=True)
        info["lspci_vga"] = "\n".join(
            line for line in result.stdout.splitlines() if "vga" in line.lower()
        )
    except Exception as e:
        info["lspci_vga"] = f"Error: {e}"

    # nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=True
        )
        info["nvidia_smi"] = result.stdout.strip()
    except Exception as e:
        info["nvidia_smi"] = f"Error: {e}"

    # lshw -C video (no sudo)
    try:
        result = subprocess.run(
            ["lshw", "-C", "video"], capture_output=True, text=True, check=True
        )
        info["lshw_video"] = result.stdout.strip()
    except Exception as e:
        info["lshw_video"] = f"Error: {e}"

    # Log all system info
    for key, value in info.items():
        logger.info(f"{key}:\n{value}\n{'-' * 40}")

    return info
