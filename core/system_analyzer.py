import os
import platform
import psutil
import torch
import json
import logging
import argparse
from pathlib import Path

# Note: logging.basicConfig is configured in manager.py (entrypoint)
logger = logging.getLogger(__name__)

# Cache file location
CACHE_FILE = Path.home() / ".slmtunner_cache.json"

def load_cached_gpu_info():
    """Load cached GPU info from previous successful detection."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError, OSError):
            pass
    return None

def save_gpu_cache(gpu_info):
    """Save GPU info to cache for fallback."""
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(gpu_info, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save GPU cache: {e}")

def get_system_info(manual_vram_gb=None):
    """
    Detects system hardware capabilities including OS, CPU, RAM, and GPU/VRAM.
    
    Args:
        manual_vram_gb: Override VRAM detection with manual value (float)
    
    Returns a dictionary suitable for config merging.
    """
    info = {
        "os": platform.system(),
        "os_release": platform.release(),
        "python_version": platform.python_version(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": 0,
        "gpus": [],
        "best_gpu_vram_gb": 0,
        "detection_method": "auto"
    }

    # Manual VRAM override
    if manual_vram_gb is not None and manual_vram_gb > 0:
        info["gpu_available"] = True
        info["gpu_count"] = 1
        info["best_gpu_vram_gb"] = float(manual_vram_gb)
        info["detection_method"] = "manual"
        info["gpus"] = [{"id": 0, "name": "Manual Override", "vram_gb": float(manual_vram_gb)}]
        logger.info(f"Using manual VRAM override: {manual_vram_gb}GB")
        return info

    # Auto detection
    if info["gpu_available"]:
        try:
            info["gpu_count"] = torch.cuda.device_count()
            for i in range(info["gpu_count"]):
                props = torch.cuda.get_device_properties(i)
                vram_gb = round(props.total_memory / (1024**3), 2)
                info["gpus"].append({
                    "id": i,
                    "name": props.name,
                    "vram_gb": vram_gb,
                    "multi_processor_count": props.multi_processor_count,
                    "major_version": props.major,
                    "minor_version": props.minor
                })
                if vram_gb > info["best_gpu_vram_gb"]:
                    info["best_gpu_vram_gb"] = vram_gb
            
            # Save successful detection to cache
            save_gpu_cache({
                "gpus": info["gpus"],
                "best_gpu_vram_gb": info["best_gpu_vram_gb"],
                "gpu_count": info["gpu_count"]
            })
            
        except Exception as e:
            logger.error(f"Error detecting GPU properties: {e}")
            info["gpu_available"] = False
    
    # Fallback to cache if auto detection failed
    if not info["gpu_available"] or info["best_gpu_vram_gb"] == 0:
        cached = load_cached_gpu_info()
        if cached:
            logger.warning("GPU detection failed. Using cached GPU info.")
            info["gpus"] = cached.get("gpus", [])
            info["best_gpu_vram_gb"] = cached.get("best_gpu_vram_gb", 0)
            info["gpu_count"] = cached.get("gpu_count", 0)
            info["gpu_available"] = True
            info["detection_method"] = "cached"
    
    return info

def print_report(info):
    """Prints a formatted report of the system analysis."""
    print("\n" + "="*60)
    print("           SLM TUNNER - SYSTEM DIAGNOSTICS")
    print("="*60)
    print(f"OS          : {info['os']} {info['os_release']}")
    print(f"CPU Cores   : {info['cpu_count_physical']} Physical / {info['cpu_count_logical']} Logical")
    print(f"System RAM  : {info['ram_available_gb']}GB Available / {info['ram_total_gb']}GB Total")
    print("-" * 60)
    
    if info["gpu_available"]:
        method_str = f" [{info['detection_method'].upper()}]" if info['detection_method'] != 'auto' else ""
        print(f"GPU Status  : DETECTED ({info['gpu_count']} devices){method_str}")
        for gpu in info["gpus"]:
            print(f"  [{gpu['id']}] {gpu['name']}")
            print(f"      VRAM: {gpu['vram_gb']} GB")
    else:
        print("GPU Status  : NOT DETECTED")
        print("  >> TIP: Use --manual-vram <GB> to override")
    print("="*60 + "\n")

def get_tier_recommendation(vram_gb):
    """Returns tier name and basic recommendation."""
    if vram_gb >= 24:
        return "HIGH-END", "8-bit or 16-bit, Batch 8+"
    elif vram_gb >= 16:
        return "MID-HIGH", "8-bit, Batch 4-8"
    elif vram_gb >= 10:
        return "MID", "4-bit, Batch 4"
    elif vram_gb >= 6:
        return "ENTRY", "4-bit (QLoRA), Batch 1-2"
    else:
        return "LOW/CPU", "4-bit minimum, Batch 1 only"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SlmTunner System Analyzer")
    parser.add_argument("--manual-vram", type=float, default=None,
                        help="Manually specify GPU VRAM in GB (overrides auto-detection)")
    args = parser.parse_args()
    
    try:
        report = get_system_info(manual_vram_gb=args.manual_vram)
        print_report(report)
        
        # Recommendation
        print(">> Hardware Tier & Recommendation:")
        tier, rec = get_tier_recommendation(report["best_gpu_vram_gb"])
        print(f"   Tier: {tier}")
        print(f"   Strategy: {rec}")
        
    except ImportError as e:
        print(f"CRITICAL ERROR: Missing Dependency - {e}")
        print("Please run: pip install -r requirements.txt")
