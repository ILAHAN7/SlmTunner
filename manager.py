import os
import sys
import subprocess
import logging
import yaml
import json
from datetime import datetime
from pathlib import Path
from core.system_analyzer import get_system_info, print_report, save_gpu_cache, CACHE_FILE
from core.config_loader import ConfigLoader

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Global state for manual VRAM override
MANUAL_VRAM_OVERRIDE = None

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    print("""
+-------------------------------------------------------------+
|            SLM TUNNER MANAGEMENT SYSTEM v2.1                |
+-------------------------------------------------------------+
    """)

def run_setup_wizard():
    print("\n[ SETUP WIZARD ]")
    model = input("Enter HuggingFace Model Name (default: google/gemma-2b): ").strip() or "google/gemma-2b"
    rows = input("Approximate Dataset Rows (default: 1000): ").strip() or "1000"
    
    config = {
        "model": {
            "name": model,
            "output_dir": f"./{model.replace('/', '-')}-lora"
        },
        "dataset": {
            "train_path": "train.jsonl",
            "valid_path": "valid.jsonl",
            "rows": int(rows)
        },
        "training": {
            "per_device_train_batch_size": "auto",
            "gradient_accumulation_steps": "auto",
            "quantization": "auto",
            "num_train_epochs": "auto"
        }
    }
    
    with open("config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(">> 'config.yaml' generated successfully!")
    input("[Press Enter to Continue]")

def manual_hardware_setup():
    """Allow user to manually configure hardware settings."""
    global MANUAL_VRAM_OVERRIDE
    
    print("\n[ MANUAL HARDWARE SETUP ]")
    print("Use this if GPU detection fails or to test specific configurations.\n")
    
    print("Current Status:")
    sys_info = get_system_info(manual_vram_gb=MANUAL_VRAM_OVERRIDE)
    print(f"  Detection Method: {sys_info.get('detection_method', 'auto')}")
    print(f"  VRAM: {sys_info['best_gpu_vram_gb']}GB")
    print()
    
    print("Options:")
    print("  1. Set Manual VRAM (GB)")
    print("  2. Clear Manual Override (Use Auto-Detection)")
    print("  3. Clear GPU Cache")
    print("  0. Back")
    
    choice = input("\nSelect >> ").strip()
    
    if choice == '1':
        try:
            vram = float(input("Enter VRAM in GB (e.g., 8, 12, 24): ").strip())
            MANUAL_VRAM_OVERRIDE = vram
            print(f">> Manual VRAM set to {vram}GB")
        except ValueError:
            print(">> Invalid input. Please enter a number.")
    elif choice == '2':
        MANUAL_VRAM_OVERRIDE = None
        print(">> Manual override cleared. Using auto-detection.")
    elif choice == '3':
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
            print(">> GPU cache cleared.")
        else:
            print(">> No cache file found.")
    
    input("[Press Enter to Continue]")

def manual_training():
    """Manual training mode with user-specified parameters."""
    print("\n[ MANUAL TRAINING MODE ]")
    print("Specify your own training parameters.\n")
    
    if not os.path.exists("config.yaml"):
        print(">> Error: 'config.yaml' not found. Run Setup Wizard first.")
        input("[Press Enter]")
        return
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    print("Current config.yaml found. Enter new values (press Enter to keep current):\n")
    
    # Model
    current_model = config.get("model", {}).get("name", "google/gemma-2b")
    model = input(f"Model [{current_model}]: ").strip() or current_model
    
    # Training params
    batch = input("Batch Size [1]: ").strip() or "1"
    epochs = input("Epochs [3]: ").strip() or "3"
    quant = input("Quantization (4bit/8bit/none) [4bit]: ").strip() or "4bit"
    accum = input("Gradient Accumulation Steps [16]: ").strip() or "16"
    lr = input("Learning Rate [2e-4]: ").strip() or "2e-4"
    
    # Update config
    config["model"]["name"] = model
    config["model"]["output_dir"] = f"./{model.replace('/', '-')}-lora"
    config["training"] = {
        "per_device_train_batch_size": int(batch),
        "gradient_accumulation_steps": int(accum),
        "quantization": quant,
        "num_train_epochs": int(epochs),
        "learning_rate": float(lr)
    }
    
    with open("config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("\n>> config.yaml updated with manual settings!")
    print("\n[ TRAINING PLAN - MANUAL ]")
    print(f"Model: {model}")
    print(f"Batch Size: {batch}")
    print(f"Accumulation: {accum}")
    print(f"Effective Batch: {int(batch) * int(accum)}")
    print(f"Epochs: {epochs}")
    print(f"Quantization: {quant}")
    print(f"Learning Rate: {lr}")
    
    confirm = input("\nStart Training with these settings? (y/n): ").lower()
    if confirm != 'y':
        return
    
    # Launch training
    _execute_training()

def start_training():
    """Auto-optimized training mode."""
    global MANUAL_VRAM_OVERRIDE
    
    if not os.path.exists("config.yaml"):
        print(">> Error: 'config.yaml' not found. Please run Setup Wizard first.")
        input("[Press Enter]")
        return

    # 1. Load System Info & Config
    sys_info = get_system_info(manual_vram_gb=MANUAL_VRAM_OVERRIDE)
    try:
        with open("config.yaml", "r") as f:
             raw_config = yaml.safe_load(f)
             dataset_rows = raw_config.get("dataset", {}).get("rows", 1000)
             
        final_config = ConfigLoader.get_final_config("config.yaml", dataset_rows=dataset_rows)
        
        print("\n[ TRAINING PLAN - AUTO ]")
        print(f"Model: {final_config.get('model_name')}")
        print(f"Target Global Batch Size: {final_config.get('target_global_batch_size')}")
        print(f"Quantization: {final_config.get('quantization')}")
        print(f"Micro-Batch: {final_config.get('per_device_train_batch_size')}")
        print(f"Accumulation Steps: {final_config.get('gradient_accumulation_steps')}")
        print(f"Epochs: {final_config.get('num_train_epochs')}")
        print(f"Detection Method: {sys_info.get('detection_method', 'auto')}")
        
        confirm = input("\nStart Training? (y/n): ").lower()
        if confirm != 'y': return
        
    except Exception as e:
        print(f">> Error preparing config: {e}")
        input("[Press Enter]")
        return

    _execute_training()

def _execute_training():
    """Common training execution logic."""
    global MANUAL_VRAM_OVERRIDE
    sys_info = get_system_info(manual_vram_gb=MANUAL_VRAM_OVERRIDE)
    gpu_count = sys_info.get("gpu_count", 0)
    
    cmd = []
    if gpu_count > 1:
        print(f">> Multi-GPU Detected ({gpu_count}). Using 'accelerate launch'...")
        cmd = ["accelerate", "launch", "--multi_gpu", "--num_processes", str(gpu_count), "-m", "core.trainer"]
    else:
        print(">> Single GPU/Manual Mode. Running directly...")
        cmd = [sys.executable, "-m", "core.trainer"]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n>> Training Failed with error code {e.returncode}")
    except KeyboardInterrupt:
        print("\n>> Training Interrupted.")
    
    input("[Press Enter to Return]")

def main():
    global MANUAL_VRAM_OVERRIDE
    
    while True:
        clear_screen()
        print_banner()
        
        # Mini Sys Info
        sys_info = get_system_info(manual_vram_gb=MANUAL_VRAM_OVERRIDE)
        method = sys_info.get('detection_method', 'auto')
        method_tag = f" [{method.upper()}]" if method != 'auto' else ""
        gpu_status = f"{sys_info['best_gpu_vram_gb']}GB VRAM{method_tag}" if sys_info['gpu_available'] else "CPU ONLY"
        print(f" System: {sys_info['os']} | {gpu_status}")
        print("-" * 61)
        
        print(" 1. Run Setup Wizard (Create config.yaml)")
        print(" 2. Start Training (Auto-Optimize)")
        print(" 3. Manual Training (Custom Settings)")
        print(" 4. Manual Hardware Setup (Override GPU Detection)")
        print(" 5. System Diagnostics (Full Report)")
        print(" 0. Exit")
        print("-" * 61)
        
        choice = input(" Select option >> ").strip()
        
        if choice == '1':
            run_setup_wizard()
        elif choice == '2':
            start_training()
        elif choice == '3':
            manual_training()
        elif choice == '4':
            manual_hardware_setup()
        elif choice == '5':
            print_report(sys_info)
            input("[Press Enter]")
        elif choice == '0':
            print("Goodbye.")
            break
        else:
            pass

if __name__ == "__main__":
    main()
