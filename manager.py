"""
SlmTunner Management System v2.2
Main CLI entrypoint for LoRA tuning pipeline.
"""
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
from core.experiment_tracker import ExperimentTracker

# Setup Logging (single entrypoint for basicConfig)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class SlmManager:
    """Main application class encapsulating CLI state and operations."""

    def __init__(self):
        self.manual_vram_override = None
        self.experiment_tracker = ExperimentTracker()

    def clear_screen(self):
        cmd = 'cls' if os.name == 'nt' else 'clear'
        subprocess.run(cmd, shell=True, check=False)

    def print_banner(self):
        print("""
+-------------------------------------------------------------+
|            SLM TUNNER MANAGEMENT SYSTEM v2.2                |
+-------------------------------------------------------------+
        """)

    def _safe_int_input(self, prompt, default):
        """Safely get integer input with validation."""
        raw = input(prompt).strip()
        if not raw:
            return default
        try:
            value = int(raw)
            if value <= 0:
                print(">> Value must be positive. Using default.")
                return default
            return value
        except ValueError:
            print(f">> Invalid number '{raw}'. Using default: {default}")
            return default

    def _safe_float_input(self, prompt, default):
        """Safely get float input with validation."""
        raw = input(prompt).strip()
        if not raw:
            return default
        try:
            value = float(raw)
            if value <= 0:
                print(">> Value must be positive. Using default.")
                return default
            return value
        except ValueError:
            print(f">> Invalid number '{raw}'. Using default: {default}")
            return default

    def run_setup_wizard(self):
        print("\n[ SETUP WIZARD ]")
        model = input("Enter HuggingFace Model Name (default: google/gemma-2b): ").strip() or "google/gemma-2b"
        rows = self._safe_int_input("Approximate Dataset Rows (default: 1000): ", 1000)
        max_length = self._safe_int_input("Max Sequence Length (default: 512): ", 512)

        config = {
            "model": {
                "name": model,
                "output_dir": f"./{model.replace('/', '-')}-lora"
            },
            "dataset": {
                "train_path": "train.jsonl",
                "valid_path": "valid.jsonl",
                "rows": rows
            },
            "training": {
                "per_device_train_batch_size": "auto",
                "gradient_accumulation_steps": "auto",
                "quantization": "auto",
                "num_train_epochs": "auto",
                "max_length": max_length
            }
        }

        with open("config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False)
        print(">> 'config.yaml' generated successfully!")
        input("[Press Enter to Continue]")

    def manual_hardware_setup(self):
        """Allow user to manually configure hardware settings."""
        print("\n[ MANUAL HARDWARE SETUP ]")
        print("Use this if GPU detection fails or to test specific configurations.\n")

        print("Current Status:")
        sys_info = get_system_info(manual_vram_gb=self.manual_vram_override)
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
            vram = self._safe_float_input("Enter VRAM in GB (e.g., 8, 12, 24): ", 0)
            if vram > 0:
                self.manual_vram_override = vram
                print(f">> Manual VRAM set to {vram}GB")
            else:
                print(">> Invalid VRAM value.")
        elif choice == '2':
            self.manual_vram_override = None
            print(">> Manual override cleared. Using auto-detection.")
        elif choice == '3':
            if CACHE_FILE.exists():
                CACHE_FILE.unlink()
                print(">> GPU cache cleared.")
            else:
                print(">> No cache file found.")

        input("[Press Enter to Continue]")

    def manual_training(self):
        """Manual training mode with user-specified parameters."""
        print("\n[ MANUAL TRAINING MODE ]")
        print("Specify your own training parameters.\n")

        if not os.path.exists("config.yaml"):
            print(">> Error: 'config.yaml' not found. Run Setup Wizard first.")
            input("[Press Enter]")
            return

        config = ConfigLoader.load_config("config.yaml")

        print("Current config.yaml found. Enter new values (press Enter to keep current):\n")

        # Model
        current_model = config.get("model", {}).get("name", "google/gemma-2b")
        model = input(f"Model [{current_model}]: ").strip() or current_model

        # Training params with validation
        batch = self._safe_int_input("Batch Size [1]: ", 1)
        epochs = self._safe_int_input("Epochs [3]: ", 3)
        max_length = self._safe_int_input("Max Sequence Length [512]: ", 512)

        quant_input = input("Quantization (4bit/8bit/none) [4bit]: ").strip() or "4bit"
        if quant_input not in ("4bit", "8bit", "none"):
            print(f">> Invalid quantization '{quant_input}'. Using default: 4bit")
            quant_input = "4bit"

        accum = self._safe_int_input("Gradient Accumulation Steps [16]: ", 16)
        lr = self._safe_float_input("Learning Rate [2e-4]: ", 2e-4)

        # Update config
        config["model"] = config.get("model", {})
        config["model"]["name"] = model
        config["model"]["output_dir"] = f"./{model.replace('/', '-')}-lora"
        config["training"] = {
            "per_device_train_batch_size": batch,
            "gradient_accumulation_steps": accum,
            "quantization": quant_input,
            "num_train_epochs": epochs,
            "learning_rate": lr,
            "max_length": max_length
        }

        with open("config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False)

        print("\n>> config.yaml updated with manual settings!")
        print("\n[ TRAINING PLAN - MANUAL ]")
        print(f"Model: {model}")
        print(f"Batch Size: {batch}")
        print(f"Accumulation: {accum}")
        print(f"Effective Batch: {batch * accum}")
        print(f"Epochs: {epochs}")
        print(f"Max Length: {max_length}")
        print(f"Quantization: {quant_input}")
        print(f"Learning Rate: {lr}")

        confirm = input("\nStart Training with these settings? (y/n): ").lower()
        if confirm != 'y':
            return

        self._execute_training()

    def start_training(self):
        """Auto-optimized training mode."""
        if not os.path.exists("config.yaml"):
            print(">> Error: 'config.yaml' not found. Please run Setup Wizard first.")
            input("[Press Enter]")
            return

        # 1. Load System Info & Config
        sys_info = get_system_info(manual_vram_gb=self.manual_vram_override)
        try:
            raw_config = ConfigLoader.load_config("config.yaml")
            dataset_rows = raw_config.get("dataset", {}).get("rows", 1000)

            final_config = ConfigLoader.get_final_config(
                "config.yaml", 
                dataset_rows=dataset_rows,
                manual_vram_gb=self.manual_vram_override
            )

            print("\n[ TRAINING PLAN - AUTO ]")
            print(f"Model: {final_config.get('model_name')}")
            print(f"Target Global Batch Size: {final_config.get('target_global_batch_size')}")
            print(f"Quantization: {final_config.get('quantization')}")
            print(f"Micro-Batch: {final_config.get('per_device_train_batch_size')}")
            print(f"Accumulation Steps: {final_config.get('gradient_accumulation_steps')}")
            print(f"Epochs: {final_config.get('num_train_epochs')}")
            print(f"Max Length: {final_config.get('max_length')}")
            print(f"Detection Method: {sys_info.get('detection_method', 'auto')}")

            confirm = input("\nStart Training? (y/n): ").lower()
            if confirm != 'y':
                return

        except Exception as e:
            print(f">> Error preparing config: {e}")
            input("[Press Enter]")
            return

        self._execute_training()

    def _execute_training(self):
        """Common training execution logic."""
        sys_info = get_system_info(manual_vram_gb=self.manual_vram_override)
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
        except FileNotFoundError:
            print("\n>> Error: Command not found. Check your Python/accelerate installation.")

        input("[Press Enter to Return]")

    def view_experiments(self):
        """Display past experiment runs."""
        print("\n[ EXPERIMENT HISTORY ]")
        experiments = self.experiment_tracker.list_experiments()

        if not experiments:
            print("No experiments found.")
            input("[Press Enter]")
            return

        print(f"Found {len(experiments)} experiment(s):\n")
        for i, exp in enumerate(experiments):
            status = exp.get("final_metrics", {}).get("status", exp.get("status", "unknown"))
            model = exp.get("model", "N/A")
            duration = exp.get("duration_human", "N/A")
            final_loss = exp.get("final_loss", "N/A")
            print(f"  {i+1}. {exp['run_name']}")
            print(f"     Model: {model} | Status: {status}")
            print(f"     Duration: {duration} | Final Loss: {final_loss}")
            print()

        input("[Press Enter to Continue]")

    def run(self):
        """Main application loop."""
        while True:
            self.clear_screen()
            self.print_banner()

            # Mini Sys Info
            sys_info = get_system_info(manual_vram_gb=self.manual_vram_override)
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
            print(" 6. View Past Experiments")
            print(" 0. Exit")
            print("-" * 61)

            choice = input(" Select option >> ").strip()

            if choice == '1':
                self.run_setup_wizard()
            elif choice == '2':
                self.start_training()
            elif choice == '3':
                self.manual_training()
            elif choice == '4':
                self.manual_hardware_setup()
            elif choice == '5':
                print_report(sys_info)
                input("[Press Enter]")
            elif choice == '6':
                self.view_experiments()
            elif choice == '0':
                print("Goodbye.")
                break


if __name__ == "__main__":
    manager = SlmManager()
    manager.run()
