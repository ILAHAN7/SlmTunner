import logging
import math

logger = logging.getLogger(__name__)

class OptimizationStrategy:
    """
    Implements the 'Constraint-Target' optimization logic.
    
    Constraint: Hardware limits (VRAM, Model Size, Context Length) -> Micro Batch Size & Quantization
    Target: Data characteristics -> Global Batch Size & Epochs
    
    Key improvement: Now considers model size and context length for accurate VRAM estimation.
    """
    
    # Approximate VRAM usage per billion parameters (in GB)
    VRAM_PER_B_FP16 = 2.0      # ~2GB per 1B params in fp16
    VRAM_PER_B_8BIT = 1.0      # ~1GB per 1B params in 8-bit
    VRAM_PER_B_4BIT = 0.5      # ~0.5GB per 1B params in 4-bit
    
    # Context length scaling factor (base 512)
    BASE_CONTEXT = 512
    
    @staticmethod
    def estimate_model_size(model_name: str) -> float:
        """
        Estimate model size in billions from model name.
        Returns conservative estimate if unknown.
        """
        model_lower = model_name.lower()
        
        # Common patterns: "7b", "2b", "13b", etc.
        import re
        match = re.search(r'(\d+\.?\d*)b', model_lower)
        if match:
            return float(match.group(1))
        
        # Known models without size in name
        if "gemma-2" in model_lower or "gemma2" in model_lower:
            return 2.0
        if "phi" in model_lower:
            return 2.7
        if "tinyllama" in model_lower:
            return 1.1
        
        # Conservative default for unknown models
        logger.warning(f"Could not determine size for '{model_name}'. Assuming 7B (conservative).")
        return 7.0

    @staticmethod
    def get_recommendation(system_info, dataset_rows, is_windows=False, 
                           model_name=None, max_length=512):
        """
        Returns a dictionary of recommended training arguments.
        
        Args:
            system_info: Dict from system_analyzer.get_system_info()
            dataset_rows: Approximate number of training samples
            is_windows: True if running on Windows
            model_name: HuggingFace model name (for size estimation)
            max_length: Context length for training
        """
        # --- 0. Model Size Estimation ---
        model_size_b = 7.0  # Default conservative
        if model_name:
            model_size_b = OptimizationStrategy.estimate_model_size(model_name)
        
        # Context scaling factor
        context_factor = max_length / OptimizationStrategy.BASE_CONTEXT
        
        # --- 1. System Constraints (The Speed Limit) ---
        vram_gb = system_info.get("best_gpu_vram_gb", 0)
        gpu_count = max(1, system_info.get("gpu_count", 1))
        
        # Calculate VRAM needed for different quantization modes
        vram_fp16 = model_size_b * OptimizationStrategy.VRAM_PER_B_FP16
        vram_8bit = model_size_b * OptimizationStrategy.VRAM_PER_B_8BIT
        vram_4bit = model_size_b * OptimizationStrategy.VRAM_PER_B_4BIT
        
        # Additional VRAM for activations/gradients (rough estimate: 2GB + context scaling)
        activation_overhead = 2.0 * context_factor
        
        # Determine quantization based on available VRAM
        if vram_gb >= (vram_fp16 + activation_overhead + 4):  # Extra 4GB headroom
            quantization = "none"  # Full precision
            base_vram = vram_fp16
        elif vram_gb >= (vram_8bit + activation_overhead + 2):
            quantization = "8bit"
            base_vram = vram_8bit
        else:
            quantization = "4bit"
            base_vram = vram_4bit
        
        # Calculate free VRAM for batching
        free_vram = vram_gb - base_vram - activation_overhead
        
        # Estimate VRAM per sample (very rough: ~0.5GB per sample at batch=1, context=512)
        vram_per_sample = 0.5 * context_factor
        
        # Calculate max micro batch size
        if free_vram > 0:
            micro_batch_size = max(1, int(free_vram / vram_per_sample))
            micro_batch_size = min(micro_batch_size, 16)  # Cap at 16
        else:
            micro_batch_size = 1
        
        # Gradient checkpointing decision
        grad_checkpointing = True if vram_gb < 16 else False
        fp16 = True
        
        # --- 2. Data Targets (The Destination) ---
        target_global_batch_size = 32  # Default
        num_epochs = 3
        
        if dataset_rows < 500:
            target_global_batch_size = 8
            num_epochs = 10
        elif dataset_rows < 1000:
            target_global_batch_size = 16
            num_epochs = 5
        elif dataset_rows < 10000:
            target_global_batch_size = 32
            num_epochs = 3
        elif dataset_rows < 100000:
            target_global_batch_size = 64 
            num_epochs = 1
        else:
            target_global_batch_size = 128
            num_epochs = 1

        # --- 3. Reconciliation (The Calculator) ---
        total_micro_batch = micro_batch_size * gpu_count
        grad_accum_steps = math.ceil(target_global_batch_size / total_micro_batch)
        grad_accum_steps = max(1, grad_accum_steps)
        
        real_global_batch_size = total_micro_batch * grad_accum_steps

        # --- 4. Parallelism & OS Specifics ---
        num_workers = 4
        if is_windows:
            num_workers = 0 
            logger.info("Windows detected: Setting num_workers=0.")
        else:
            cpu_cores = system_info.get("cpu_count_physical", 2)
            num_workers = min(cpu_cores, 8)

        # --- 5. Return Recommendation ---
        return {
            "quantization": quantization,
            "per_device_train_batch_size": micro_batch_size,
            "gradient_accumulation_steps": grad_accum_steps,
            "num_train_epochs": num_epochs,
            "per_device_eval_batch_size": micro_batch_size,
            "fp16": fp16,
            "gradient_checkpointing": grad_checkpointing,
            "dataloader_num_workers": num_workers,
            "target_global_batch_size": real_global_batch_size,
            "max_length": max_length,
            # Metadata for transparency
            "_estimated_model_size_b": model_size_b,
            "_vram_available": vram_gb,
            "_vram_model_usage": base_vram,
            "_context_factor": context_factor
        }


if __name__ == "__main__":
    # Test Scenarios
    print(">> Strategy Test (Model-Aware):\n")
    
    mock_sys_8gb = {"best_gpu_vram_gb": 8.0, "gpu_count": 1, "cpu_count_physical": 6}
    mock_sys_24gb = {"best_gpu_vram_gb": 24.0, "gpu_count": 1, "cpu_count_physical": 16}
    
    # Test 1: 8GB VRAM with 2B model
    r1 = OptimizationStrategy.get_recommendation(
        mock_sys_8gb, 1000, is_windows=True, 
        model_name="google/gemma-2b", max_length=512
    )
    print(f"8GB + Gemma-2B + 512 ctx:")
    print(f"  Quant: {r1['quantization']}, Batch: {r1['per_device_train_batch_size']}")
    print(f"  Model Size Est: {r1['_estimated_model_size_b']}B")
    
    # Test 2: 8GB VRAM with 7B model
    r2 = OptimizationStrategy.get_recommendation(
        mock_sys_8gb, 1000, is_windows=True,
        model_name="meta-llama/Llama-2-7b", max_length=512
    )
    print(f"\n8GB + Llama-7B + 512 ctx:")
    print(f"  Quant: {r2['quantization']}, Batch: {r2['per_device_train_batch_size']}")
    print(f"  Model Size Est: {r2['_estimated_model_size_b']}B")
    
    # Test 3: 24GB with 7B, long context
    r3 = OptimizationStrategy.get_recommendation(
        mock_sys_24gb, 50000, is_windows=False,
        model_name="meta-llama/Llama-2-7b", max_length=2048
    )
    print(f"\n24GB + Llama-7B + 2048 ctx:")
    print(f"  Quant: {r3['quantization']}, Batch: {r3['per_device_train_batch_size']}")
    print(f"  Context Factor: {r3['_context_factor']}x")
