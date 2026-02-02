import logging
from transformers import AutoConfig

logger = logging.getLogger(__name__)

class ModelInspector:
    """
    Analyzes a Hugging Face model to determine optimal LoRA configuration capabilities.
    """
    
    @staticmethod
    def get_lora_target_modules(model_name):
        """
        Determines the best target_modules for LoRA based on model architecture.
        Returns a list of module names (e.g., ["q_proj", "v_proj"]).
        """
        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            arch = config.architectures[0].lower() if config.architectures else ""
            model_type = config.model_type.lower()
            
            logger.info(f"Inspecting Model: {model_name} | Type: {model_type} | Arch: {arch}")

            # 1. Llama, Mistral, Gemma, Qwen family (Standard Linear Layers)
            if any(k in model_type for k in ["llama", "mistral", "gemma", "qwen", "yi"]):
                # For basic tuning, q+v is enough. For better quality, include k+o.
                # Returning standard 4-module set for optimal performance
                return ["q_proj", "k_proj", "v_proj", "o_proj"]
            
            # 2. Bloom (Fused QKV)
            if "bloom" in model_type:
                return ["query_key_value"]
            
            # 3. Falcon (Fused QKV)
            if "falcon" in model_type:
                return ["query_key_value"]
            
            # 4. GPT-NeoX / Pythia
            if "gpt_neox" in model_type:
                return ["query_key_value"]
                
            # 5. T5 / Flan-T5
            if "t5" in model_type:
                return ["q", "v"]
            
            # 6. BERT / RoBERTa (Encoder only - rare for SLM generation but possible)
            if "bert" in model_type:
                return ["query", "value"]

            # Fallback based on text heuristics if model_type is obscure
            # (Note: This is risky without loading model, so we stick to safe defaults)
            logger.warning(f"Unknown model architecture '{model_type}'. Defaulting to 'q_proj', 'v_proj'.")
            return ["q_proj", "v_proj"]

        except Exception as e:
            logger.error(f"Failed to inspect model config: {e}")
            logger.warning("Fallback to universal default ['q_proj', 'v_proj']")
            return ["q_proj", "v_proj"]

if __name__ == "__main__":
    # Test with common models
    test_models = ["google/gemma-2b", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"] 
    # Note: These require internet access to fetch config
    print(">> Model Inspection Test (Offline Mock mainly, requires HF access for real test)")
