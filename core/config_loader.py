import yaml
import os
import logging
from .strategy import OptimizationStrategy
from .system_analyzer import get_system_info

logger = logging.getLogger(__name__)

class ConfigLoader:
    """
    Loads configuration from YAML and merges it with dynamic hardware recommendations.
    Priority: CLI Args (not handled here) > YAML Config > Auto Strategy > Defaults
    """
    
    @staticmethod
    def load_config(config_path="config.yaml"):
        """
        Loads the YAML config file.
        """
        if not os.path.exists(config_path):
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return {}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error parsing config file: {e}")
            return {}

    @staticmethod
    def get_final_config(config_path="config.yaml", dataset_rows=1000, manual_vram_gb=None):
        """
        Generates the final training configuration.
        
        Args:
            config_path: Path to config.yaml
            dataset_rows: Number of training samples
            manual_vram_gb: Manual VRAM override (passed to system_analyzer)
        """
        # 1. Load User Config (YAML) first to get model info
        user_config = ConfigLoader.load_config(config_path)
        
        # Extract model info for strategy
        model_name = user_config.get("model", {}).get("name", "unknown/7b")
        max_length = user_config.get("training", {}).get("max_length", 512)
        
        # 2. Analyze System
        sys_info = get_system_info(manual_vram_gb=manual_vram_gb)
        is_windows = (sys_info["os"] == "Windows")
        
        # 3. Get Auto Recommendation (now model-aware)
        auto_config = OptimizationStrategy.get_recommendation(
            sys_info, 
            dataset_rows, 
            is_windows=is_windows,
            model_name=model_name,
            max_length=max_length
        )
        
        # 4. Merge (User overrides Auto)
        final_config = auto_config.copy()
        
        # Merge 'training' section from yaml - respect non-auto values
        if "training" in user_config:
            for k, v in user_config["training"].items():
                # Check if value is explicitly set (not "auto" string)
                if v is not None and str(v).lower() != "auto":
                    final_config[k] = v
        
        # Merge 'model' section
        if "model" in user_config:
            final_config["model_name"] = user_config["model"].get("name")
            if "output_dir" in user_config["model"]:
                final_config["output_dir"] = user_config["model"]["output_dir"]
        
        # Merge 'dataset' section
        if "dataset" in user_config:
            final_config["train_path"] = user_config["dataset"].get("train_path", "train.jsonl")
            final_config["valid_path"] = user_config["dataset"].get("valid_path", "valid.jsonl")
        
        return final_config


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(">> Config Loader Test (Model-Aware)")
    
    # Create test config
    test_config = {
        "model": {"name": "google/gemma-2b"},
        "dataset": {"rows": 5000},
        "training": {"max_length": 1024, "quantization": "auto"}
    }
    
    with open("test_config.yaml", "w") as f:
        yaml.dump(test_config, f)
    
    cfg = ConfigLoader.get_final_config("test_config.yaml", dataset_rows=5000)
    print(f"Model: {cfg.get('model_name')}")
    print(f"Quantization: {cfg.get('quantization')}")
    print(f"Batch: {cfg.get('per_device_train_batch_size')}")
    print(f"Model Size Est: {cfg.get('_estimated_model_size_b')}B")
    
    os.remove("test_config.yaml")
