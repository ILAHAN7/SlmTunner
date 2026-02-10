"""Tests for ConfigLoader in core/config_loader.py"""
import pytest
import yaml
import os
import sys
from unittest.mock import patch, MagicMock

# Mock torch before importing config_loader (torch may not be installed)
sys.modules.setdefault('torch', MagicMock())
sys.modules.setdefault('torch.cuda', MagicMock())

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.config_loader import ConfigLoader


class TestLoadConfig:
    """Tests for YAML config loading."""

    def test_loads_valid_config(self, temp_config_file):
        config = ConfigLoader.load_config(temp_config_file)
        assert config["model"]["name"] == "google/gemma-2b"
        assert config["dataset"]["rows"] == 1000

    def test_missing_file_returns_empty(self, tmp_path):
        config = ConfigLoader.load_config(str(tmp_path / "nonexistent.yaml"))
        assert config == {}

    def test_empty_file_returns_empty(self, tmp_path):
        empty_path = tmp_path / "empty.yaml"
        empty_path.write_text("")
        config = ConfigLoader.load_config(str(empty_path))
        assert config == {}

    def test_invalid_yaml_returns_empty(self, tmp_path):
        bad_path = tmp_path / "bad.yaml"
        bad_path.write_text("{{invalid: yaml: [}")
        config = ConfigLoader.load_config(str(bad_path))
        assert config == {}


class TestGetFinalConfig:
    """Tests for config merging logic."""

    @patch('core.config_loader.get_system_info')
    def test_auto_values_get_replaced(self, mock_sys_info, temp_config_file):
        mock_sys_info.return_value = {
            "os": "Windows",
            "best_gpu_vram_gb": 8.0,
            "gpu_count": 1,
            "cpu_count_physical": 6,
            "detection_method": "auto"
        }
        config = ConfigLoader.get_final_config(temp_config_file, dataset_rows=1000)
        
        # "auto" values should be replaced with actual numbers
        assert isinstance(config["per_device_train_batch_size"], int)
        assert isinstance(config["gradient_accumulation_steps"], int)
        assert isinstance(config["num_train_epochs"], int)
        assert config["quantization"] in ("4bit", "8bit", "none")

    @patch('core.config_loader.get_system_info')
    def test_manual_values_override_auto(self, mock_sys_info, tmp_path):
        mock_sys_info.return_value = {
            "os": "Windows",
            "best_gpu_vram_gb": 8.0,
            "gpu_count": 1,
            "cpu_count_physical": 6,
            "detection_method": "auto"
        }
        # Create config with manual values
        config = {
            "model": {"name": "google/gemma-2b"},
            "training": {
                "per_device_train_batch_size": 4,  # Manual override
                "quantization": "8bit",  # Manual override
                "gradient_accumulation_steps": "auto",  # Let system decide
                "max_length": 1024
            }
        }
        config_path = tmp_path / "manual_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        result = ConfigLoader.get_final_config(str(config_path), dataset_rows=1000)
        
        assert result["per_device_train_batch_size"] == 4  # User override preserved
        assert result["quantization"] == "8bit"  # User override preserved
        assert isinstance(result["gradient_accumulation_steps"], int)  # Auto resolved

    @patch('core.config_loader.get_system_info')
    def test_model_name_propagated(self, mock_sys_info, temp_config_file):
        mock_sys_info.return_value = {
            "os": "Windows",
            "best_gpu_vram_gb": 8.0,
            "gpu_count": 1,
            "cpu_count_physical": 6,
            "detection_method": "auto"
        }
        config = ConfigLoader.get_final_config(temp_config_file, dataset_rows=1000)
        assert config["model_name"] == "google/gemma-2b"

    @patch('core.config_loader.get_system_info')
    def test_dataset_paths_propagated(self, mock_sys_info, temp_config_file):
        mock_sys_info.return_value = {
            "os": "Windows",
            "best_gpu_vram_gb": 8.0,
            "gpu_count": 1,
            "cpu_count_physical": 6,
            "detection_method": "auto"
        }
        config = ConfigLoader.get_final_config(temp_config_file, dataset_rows=1000)
        assert config["train_path"] == "train.jsonl"
        assert config["valid_path"] == "valid.jsonl"
