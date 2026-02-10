"""
Shared pytest fixtures for SlmTunner tests.
All tests are designed to run without GPU or model downloads (mock-based).
"""
import pytest
import json
import os
import tempfile
from pathlib import Path


@pytest.fixture
def mock_system_info_8gb():
    """Mock system info for 8GB VRAM GPU."""
    return {
        "os": "Windows",
        "os_release": "10",
        "python_version": "3.10.0",
        "cpu_count_physical": 6,
        "cpu_count_logical": 12,
        "ram_total_gb": 32.0,
        "ram_available_gb": 20.0,
        "gpu_available": True,
        "gpu_count": 1,
        "gpus": [{"id": 0, "name": "NVIDIA RTX 4050", "vram_gb": 8.0}],
        "best_gpu_vram_gb": 8.0,
        "detection_method": "auto"
    }


@pytest.fixture
def mock_system_info_24gb():
    """Mock system info for 24GB VRAM GPU."""
    return {
        "os": "Linux",
        "os_release": "5.15",
        "python_version": "3.10.0",
        "cpu_count_physical": 16,
        "cpu_count_logical": 32,
        "ram_total_gb": 64.0,
        "ram_available_gb": 50.0,
        "gpu_available": True,
        "gpu_count": 1,
        "gpus": [{"id": 0, "name": "NVIDIA RTX 3090", "vram_gb": 24.0}],
        "best_gpu_vram_gb": 24.0,
        "detection_method": "auto"
    }


@pytest.fixture
def mock_system_info_cpu():
    """Mock system info for CPU-only environment."""
    return {
        "os": "Windows",
        "os_release": "10",
        "python_version": "3.10.0",
        "cpu_count_physical": 4,
        "cpu_count_logical": 8,
        "ram_total_gb": 16.0,
        "ram_available_gb": 10.0,
        "gpu_available": False,
        "gpu_count": 0,
        "gpus": [],
        "best_gpu_vram_gb": 0,
        "detection_method": "auto"
    }


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config.yaml for testing."""
    config = {
        "model": {
            "name": "google/gemma-2b",
            "output_dir": "./gemma-2b-lora"
        },
        "dataset": {
            "train_path": "train.jsonl",
            "valid_path": "valid.jsonl",
            "rows": 1000
        },
        "training": {
            "per_device_train_batch_size": "auto",
            "gradient_accumulation_steps": "auto",
            "quantization": "auto",
            "num_train_epochs": "auto",
            "max_length": 512
        }
    }
    import yaml
    config_path = tmp_path / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    return str(config_path)


@pytest.fixture
def temp_jsonl_valid(tmp_path):
    """Create a valid JSONL training file."""
    data = [
        {"prompt": "What is Python?", "completion": "Python is a programming language."},
        {"prompt": "What is 2+2?", "completion": "4"},
        {"prompt": "Hello", "completion": "Hi there!"},
    ]
    path = tmp_path / "train.jsonl"
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    return str(path)


@pytest.fixture
def temp_jsonl_invalid(tmp_path):
    """Create an invalid JSONL file with parse errors."""
    path = tmp_path / "bad_train.jsonl"
    with open(path, 'w', encoding='utf-8') as f:
        f.write('{"prompt": "valid", "completion": "ok"}\n')
        f.write('not json at all\n')
        f.write('{"broken": json}\n')
    return str(path)


@pytest.fixture
def temp_jsonl_text_format(tmp_path):
    """Create a JSONL file using text format."""
    data = [
        {"text": "This is a complete text sample for training."},
        {"text": "Another sample with different content."},
    ]
    path = tmp_path / "text_train.jsonl"
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    return str(path)
