"""Tests for ModelInspector in core/model_inspector.py"""
import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Mock transformers before any imports that use it
if 'transformers' not in sys.modules:
    sys.modules['transformers'] = MagicMock()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.model_inspector import ModelInspector


class TestGetLoraTargetModules:
    """Tests for LoRA target module detection."""

    def _mock_config(self, model_type, architectures=None, hidden_size=2048):
        """Helper to create mock AutoConfig."""
        config = MagicMock()
        config.model_type = model_type
        config.architectures = architectures or [f"{model_type.capitalize()}ForCausalLM"]
        config.hidden_size = hidden_size
        return config

    @patch('core.model_inspector.AutoConfig')
    def test_llama_family(self, mock_auto_config):
        mock_auto_config.from_pretrained.return_value = self._mock_config("llama")
        result = ModelInspector.get_lora_target_modules("meta-llama/Llama-2-7b")
        assert result == ["q_proj", "k_proj", "v_proj", "o_proj"]

    @patch('core.model_inspector.AutoConfig')
    def test_gemma_family(self, mock_auto_config):
        mock_auto_config.from_pretrained.return_value = self._mock_config("gemma")
        result = ModelInspector.get_lora_target_modules("google/gemma-2b")
        assert result == ["q_proj", "k_proj", "v_proj", "o_proj"]

    @patch('core.model_inspector.AutoConfig')
    def test_mistral_family(self, mock_auto_config):
        mock_auto_config.from_pretrained.return_value = self._mock_config("mistral")
        result = ModelInspector.get_lora_target_modules("mistralai/Mistral-7B")
        assert result == ["q_proj", "k_proj", "v_proj", "o_proj"]

    @patch('core.model_inspector.AutoConfig')
    def test_phi_small(self, mock_auto_config):
        """Phi-2 (hidden_size=2560) should use q_proj/v_proj/dense."""
        mock_auto_config.from_pretrained.return_value = self._mock_config(
            "phi", hidden_size=2560
        )
        result = ModelInspector.get_lora_target_modules("microsoft/phi-2")
        assert "q_proj" in result
        assert "dense" in result

    @patch('core.model_inspector.AutoConfig')
    def test_phi_large(self, mock_auto_config):
        """Phi-3 (hidden_size=3072+) should use fused qkv_proj."""
        mock_auto_config.from_pretrained.return_value = self._mock_config(
            "phi", hidden_size=3072
        )
        result = ModelInspector.get_lora_target_modules("microsoft/phi-3-mini")
        assert "qkv_proj" in result
        assert "o_proj" in result

    @patch('core.model_inspector.AutoConfig')
    def test_bloom(self, mock_auto_config):
        mock_auto_config.from_pretrained.return_value = self._mock_config("bloom")
        result = ModelInspector.get_lora_target_modules("bigscience/bloom-560m")
        assert result == ["query_key_value"]

    @patch('core.model_inspector.AutoConfig')
    def test_falcon(self, mock_auto_config):
        mock_auto_config.from_pretrained.return_value = self._mock_config("falcon")
        result = ModelInspector.get_lora_target_modules("tiiuae/falcon-7b")
        assert result == ["query_key_value"]

    @patch('core.model_inspector.AutoConfig')
    def test_gpt_neox(self, mock_auto_config):
        mock_auto_config.from_pretrained.return_value = self._mock_config("gpt_neox")
        result = ModelInspector.get_lora_target_modules("EleutherAI/pythia-1b")
        assert result == ["query_key_value"]

    @patch('core.model_inspector.AutoConfig')
    def test_gpt2(self, mock_auto_config):
        mock_auto_config.from_pretrained.return_value = self._mock_config("gpt2")
        result = ModelInspector.get_lora_target_modules("gpt2")
        assert result == ["c_attn"]

    @patch('core.model_inspector.AutoConfig')
    def test_stablelm(self, mock_auto_config):
        mock_auto_config.from_pretrained.return_value = self._mock_config("stablelm")
        result = ModelInspector.get_lora_target_modules("stabilityai/stablelm-2-zephyr-1.6b")
        assert result == ["c_attn"]

    @patch('core.model_inspector.AutoConfig')
    def test_t5(self, mock_auto_config):
        mock_auto_config.from_pretrained.return_value = self._mock_config("t5")
        result = ModelInspector.get_lora_target_modules("google/flan-t5-base")
        assert result == ["q", "v"]

    @patch('core.model_inspector.AutoConfig')
    def test_unknown_model_fallback(self, mock_auto_config):
        mock_auto_config.from_pretrained.return_value = self._mock_config("unknown_model_type")
        result = ModelInspector.get_lora_target_modules("some/unknown-model")
        assert result == ["q_proj", "v_proj"]

    @patch('core.model_inspector.AutoConfig')
    def test_config_fetch_failure_fallback(self, mock_auto_config):
        """When AutoConfig fails, should return safe default."""
        mock_auto_config.from_pretrained.side_effect = Exception("Network error")
        result = ModelInspector.get_lora_target_modules("unreachable/model")
        assert result == ["q_proj", "v_proj"]
