"""Tests for OptimizationStrategy in core/strategy.py"""
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.strategy import OptimizationStrategy


class TestEstimateModelSize:
    """Tests for model size estimation from model names."""

    def test_gemma_2b(self):
        assert OptimizationStrategy.estimate_model_size("google/gemma-2b") == 2.0

    def test_gemma_2b_it(self):
        assert OptimizationStrategy.estimate_model_size("google/gemma-2b-it") == 2.0

    def test_llama_7b(self):
        assert OptimizationStrategy.estimate_model_size("meta-llama/Llama-2-7b") == 7.0

    def test_llama_13b(self):
        assert OptimizationStrategy.estimate_model_size("meta-llama/Llama-2-13b-chat") == 13.0

    def test_llama_70b(self):
        assert OptimizationStrategy.estimate_model_size("meta-llama/Llama-2-70b") == 70.0

    def test_tinyllama(self):
        result = OptimizationStrategy.estimate_model_size("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        assert result == 1.1

    def test_phi_2(self):
        assert OptimizationStrategy.estimate_model_size("microsoft/phi-2") == 2.7

    def test_phi_3_mini(self):
        assert OptimizationStrategy.estimate_model_size("microsoft/phi-3-mini-4k") == 3.8

    def test_opt_125m(self):
        result = OptimizationStrategy.estimate_model_size("facebook/opt-125m")
        assert result == 0.125

    def test_pythia_70m(self):
        result = OptimizationStrategy.estimate_model_size("EleutherAI/pythia-70m")
        assert result == 0.07

    def test_unknown_model_defaults_to_7b(self):
        """Unknown models should conservatively default to 7B."""
        result = OptimizationStrategy.estimate_model_size("some/unknown-model")
        assert result == 7.0

    def test_model_with_version_number(self):
        """Should not confuse version numbers with param count."""
        result = OptimizationStrategy.estimate_model_size("model/test-v2-7b")
        assert result == 7.0


class TestGetRecommendation:
    """Tests for training parameter recommendations."""

    def test_8gb_small_model_gets_reasonable_batch(self, mock_system_info_8gb):
        rec = OptimizationStrategy.get_recommendation(
            mock_system_info_8gb, 1000, is_windows=True,
            model_name="google/gemma-2b", max_length=512
        )
        assert rec["per_device_train_batch_size"] >= 1
        assert rec["per_device_train_batch_size"] <= 16
        assert rec["quantization"] in ("4bit", "8bit", "none")

    def test_8gb_large_model_uses_4bit(self, mock_system_info_8gb):
        rec = OptimizationStrategy.get_recommendation(
            mock_system_info_8gb, 1000, is_windows=True,
            model_name="meta-llama/Llama-2-7b", max_length=512
        )
        assert rec["quantization"] == "4bit"

    def test_24gb_small_model_can_skip_quantization(self, mock_system_info_24gb):
        rec = OptimizationStrategy.get_recommendation(
            mock_system_info_24gb, 1000, is_windows=False,
            model_name="google/gemma-2b", max_length=512
        )
        # 24GB should handle 2B model without quantization
        assert rec["quantization"] in ("none", "8bit")

    def test_windows_sets_zero_workers(self, mock_system_info_8gb):
        rec = OptimizationStrategy.get_recommendation(
            mock_system_info_8gb, 1000, is_windows=True,
            model_name="google/gemma-2b"
        )
        assert rec["dataloader_num_workers"] == 0

    def test_linux_sets_nonzero_workers(self, mock_system_info_24gb):
        rec = OptimizationStrategy.get_recommendation(
            mock_system_info_24gb, 1000, is_windows=False,
            model_name="google/gemma-2b"
        )
        assert rec["dataloader_num_workers"] > 0

    def test_small_dataset_more_epochs(self, mock_system_info_8gb):
        rec = OptimizationStrategy.get_recommendation(
            mock_system_info_8gb, 100, is_windows=True,
            model_name="google/gemma-2b"
        )
        assert rec["num_train_epochs"] >= 5

    def test_large_dataset_fewer_epochs(self, mock_system_info_8gb):
        rec = OptimizationStrategy.get_recommendation(
            mock_system_info_8gb, 50000, is_windows=True,
            model_name="google/gemma-2b"
        )
        assert rec["num_train_epochs"] <= 3

    def test_cpu_only_uses_4bit(self, mock_system_info_cpu):
        rec = OptimizationStrategy.get_recommendation(
            mock_system_info_cpu, 1000, is_windows=True,
            model_name="google/gemma-2b"
        )
        assert rec["quantization"] == "4bit"
        assert rec["per_device_train_batch_size"] == 1

    def test_gradient_checkpointing_low_vram(self, mock_system_info_8gb):
        rec = OptimizationStrategy.get_recommendation(
            mock_system_info_8gb, 1000, is_windows=True,
            model_name="google/gemma-2b"
        )
        assert rec["gradient_checkpointing"] is True

    def test_long_context_reduces_batch(self, mock_system_info_8gb):
        rec_512 = OptimizationStrategy.get_recommendation(
            mock_system_info_8gb, 1000, is_windows=True,
            model_name="google/gemma-2b", max_length=512
        )
        rec_2048 = OptimizationStrategy.get_recommendation(
            mock_system_info_8gb, 1000, is_windows=True,
            model_name="google/gemma-2b", max_length=2048
        )
        # Longer context should result in same or smaller batch size
        assert rec_2048["per_device_train_batch_size"] <= rec_512["per_device_train_batch_size"]

    def test_recommendation_includes_metadata(self, mock_system_info_8gb):
        rec = OptimizationStrategy.get_recommendation(
            mock_system_info_8gb, 1000, is_windows=True,
            model_name="google/gemma-2b"
        )
        assert "_estimated_model_size_b" in rec
        assert "_vram_available" in rec
        assert "_context_factor" in rec
