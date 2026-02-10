"""Tests for SlmEvaluator and Callbacks in core/evaluator.py

Note: compute_metrics requires torch tensors, so those tests are skipped
when torch is not available. Callback tests work with mocked state.
"""
import pytest
import math
import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Check if REAL torch is available (not just a MagicMock from other tests)
HAS_TORCH = False
try:
    import torch
    # MagicMock won't have a real __file__ attribute with a string path
    if hasattr(torch, '__file__') and isinstance(torch.__file__, str):
        _ = torch.tensor([1, 2, 3])
        HAS_TORCH = True
except (ImportError, TypeError, AttributeError, Exception):
    HAS_TORCH = False

# For callback tests, we need real TrainerCallback base class behavior.
# When transformers is not installed, we create a minimal stub.
if 'transformers' not in sys.modules:
    transformers_mock = MagicMock()
    # TrainerCallback needs to be a real class so subclassing works
    class _StubTrainerCallback:
        def on_log(self, args, state, control, logs=None, **kwargs):
            pass
    transformers_mock.TrainerCallback = _StubTrainerCallback
    sys.modules['transformers'] = transformers_mock

if not HAS_TORCH:
    mock_torch = MagicMock()
    mock_torch.Tensor = type('FakeTensor', (), {})
    sys.modules['torch'] = mock_torch
    sys.modules['torch.cuda'] = MagicMock()

# NOW import evaluator (after mocks are in place)
from core.evaluator import SlmEvaluator, LoggingCallback, TrackingCallback


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestComputeMetrics:
    """Tests for token-level accuracy computation (requires torch)."""

    def test_perfect_accuracy(self):
        predictions = torch.tensor([[1, 2, 3, 4, 5]])
        labels = torch.tensor([[0, 1, 2, 3, 4]])
        result = SlmEvaluator.compute_metrics((predictions, labels))
        assert result["token_accuracy"] == 1.0

    def test_zero_accuracy(self):
        predictions = torch.tensor([[10, 20, 30, 40, 50]])
        labels = torch.tensor([[0, 1, 2, 3, 4]])
        result = SlmEvaluator.compute_metrics((predictions, labels))
        assert result["token_accuracy"] == 0.0

    def test_ignore_padding_tokens(self):
        predictions = torch.tensor([[1, 2, 3, 4, 5]])
        labels = torch.tensor([[0, 1, -100, -100, 4]])
        result = SlmEvaluator.compute_metrics((predictions, labels))
        assert result["token_accuracy"] == 1.0

    def test_all_labels_masked(self):
        predictions = torch.tensor([[1, 2, 3]])
        labels = torch.tensor([[-100, -100, -100]])
        result = SlmEvaluator.compute_metrics((predictions, labels))
        assert result["token_accuracy"] == 0.0


class TestLoggingCallback:
    """Tests for LoggingCallback perplexity computation."""

    def test_perplexity_from_loss(self):
        callback = LoggingCallback()
        state = MagicMock()
        state.global_step = 100
        logs = {"loss": 2.0}
        
        callback.on_log(None, state, None, logs=logs)
        
        assert "perplexity" in logs
        assert abs(logs["perplexity"] - math.exp(2.0)) < 0.01

    def test_overflow_loss_gives_inf(self):
        callback = LoggingCallback()
        state = MagicMock()
        state.global_step = 100
        logs = {"loss": 1000.0}
        
        callback.on_log(None, state, None, logs=logs)
        assert logs["perplexity"] == float("inf")

    def test_eval_perplexity(self):
        callback = LoggingCallback()
        state = MagicMock()
        logs = {"eval_loss": 1.5}
        
        callback.on_log(None, state, None, logs=logs)
        assert "eval_perplexity" in logs
        assert abs(logs["eval_perplexity"] - math.exp(1.5)) < 0.01

    def test_eval_token_accuracy_logged(self):
        callback = LoggingCallback()
        state = MagicMock()
        logs = {"eval_loss": 1.0, "eval_token_accuracy": 0.85}
        
        callback.on_log(None, state, None, logs=logs)
        assert "eval_perplexity" in logs

    def test_no_crash_on_empty_logs(self):
        callback = LoggingCallback()
        state = MagicMock()
        callback.on_log(None, state, None, logs=None)
        callback.on_log(None, state, None, logs={})


class TestTrackingCallback:
    """Tests for TrackingCallback integration with ExperimentTracker."""

    def test_logs_metrics_to_tracker(self):
        tracker = MagicMock()
        tracker.current_run_dir = "/fake/path"
        callback = TrackingCallback(tracker)
        
        state = MagicMock()
        state.global_step = 50
        logs = {"loss": 2.5, "learning_rate": 0.0002, "epoch": 1.0}
        
        callback.on_log(None, state, None, logs=logs)
        
        tracker.log_metrics.assert_called_once()
        call_args = tracker.log_metrics.call_args
        assert call_args[0][0] == 50  # step
        metrics = call_args[0][1]
        assert "loss" in metrics
        assert "perplexity" in metrics
        assert "learning_rate" in metrics

    def test_no_crash_when_no_run(self):
        tracker = MagicMock()
        tracker.current_run_dir = None
        callback = TrackingCallback(tracker)
        
        state = MagicMock()
        callback.on_log(None, state, None, logs={"loss": 1.0})
        tracker.log_metrics.assert_not_called()
