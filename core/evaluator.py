"""
Evaluator module for training metrics and callbacks.
Provides token-level accuracy, perplexity tracking, and experiment integration.
"""
import math
import logging
import torch
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class SlmEvaluator:
    """Handles metric calculation and reporting."""

    @staticmethod
    def compute_metrics(eval_preds):
        """
        Compute token-level accuracy for Causal LM evaluation.
        
        Note: This expects logits preprocessed by preprocess_logits_for_metrics
        (i.e., already argmax'd to predictions).
        """
        predictions, labels = eval_preds
        
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        # Convert to tensors if numpy arrays
        if not isinstance(predictions, torch.Tensor):
            predictions = torch.tensor(predictions)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        
        # Shift: predictions[:-1] should match labels[1:] for causal LM
        shift_preds = predictions[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        # Mask out padding tokens (-100 is the standard ignore index)
        mask = shift_labels != -100
        
        if mask.sum() == 0:
            return {"token_accuracy": 0.0}
        
        correct = (shift_preds[mask] == shift_labels[mask]).float().sum()
        total = mask.sum().float()
        accuracy = (correct / total).item()
        
        return {"token_accuracy": round(accuracy, 4)}

    @staticmethod
    def preprocess_logits_for_metrics(logits, labels):
        """
        Memory-efficient logits handling.
        Reduces logits to argmax predictions to save memory during eval.
        """
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)


class LoggingCallback(TrainerCallback):
    """Callback to log training stats to console with Perplexity."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            try:
                ppl = math.exp(logs["loss"])
            except OverflowError:
                ppl = float("inf")
            logs["perplexity"] = ppl
            logger.info(f"Step {state.global_step}: Loss {logs['loss']:.4f}, PPL {ppl:.2f}")
        
        if logs and "eval_loss" in logs:
            try:
                eval_ppl = math.exp(logs["eval_loss"])
            except OverflowError:
                eval_ppl = float("inf")
            logs["eval_perplexity"] = eval_ppl
            logger.info(f"Eval Loss {logs['eval_loss']:.4f}, Eval PPL {eval_ppl:.2f}")
            if "eval_token_accuracy" in logs:
                logger.info(f"Eval Token Accuracy: {logs['eval_token_accuracy']:.4f}")


class TrackingCallback(TrainerCallback):
    """Callback that integrates with ExperimentTracker to log metrics."""

    def __init__(self, tracker):
        self.tracker = tracker

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and self.tracker and self.tracker.current_run_dir:
            metrics = {}
            if "loss" in logs:
                metrics["loss"] = logs["loss"]
                try:
                    metrics["perplexity"] = math.exp(logs["loss"])
                except OverflowError:
                    metrics["perplexity"] = float("inf")
            if "eval_loss" in logs:
                metrics["eval_loss"] = logs["eval_loss"]
                try:
                    metrics["eval_perplexity"] = math.exp(logs["eval_loss"])
                except OverflowError:
                    metrics["eval_perplexity"] = float("inf")
            if "eval_token_accuracy" in logs:
                metrics["eval_token_accuracy"] = logs["eval_token_accuracy"]
            if "learning_rate" in logs:
                metrics["learning_rate"] = logs["learning_rate"]
            if "epoch" in logs:
                metrics["epoch"] = logs["epoch"]

            if metrics:
                self.tracker.log_metrics(state.global_step, metrics)
