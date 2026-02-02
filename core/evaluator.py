import math
import logging
from transformers import TrainerCallback

logger = logging.getLogger(__name__)

class SlmEvaluator:
    """
    Handles metric calculation and reporting.
    """
    
    @staticmethod
    def compute_metrics(eval_preds):
        """
        Compute metrics for validation (e.g. accuracy) if needed.
        For Causal LM, we mostly care about Perplexity which is derived from loss.
        """
        return {}

    @staticmethod
    def preprocess_logits_for_metrics(logits, labels):
        """
        Memory-efficient logits handling.
        """
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)


class LoggingCallback(TrainerCallback):
    """
    Callback to log training stats to console with Perplexity.
    """
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            try:
                ppl = math.exp(logs["loss"])
            except OverflowError:
                ppl = float("inf")
            logs["perplexity"] = ppl
            logger.info(f"Step {state.global_step}: Loss {logs['loss']:.4f}, PPL {ppl:.4f}")


class TrackingCallback(TrainerCallback):
    """
    Callback that integrates with ExperimentTracker to log metrics.
    """
    
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
            if "learning_rate" in logs:
                metrics["learning_rate"] = logs["learning_rate"]
            if "epoch" in logs:
                metrics["epoch"] = logs["epoch"]
            
            if metrics:
                self.tracker.log_metrics(state.global_step, metrics)
