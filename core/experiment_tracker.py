import os
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class ExperimentTracker:
    """
    Manages experiment runs, logging configurations and metrics.
    Creates a structured folder per run with config, metrics, and summary.
    """
    
    def __init__(self, base_dir="experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.current_run_dir = None
        self.metrics = []
        self.config = {}
        self.start_time = None
        
    def start_run(self, config: dict, run_name: str = None):
        """
        Start a new experiment run.
        Creates a folder: experiments/run_YYYYMMDD_HHMMSS/ or experiments/{run_name}/
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = run_name or f"run_{timestamp}"
        
        self.current_run_dir = self.base_dir / run_name
        self.current_run_dir.mkdir(exist_ok=True)
        
        self.config = config
        self.metrics = []
        self.start_time = datetime.now()
        
        # Save initial config
        config_path = self.current_run_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Experiment started: {self.current_run_dir}")
        return self.current_run_dir
    
    def log_metrics(self, step: int, metrics: dict):
        """Log metrics for a training step."""
        entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        self.metrics.append(entry)
        
        # Append to metrics file (streaming)
        metrics_path = self.current_run_dir / "metrics.jsonl"
        with open(metrics_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')
    
    def end_run(self, final_metrics: dict = None):
        """
        End the experiment run and generate summary.
        """
        if not self.current_run_dir:
            logger.warning("No active run to end.")
            return
        
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        # Calculate summary stats
        summary = {
            "run_name": self.current_run_dir.name,
            "model": self.config.get("model_name", "unknown"),
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": round(duration, 2),
            "duration_human": self._format_duration(duration),
            "total_steps": len(self.metrics),
            "config": self.config
        }
        
        # Add final metrics if provided
        if final_metrics:
            summary["final_metrics"] = final_metrics
        
        # Calculate average loss if available
        losses = [m.get("loss") for m in self.metrics if m.get("loss") is not None]
        if losses:
            summary["avg_loss"] = round(sum(losses) / len(losses), 4)
            summary["final_loss"] = losses[-1] if losses else None
        
        # Save summary JSON
        summary_path = self.current_run_dir / "summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Generate Markdown report
        self._generate_report(summary)
        
        logger.info(f"Experiment ended: {self.current_run_dir}")
        logger.info(f"Duration: {summary['duration_human']}")
        
        return summary
    
    def _format_duration(self, seconds: float) -> str:
        """Format seconds into human-readable string."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def _generate_report(self, summary: dict):
        """Generate a markdown training report."""
        report_path = self.current_run_dir / "training_report.md"
        
        config = summary.get("config", {})
        
        content = f"""# Training Report: {summary['run_name']}

## Overview
- **Model**: {summary.get('model', 'N/A')}
- **Duration**: {summary.get('duration_human', 'N/A')}
- **Total Steps**: {summary.get('total_steps', 0)}
- **Final Loss**: {summary.get('final_loss', 'N/A')}

## Configuration
| Parameter | Value |
|-----------|-------|
| Batch Size | {config.get('per_device_train_batch_size', 'N/A')} |
| Accumulation | {config.get('gradient_accumulation_steps', 'N/A')} |
| Epochs | {config.get('num_train_epochs', 'N/A')} |
| Quantization | {config.get('quantization', 'N/A')} |
| Learning Rate | {config.get('learning_rate', 'N/A')} |

## Files
- `config.json` - Full configuration snapshot
- `metrics.jsonl` - Per-step training metrics
- `summary.json` - Run summary

---
*Generated by SlmTunner v2.1*
"""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def list_experiments(self) -> list:
        """List all past experiments."""
        experiments = []
        if self.base_dir.exists():
            for run_dir in sorted(self.base_dir.iterdir(), reverse=True):
                if run_dir.is_dir():
                    summary_file = run_dir / "summary.json"
                    if summary_file.exists():
                        with open(summary_file, 'r') as f:
                            experiments.append(json.load(f))
                    else:
                        experiments.append({"run_name": run_dir.name, "status": "incomplete"})
        return experiments


# Global tracker instance
tracker = ExperimentTracker()

if __name__ == "__main__":
    print(">> Experiment Tracker Test")
    
    # Mock test
    t = ExperimentTracker(base_dir="./test_experiments")
    t.start_run({"model_name": "test-model", "per_device_train_batch_size": 2})
    t.log_metrics(100, {"loss": 2.5, "learning_rate": 0.0002})
    t.log_metrics(200, {"loss": 1.8, "learning_rate": 0.0001})
    summary = t.end_run()
    print(f"Test run completed: {summary}")
