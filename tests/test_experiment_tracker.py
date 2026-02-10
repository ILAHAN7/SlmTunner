"""Tests for ExperimentTracker in core/experiment_tracker.py"""
import pytest
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.experiment_tracker import ExperimentTracker


class TestStartRun:
    """Tests for experiment run initialization."""

    def test_creates_run_directory(self, tmp_path):
        tracker = ExperimentTracker(base_dir=str(tmp_path / "exps"))
        run_dir = tracker.start_run({"model_name": "test"})
        assert run_dir.exists()

    def test_saves_config_json(self, tmp_path):
        tracker = ExperimentTracker(base_dir=str(tmp_path / "exps"))
        tracker.start_run({"model_name": "test", "batch_size": 4})
        config_file = tracker.current_run_dir / "config.json"
        assert config_file.exists()
        data = json.loads(config_file.read_text())
        assert data["model_name"] == "test"
        assert data["batch_size"] == 4

    def test_custom_run_name(self, tmp_path):
        tracker = ExperimentTracker(base_dir=str(tmp_path / "exps"))
        tracker.start_run({"model_name": "test"}, run_name="my_run")
        assert tracker.current_run_dir.name == "my_run"

    def test_resets_metrics_on_new_run(self, tmp_path):
        tracker = ExperimentTracker(base_dir=str(tmp_path / "exps"))
        tracker.start_run({"model_name": "run1"})
        tracker.log_metrics(1, {"loss": 2.0})
        assert len(tracker.metrics) == 1

        tracker.start_run({"model_name": "run2"})
        assert len(tracker.metrics) == 0


class TestLogMetrics:
    """Tests for per-step metric logging."""

    def test_appends_to_metrics_list(self, tmp_path):
        tracker = ExperimentTracker(base_dir=str(tmp_path / "exps"))
        tracker.start_run({"model_name": "test"})
        tracker.log_metrics(10, {"loss": 2.5})
        tracker.log_metrics(20, {"loss": 1.8})
        assert len(tracker.metrics) == 2

    def test_writes_to_jsonl_file(self, tmp_path):
        tracker = ExperimentTracker(base_dir=str(tmp_path / "exps"))
        tracker.start_run({"model_name": "test"})
        tracker.log_metrics(10, {"loss": 2.5})
        tracker.log_metrics(20, {"loss": 1.8})

        metrics_file = tracker.current_run_dir / "metrics.jsonl"
        assert metrics_file.exists()
        lines = metrics_file.read_text().strip().split('\n')
        assert len(lines) == 2
        assert json.loads(lines[0])["step"] == 10

    def test_metrics_include_timestamp(self, tmp_path):
        tracker = ExperimentTracker(base_dir=str(tmp_path / "exps"))
        tracker.start_run({"model_name": "test"})
        tracker.log_metrics(1, {"loss": 1.0})
        assert "timestamp" in tracker.metrics[0]


class TestEndRun:
    """Tests for experiment finalization."""

    def test_creates_summary_json(self, tmp_path):
        tracker = ExperimentTracker(base_dir=str(tmp_path / "exps"))
        tracker.start_run({"model_name": "test-model"})
        tracker.log_metrics(10, {"loss": 2.0})
        tracker.log_metrics(20, {"loss": 1.5})
        summary = tracker.end_run({"status": "completed"})

        assert summary["model"] == "test-model"
        assert summary["total_steps"] == 2
        assert summary["final_loss"] == 1.5
        assert "duration_human" in summary

        summary_file = tracker.current_run_dir / "summary.json"
        assert summary_file.exists()

    def test_creates_training_report_md(self, tmp_path):
        tracker = ExperimentTracker(base_dir=str(tmp_path / "exps"))
        tracker.start_run({"model_name": "test"})
        tracker.end_run()
        report = tracker.current_run_dir / "training_report.md"
        assert report.exists()
        content = report.read_text()
        assert "Training Report" in content
        assert "v2.2" in content

    def test_no_active_run_warns(self, tmp_path):
        tracker = ExperimentTracker(base_dir=str(tmp_path / "exps"))
        result = tracker.end_run()
        assert result is None

    def test_avg_loss_calculated(self, tmp_path):
        tracker = ExperimentTracker(base_dir=str(tmp_path / "exps"))
        tracker.start_run({"model_name": "test"})
        tracker.log_metrics(1, {"loss": 4.0})
        tracker.log_metrics(2, {"loss": 2.0})
        summary = tracker.end_run()
        assert summary["avg_loss"] == 3.0


class TestListExperiments:
    """Tests for experiment listing."""

    def test_empty_directory(self, tmp_path):
        tracker = ExperimentTracker(base_dir=str(tmp_path / "empty_exps"))
        exps = tracker.list_experiments()
        assert exps == []

    def test_lists_completed_experiments(self, tmp_path):
        tracker = ExperimentTracker(base_dir=str(tmp_path / "exps"))
        tracker.start_run({"model_name": "m1"}, run_name="run_001")
        tracker.end_run({"status": "completed"})
        tracker.start_run({"model_name": "m2"}, run_name="run_002")
        tracker.end_run({"status": "completed"})

        exps = tracker.list_experiments()
        assert len(exps) == 2

    def test_incomplete_runs_listed(self, tmp_path):
        tracker = ExperimentTracker(base_dir=str(tmp_path / "exps"))
        tracker.start_run({"model_name": "m1"}, run_name="incomplete_run")
        # Don't end run - no summary.json

        exps = tracker.list_experiments()
        assert len(exps) == 1
        assert exps[0]["status"] == "incomplete"


class TestFormatDuration:
    """Tests for duration formatting helper."""

    def test_seconds_only(self, tmp_path):
        tracker = ExperimentTracker(base_dir=str(tmp_path / "exps"))
        assert tracker._format_duration(45) == "45s"

    def test_minutes_and_seconds(self, tmp_path):
        tracker = ExperimentTracker(base_dir=str(tmp_path / "exps"))
        assert tracker._format_duration(125) == "2m 5s"

    def test_hours_minutes_seconds(self, tmp_path):
        tracker = ExperimentTracker(base_dir=str(tmp_path / "exps"))
        assert tracker._format_duration(3661) == "1h 1m 1s"
