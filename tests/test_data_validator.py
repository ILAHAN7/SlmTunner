"""Tests for DataValidator in core/data_validator.py"""
import pytest
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.data_validator import DataValidator


class TestValidateJsonl:
    """Tests for JSONL file validation."""

    def test_valid_prompt_completion_format(self, temp_jsonl_valid):
        report = DataValidator.validate(temp_jsonl_valid)
        assert report["valid"] is True
        assert len(report["errors"]) == 0
        assert report["stats"]["train"]["valid_lines"] == 3
        assert report["stats"]["train"]["format_detected"] == "prompt/completion"

    def test_valid_text_format(self, temp_jsonl_text_format):
        report = DataValidator.validate(temp_jsonl_text_format)
        assert report["valid"] is True
        assert report["stats"]["train"]["format_detected"] == "text"

    def test_invalid_json_lines(self, temp_jsonl_invalid):
        report = DataValidator.validate(temp_jsonl_invalid)
        assert report["valid"] is False
        assert len(report["errors"]) > 0

    def test_missing_file(self):
        report = DataValidator.validate("/nonexistent/train.jsonl")
        assert report["valid"] is False
        assert any("not found" in err for err in report["errors"])

    def test_empty_file(self, tmp_path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        report = DataValidator.validate(str(empty))
        assert report["valid"] is False

    def test_optional_valid_path_missing(self, temp_jsonl_valid):
        report = DataValidator.validate(temp_jsonl_valid, valid_path="/nonexistent/valid.jsonl")
        # Should be valid (train exists) but with warning about missing valid file
        assert report["valid"] is True
        assert any("not found" in w for w in report["warnings"])

    def test_small_dataset_warning(self, tmp_path):
        """Datasets with < 10 samples should generate a warning."""
        path = tmp_path / "tiny.jsonl"
        with open(path, 'w') as f:
            for i in range(3):
                f.write(json.dumps({"prompt": f"q{i}", "completion": f"a{i}"}) + '\n')
        report = DataValidator.validate(str(path))
        assert report["valid"] is True
        assert any("small dataset" in w.lower() or "Very small" in w for w in report["warnings"])

    def test_avg_text_length_calculated(self, temp_jsonl_valid):
        report = DataValidator.validate(temp_jsonl_valid)
        assert report["stats"]["train"]["avg_text_length"] > 0

    def test_instruction_response_format(self, tmp_path):
        """Should recognize instruction/response format."""
        path = tmp_path / "instruct.jsonl"
        with open(path, 'w') as f:
            for i in range(5):
                f.write(json.dumps({"instruction": f"Do task {i}", "response": f"Done {i}"}) + '\n')
        report = DataValidator.validate(str(path))
        assert report["valid"] is True
        assert report["stats"]["train"]["format_detected"] == "prompt/completion"


class TestPrintReport:
    """Test that print_report doesn't crash."""

    def test_print_valid_report(self, temp_jsonl_valid, capsys):
        report = DataValidator.validate(temp_jsonl_valid)
        DataValidator.print_report(report)
        captured = capsys.readouterr()
        assert "VALID" in captured.out

    def test_print_invalid_report(self, capsys):
        report = DataValidator.validate("/nonexistent/file.jsonl")
        DataValidator.print_report(report)
        captured = capsys.readouterr()
        assert "INVALID" in captured.out
