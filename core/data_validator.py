"""
Data Validator for SlmTunner training pipeline.
Validates JSONL dataset files before training to catch errors early.
"""
import json
import os
import logging
from pathlib import Path

from .constants import PROMPT_KEYS, COMPLETION_KEYS, TEXT_KEY

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates training and validation dataset files."""

    # Use shared constants
    PROMPT_KEYS = PROMPT_KEYS
    COMPLETION_KEYS = COMPLETION_KEYS
    TEXT_KEY = TEXT_KEY

    @staticmethod
    def validate(train_path: str, valid_path: str = None) -> dict:
        """
        Validate dataset files and return a report.

        Args:
            train_path: Path to training JSONL file
            valid_path: Optional path to validation JSONL file

        Returns:
            dict with keys: valid (bool), errors (list), warnings (list), stats (dict)
        """
        report = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {}
        }

        # Validate training file (required)
        train_result = DataValidator._validate_jsonl(train_path, "train")
        report["errors"].extend(train_result["errors"])
        report["warnings"].extend(train_result["warnings"])
        report["stats"]["train"] = train_result["stats"]

        # Validate validation file (optional)
        if valid_path:
            if os.path.exists(valid_path):
                valid_result = DataValidator._validate_jsonl(valid_path, "valid")
                report["errors"].extend(valid_result["errors"])
                report["warnings"].extend(valid_result["warnings"])
                report["stats"]["valid"] = valid_result["stats"]
            else:
                report["warnings"].append(f"Validation file '{valid_path}' not found. Training will proceed without validation.")

        if report["errors"]:
            report["valid"] = False

        return report

    @staticmethod
    def _validate_jsonl(file_path: str, file_label: str) -> dict:
        """Validate a single JSONL file."""
        result = {
            "errors": [],
            "warnings": [],
            "stats": {
                "total_lines": 0,
                "valid_lines": 0,
                "empty_lines": 0,
                "format_detected": None,
                "avg_text_length": 0
            }
        }

        if not os.path.exists(file_path):
            result["errors"].append(f"[{file_label}] File not found: {file_path}")
            return result

        file_size = os.path.getsize(file_path)
        if file_size == 0:
            result["errors"].append(f"[{file_label}] File is empty: {file_path}")
            return result

        text_lengths = []
        format_counts = {"prompt_completion": 0, "text": 0, "unknown": 0}
        parse_errors = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    result["stats"]["total_lines"] += 1
                    stripped = line.strip()

                    if not stripped:
                        result["stats"]["empty_lines"] += 1
                        continue

                    try:
                        data = json.loads(stripped)
                    except json.JSONDecodeError as e:
                        parse_errors.append(f"[{file_label}] Line {line_num}: Invalid JSON - {e}")
                        continue

                    if not isinstance(data, dict):
                        parse_errors.append(f"[{file_label}] Line {line_num}: Expected JSON object, got {type(data).__name__}")
                        continue

                    result["stats"]["valid_lines"] += 1

                    # Detect format
                    has_prompt = any(k in data for k in DataValidator.PROMPT_KEYS)
                    has_completion = any(k in data for k in DataValidator.COMPLETION_KEYS)
                    has_text = DataValidator.TEXT_KEY in data

                    if has_prompt and has_completion:
                        format_counts["prompt_completion"] += 1
                        # Estimate text length
                        prompt = next((data[k] for k in DataValidator.PROMPT_KEYS if k in data), "")
                        completion = next((data[k] for k in DataValidator.COMPLETION_KEYS if k in data), "")
                        text_lengths.append(len(str(prompt)) + len(str(completion)))
                    elif has_text:
                        format_counts["text"] += 1
                        text_lengths.append(len(str(data[DataValidator.TEXT_KEY])))
                    elif has_prompt:
                        format_counts["prompt_completion"] += 1
                        result["warnings"].append(
                            f"[{file_label}] Line {line_num}: Has prompt but no completion field"
                        )
                        # Only warn for first few
                        if len(result["warnings"]) > 5:
                            break
                    else:
                        format_counts["unknown"] += 1

        except UnicodeDecodeError as e:
            result["errors"].append(f"[{file_label}] File encoding error: {e}. Ensure UTF-8 encoding.")
            return result
        except IOError as e:
            result["errors"].append(f"[{file_label}] Cannot read file: {e}")
            return result

        # Report parse errors (limit to first 5)
        if parse_errors:
            result["errors"].extend(parse_errors[:5])
            if len(parse_errors) > 5:
                result["errors"].append(f"[{file_label}] ... and {len(parse_errors) - 5} more parse errors")

        # Determine dominant format
        if format_counts["prompt_completion"] > 0:
            result["stats"]["format_detected"] = "prompt/completion"
        elif format_counts["text"] > 0:
            result["stats"]["format_detected"] = "text"
        else:
            result["warnings"].append(
                f"[{file_label}] Could not detect data format. "
                f"Expected keys: {DataValidator.PROMPT_KEYS} + {DataValidator.COMPLETION_KEYS} or '{DataValidator.TEXT_KEY}'"
            )

        if format_counts["unknown"] > 0:
            result["warnings"].append(
                f"[{file_label}] {format_counts['unknown']} lines have unrecognized format"
            )

        # Stats
        if text_lengths:
            result["stats"]["avg_text_length"] = round(sum(text_lengths) / len(text_lengths), 1)

        if result["stats"]["valid_lines"] == 0:
            result["errors"].append(f"[{file_label}] No valid data lines found in {file_path}")

        # Warnings for small datasets
        if result["stats"]["valid_lines"] < 10:
            result["warnings"].append(
                f"[{file_label}] Very small dataset ({result['stats']['valid_lines']} samples). "
                f"Consider at least 50-100 samples for meaningful fine-tuning."
            )

        return result

    @staticmethod
    def print_report(report: dict):
        """Print a human-readable validation report."""
        print("\n" + "=" * 50)
        print("    DATA VALIDATION REPORT")
        print("=" * 50)

        if report["valid"]:
            print("Status: ✓ VALID\n")
        else:
            print("Status: ✗ INVALID\n")

        if report["errors"]:
            print("ERRORS:")
            for err in report["errors"]:
                print(f"  ✗ {err}")
            print()

        if report["warnings"]:
            print("WARNINGS:")
            for warn in report["warnings"]:
                print(f"  ! {warn}")
            print()

        for split, stats in report.get("stats", {}).items():
            if isinstance(stats, dict):
                print(f"{split.upper()} Dataset:")
                print(f"  Total lines: {stats.get('total_lines', 0)}")
                print(f"  Valid lines: {stats.get('valid_lines', 0)}")
                print(f"  Format: {stats.get('format_detected', 'unknown')}")
                print(f"  Avg text length: {stats.get('avg_text_length', 0)} chars")
                print()

        print("=" * 50)
