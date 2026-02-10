"""
Shared constants for the SlmTunner pipeline.
Centralizes column name mappings used across data_validator and trainer.
"""

# Supported prompt/completion column names for dataset formats
PROMPT_KEYS = ["prompt", "instruction", "input", "question"]
COMPLETION_KEYS = ["completion", "response", "output", "answer"]
TEXT_KEY = "text"
