"""smoke tests for orchestrator.py."""

import pathlib

import pytest

from mobi_motion_tracking.core import orchestrator


def test_orchestrator_good() -> None:
    """Smoke test for the orchestrator run function."""
    experimental_path = pathlib.Path("tests/sample_data/100.xlsx")
    gold_path = pathlib.Path("tests/sample_data/Gold.xlsx")
    sequence = [1]
    algorithm = "dtw"

    orchestrator.run(experimental_path, gold_path, sequence, algorithm)
