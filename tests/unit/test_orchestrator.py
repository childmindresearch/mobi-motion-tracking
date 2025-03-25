"""test orchestrator.py functions."""

import pathlib

import pytest

from mobi_motion_tracking.core import orchestrator


def test_run_file_bad_alg() -> None:
    """Tests the run_file function with an unexpected algorithm."""
    file_path = pathlib.Path("tests/sample_data/100.xlsx")
    gold_path = pathlib.Path("tests/sample_data/Gold.xlsx")
    output_dir = pathlib.Path("tests/sample_data")
    sequence = [1, 2, 3]

    with pytest.raises(ValueError, match="Unsupported algorithm."):
        orchestrator.run_file(file_path, gold_path, output_dir, sequence, "bad_alg")


def test_run_empty_sequence() -> None:
    """Tests the run function with an empty sequence list."""
    file_path = pathlib.Path("tests/sample_data/100.xlsx")
    gold_path = pathlib.Path("tests/sample_data/Gold.xlsx")
    sequence = []

    with pytest.raises(
        ValueError, match="Input sequence list is empty. Must have at least 1 sequence."
    ):
        orchestrator.run(file_path, gold_path, sequence, "dtw")
