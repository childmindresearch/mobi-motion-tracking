"""test orchestrator.py functions."""

import os
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


def test_run_bad_input_path() -> None:
    """Tests the run function with an input path that is not a file nor a directory."""
    file_path = pathlib.Path("/tmp/fake_fifo")
    gold_path = pathlib.Path("tests/sample_data/Gold.xlsx")
    sequence = [1, 2, 3]
    os.mkfifo(file_path)

    with pytest.raises(TypeError, match="Input path is not a file nor a directory."):
        orchestrator.run(file_path, gold_path, sequence, "dtw")
