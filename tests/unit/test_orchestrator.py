"""test orchestrator.py functions."""

import pathlib

import pytest

from mobi_motion_tracking.core import orchestrator


def test_run_fake_input_path() -> None:
    """Tests the run function with a nonexistent input path."""
    file_path = pathlib.Path("fake/path")
    gold_path = pathlib.Path("tests/sample_data/Gold.xlsx")
    sequence = [1, 2, 3]

    with pytest.raises(FileNotFoundError, match="Input path does not exist."):
        orchestrator.run(file_path, gold_path, sequence, "dtw")


def test_run_bad_algorithm() -> None:
    """Tests the run function with an unsupported algorithm."""
    file_path = pathlib.Path("tests/sample_data/100.xlsx")
    gold_path = pathlib.Path("tests/sample_data/Gold.xlsx")
    sequence = [1, 2, 3]

    with pytest.raises(ValueError, match="Unsupported algorithm provided."):
        orchestrator.run(file_path, gold_path, sequence, "fake_alg")  # type: ignore[arg-type] # Failing on purpose to test ValueError


def test_run_file_bad_algorithm() -> None:
    """Tests run_file with an unsupported algorithm."""
    file_path = pathlib.Path("tests/sample_data/100.xlsx")
    gold_path = pathlib.Path("tests/sample_data/Gold.xlsx")
    output_dir = pathlib.Path("tests/sample_data")
    sequence = [1, 2, 3]

    with pytest.raises(ValueError, match="Unsupported algorithm selected."):
        orchestrator.run_file(file_path, gold_path, output_dir, sequence, "fake_alg")  # type: ignore[arg-type] # Failing on purpose to test ValueError


def test_run_file_invalid_extension() -> None:
    """Tests run_file with an invalid file extension."""
    file_path = pathlib.Path("tests/sample_data/csv_file.csv")
    gold_path = pathlib.Path("tests/sample_data/Gold.xlsx")
    output_dir = pathlib.Path("tests/sample_data")
    sequence = [1, 2, 3]

    with pytest.raises(
        ValueError, match=f"Invalid file extension: {file_path}. Expected '.xlsx'."
    ):
        orchestrator.run_file(file_path, gold_path, output_dir, sequence, "dtw")


def test_run_file_wrong_basename() -> None:
    """Tests run_file with input file with wrong basename."""
    file_path = pathlib.Path("tests/sample_data/valid_file.xlsx")
    gold_path = pathlib.Path("tests/sample_data/Gold.xlsx")
    output_dir = pathlib.Path("tests/sample_data")
    sequence = [1, 2, 3]

    with pytest.raises(ValueError, match="The input file is named incorrectly."):
        orchestrator.run_file(file_path, gold_path, output_dir, sequence, "dtw")
