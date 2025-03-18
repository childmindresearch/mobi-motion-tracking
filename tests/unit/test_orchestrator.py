"""Tests orchestrator.py."""

import pathlib
from typing import Tuple

import numpy as np
import pytest

from mobi_motion_tracking.core import orchestrator


@pytest.fixture
def mock_paths(tmp_path: pathlib.Path) -> Tuple[pathlib.Path, pathlib.Path]:
    """Creates temporary mock paths for gold and experimental files."""
    gold_path = tmp_path / "gold.xlsx"
    experimental_path = tmp_path / "experimental"
    experimental_path.mkdir()
    (experimental_path / "001.xlsx").touch()

    return gold_path, experimental_path


def test_run_with_valid_directory(
    mock_paths: Tuple[pathlib.Path, pathlib.Path],
) -> None:
    """Test run funciton with a valid directory."""
    gold_path, experimental_path = mock_paths
    experimental_dir = experimental_path.parent
    sequence = 1
    algorithm = "DTW"

    orchestrator.run(gold_path, experimental_dir, sequence, algorithm)


def test_run_with_valid_file(mock_paths: Tuple[pathlib.Path, pathlib.Path]) -> None:
    """Test run function with a valid file."""
    gold_path, experimental_file = mock_paths
    sequence = 1
    algorithm = "DTW"

    orchestrator.run(gold_path, experimental_file, sequence, algorithm)


def test_run_with_invalid_path() -> None:
    """Test run function with invalid experimental path."""
    gold_path = pathlib.Path("/invalid/gold.xlsx")
    experimental_path = pathlib.Path("/invalid/experimental")
    sequence = 1
    algorithm = "DTW"

    with pytest.raises(
        ValueError,
        match=f"Path '{experimental_path}' is neither a \
                       file nor a directory.",
    ):
        orchestrator.run(gold_path, experimental_path, sequence, algorithm)


def test_run_with_invalid_sequence_type(
    mock_paths: Tuple[pathlib.Path, pathlib.Path],
) -> None:
    """Test run function with invalid sequence."""
    gold_path, experimental_path = mock_paths
    sequence = "invalid_type"
    algorithm = "DTW"

    with pytest.raises(
        TypeError,
        match=f"Unsupported type: \
                    {type(sequence).__name__}. Expected list or int.",
    ):
        orchestrator.run(gold_path, experimental_path, sequence, algorithm)


def test_invalid_algorithm() -> None:
    """Test run_algorithm with an invalid algorithm."""
    gold_data = np.array([[1, 2], [3, 4]])
    subject_data = np.array([[1, 2], [3, 4]])
    algorithm = "INVALID"

    with pytest.raises(ValueError, match=f"Unsupported algorithm '{algorithm}'."):
        orchestrator.run_algorithm(algorithm, gold_data, subject_data)
