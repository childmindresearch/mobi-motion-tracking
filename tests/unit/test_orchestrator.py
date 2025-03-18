"""Tests orchestrator.py."""

import pathlib
from typing import Tuple

import numpy as np
import pytest
import pytest_mock

from mobi_motion_tracking.core import models, orchestrator
from mobi_motion_tracking.io.readers import readers


@pytest.fixture
def mock_paths(tmp_path: pathlib.Path) -> Tuple[pathlib.Path, pathlib.path]:
    """Creates temporary mock paths for gold and experimental files."""
    gold_path = tmp_path / "gold.xlsx"
    experimental_path = tmp_path / "experimental"
    experimental_path.mkdir()
    (experimental_path / "001.xlsx").touch()

    return gold_path, experimental_path


@pytest.fixture
def mock_dependencies(mocker: pytest_mock.MockerFixture) -> None:
    """Mocks external dependencies of the run function."""
    mocker.patch(
        "your_module.models.Metadata.get_metadata",
        return_value={"sequence_sheetname": "Sheet1"},
    )
    mocker.patch("your_module.readers.read_sheet", return_value="mock_data")
    mocker.patch(
        "your_module.preprocessing.center_joints_to_hip", return_value="centered_data"
    )
    mocker.patch(
        "your_module.preprocessing.get_average_length", return_value="avg_lengths"
    )
    mocker.patch(
        "your_module.preprocessing.normalize_segments", return_value="normalized_data"
    )
    mocker.patch("your_module.run_algorithm", return_value="mock_similarity_metric")
    mocker.patch("your_module.writers.save_results_to_ndjson")


def test_run_with_valid_directory(
    mock_paths: Tuple[pathlib.Path, pathlib.Path], mock_dependencies
) -> None:
    """Test run funciton with a valid directory."""
    gold_path, experimental_path = mock_paths
    experimental_dir = experimental_path.parent
    sequence = 1
    algorithm = "DTW"

    orchestrator.run(gold_path, experimental_dir, sequence, algorithm)

    mock_dependencies["models.Metadata.get_metadata"].assert_called()
    mock_dependencies["readers.read_sheet"].assert_called()
    mock_dependencies["preprocessing.center_joints_to_hip"].assert_called()
    mock_dependencies["run_algorithm"].assert_called()
    mock_dependencies["writers.save_results_to_ndjson"].assert_called()


def test_run_with_valid_file(
    mock_paths: Tuple[pathlib.Path, pathlib.Path], mock_dependencies
) -> None:
    """Test run function with a valid file."""
    gold_path, experimental_file = mock_paths
    sequence = 1
    algorithm = "DTW"

    orchestrator.run(gold_path, experimental_file, sequence, algorithm)

    mock_dependencies["models.Metadata.get_metadata"].assert_called()
    mock_dependencies["readers.read_sheet"].assert_called()
    mock_dependencies["preprocessing.center_joints_to_hip"].assert_called()
    mock_dependencies["run_algorithm"].assert_called()
    mock_dependencies["writers.save_results_to_ndjson"].assert_called()


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
