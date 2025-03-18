"""Tests orchestrator.py."""

import pathlib
from typing import Tuple

import numpy as np
import pytest
import pytest_mock

from mobi_motion_tracking.core import orchestrator


class MockMetadata:
    def __init__(self, sequence_sheetname: str):
        self.sequence_sheetname = sequence_sheetname


@pytest.fixture
def mock_paths(tmp_path: pathlib.Path) -> Tuple[pathlib.Path, pathlib.Path]:
    """Creates temporary mock paths for gold and experimental files."""
    gold_path = tmp_path / "gold.xlsx"
    experimental_path = tmp_path / "001.xlsx"

    return gold_path, experimental_path


@pytest.fixture
def mock_dependencies(mocker: pytest_mock.MockerFixture) -> dict:
    """Mocks external dependencies of the run function."""
    mock_metadata = MockMetadata(sequence_sheetname="seq1")

    get_metadata_mock = mocker.patch(
        "mobi_motion_tracking.core.models.Metadata.get_metadata",
        return_value=mock_metadata,
    )
    mocker.patch(
        "mobi_motion_tracking.io.readers.readers.read_sheet", return_value="mock_data"
    )
    mocker.patch(
        "mobi_motion_tracking.preprocessing.preprocessing.center_joints_to_hip",
        return_value="centered_data",
    )
    mocker.patch(
        "mobi_motion_tracking.preprocessing.preprocessing.get_average_length",
        return_value="avg_lengths",
    )
    mocker.patch(
        "mobi_motion_tracking.preprocessing.preprocessing.normalize_segments",
        return_value="normalized_data",
    )
    mocker.patch(
        "mobi_motion_tracking.core.orchestrator.run_algorithm",
        return_value="mock_similarity_metric",
    )
    mocker.patch("mobi_motion_tracking.io.writers.writers.save_results_to_ndjson")

    return {
        "get_metadata": get_metadata_mock,
    }


def test_run_with_valid_directory(
    mock_paths: Tuple[pathlib.Path, pathlib.Path], mock_dependencies: dict
) -> None:
    """Test run funciton with a valid directory."""
    gold_path, experimental_path = mock_paths
    experimental_dir = experimental_path.parent
    sequence = 1
    algorithm = "DTW"

    orchestrator.run(gold_path, experimental_dir, sequence, algorithm)

    mock_dependencies["get_metadata"].assert_called()
    mock_dependencies["read_sheet"].assert_called()
    mock_dependencies["center_joints_to_hip"].assert_called()
    mock_dependencies["run_algorithm"].assert_called()
    mock_dependencies["save_results_to_ndjson"].assert_called()


def test_run_with_valid_file(
    mock_paths: Tuple[pathlib.Path, pathlib.Path], mock_dependencies: dict
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
