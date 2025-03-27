"""Test models.py functions."""

import pathlib

import numpy as np
import pytest

from mobi_motion_tracking.core import models


def test_get_metadata_good() -> None:
    """Test get_metadata works."""
    expected_ID = "100"
    expected_seq = "seq1"

    metadata = models.Metadata.get_metadata(
        pathlib.Path("tests/sample_data/100.xlsx"), 1
    )

    assert isinstance(metadata.participant_ID, str), (
        "participant_ID should be a string."
    )
    assert isinstance(metadata.sequence_sheetname, str), "sequence should be a string."
    assert expected_ID == metadata.participant_ID, (
        "extracted ID does not match expected value."
    )
    assert expected_seq == metadata.sequence_sheetname, (
        "extracted sequence does not match expected value."
    )


def test_get_metadata_file_not_found() -> None:
    """Test FileNotFoundError when file does not exist."""
    metadata = models.Metadata.get_metadata(pathlib.Path("/dummy/path/100.xlsx"), 1)

    assert metadata.participant_ID == "None", (
        "Expected output should be None when a file does not \
        exist."
    )


def test_read_sheet_invalid_file_extension() -> None:
    """Test ValueError when file is not .xlsx."""
    metadata = models.Metadata.get_metadata(pathlib.Path("/dummy/path/100.csv"), 1)

    assert metadata.participant_ID == "None", (
        "Expected output should be None with an invalid file \
        extension."
    )


def test_get_metadata_incorrect_filename() -> None:
    """Test get_metadata with an incorrect filename."""
    metadata = models.Metadata.get_metadata(pathlib.Path("/dummy/path/100_01.xlsx"), 1)

    assert metadata.participant_ID == "None", (
        "Expected output should be None when a file is named \
        incorrectly."
    )


@pytest.mark.parametrize(
    "distance, warping_path, expected_method, expected_distance, expected_target_path, \
        expected_experimental_path",
    [
        (1.0, [(0, 1), (2, 3)], "DTW", 1.0, np.array([0, 2]), np.array([1, 3])),
        (2.5, [(1, 2), (3, 4)], "DTW", 2.5, np.array([1, 3]), np.array([2, 4])),
        (0.0, [(0, 0)], "DTW", 0.0, np.array([0]), np.array([0])),
    ],
)
def test_from_dtw_good(
    distance: float,
    warping_path: list,
    expected_method: str,
    expected_distance: float,
    expected_target_path: np.ndarray,
    expected_experimental_path: np.ndarray,
) -> None:
    """Test from_dtw works with parameterized inputs."""
    similaritymetrics = models.SimilarityMetrics.from_dtw(
        distance=distance, warping_path=warping_path
    )

    assert similaritymetrics.method == expected_method, (
        f"Returned method {similaritymetrics.method} does not equal expected method \
        {expected_method}."
    )
    assert similaritymetrics.metrics["distance"] == expected_distance, (
        f"Calculated distance {similaritymetrics.metrics['distance']} does not match \
        expected output {expected_distance}."
    )
    assert np.array_equal(
        similaritymetrics.metrics["target_path"], expected_target_path
    ), (
        f"Calculated target path {similaritymetrics.metrics['target_path']} does not \
        match expected output {expected_target_path}."
    )
    assert np.array_equal(
        similaritymetrics.metrics["experimental_path"], expected_experimental_path
    ), (
        f"Calculated experimental path \
            {similaritymetrics.metrics['experimental_path']} \
        does not match expected output {expected_experimental_path}."
    )
    assert isinstance(similaritymetrics.method, str), (
        "Returned method should be a string."
    )
