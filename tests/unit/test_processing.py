"""Test processing.py functions."""

import numpy as np
import pytest

from mobi_motion_tracking.processing import processing


def test_dtw_good() -> None:
    """Test that the dynamic time warping funciton extracts correct known values."""
    preprocessed_target_data = np.array([
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 2],
        [0, 0, 0, 0, 3],
        [0, 0, 0, 0, 4],
    ])

    preprocessed_experimental_data = np.array([
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 2],
        [0, 0, 0, 0, 3],
        [0, 0, 0, 0, 4],
    ])

    expected_target_path = np.array([0, 0, 1, 2, 3])
    expected_experimental_path = np.array([0, 1, 2, 3, 4])
    expected_distance = 0

    distance, target_path, experimental_path = processing.dynamic_time_warping(
        preprocessed_target_data, preprocessed_experimental_data
    )

    assert distance == expected_distance, (
        f"Calculated distance {distance} does not \
        match expected output {expected_distance}."
    )
    assert np.array_equal(target_path, expected_target_path), (
        f"Calculated target path \
        {target_path} does not match expected output {expected_target_path}."
    )
    assert np.array_equal(experimental_path, expected_experimental_path), (
        f"Calculated \
        experimental path {experimental_path} does not match expected output \
            {expected_experimental_path}."
    )
    assert isinstance(distance, float), "Ouput distance should be a float."
    assert isinstance(target_path, np.ndarray), (
        "Output target path should be a NumPy \
        array."
    )
    assert isinstance(experimental_path, np.ndarray), (
        "Output experimental path should \
        be a NumPy array."
    )


def test_dtw_empty_warping_path() -> None:
    """Test that the dtw function when the DTW warping path is empty."""
    preprocessed_target_data = np.array([[]])
    preprocessed_experimental_data = np.array([[0]])
    with pytest.raises(
        ValueError, match="DTW warping path is empty. Check input sequences."
    ):
        processing.dynamic_time_warping(
            preprocessed_target_data, preprocessed_experimental_data
        )


def test_dtw_identical_sequences() -> None:
    """Test that the dtw function when target and experimental sequences are equal."""
    preprocessed_target_data = np.array([[0, 1, 2]])
    preprocessed_experimental_data = np.array([[0, 1, 2]])
    with pytest.raises(
        ValueError,
        match="Target and experimental data are identical. DTW is not needed.",
    ):
        processing.dynamic_time_warping(
            preprocessed_target_data, preprocessed_experimental_data
        )
