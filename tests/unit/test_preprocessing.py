"""test readers.py functions."""

import numpy as np
import pytest

from mobi_motion_tracking.preprocessing import preprocessing


def test_center_joints_to_hip_good() -> None:
    """Test that the center joints function extracts correct known values."""
    data = np.array([
        list(range(1, 11)),
        list(range(2, 12)),
        list(range(3, 13)),
    ])
    expected_output = np.array([
        [1, 0, 0, 0, 3, 3, 3, 6, 6, 6],
        [2, 0, 0, 0, 3, 3, 3, 6, 6, 6],
        [3, 0, 0, 0, 3, 3, 3, 6, 6, 6],
    ])

    normalized_data = preprocessing.center_joints_to_hip(data)

    assert isinstance(normalized_data, np.ndarray), "Output should be a NumPy array."
    assert np.array_equal(normalized_data, expected_output), (
        "Extracted data does not \
        match expected values."
    )


def test_get_average_length_good() -> None:
    """Test that the average length function calculates the expected value."""
    data = np.array([[1, 0, 0, 0, 1, 1, 1, 2, 2, 2], [2, 0, 0, 0, 1, 1, 1, 2, 2, 2]])
    segment_list = np.array([[0, 1], [0, 2], [1, 2]])
    expected_output = np.array([
        [1.7320508075688772],
        [3.4641016151377544],
        [1.7320508075688772],
    ])

    average_length = preprocessing.get_average_length(data, segment_list)

    assert np.array_equal(average_length, expected_output), (
        "Calculated data does not \
        match expected values."
    )
    assert average_length.shape == (3, 1), (
        f"Expected shape (3, 1), but got \
        {average_length.shape}"
    )
    assert isinstance(average_length, np.ndarray), "Output should be a NumPy array."


def test_get_average_length_empty_list() -> None:
    """Test the get_average_length function with an empty segment_list."""
    data = np.array([[1, 0, 0, 0, 1, 1, 1, 2, 2, 2], [2, 0, 0, 0, 1, 1, 1, 2, 2, 2]])
    segment_list = np.array([])
    with pytest.raises(ValueError, match="segment_list cannot be empty."):
        preprocessing.get_average_length(data, segment_list)


def test_get_average_length_out_of_range() -> None:
    """Test the get_average_length function with a joint index out of range."""
    data = np.array([[1, 0, 0, 0, 1, 1, 1, 2, 2, 2], [2, 0, 0, 0, 1, 1, 1, 2, 2, 2]])
    segment_list = np.array([[0, 1], [1, 3]])
    with pytest.raises(
        IndexError, match="Joint index in segment_list is out of range."
    ):
        preprocessing.get_average_length(data, segment_list)
