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
    segment_list = [
        np.array([np.array([1, 4]), np.array([2, 5]), np.array([3, 6])]),
        np.array([np.array([4, 7]), np.array([5, 8]), np.array([6, 9])]),
    ]
    expected_output = np.array([
        [np.sqrt(3)],
        [np.sqrt(3)],
    ])

    average_length = preprocessing.get_average_length(data, segment_list)

    assert np.array_equal(average_length, expected_output), (
        f"Calculated data {average_length} does not match expected values \
            {expected_output}."
    )
    assert average_length.shape == (2, 1), (
        f"Expected shape (2, 1), but got \
        {average_length.shape}"
    )
    assert isinstance(average_length, np.ndarray), "Output should be a NumPy array."


def test_get_average_length_out_of_range() -> None:
    """Test the get_average_length function with a joint index out of range."""
    data = np.array([[1, 0, 0, 0, 1, 1, 1, 2, 2, 2], [2, 0, 0, 0, 1, 1, 1, 2, 2, 2]])
    segment_list = segment_list = [
        np.array([np.array([1, 4]), np.array([2, 5]), np.array([3, 6])]),
        np.array([np.array([4, 7]), np.array([5, 8]), np.array([6, 10])]),
    ]
    with pytest.raises(
        IndexError,
        match="Incorrect JOINT_INDEX_LIST.py. Joint index in \
                         JOINT_INDEX_LIST.segments is out of range for data.",
    ):
        preprocessing.get_average_length(data, segment_list)
