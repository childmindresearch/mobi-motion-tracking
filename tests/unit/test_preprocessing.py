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


def test_get_average_length_good_with_dummy_data() -> None:
    """Test that the average length function calculates the expected value."""
    data = np.array([
        [1.0, 0, 0, 0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        [2.0, 0, 0, 0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
    ])
    segment_list = [
        np.array([np.array([1, 4]), np.array([2, 5]), np.array([3, 6])]),
        np.array([np.array([4, 7]), np.array([5, 8]), np.array([6, 9])]),
    ]
    expected_output = np.array([np.sqrt([3]), np.sqrt([3])])

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


def test_get_average_length_good_with_default() -> None:
    """Test that the average length function calculates the expected value."""
    data = np.ones((2, 61))
    expected_output = np.zeros((19, 1))

    average_length = preprocessing.get_average_length(data)

    assert np.array_equal(average_length, expected_output), (
        f"Calculated data {average_length} does not match expected values \
            {expected_output}."
    )
    assert average_length.shape == (19, 1), (
        f"Expected shape (19, 1), but got \
        {average_length.shape}"
    )
    assert isinstance(average_length, np.ndarray), "Output should be a NumPy array."


def test_get_average_length_out_of_range() -> None:
    """Test the get_average_length function with a joint index out of range."""
    data = np.array([
        [1.0, 0, 0, 0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        [2.0, 0, 0, 0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
    ])
    segment_list = segment_list = [
        np.array([np.array([1, 4]), np.array([2, 5]), np.array([3, 6])]),
        np.array([np.array([4, 7]), np.array([5, 8]), np.array([6, 10])]),
    ]
    with pytest.raises(
        IndexError,
        match="Incorrect joint index list. Joint index in \
                         segment_list is out of range for data.",
    ):
        preprocessing.get_average_length(data, segment_list)


def test_normalize_segments_good_with_dummy_data() -> None:
    """Test that the normalize segments function calculates the expected value."""
    data = np.array([
        [1.0, 0, 0, 0, 1.0, 0, 0, 2.0, 0, 0],
        [2.0, 0, 0, 0, 2.0, 0, 0, 4.0, 0, 0],
    ])
    segment_list = [
        np.array([np.array([1, 4]), np.array([2, 5]), np.array([3, 6])]),
        np.array([np.array([4, 7]), np.array([5, 8]), np.array([6, 9])]),
    ]
    average_lengths = np.array([1.5, 2.0])
    expected_output = np.array([
        [1, 0, 0, 0, 1.5, 0, 0, 3.5, 0, 0],
        [2, 0, 0, 0, 1.5, 0, 0, 3.5, 0, 0],
    ])

    normalized_data = preprocessing.normalize_segments(
        data, average_lengths, segment_list
    )

    assert np.array_equal(normalized_data, expected_output), (
        f"Calculated data {normalized_data} does not match expected values \
            {expected_output}."
    )
    assert normalized_data.shape == (2, 10), (
        f"Expected shape (2, 10), but got \
        {normalized_data.shape}"
    )
    assert isinstance(normalized_data, np.ndarray), "Output should be a NumPy array."


def test_normalize_segments_good_with_default() -> None:
    """Test that the normalize segments function calculates the expected value."""
    data = np.zeros((1, 61), dtype=float)
    data[:, 4:] = np.repeat(np.arange(1, (61 - 4) // 3 + 1), 3)
    average_lengths = np.ones((19, 1)) * np.sqrt(3)
    expected_output = np.concatenate(
        [
            [0],
            np.repeat([0, 1, 2, 3], 3),
            np.tile(np.repeat([4, 5], 3), 2),
            np.repeat([6, 7], 3),
            np.repeat([4, 5, 6, 7], 3),
            np.tile(np.repeat([1, 2, 3], 3), 2),
        ],
        dtype=float,
    )
    expected_output = expected_output.reshape((1, 61))

    normalized_data = preprocessing.normalize_segments(data, average_lengths)

    assert np.array_equal(normalized_data, expected_output), (
        f"Calculated data {normalized_data} does not match expected values \
            {expected_output}."
    )
    assert normalized_data.shape == (1, 61), (
        f"Expected shape (1, 61), but got \
        {normalized_data.shape}"
    )
    assert isinstance(normalized_data, np.ndarray), "Output should be a NumPy array."


def test_normalize_segments_mismatch_shape_error() -> None:
    """Test the normalize segments function with a mismatch in lengths of inputs."""
    data = np.zeros((1, 61), dtype=float)
    average_lengths = np.ones((10, 1))
    with pytest.raises(
        ValueError, match="Mismatch in shape for segment_list and average_lengths."
    ):
        preprocessing.normalize_segments(data, average_lengths)


def test_normalize_segments_dimension_error() -> None:
    """Test the normalize segments function with incorrect data dimensions."""
    data = np.zeros((1, 6), dtype=float)
    average_lengths = np.ones((19, 1))
    with pytest.raises(
        ValueError,
        match="The shape of centered_data does not match the expected dimensions.",
    ):
        preprocessing.normalize_segments(data, average_lengths)
