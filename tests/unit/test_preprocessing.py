"""test readers.py functions."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mobi_motion_tracking.preprocessing import preprocessing


def create_dummy_ndarray() -> np.ndarray:
    """Create a known 3x9 dummy DataFrame for testing.

    Returns:
            data: np.ndarray, A 3x9 array with numeric values.
    """
    data = pd.DataFrame([
        list(range(1, 9)),
        list(range(10, 18)),
        list(range(19, 27)),
    ])
    return data


def test_normalize_joints_good() -> np.ndarray:
    """Test that the normalize joints function extracts correct known values."""
    data = np.array([
        list(range(1, 11)),
        list(range(1, 11)),
        list(range(1, 11)),
    ])
    expected_output = np.array([
        [1, 0, 0, 0, 3, 3, 3, 6, 6, 6],
        [1, 0, 0, 0, 3, 3, 3, 6, 6, 6],
        [1, 0, 0, 0, 3, 3, 3, 6, 6, 6],
    ])

    normalized_data = preprocessing.normalize_all_joints_to_hip(data)

    assert isinstance(normalized_data, np.ndarray), "Output should be a NumPy array."
    assert np.array_equal(normalized_data, expected_output), (
        "Extracted data does not \
        match expected values."
    )
