"""test readers.py functions."""

import numpy as np

from mobi_motion_tracking.preprocessing import preprocessing


def test_center_joints_to_hip_good() -> None:
    """Test that the center joints function extracts correct known values."""
    data = np.array(
        [
            list(range(1, 11)),
            list(range(2, 12)),
            list(range(3, 13)),
        ]
    )
    expected_output = np.array(
        [
            [1, 0, 0, 0, 3, 3, 3, 6, 6, 6],
            [2, 0, 0, 0, 3, 3, 3, 6, 6, 6],
            [3, 0, 0, 0, 3, 3, 3, 6, 6, 6],
        ]
    )

    normalized_data = preprocessing.center_joints_to_hip(data)

    assert isinstance(normalized_data, np.ndarray), "Output should be a NumPy array."
    assert np.array_equal(normalized_data, expected_output), "Extracted data does not \
        match expected values."
