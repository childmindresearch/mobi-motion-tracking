"""Test processing.py functions."""

import numpy as np

from mobi_motion_tracking.processing import processing


def test_dtw_good() -> None:
    """Test that the dynamic time warping funciton extracts correct known values."""
    preprocessed_target_data = np.array([
        [0, 0, 0, 0, 0, 2, 4],
    ])

    preprocessed_experimental_data = np.array([
        [0, 0, 0, 0, 1, 3, 5],
    ])

    expected_target_path = np.array([0, 1, 2])
    expected_experimental_path = np.array([0, 1, 2])
    expected_distance = np.sqrt(3)

    output = processing.dynamic_time_warping(
        preprocessed_target_data, preprocessed_experimental_data
    )

    assert output.metrics["distance"] == expected_distance, (
        f"Calculated distance {output.distance} does not \
        match expected output {expected_distance}."
    )
    assert np.array_equal(output.metrics["target_path"], expected_target_path), (
        f"Calculated target path {output.metrics['target_path']} does not match \
            expected output {expected_target_path}."
    )
    assert np.array_equal(
        output.metrics["experimental_path"], expected_experimental_path
    ), (
        f"Calculated experimental path {output.metrics['experimental_path']} \
            does not match expected output {expected_experimental_path}."
    )
    assert isinstance(output.metrics["distance"], float), (
        "Ouput distance should be a float."
    )
    assert isinstance(output.metrics["target_path"], np.ndarray), (
        "Output target path should be a NumPy \
        array."
    )
    assert isinstance(output.metrics["experimental_path"], np.ndarray), (
        "Output experimental path should \
        be a NumPy array."
    )


def test_dtw_identical_sequences() -> None:
    """Test the dtw function when target and experimental sequences are equal."""
    preprocessed_target_data = np.array([
        [0, 0, 0, 0, 1, 2, 3],
    ])

    preprocessed_experimental_data = np.array([
        [0, 0, 0, 0, 1, 2, 3],
    ])

    expected_target_path = np.array([0, 1, 2])
    expected_experimental_path = np.array([0, 1, 2])
    expected_distance = 0.0

    output = processing.dynamic_time_warping(
        preprocessed_target_data, preprocessed_experimental_data
    )

    assert output.metrics["distance"] == expected_distance, (
        f"Calculated distance {output.distance} does not \
        match expected output {expected_distance}."
    )
    assert np.array_equal(output.metrics["target_path"], expected_target_path), (
        f"Calculated target path {output.metrics['target_path']} does not match \
            expected output {expected_target_path}."
    )
    assert np.array_equal(
        output.metrics["experimental_path"], expected_experimental_path
    ), (
        f"Calculated experimental path {output.metrics['experimental_path']} \
            does not match expected output {expected_experimental_path}."
    )
    assert isinstance(output.metrics["distance"], float), (
        "Ouput distance should be a float."
    )
    assert isinstance(output.metrics["target_path"], np.ndarray), (
        "Output target path should be a NumPy \
        array."
    )
    assert isinstance(output.metrics["experimental_path"], np.ndarray), (
        "Output experimental path should \
        be a NumPy array."
    )
