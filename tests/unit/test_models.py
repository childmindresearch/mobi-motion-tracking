"""Test models.py functions."""

import numpy as np
import pytest

from mobi_motion_tracking.core import models


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
