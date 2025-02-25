"""Functions for calculating similarity metrics on preprocessed data."""

import numpy as np
from dtaidistance import dtw


def dynamic_time_warping(
    preprocessed_target_data: np.ndarray, preprocessed_experimental_data: np.ndarray
) -> float | np.ndarray | np.ndarray:
    """Perform dynamic time warping.

    This function calculates the cumulative distance and warping paths between the
    target and experimental sequences. Both sequences have been cenetered to the
    hip and the experimental sequence data has been normalized to the target lengths.

    Args:
        preprocessed_target_data: cleaned and centered target data.
        preprocessed_experimental_data: cleaned, centered, and normalized
            experimental data.

    Returns:
        distance: float, cumulative distance between experimental and target.
        target_path: np.ndarray, warping path for preprocessed target data.
        experimental_path: np.ndarray, warping path for preprocessed experimental data.
    """
    distance, paths = dtw.warping_paths(
        preprocessed_target_data[:, 4:].flatten(),
        preprocessed_experimental_data[:, 4:].flatten(),
    )

    path = dtw.best_path(paths)

    target_path = np.array([p[0] for p in path])
    experimental_path = np.array([p[1] for p in path])

    return distance, target_path, experimental_path
