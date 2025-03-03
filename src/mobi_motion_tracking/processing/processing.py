"""Functions for calculating similarity metrics on preprocessed data."""

import numpy as np
from dtaidistance import dtw
from models import SimilarityMetrics


def dynamic_time_warping(
    preprocessed_target_data: np.ndarray, preprocessed_experimental_data: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Perform dynamic time warping.

    This function calculates the cumulative distance and warping paths between the
    target and experimental sequences. Both sequences have been centered to the
    hip and the experimental sequence has been normalized to the target lengths.
    Both input sequences have 61 columns. Column 0 contains frame number. Columns 1-3
    contain the x, y, and z coordinates of the hip. Since the data has already been
    centered to the hip, when calculating the distance and paths between the sequences,
    we index the data starting at the 4th column until the end. This function returns
    a dataclass which stores the DTW similarity metrics, distance and paths.


    Args:
        preprocessed_target_data: cleaned and centered target data.
        preprocessed_experimental_data: cleaned, centered, and normalized
            experimental data.

    Returns:
        SimilarityMetrics: a dataclass which stores the DTW similarity metrics.

    Raises:
        ValueError: if the input target data and experimetnal data are identical.
        ValueError: if path returns empty. dtaidistance.dtw.warping_paths() returns an
            empty path when input data is missing or empty.
    """
    distance, paths = dtw.warping_paths(
        preprocessed_target_data[:, 4:].flatten(),
        preprocessed_experimental_data[:, 4:].flatten(),
    )

    path = dtw.best_path(paths)

    if path is None or len(path) == 0:
        raise ValueError("DTW warping path is empty. Check input sequences.")

    return SimilarityMetrics.from_dtw(distance, path)
