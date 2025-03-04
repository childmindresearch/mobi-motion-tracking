"""Functions for calculating similarity metrics on preprocessed data."""

import numpy as np
from dtaidistance import dtw

14-task-write-get_metadata-function
from mobi_motion_tracking.core import models


def dynamic_time_warping(
    preprocessed_target_data: np.ndarray, preprocessed_experimental_data: np.ndarray
) -> models.SimilarityMetrics:
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
    """
    distance, paths = dtw.warping_paths(
        preprocessed_target_data[:, 4:].flatten(),
        preprocessed_experimental_data[:, 4:].flatten(),
    )
    warping_path = dtw.best_path(paths)

    return models.SimilarityMetrics.from_dtw(
        distance=distance, warping_path=warping_path
    )
