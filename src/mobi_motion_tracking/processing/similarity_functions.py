"""Functions for calculating similarity metrics on preprocessed data."""

from typing import Optional

import numpy as np

from mobi_motion_tracking.core import models


def dynamic_time_warping(
    preprocessed_target_data: np.ndarray,
    preprocessed_subject_data: np.ndarray,
    window_size: Optional[int] = None,
) -> models.SimilarityMetrics:
    """Perform dynamic time warping.

    This function calculates the cumulative distance and optimal warping paths between
    the subject and target sequences. Both sequences have been centered to the
    hip and the experimental sequence has been normalized to the target lengths.
    Both input sequences have 61 columns. Column 0 contains frame number. Columns 1-3
    contain the x, y, and z coordinates of the hip. Since the data has already been
    centered to the hip, when calculating the distance and paths between the sequences,
    we index the data starting at the 4th column until the end. This function returns
    a dataclass which stores the DTW similarity metrics, distance and paths.

    Args:
        preprocessed_target_data: cleaned and centered target data.
        preprocessed_subject_data: cleaned, centered, and normalized subject data.
        window_size: constraint for matching points, ensuring
            |num_frames_subject - num_frames_target| <= window_size. If None, the
            window size is set to the maximum amount of rows of the two sequences.

    Returns:
        SimilarityMetrics: a dataclass which stores the DTW similarity metrics.

    Raises:
        ValueError: when dimensions of the two inputs do not match.
    """
    preprocessed_subject_data = preprocessed_subject_data[:, 4:]
    preprocessed_target_data = preprocessed_target_data[:, 4:]

    num_frames_subject, num_joints_subject = preprocessed_subject_data.shape
    num_frames_target, num_joints_target = preprocessed_target_data.shape

    if num_joints_subject != num_joints_target:
        raise ValueError(
            "Error in dtw(): the dimensions of the two input signals do not match."
        )

    if window_size is None:
        window_size = max(num_frames_subject, num_frames_target)
    else:
        window_size = max(window_size, abs(num_frames_subject - num_frames_target))

    cost_matrix = np.full((num_frames_subject + 1, num_frames_target + 1), float("inf"))
    cost_matrix[0, 0] = 0

    for row in range(1, num_frames_subject + 1):
        for column in range(
            max(1, row - window_size), min(num_frames_target + 1, row + window_size + 1)
        ):
            cost_value = np.linalg.norm(
                preprocessed_subject_data[row - 1]
                - preprocessed_target_data[column - 1]
            )
            cost_matrix[row, column] = cost_value + min(
                cost_matrix[row - 1, column],
                cost_matrix[row, column - 1],
                cost_matrix[row - 1, column - 1],
            )

    distance = cost_matrix[num_frames_subject, num_frames_target]

    subject_idx, target_idx = num_frames_subject, num_frames_target
    path = [(subject_idx, target_idx)]

    while subject_idx > 0 or target_idx > 0:
        if subject_idx == 0:
            target_idx -= 1
        elif target_idx == 0:
            subject_idx -= 1
        else:
            min_cost_index = np.argmin(
                [
                    cost_matrix[subject_idx - 1, target_idx],
                    cost_matrix[subject_idx, target_idx - 1],
                    cost_matrix[subject_idx - 1, target_idx - 1],
                ]
            )
            if min_cost_index == 0:
                subject_idx -= 1
            elif min_cost_index == 1:
                target_idx -= 1
            else:
                subject_idx -= 1
                target_idx -= 1
        path.append((subject_idx, target_idx))

    path.reverse()

    return models.SimilarityMetrics.from_dtw(distance=distance, warping_path=path)
