"""Performs preprocessing steps for raw data."""

import numpy as np


def center_joints_to_hip(data: np.ndarray) -> np.ndarray:
    """Center all joints to the hip as origin.

    This function sets the coordinates of the hip (x,y,z) as a new
    relative origin (0,0,0) for each frame. The x,y,and z coordinates
    of the hip will be subtracted from the x, y, and z coordinates of
    all joints for every frame.

    Args:
        data: ndarray, cleaned raw data.

    Returns:
        normalized_data: ndarray, data normalized to the hip.
    """
    normalized_data = data.copy()
    x_pelvis = data[:, 1]
    y_pelvis = data[:, 2]
    z_pelvis = data[:, 3]

    normalized_data[:, 1::3] -= x_pelvis[:, np.newaxis]
    normalized_data[:, 2::3] -= y_pelvis[:, np.newaxis]
    normalized_data[:, 3::3] -= z_pelvis[:, np.newaxis]

    return normalized_data


def get_average_length(
    centered_data: np.ndarray, segment_list: np.ndarray
) -> np.ndarray:
    """Calculate the average lengths of all joint segments.

    This function calculates the average length across all frames of all connecting
    joint segments in the skeleton. The x, y,and z coordinates of specified starting
    and ending joints based on segment_list are used to calculate the average distance
    between the two joints for all frames.

    Args:
        centered_data: ndarray, centered data output from center_joints_to_hip. The
            first column in centered data contains frame number, the following 60
            contain joint coordinates.
        segment_list: ndarray [X,2], containing starting and ending joint pairs for all
            skeleton segments. This array should be 0-indexed.

    Returns:
        ndarray [X,1], average distance between joints for all segments.

    Raises:
        ValueError: when segment_list is empty.
        IndexError: when a joint index in segment_list is out of range of total
            number of joints.
    """
    if segment_list.size == 0:
        raise ValueError("segment_list cannot be empty.")

    num_joints = (centered_data.shape[1] - 1) // 3

    if np.any(segment_list < 0) or np.any(segment_list >= num_joints):
        raise IndexError("Joint index in segment_list is out of range.")

    starting_joint = 3 * segment_list[:, 0]
    ending_joint = 3 * segment_list[:, 1]

    starting_points = centered_data[:, starting_joint[:, None] + np.arange(1, 4)]
    ending_points = centered_data[:, ending_joint[:, None] + np.arange(1, 4)]

    distances = np.linalg.norm(starting_points - ending_points, axis=2)

    return distances.mean(axis=0, keepdims=True).T
