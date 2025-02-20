"""Performs preprocessing steps for raw data."""

import numpy as np

from mobi_motion_tracking.preprocessing.JOINT_INDEX_LIST import segments


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
    centered_data: np.ndarray, segment_list: list = segments
) -> np.ndarray:
    """Calculate the average lengths of all joint segments.

    This function calculates the average length across all frames of all connecting
    joint segments in the skeleton. The x, y,and z coordinates of specified starting
    and ending joints based on JOINT_INDEX_LIST are used to calculate the average
    distance between the two joints for all frames.

    Args:
        centered_data: centered data output from center_joints_to_hip. The
            first column in centered data contains frame number, the following 60
            contain joint coordinates.
        segment_list: defaults to list from JOINT_INDEX_LIST.py containing all
            coordinate index pairs for all joint segments in skeleton. Can be
            overwritten for testing purposes.

    Returns:
        ndarray [N,1], average distance between joints for all segments.

    Raises:
        IndexError: when a joint index in JOINT_INDEX_LIST is out of range of total
            number of joints.
    """
    num_segments = len(segment_list)
    num_joint_coordinates = centered_data.shape[1]
    all_distances = np.zeros((centered_data.shape[0], num_segments))

    if np.any(np.array(segment_list) >= num_joint_coordinates):
        raise IndexError(
            "Incorrect JOINT_INDEX_LIST.py. Joint index in \
                         JOINT_INDEX_LIST.segments is out of range for data."
        )

    for i, segment in enumerate(segment_list):
        start_indices = np.array([segment[0][0], segment[1][0], segment[2][0]])
        end_indices = np.array([segment[0][1], segment[1][1], segment[2][1]])

        start_points = centered_data[:, start_indices]
        end_points = centered_data[:, end_indices]

        distances = np.linalg.norm(start_points - end_points, axis=1)
        all_distances[:, i] = distances

    average_distances = all_distances.mean(axis=0, keepdims=True).T

    return average_distances
