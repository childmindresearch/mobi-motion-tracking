"""Performs preprocessing steps for raw data."""

import numpy as np

from mobi_motion_tracking.preprocessing.joint_index_list import DEFAULT_JOINT_SEGMENTS


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
    centered_data: np.ndarray, segment_list: list = DEFAULT_JOINT_SEGMENTS
) -> np.ndarray:
    """Calculate the average lengths of all joint segments.

    This function calculates the average length across all frames of all connecting
    joint segments in the skeleton. The x, y,and z coordinates of specified starting
    and ending joints based on a user provided segment_list or default_joint_segments
    in JOINT_INDEX_LIST are used to calculate the average distance between the two
    joints for all frames.

    Args:
        centered_data: centered data output from center_joints_to_hip. The
            first column in centered data contains frame number, the following 60
            contain joint coordinates.
        segment_list: defaults to list from JOINT_INDEX_LIST.py containing all
            coordinate index pairs for all joint segments in skeleton.

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
            "Incorrect joint index list. Joint index in \
                         segment_list is out of range for data."
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


def normalize_segments(
    centered_data: np.ndarray,
    average_lengths: np.ndarray,
    segment_list: list = DEFAULT_JOINT_SEGMENTS,
) -> np.ndarray:
    """Normalize skeleton segments to maintain consistent bone lengths across frames.

    This function processes a motion sequence by adjusting the length of each skeleton
    segment to match the target lengths of the gold standard, while preserving the
    starting joint positions. The x, y, and z coordinates of specified starting
    and ending joints based on a user provided segment_list or default_joint_segments
    in JOINT_INDEX_LIST are used.

    Args:
        centered_data: centered data output from center_joints_to_hip. The
            first column in centered data contains frame number, the following 60
            contain joint coordinates.
        average_lengths: output from get_average_length. Array of shape
            (len(segment_list), 1) containing target lengths for each skeleton segment.
        segment_list: defaults to list from JOINT_INDEX_LIST.py containing all
            coordinate index pairs for all joint segments in skeleton.

    Returns:
        np.ndarray: Normalized motion data with consistent bone lengths, same shape as
            centered_data.

    Raises:
        ValueError: when length of segment_list and length of average_lengths do not
            match.
    """
    normalized_data = centered_data.copy()

    if len(segment_list) != len(average_lengths):
        raise ValueError(
            "Length of segment_list must match the \
                length of average_lengths."
        )

    for frame in range(centered_data.shape[0]):
        for i, segment in enumerate(segment_list):
            start_indices = np.array([segment[0][0], segment[1][0], segment[2][0]])
            end_indices = np.array([segment[0][1], segment[1][1], segment[2][1]])

            start_point = centered_data[frame, start_indices].T
            end_point = centered_data[frame, end_indices].T

            segment_vector = end_point - start_point

            scaled_segment_vector = (
                average_lengths[i] / np.linalg.norm(segment_vector)
            ) * segment_vector

            fixed_start_point = normalized_data[frame, start_indices].T

            new_end_point = fixed_start_point + scaled_segment_vector

            normalized_data[frame, end_indices] = new_end_point.T

    return normalized_data
