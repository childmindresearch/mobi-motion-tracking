"""Performs preprocessing steps for raw data."""

import numpy as np


def normalize_all_joints_to_hip(data: np.ndarray) -> np.ndarray:
    """Normalize all joints to the central hip.

    This function normalizes all joints in a skeleton to the central hip joint.
    The x,y,and z coordinates of the hip will be subtracted from the
    x, y,and z coordinates of all joints for every frame.

    Args:
        data: ndarray, cleaned raw data.

    Returns:
        data: ndarray, data normalized to the hip.
    """
    x_pelvis = data[:, 1]
    y_pelvis = data[:, 2]
    z_pelvis = data[:, 3]

    data[:, 1::3] -= x_pelvis[:, np.newaxis]
    data[:, 2::3] -= y_pelvis[:, np.newaxis]
    data[:, 3::3] -= z_pelvis[:, np.newaxis]

    return data
