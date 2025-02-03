"""Functions to read motion tracking data from a file."""

import numpy as np
import pandas as pd


def data_cleaner(data: pd.DataFrame) -> np.ndarray:
    """Select applicable data from a dataframe.

    Currently motion tracking data for Kinect and Zed are saved into xlsx files.
    This function searches through dataframe extracted from the read_sheet function for
    x_hip, then extracts all rows below and the neighboring 61 columns.

    Args:
        data: dataframe from read_sheet.

    Returns:
        cleaned_data: np.array of [x,61] where x is the total number of rows
            representing all frames and columns representing the frame number and 60 joint
            coordinates.

    Raises:
        ValueError: when x_Hip is not found in dataframe.
        IndexError: when column index is out of range.
    """
    result = data.where(data == "x_Hip").stack().index

    if result.empty:
        raise ValueError("x_Hip not found in DataFrame.")

    row = result[0][0]
    col_idx = result[0][1]

    start_col = data.columns.get_loc(col_idx) - 1
    end_col = data.columns.get_loc(col_idx) + 60

    if start_col < 0 or end_col > data.shape[1]:
        raise IndexError("Column index out of range.")

    cleaned_data = data.iloc[
        row + 1 :,
        start_col:end_col,
    ].to_numpy()

    return cleaned_data
