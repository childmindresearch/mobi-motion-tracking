"""Functions to read motion tracking data from a file."""

import os
from pathlib import Path

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
            representing all frames and columns representing the frame number and 60
            joint coordinates.

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


def read_sheet(path: Path, sequence_sheetname: str) -> np.ndarray:
    """Read data from specific sheet.

    Currently motion tracking data for Kinect and Zed are saved into xlsx files.
    This function reads in the data from 1 sheet as a dataframe. Then passes the
    raw dataframe to data_cleaner and returns the output from data_cleaner.

    Args:
        path: Path to .xlsx file.
        sequence: str, determines which sequence is processed.

    Raises:
        FileNotFoundError: File path does not exist.
        ValueError: Invalid file extension.
        ValueError: Sheet name was not found.
    """
    if not os.path.exists(path):
        raise FileNotFoundError("File not found.")
    if ".xlsx" != path.suffix:
        raise ValueError("Invalid file extension. Expected '.xlsx'.")

    try:
        motion_tracking_data = pd.read_excel(
            path, sheet_name=sequence_sheetname, engine="openpyxl"
        )
    except ValueError:
        raise ValueError("Error: Sheet name does not exist.")

    return data_cleaner(motion_tracking_data)
