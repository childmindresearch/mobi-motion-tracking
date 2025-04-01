"""Functions to read motion tracking data from a file."""

import pathlib

import numpy as np
import pandas as pd

from mobi_motion_tracking.core import models


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

    cleaned_data = (
        data.iloc[
            row + 1 :,
            start_col:end_col,
        ]
        .to_numpy()
        .astype(np.float64)
    )

    return cleaned_data


def read_participant_data(
    subject_path: pathlib.Path, sequence: int
) -> models.ParticipantData:
    """Calls get_metadata and read sheet.

    This function calls get_metadata to extract the participant_ID value and the
    sequence_sheetname. read_sheet is then called to create the subject_data array.
    The participant_ID, sequence_sheetname, and subject_data are passed to the
    ParticipantData class and returned.

    Args:
        subject_path: file path to the participant file.
        sequence: integer value indiciating the sequence currently being tested.

    Returns:
        models.ParticipantData: containing participant_ID (str), sheetname (str),
            and data (np.ndarray).
    """
    participant_ID = subject_path.stem
    sequence_sheetname = f"seq{sequence}"

    try:
        motion_tracking_data = pd.read_excel(
            subject_path, sheet_name=sequence_sheetname, engine="openpyxl"
        )
    except ValueError as ve:
        print(f"Sheet doesn't exist: {ve}")
        return models.ParticipantData(
            participant_ID=participant_ID,
            sequence_sheetname=sequence_sheetname,
            data=np.array([]),
        )

    subject_data = data_cleaner(motion_tracking_data)

    return models.ParticipantData(
        participant_ID=participant_ID,
        sequence_sheetname=sequence_sheetname,
        data=subject_data,
    )
