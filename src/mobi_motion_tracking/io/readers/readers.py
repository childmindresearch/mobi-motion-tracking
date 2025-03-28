"""Functions to read motion tracking data from a file."""

import pathlib
import os
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


def read_sheet(path: pathlib.Path, sequence_sheetname: str) -> np.ndarray:
    """Read data from specific sheet.

    Currently motion tracking data for Kinect and Zed are saved into xlsx files.
    This function reads in the data from 1 sheet as a dataframe. Then passes the
    raw dataframe to data_cleaner and returns the output from data_cleaner.

    Args:
        path: Path to .xlsx file.
        sequence_sheetname: str, determines which sequence is processed.

    Returns:
        np.ndarray: Data is passed to data_cleaner which returns an np.ndarray, or an
            empty array is returned if the sheet name does not exist.
    """
    try:
        motion_tracking_data = pd.read_excel(
            path, sheet_name=sequence_sheetname, engine="openpyxl"
        )
    except ValueError:
        print(
            f"Skipping sheet {sequence_sheetname} in {path}: Sheet name does not exist."
        )
        return np.array([])

    return data_cleaner(motion_tracking_data)


def get_metadata(subject_path: pathlib.Path, sequence: int) -> tuple[str, str]:
    """Strip path name for participant ID and create sequence sheet name.

    This function strips the basename without the file extension per
    participant to extract each participant ID (int or "gold") and saves the
    sequence (int) as a string with the preface 'seq' for the sheet name.

    Args:
        subject_path: Path, full filepath per participant.
        sequence: int, sequence number.

    Returns:
        participant_ID: basename of file.
        sequence_str: sheetname in file indicating sequence.
    """
    try:
        if not os.path.exists(subject_path):
            raise FileNotFoundError("File not found.")
        if ".xlsx" != subject_path.suffix:
            raise ValueError(
                f"Invalid file extension: {subject_path}. Expected '.xlsx'."
            )

        participant_ID = subject_path.stem

    except FileNotFoundError as fnf_error:
        print(f"Skipping {subject_path}: {fnf_error}")
        return "None", "None"
    except ValueError as ve:
        print(f"Skipping file {subject_path}: {ve} (Wrong file type)")
        return "None", "None"

    try:
        if not (participant_ID.isdigit() or "gold" in participant_ID.lower()):
            raise ValueError("The input file is named incorrectly.")

        sequence_str = f"seq{sequence}"

    except ValueError as err:
        print(f"Skipping file {subject_path}: {err}")
        return "None", "None"

    return participant_ID, sequence_str


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
    participant_ID, sequence_sheetname = get_metadata(subject_path, sequence)

    subject_data = read_sheet(subject_path, sequence_sheetname)

    return models.ParticipantData(
        participant_ID=participant_ID,
        sequence_sheetname=sequence_sheetname,
        data=subject_data,
    )
