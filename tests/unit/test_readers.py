"""test readers.py functions."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from mobi_motion_tracking.io.readers import readers


def create_dummy_dataframe(valid: bool = True) -> pd.DataFrame:
    """Create a known 5x70 dummy DataFrame for testing.

    Args:
        valid: When valid is True (default), a valid dataframe containing
        'x_Hip' at [2,5] will be generated. When valid is False, an invalid dataframe
        not containing 'x_Hip' will be generated.

    Returns:
        pd.DataFrame: A 5x70 DataFrame with numeric values and either a known 'x_Hip'
        placement or a missing 'x_Hip' for testing.
    """
    if valid:
        data = pd.DataFrame(
            [
                list(range(1, 71)),
                ["a", "b", "c", "d", "e", "x_Hip", "g"] + list(range(8, 71)),
                list(range(71, 141)),
            ]
        )
    else:
        data = pd.DataFrame(
            [
                list(range(1, 71)),
                ["a", "b", "c", "d", "e", "f", "g"] + list(range(8, 71)),
                list(range(71, 141)),
            ]
        )
    return data


def test_data_cleaner_invalid_dataframe() -> None:
    """Test the data_cleaner function with an invalid dataframe."""
    data = create_dummy_dataframe(valid=False)
    with pytest.raises(ValueError, match="x_Hip not found in DataFrame."):
        readers.data_cleaner(data)


def test_data_cleaner_good() -> None:
    """Test that data_cleaner extracts the correct data from valid dummy data."""
    data = create_dummy_dataframe()
    expected_rows = 1
    expected_cols = 61
    expected_output = [np.array(range(75, 136))]

    cleaned_data = readers.data_cleaner(data)

    assert isinstance(cleaned_data, np.ndarray), "Output should be a NumPy array."
    assert cleaned_data.shape == (
        expected_rows,
        expected_cols,
    ), f"Expected shape ({expected_rows}, {expected_cols}), \
            but got {cleaned_data.shape}"
    assert np.array_equal(cleaned_data, expected_output), "Extracted data does not \
        match expected values."


def test_data_cleaner_index_error() -> None:
    """Test the data_cleaner function with an incorrect index length."""
    data = create_dummy_dataframe()
    data = data.iloc[:, :10]
    with pytest.raises(IndexError, match="Column index out of range."):
        readers.data_cleaner(data)


def test_read_sheet_file_not_found() -> None:
    """Test FileNotFoundError when file does not exist."""
    with pytest.raises(FileNotFoundError, match="File not found."):
        readers.read_sheet(Path("non_existent_file.xlsx"), "seq1")


def test_read_sheet_invalid_file_extension() -> None:
    """Test ValueError when file is not .xlsx."""
    with pytest.raises(ValueError, match="Invalid file extension. Expected '.xlsx'."):
        readers.read_sheet(Path("tests/sample_data/csv_file.csv"), "seq1")


def test_read_sheet_invalid_sheet_name() -> None:
    """Test ValueError when sheet name does not exist."""
    with pytest.raises(ValueError, match="Error: Sheet name may not exist."):
        readers.read_sheet(Path("tests/sample_data/valid_file.xlsx"), "InvalidSheet")
