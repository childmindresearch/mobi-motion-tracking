"""test readers.py functions."""

import numpy as np
import pandas as pd
import pytest

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
        data = pd.DataFrame([
            list(range(1, 71)),
            list(range(71, 141)),
            ['a', 'b', 'c', 'd', 'e', 'x_Hip', 'g'] + list(range(8, 71)),
            list(range(141, 211)),
            list(range(211, 281)),
        ])
    else:
        data = pd.DataFrame([
            list(range(1, 71)),
            list(range(71, 141)),
            ['a', 'b', 'c', 'd', 'e', 'f', 'g'] + list(range(8, 71)),
            list(range(141, 211)),
            list(range(211, 281)),
        ])
    return data

def test_data_cleaner_invalid_dataframe() -> None:
    """Test the data_cleaner function with an invalid dataframe."""
    data = create_dummy_dataframe(valid=False)
    with pytest.raises(ValueError, match="x_Hip not found in DataFrame"):
        readers.data_cleaner(data)

def test_data_cleaner_correct_size() -> None:
    """Test that data_cleaner extracts the correct shape from valid dummy data."""
    data = create_dummy_dataframe()  # Generate the known dummy data
    cleaned_data = readers.data_cleaner(data)

    expected_rows = 2
    expected_cols = 61

    assert isinstance(cleaned_data, np.ndarray), "Output should be a NumPy array"
    assert cleaned_data.shape == (expected_rows, expected_cols), (
        f"Expected shape ({expected_rows}, {expected_cols}), \
            but got {cleaned_data.shape}")

def test_data_cleaner_correct_values() -> None:
    """Test that data_cleaner extracts the correct known values from dummy data."""
    data = create_dummy_dataframe()  # Generate the known dummy data
    cleaned_data = readers.data_cleaner(data)

    expected_output = np.array([
        [141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 
         156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 
         171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 
         186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 
         201, 202],
        
        [211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 
         226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 
         241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 
         256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 
         271, 272]
    ])

    assert np.array_equal(cleaned_data, expected_output), "Extracted data does not \
        match expected values"
