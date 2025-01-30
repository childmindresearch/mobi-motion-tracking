"""test readers.py functions."""


import pandas as pd
import pytest

from mobi_motion_tracking.io.readers import readers


def test_data_cleaner_invalid_dataframe(sample_data_participant: pd.DataFrame) -> None:
    """Test the data_cleaner function with an invalid dataframe."""
    with pytest.raises(ValueError, match="x_Hip not found in DataFrame"):
        readers.data_cleaner(sample_data_participant)

# check it grabs numpy array of right size

# check it grabs correct known values
