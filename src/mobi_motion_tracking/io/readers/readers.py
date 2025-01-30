"""Functions to read motion tracking data from a file."""

import numpy as np
import pandas as pd


def data_cleaner(data: pd.DataFrame) -> np.ndarray:
    """Select applicable data from a dataframe.
    
    Currently motion tracking data for Kinect and Zed are saved into xlsx files.
    Searches through dataframe extracted from the read_sheet function for 
    x_hip, then extracts all rows below and the neighboring 61 columns.
    
    Args:
        data: dataframe from read_sheet.
    
    Returns:
        cleaned_data: np.array of [x,61] where x is the total number of rows 
            representing all frames and columns representing the frames and 60 joint 
            coordinates.
        
    Raises:
        ValueError: when x_Hip is not found in dataframe.
    """
    value_to_search = 'x_Hip'
    result = data.where(data == value_to_search).stack().index
    
    if result.empty:
        raise ValueError("x_Hip not found in DataFrame.")
    
    x_idx = result[0][0]
    y_idx = result[0][1]
    cleaned_data = data.iloc[
        x_idx+1:, data.columns.get_loc(y_idx)-1:data.columns.get_loc(y_idx)+60
        ].to_numpy()
    
    return cleaned_data

