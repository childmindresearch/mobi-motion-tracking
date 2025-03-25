"""test orchestrator.py functions."""

import os
import pathlib

import pytest

from mobi_motion_tracking.core import orchestrator


def test_run_bad_input_path() -> None:
    """Tests the run function with an input path that is not a file nor a directory."""
    file_path = pathlib.Path("/tmp/fake_fifo")
    gold_path = pathlib.Path("tests/sample_data/Gold.xlsx")
    sequence = [1, 2, 3]
    os.mkfifo(file_path)

    with pytest.raises(TypeError, match="Input path is not a file nor a directory."):
        orchestrator.run(file_path, gold_path, sequence, "dtw")
