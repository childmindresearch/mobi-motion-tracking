"""Test organizer.py functions."""

import pathlib

import pytest

from mobi_motion_tracking.core import models


def test_get_metadata_incorrect_filename() -> None:
    """Test get_metadata with an incorrect filename."""
    with pytest.raises(ValueError, match="The participant file is named incorrectly."):
        models.Metadata.get_metadata(pathlib.Path("/dummy/path/100_01.xlsx"), 1)


def test_get_metadata_good() -> None:
    """Test get_metadata works."""
    expected_ID = "100"
    expected_seq = "seq1"

    metadata = models.Metadata.get_metadata(pathlib.Path("/dummy/path/100.xlsx"), 1)

    assert isinstance(
        metadata.participant_ID, str
    ), "participant_ID should be a string."
    assert isinstance(metadata.sequence_sheetname, str), "sequence should be a string."
    assert (
        expected_ID == metadata.participant_ID
    ), "extracted ID does not match expected value."
    assert (
        expected_seq == metadata.sequence_sheetname
    ), "extracted sequence does not match expected value."
