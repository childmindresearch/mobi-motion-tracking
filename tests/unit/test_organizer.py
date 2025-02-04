"""Test organizer.py functions."""

from pathlib import Path

import pytest

from mobi_motion_tracking.core import organizer


def test_get_metadata_incorrect_filename() -> None:
    """Test get_metadata with an incorrect filename."""
    with pytest.raises(ValueError, match="The participant file is named incorrectly."):
        organizer.get_metadata(Path("/dummy/path/100_01.xlsx"), 1)


def test_get_metadata_good() -> None:
    """Test get_metadata with an incorrect filename."""
    expected_ID = "100"
    expected_seq = "seq1"

    participant_ID, sequence = organizer.get_metadata(Path("/dummy/path/100.xlsx"), 1)

    assert isinstance(participant_ID, str), "participant_ID should be a string."
    assert isinstance(sequence, str), "sequence should be a string."
    assert expected_ID == participant_ID, "extracted ID does not match expected value."
    assert expected_seq == sequence, "extracted sequence does not match expected value."
