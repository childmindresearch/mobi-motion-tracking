"""Data organizer for mobi-motion-tracking."""

import os
from pathlib import Path


def get_metadata(subject_path: Path, sequence: int) -> tuple[str, str]:
    """Strip path name for participant ID and create sequence sheet name.

    This function strips the basename without the file extension per
    participant to extract each participant ID and saves the sequence (int)
    as a string with the preface 'seq' for the sheetname.

    Args:
        subject_path: Path, full filepath per participant.
        sequence: int, sequence number.

    Returns:
        participant_ID: str, basename from filepath and participant ID number.
        sequence: str, 'seq'+ sequence number.

    Raises:
        ValueError: subject_file named incorrectly.
    """
    participant_ID = os.path.splitext(os.path.basename(subject_path))[0]

    if not participant_ID.isdigit():
        raise ValueError("The participant file is named incorrectly.")

    sequence_str = f"seq{sequence}"

    return participant_ID, sequence_str
