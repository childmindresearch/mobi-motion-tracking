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
    participant_ID = os.path.splitext(os.path.basename("/path/to/file.txt"))[0]
    try:
        int(participant_ID)
    except ValueError:
        raise ValueError(
            "The participant file is named incorrectly. Make sure the \
                         filename is the participant_ID."
        )

    sequence = f"seq{sequence}"

    return participant_ID, sequence
