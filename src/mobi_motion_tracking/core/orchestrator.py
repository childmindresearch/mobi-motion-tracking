"""Python based runner."""

import pathlib

from mobi_motion_tracking.core import models
from mobi_motion_tracking.io.readers import readers


def read_subject(path: pathlib.Path, sequence: int | list[int]) -> None:
    """Runs get_metadata and read_sheet.

    This function checks if the input sequence is a list of integers or a single
    integer. If sequence is a list, it iterates over each integer in the list and runs
    get_metadata and read_sheet for each sequence for each subject.

    Args:
        path: File path for a single subject.
        sequence: Integer or integers representing each sequence for each subject.
    """
    if sequence is list[int]:
        for seq in sequence:
            subject_metadata = models.Metadata.get_metadata(path, seq)
            readers.read_sheet(path, subject_metadata.sequence_sheetname)
    elif sequence is int:
        subject_metadata = models.Metadata.get_metadata(path, seq)
        readers.read_sheet(path, subject_metadata.sequence_sheetname)


def read_path(path: pathlib.Path, sequence: int | list[int]) -> None:
    """Runs read_subject for every file.

    This function checks if the input path is a directory of subject files, or a single
    subject file. For each subject, read_subject is called.

    Args:
        path: Directory of subject files or single subject file.
        sequence: Integer or integers representing each sequence for each subject.
    """
    if path.is_dir():
        for filepath in path.iterdir():
            if filepath.is_file() and filepath.suffix == ".xlsx":
                read_subject(filepath, sequence)
    elif path.is_file() and filepath.suffix == ".xlsx":
        read_subject(path, sequence)
