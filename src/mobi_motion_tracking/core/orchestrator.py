"""Python based runner."""

import pathlib

from mobi_motion_tracking.core import models
from mobi_motion_tracking.io.readers import readers
from mobi_motion_tracking.io.writers import writers
from mobi_motion_tracking.preprocessing import preprocessing
from mobi_motion_tracking.processing import similarity_functions


def run(
    experimental_path: pathlib.Path,
    gold_path: pathlib.Path,
    sequence: list[int],
    algorithm: str,
) -> None:
    """Checks if experimental path is a directory or file, calls run_file.

    This function determines whether the experimental path is a directory or a single
    file and processes each subject's data accordingly by calling `run_file`.

    Args:
        experimental_path (pathlib.Path): Path to the subject's motion tracking data
            file or directory.
        gold_path (pathlib.Path): Path to the gold-standard motion tracking data file.
        sequence (list[int]): List of sequence numbers to process.
        algorithm (str): Name of the algorithm to use for similarity computation.

    Raises:
        TypeError: If `experimental_path` is not a file or directory.
    """
    if experimental_path.is_dir():
        for file in experimental_path.iterdir():
            run_file(file, gold_path, experimental_path, sequence, algorithm)
    elif experimental_path.is_file():
        run_file(
            experimental_path, gold_path, experimental_path.parent, sequence, algorithm
        )
    else:
        raise TypeError(
            f"Unsupported type: \
                {type(sequence).__name__}. Expected list or int."
        )


def run_file(
    file_path: pathlib.Path,
    gold_path: pathlib.Path,
    output_path: pathlib.Path,
    sequence: list[int],
    algorithm: str,
) -> None:
    """Performs main processing steps for a subject, per sequence.

    This function reads motion tracking data from the specified subject and gold-
    standard files, applies preprocessing steps, computes similarity metrics using the
    specified algorithm, and saves the results.

    Args:
        file_path (pathlib.Path): Path to the subject's motion tracking data file.
        gold_path (pathlib.Path): Path to the gold-standard motion tracking data file.
        output_path (pathlib.Path): Directory where similarity results should be saved.
        sequence (list[int]): List of sequence numbers to process.
        algorithm (str): Name of the algorithm to use for similarity computation.

    Raises:
        ValueError: If an unsupported algorithm is provided.
    """
    for seq in sequence:
        gold_metadata = models.Metadata.get_metadata(gold_path, seq)
        gold_data = readers.read_sheet(gold_path, gold_metadata.sequence_sheetname)
        subject_metadata = models.Metadata.get_metadata(file_path, seq)
        subject_data = readers.read_sheet(
            file_path, subject_metadata.sequence_sheetname
        )

        centered_gold_data = preprocessing.center_joints_to_hip(gold_data)
        centered_subject_data = preprocessing.center_joints_to_hip(subject_data)
        gold_average_lengths = preprocessing.get_average_length(centered_gold_data)
        normalized_subject_data = preprocessing.normalize_segments(
            centered_subject_data, gold_average_lengths
        )

        if algorithm.lower() == "dtw":
            similarity_metric = similarity_functions.dynamic_time_warping(
                centered_gold_data, normalized_subject_data
            )
        else:
            raise ValueError(f"Unsupported algorithm '{algorithm}'.")

        writers.save_results_to_ndjson(
            gold_metadata,
            subject_metadata,
            similarity_metric,
            output_path,
            selected_metrics=["distance"],
        )
