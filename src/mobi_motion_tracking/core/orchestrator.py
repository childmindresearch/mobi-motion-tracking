"""Python based runner."""

import pathlib

import numpy as np

from mobi_motion_tracking.core import models
from mobi_motion_tracking.io.readers import readers
from mobi_motion_tracking.io.writers import writers
from mobi_motion_tracking.preprocessing import preprocessing
from mobi_motion_tracking.processing import similarity_functions


def run_algorithm(
    algorithm: str,
    prerocessed_gold_data: np.ndarray,
    preprocessed_subject_data: np.ndarray,
) -> models.SimilarityMetrics:
    """Runs similarity metric algorithm."""
    if algorithm.lower() == "dtw":
        simililarity_metric = similarity_functions.dynamic_time_warping(
            prerocessed_gold_data, preprocessed_subject_data
        )
    else:
        raise ValueError(f"Unsupported algorithm '{algorithm}'.")
    return simililarity_metric


def run(
    gold_path: pathlib.Path,
    experimental_path: pathlib.Path,
    sequence: int | list[int],
    algorithm: str,
) -> None:
    """Runs main processing steps on single files, or directories."""
    if sequence is list:
        for seq in sequence:
            gold_metadata = models.Metadata.get_metadata(gold_path, seq)
            gold_data = readers.read_sheet(gold_path, gold_metadata.sequence_sheetname)

            if experimental_path.is_dir():
                files = [f for f in experimental_path.iterdir()]
                output_dir = experimental_path
            elif experimental_path.is_file():
                files = [experimental_path]
                output_dir = experimental_path.parent
            else:
                raise ValueError(
                    f"Path '{experimental_path}' is neither a \
                        file nor a directory."
                )

            for filepath in files:
                subject_metadata = models.Metadata.get_metadata(filepath, seq)
                subject_data = readers.read_sheet(
                    filepath, subject_metadata.sequence_sheetname
                )

                centered_gold_data = preprocessing.center_joints_to_hip(gold_data)
                centered_subject_data = preprocessing.center_joints_to_hip(subject_data)
                gold_average_lengths = preprocessing.get_average_length(
                    centered_gold_data
                )
                normalized_subject_data = preprocessing.normalize_segments(
                    centered_subject_data, gold_average_lengths
                )

                similarity_metric = run_algorithm(
                    algorithm, centered_gold_data, normalized_subject_data
                )

                writers.save_results_to_ndjson(
                    gold_metadata, subject_metadata, similarity_metric, output_dir
                )

    elif sequence is int:
        print(
            f"Calling get_metadata with gold_path={gold_path} and sequence={sequence}"
        )
        gold_metadata = models.Metadata.get_metadata(gold_path, sequence)
        gold_data = readers.read_sheet(gold_path, gold_metadata.sequence_sheetname)

        if experimental_path.is_dir():
            files = [f for f in experimental_path.iterdir()]
            output_dir = experimental_path
        elif experimental_path.is_file():
            files = [experimental_path]
            output_dir = experimental_path.parent
        else:
            raise ValueError(
                f"Path '{experimental_path}' is neither a file nor a \
                                directory."
            )

        for filepath in files:
            subject_metadata = models.Metadata.get_metadata(filepath, sequence)
            subject_data = readers.read_sheet(
                filepath, subject_metadata.sequence_sheetname
            )

            centered_gold_data = preprocessing.center_joints_to_hip(gold_data)
            centered_subject_data = preprocessing.center_joints_to_hip(subject_data)
            gold_average_lengths = preprocessing.get_average_length(centered_gold_data)
            normalized_subject_data = preprocessing.normalize_segments(
                centered_subject_data, gold_average_lengths
            )

            similarity_metric = run_algorithm(
                algorithm, centered_gold_data, normalized_subject_data
            )

            writers.save_results_to_ndjson(
                gold_metadata, subject_metadata, similarity_metric, output_dir
            )
    else:
        raise TypeError(
            f"Unsupported type: \
                {type(sequence).__name__}. Expected list or int."
        )
