"""Python based runner."""

import pathlib

import numpy as np

from mobi_motion_tracking.core import models
from mobi_motion_tracking.io.readers import readers
from mobi_motion_tracking.preprocessing import preprocessing
from mobi_motion_tracking.processing import similarity_functions


def run_algorithm(
    algorithm: str,
    prerocessed_gold_data: np.ndarray,
    preprocessed_subject_data: np.ndarray,
) -> models.SimilarityMetrics:
    """Runs similarity metric algorithm."""
    if algorithm == "DTW":
        simililarity_metric = similarity_functions.dynamic_time_warping(
            prerocessed_gold_data, preprocessed_subject_data
        )

    return simililarity_metric


def run(
    gold_path: pathlib.Path,
    experimental_path: pathlib.Path,
    sequence: int | list[int],
    algorithm: str,
) -> models.SimilarityMetrics:
    """Runs main processing steps on single files, or directories."""
    if sequence is list:
        for seq in sequence:
            gold_metadata = models.Metadata.get_metadata(gold_path, seq)
            gold_data = readers.read_sheet(gold_path, gold_metadata.sequence_sheetname)

            if experimental_path.is_dir():
                for filepath in experimental_path.iterdir():
                    if filepath.is_file() and filepath.suffix == ".xlsx":
                        subject_metadata = models.Metadata.get_metadata(
                            experimental_path, sequence
                        )
                        subject_data = readers.read_sheet(
                            experimental_path, subject_metadata.sequence_sheetname
                        )

                        centered_gold_data = preprocessing.center_joints_to_hip(
                            gold_data
                        )
                        centered_subject_data = preprocessing.center_joints_to_hip(
                            subject_data
                        )
                        gold_average_lengths = preprocessing.get_average_length(
                            centered_gold_data
                        )
                        normalized_subject_data = preprocessing.normalize_segments(
                            centered_subject_data, gold_average_lengths
                        )

                        similarity_metric = run_algorithm(
                            algorithm, centered_gold_data, normalized_subject_data
                        )

            elif experimental_path.is_file() and filepath.suffix == ".xlsx":
                subject_metadata = models.Metadata.get_metadata(
                    experimental_path, sequence
                )
                subject_data = readers.read_sheet(
                    experimental_path, subject_metadata.sequence_sheetname
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

    elif sequence is int:
        gold_metadata = models.Metadata.get_metadata(gold_path, seq)
        gold_data = readers.read_sheet(gold_path, gold_metadata.sequence_sheetname)
        if experimental_path.is_dir():
            for filepath in experimental_path.iterdir():
                if filepath.is_file() and filepath.suffix == ".xlsx":
                    subject_metadata = models.Metadata.get_metadata(
                        experimental_path, sequence
                    )
                    subject_data = readers.read_sheet(
                        experimental_path, subject_metadata.sequence_sheetname
                    )

                    centered_gold_data = preprocessing.center_joints_to_hip(gold_data)
                    centered_subject_data = preprocessing.center_joints_to_hip(
                        subject_data
                    )
                    gold_average_lengths = preprocessing.get_average_length(
                        centered_gold_data
                    )
                    normalized_subject_data = preprocessing.normalize_segments(
                        centered_subject_data, gold_average_lengths
                    )

                    similarity_metric = run_algorithm(
                        algorithm, centered_gold_data, normalized_subject_data
                    )

        elif experimental_path.is_file() and filepath.suffix == ".xlsx":
            subject_metadata = models.Metadata.get_metadata(experimental_path, sequence)
            subject_data = readers.read_sheet(
                experimental_path, subject_metadata.sequence_sheetname
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

    return similarity_metric
