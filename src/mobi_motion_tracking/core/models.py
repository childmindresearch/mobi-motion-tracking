"""Dataclass storing all similarity metrics."""

import os
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class SimilarityMetrics:
    """Stores similarity metrics between two time-series sequences.

    Attributes:
        method: A string containing the name of the similarity function used.
        metrics A list containing speciifc outputs for each
            similarity function.
    """

    method: str
    metrics: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dtw(
        cls, distance: float, warping_path: list[tuple[int, int]]
    ) -> "SimilarityMetrics":
        """Creates a SimilarityMetrics instance from DTW output.

        Args:
            distance: The cumulative distance between experimental and target sequences.
            warping_path: A list of tuples representing the warping path.

        Returns:
            A SimilarityMetrics instance storing DTW-specific metrics.
        """
        return cls(
            method="DTW",
            metrics={
                "distance": distance,
                "target_path": ([p[0] for p in warping_path]),
                "experimental_path": ([p[1] for p in warping_path]),
            },
        )


@dataclass
class Metadata:
    """Stores relevant participant information.

    Attributes:
        participant_ID: Integer value unique to every participant.
        sequence: Integer value associated with each iteration of a test.
    """

    participant_ID: str
    sequence_sheetname: str

    @classmethod
    def get_metadata(cls, subject_path: pathlib.Path, sequence: int) -> "Metadata":
        """Strip path name for participant ID and create sequence sheet name.

        This function strips the basename without the file extension per
        participant to extract each participant ID (int or "gold") and saves the
        sequence (int) as a string with the preface 'seq' for the sheet name.

        Args:
            subject_path: Path, full filepath per participant.
            sequence: int, sequence number.

        Returns:
            Metadata: Dataclass with participant_ID and sequence.

        Raises:
            FileNotFoundError: subject_file doesn't exist.
            ValueError: Invalid file extension.
            ValueError: subject_file named incorrectly.
        """
        try:
            if not os.path.exists(subject_path):
                raise FileNotFoundError("File not found.")
            if ".xlsx" != subject_path.suffix:
                raise ValueError(
                    f"Invalid file extension: {subject_path}. Expected '.xlsx'."
                )

            participant_ID = subject_path.stem

        except FileNotFoundError as fnf_error:
            print(f"Skipping {subject_path}: {fnf_error}")
            return cls(participant_ID=None, sequence_sheetname=None)
        except ValueError as ve:
            print(f"Skipping file {subject_path}: {ve} (Wrong file type)")
            return cls(participant_ID=None, sequence_sheetname=None)

        try:
            if not (participant_ID.isdigit() or "gold" in participant_ID.lower()):
                raise ValueError("The input file is named incorrectly.")

            sequence_str = f"seq{sequence}"

        except ValueError as err:
            print(f"Skipping file {subject_path}: {err}")
            return cls(participant_ID=None, sequence_sheetname=None)

        return cls(participant_ID=participant_ID, sequence_sheetname=sequence_str)
