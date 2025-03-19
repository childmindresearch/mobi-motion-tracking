"""Dataclass storing all similarity metrics."""

import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


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
        participant to extract each participant ID and saves the sequence (int)
        as a string with the preface 'seq' for the sheet name.

        Args:
            subject_path: Path, full filepath per participant.
            sequence: int, sequence number.

        Returns:
            Metadata: Dataclass with participant_ID and sequence.

        Raises:
            ValueError: subject_file named incorrectly.
        """
        participant_ID = subject_path.stem

        if not (participant_ID.isdigit() or "gold" in participant_ID.lower()):
            raise ValueError("The input file is named incorrectly.")

        sequence_str = f"seq{sequence}"

        return cls(participant_ID=participant_ID, sequence_sheetname=sequence_str)
