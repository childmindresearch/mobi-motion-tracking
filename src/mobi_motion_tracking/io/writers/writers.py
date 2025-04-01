"""Functions to write calculated outputs to a file."""

import datetime
import json
import pathlib
from typing import Optional

from mobi_motion_tracking.core import models


def generate_output_filename(
    gold_participant_ID: str, output_dir: pathlib.Path
) -> pathlib.Path:
    """Generates a unique filename based on gold participant ID and date.

    The filename follows the format: `results_<gold_participant_ID>_<MMDDYYYY>.ndjson`.
    If the directory does not exist, it is created. If the file does not exist, it is
    created.

    Args:
        gold_participant_ID: The identifier for the gold-standard participant.
        output_dir: The directory where the NDJSON file should be stored.

    Returns:
        pathlib.Path: The full path to the generated NDJSON file.
    """
    date_str = datetime.datetime.now().strftime("%m%d%Y")
    base_filename = f"results_{gold_participant_ID}_{date_str}.ndjson"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / base_filename

    if not output_file.exists():
        output_file.touch()

    return output_file


def save_results_to_ndjson(
    gold: models.ParticipantData,
    subject: models.ParticipantData,
    similarity_metrics: models.SimilarityMetrics,
    output_dir: pathlib.Path,
    selected_metrics: Optional[list[str]] = None,
) -> dict:
    """Appends results to a NDJSON file with selected or all similarity metrics.

    The function writes a new JSON object containing the subject's metadata,
    the method used, and the selected similarity metrics. If `selected_metrics`
    is not provided, all available metrics from `similarity_metrics.metrics` are
    written.

    The data is appended to an NDJSON file specific to the gold participant,
    with the filename generated using `generate_output_filename()`.

    Args:
        gold: data for the gold-standard participant.
        subject: data for the subject being analyzed.
        similarity_metrics: Object containing the similarity
            method and metrics.
        output_dir: Directory where the results file should be saved.
        selected_metrics: List of metric keys to include in the
            output. If None, all available metrics are written.

    Returns:
        dict: entry to be written to output file.

    Raises:
        ValueError: If any selected metric is not available in
            `similarity_metrics.metrics`.
    """
    new_entry = {
        "participant_ID": subject.participant_ID,
        "sheetname": subject.sequence_sheetname,
        "method": similarity_metrics.method,
    }

    if selected_metrics is None:
        selected_metrics = list(similarity_metrics.metrics.keys())

    for metric_key in selected_metrics:
        if metric_key in list(similarity_metrics.metrics.keys()):
            new_entry[metric_key] = similarity_metrics.metrics[metric_key]
        else:
            raise ValueError("Selected metrics are not eligible for selected method.")

    output_path = generate_output_filename(gold.participant_ID, output_dir)

    with open(output_path, "a") as f:
        json.dump(new_entry, f)
        f.write("\n")

    return new_entry
