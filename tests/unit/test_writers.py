"""Test writers.py functions."""

import datetime
import pathlib
from typing import List, Optional

import numpy as np
import pytest

from mobi_motion_tracking.core import models
from mobi_motion_tracking.io.writers import writers


def test_generate_output_filename_good() -> None:
    """Test that the generated filename follows the expected format and is created."""
    gold_id = "Gold"
    output_dir = pathlib.Path("tests/sample_data")
    date_str = datetime.datetime.now().strftime("%m%d%Y")
    expected_file = output_dir / f"results_Gold_{date_str}.ndjson"

    result = writers.generate_output_filename(gold_id, output_dir)

    assert result == pathlib.Path(
        expected_file
    ), "Generated filename does not match the expected format."
    assert result.exists(), "Output file was not created."


@pytest.mark.parametrize(
    "selected_metrics, expected_keys",
    [
        (["metric1"], {"participant_ID", "sheetname", "method", "metric1"}),
        (None, {"participant_ID", "sheetname", "method", "metric1", "metric2"}),
    ],
)
def test_save_results_good(
    selected_metrics: Optional[List[str]], expected_keys: set
) -> None:
    """Test that a single entry is correctly written to the NDJSON file."""
    gold = models.ParticipantData("Gold", "seq1", np.array([]))
    subject = models.ParticipantData("123", "seq1", np.array([]))
    similarity_metrics = models.SimilarityMetrics(
        "fake_method", {"metric1": 1, "metric2": 2}
    )
    output_dir = pathlib.Path("tests/sample_data")
    date_str = datetime.datetime.now().strftime("%m%d%Y")
    expected_output_path = output_dir / f"results_Gold_{date_str}.ndjson"

    output_dict = writers.save_results_to_ndjson(
        gold,
        subject,
        similarity_metrics,
        output_dir,
        selected_metrics=selected_metrics,
    )
    assert expected_output_path.exists(), "Expected output file was not generated."
    assert output_dict.keys() == expected_keys, "Selected metrics do not match \
        expected metrics."


def test_save_results_wrong_metric() -> None:
    """Test save_results when an incorrect metric is selected."""
    gold = models.ParticipantData("Gold", "seq1", np.array([]))
    subject = models.ParticipantData("123", "seq1", np.array([]))
    similarity_metrics = models.SimilarityMetrics(
        "fake_method", {"metric1": 1, "metric2": 2}
    )
    output_dir = pathlib.Path("tests/sample_data")
    selected_metrics = ["metric1", "false_metric"]

    with pytest.raises(
        ValueError,
        match="Selected metrics are not eligible for selected method.",
    ):
        writers.save_results_to_ndjson(
            gold,
            subject,
            similarity_metrics,
            output_dir,
            selected_metrics=selected_metrics,
        )
