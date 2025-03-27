"""smoke tests for orchestrator.py."""

import datetime
import pathlib

from mobi_motion_tracking.core import orchestrator


def test_orchestrator_good_file() -> None:
    """Smoke test for the orchestrator run function."""
    experimental_path = pathlib.Path("tests/sample_data/100.xlsx")
    gold_path = pathlib.Path("tests/sample_data/Gold.xlsx")
    sequence = [1]
    date_str = datetime.datetime.now().strftime("%m%d%Y")
    expected_output_file = pathlib.Path(
        f"tests/sample_data/results_Gold_{date_str}.ndjson"
    )
    expected_keys = {"participant_ID", "sheetname", "method", "distance"}

    outputs = orchestrator.run(experimental_path, gold_path, sequence, "dtw")

    assert expected_output_file.exists(), "Expected file was not created."
    assert (
        outputs[0].keys() == expected_keys
    ), "Saved dictionary keys do not match expected results."


def test_orchestrator_good_dir() -> None:
    """Smoke test for the orchestrator run function."""
    experimental_path = pathlib.Path("tests/sample_data/sample_directory")
    gold_path = pathlib.Path("tests/sample_data/sample_directory/Gold.xlsx")
    sequence = [1]
    date_str = datetime.datetime.now().strftime("%m%d%Y")
    expected_output_file = pathlib.Path(
        f"tests/sample_data/results_Gold_{date_str}.ndjson"
    )
    expected_keys = {"participant_ID", "sheetname", "method", "distance"}

    outputs = orchestrator.run(experimental_path, gold_path, sequence, "dtw")

    assert expected_output_file.exists(), "Expected file was not created."
    assert (
        outputs[0].keys() == expected_keys
    ), "Saved dictionary keys do not match expected results."
