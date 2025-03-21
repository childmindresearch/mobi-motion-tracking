"""smoke tests for orchestrator.py."""

import pathlib

import pytest

from mobi_motion_tracking.core import orchestrator


def test_orchestrator_good() -> None:
    """Smoke test for the orchestrator run function.

    This test ensures that the orchestrator can execute without errors using minimal,
    valid input data.

    It creates temporary files and directories simulating an experimental dataset and a
    gold standard reference. The test then invokes the `orchestrator.run` function with a sample sequence and
    algorithm.

    Args:
        tmp_path (Path): A temporary directory provided by pytest for file operations.

    Raises:
        pytest.Fail: If an exception occurs during execution.
    """
    experimental_path = pathlib.Path("tests/sample_data/100.xlsx")
    gold_path = pathlib.Path("tests/sample_data/Gold.xlsx")
    sequence = [1]
    algorithm = "dtw"

    try:
        orchestrator.run(experimental_path, gold_path, sequence, algorithm)
    except Exception as e:
        pytest.fail(f"Smoke test failed with exception: {e}")
