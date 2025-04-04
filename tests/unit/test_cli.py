"""test cli.py functions."""

import pathlib

import pytest
import pytest_mock

from mobi_motion_tracking.core import cli, orchestrator


def test_parse_arguments() -> None:
    """Test the basic running of argparse with manadatory args."""
    args = cli.parse_arguments(
        [
            "-d",
            "path/to/subject",
            "-g",
            "path/to/gold",
            "-s",
            "1,2,3",
            "-a",
            "dtw",
        ]
    )

    assert args.data == pathlib.Path("path/to/subject")
    assert args.gold == pathlib.Path("path/to/gold")
    assert args.sequence == [1, 2, 3]
    assert args.algorithm == "dtw"


def test_parse_arguments_no_inputs() -> None:
    """Test the error when required argument is missing."""
    with pytest.raises(SystemExit):
        cli.parse_arguments([])


def test_main_default(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Test cli with only necessary arguments."""
    mock_run = mocker.patch.object(orchestrator, "run")

    cli.main(
        [
            "-d",
            "tests/sample_data/100.xlsx",
            "-g",
            "tests/sample_data/Gold.xlsx",
            "-s",
            "1",
            "-a",
            "dtw",
        ]
    )

    mock_run.assert_called_once_with(
        experimental_path=pathlib.Path("tests/sample_data/100.xlsx"),
        gold_path=pathlib.Path("tests/sample_data/Gold.xlsx"),
        sequence=[1],
        algorithm="dtw",
    )
