"""test cli.py functions."""

import argparse
import pathlib
from typing import List

from mobi_motion_tracking.core import cli, orchestrator


def test_parse_arguments() -> None:
    """Test the basic running of argparse with manadatory args."""
    args = cli.parse_arguments(["path/to/subject", "path/to/gold", "[1, 2, 3]", "dtw"])

    assert args.data == pathlib.Path("path/to/subject")
    assert args.gold == pathlib.Path("path/to/gold")
    assert args.sequence == List[1, 2, 3]
    assert args.algorithm == "dtw"
