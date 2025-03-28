"""CLI for mobi-motion-tracking."""

import argparse
import pathlib
from typing import List, Optional

from mobi_motion_tracking.core import orchestrator


def parse_arguments(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Argument parser for mobi-motion-tracking cli.

    Args:
        args: A list of command line arguments given as strings. If None, the parser
            will take the args from `sys.argv`.

    Returns:
        Namespace object with all the input arguments and default values.
    """
    parser = argparse.ArgumentParser(
        description="Run the main motion tracking pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Please report issues at https://github.com/childmindresearch/mobi-motion-tracking.",
    )
    parser.add_argument(
        "--d", "--data", type=pathlib.Path, help="Path to the subject(s) data."
    )

    parser.add_argument(
        "-g", "--gold", type=pathlib.Path, help="Path to the gold data file."
    )

    parser.add_argument(
        "--s",
        "--sequence",
        type=List[int],
        help="List of integer(s) indicating which sequences to run the pipeline for.",
    )

    parser.add_argument(
        "-a",
        "--algorithm",
        type=str,
        choices=["dtw"],
        default="dtw",
        help="Pick which algorithm to use. Can be 'dtw'.",
    )
