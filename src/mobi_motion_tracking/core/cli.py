"""CLI for mobi-motion-tracking."""

import argparse
import pathlib
from typing import List, Optional

from mobi_motion_tracking.core import orchestrator


def parse_sequence_list(sequence_str: str) -> List[int]:
    """Converts input sequence string to List[int]."""
    return [int(seq.strip()) for seq in sequence_str.split(",")]


def parse_arguments(args: Optional[List[str]]) -> argparse.Namespace:
    """Argument parser for mobi-motion-tracking cli.

    Args:
        args: A list of command line arguments given as strings. If None, the parser
            will take the args from `sys.argv`.

    Returns:
        Namespace object with all the input arguments and default values.

    Raises:
        SystemExit: if arguments are None.
    """
    parser = argparse.ArgumentParser(
        description="Run the main motion tracking pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Please report issues at https://github.com/childmindresearch/mobi-motion-tracking.",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=pathlib.Path,
        required=True,
        help="Path to the subject(s) data.",
    )

    parser.add_argument(
        "-g",
        "--gold",
        type=pathlib.Path,
        required=True,
        help="Path to the gold data file.",
    )

    parser.add_argument(
        "-s",
        "--sequence",
        type=parse_sequence_list,
        required=True,
        help="String of comma seperated integer(s) indicating which sequences to run "
        "the pipeline for.",
    )

    parser.add_argument(
        "-a",
        "--algorithm",
        type=str,
        choices=["dtw"],
        required=True,
        help="Pick which algorithm to use. Can be 'dtw'.",
    )

    return parser.parse_args(args)


def main(
    args: Optional[List[str]],
) -> list[dict]:
    """Runs motion tracking orchestrator with command line arguments.

    Args:
         args: A list of command line arguments given as strings. If None, the parser
            will take the args from `sys.argv`.

    Returns:
        A result dict containing saved metrics for specified sequences for all subjects.
    """
    arguments = parse_arguments(args)

    results = orchestrator.run(
        experimental_path=arguments.data,
        gold_path=arguments.gold,
        sequence=arguments.sequence,
        algorithm=arguments.algorithm,
    )

    return results
