"""Main function for mobi_motion_tracking."""

from mobi_motion_tracking.core import cli


def run_main() -> None:
    """Main entry point to mobi_motion_tracking."""
    cli.main()


if __name__ == "__main__":
    cli.main()
