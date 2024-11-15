"""A very tiny CLI.

Invoke using e.g. ``python -m pygfx version``.
"""

import sys
import argparse

import pygfx


def main(argv=None):
    # Get argv so we can massage it
    if argv is None:
        argv = sys.argv
    if argv and argv[0].endswith(".py"):
        argv = argv[1:]

    # Defaults and aliases
    if not argv:
        argv = ["help"]
    if argv == ["--version"]:
        argv = ["version"]

    # Let the rest to argparse

    parser = argparse.ArgumentParser(
        prog="pygfx",
        description="The (very basic) Pygfx CLI",
    )

    parser.add_argument(
        "command", action="store", help="The command to run: 'help' or 'version'"
    )

    args = parser.parse_args(argv)
    command = args.command.lower()

    if command == "help":
        parser.print_help()
    if command == "version":
        print("pygfx v" + pygfx.__version__)
    else:
        print(f"Invalid command '{command}'")


if __name__ == "__main__":
    main()
