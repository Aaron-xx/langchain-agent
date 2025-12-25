#!/usr/bin/env python3
"""Main entry point for btliu CLI command."""
import asyncio
import sys
import os

# Ensure the package is importable
from btliu.cli import cli_main


def main() -> int:
    """Entry point for 'btliu' command.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        asyncio.run(cli_main())
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
