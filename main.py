#!/usr/bin/env python3
"""Main entry point - direct CLI access"""
import asyncio
import sys
import os

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Direct import of CLI entry point
from cli import cli_main

if __name__ == "__main__":
    asyncio.run(cli_main())