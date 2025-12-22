#!/usr/bin/env python3
"""Main entry point - direct CLI access"""
import asyncio
import sys
import os

from src.cli import cli_main

if __name__ == "__main__":
    asyncio.run(cli_main())