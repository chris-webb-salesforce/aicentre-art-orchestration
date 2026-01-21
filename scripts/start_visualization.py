#!/usr/bin/env python3
"""
Standalone visualization server.

Run this to keep the visualization page open and ready.
It will update automatically when drawings start.

Usage:
    python scripts/start_visualization.py
    python scripts/start_visualization.py --no-browser
    python scripts/start_visualization.py --port 8080
"""

import sys
import os
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.web.server import run_standalone


def main():
    parser = argparse.ArgumentParser(description="Start visualization server")
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5000, help='Port to run on (default: 5000)')
    parser.add_argument('--no-browser', action='store_true', help="Don't auto-open browser")

    args = parser.parse_args()

    run_standalone(
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser
    )


if __name__ == '__main__':
    main()
