#!/bin/bash
# Start the visualization server (keeps running)
# Open this first, then run drawings - the page will update automatically

cd "$(dirname "$0")/.."

source venv/bin/activate

python scripts/start_visualization.py
