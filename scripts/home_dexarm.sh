#!/bin/bash
# Stream Deck script to send DexArm to home position

cd "$(dirname "$0")/.."

source venv/bin/activate

python scripts/home_dexarm.py
