#!/bin/bash
# Stream Deck script to run GCode file on DexArm
# Usage: Just run this script - no arguments needed

cd "$(dirname "$0")/.."

source venv/bin/activate

python scripts/test_components.py dexarm \
    --port /dev/tty.usbmodem3187378532331 \
    --gcode-file test_output.gcode \
    -y
