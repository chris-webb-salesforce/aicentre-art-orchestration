#!/bin/bash
# Stream Deck script to stop running GCode
# Kills any running test_components.py dexarm process

pkill -f "webcam_to_dexarm.py --yes"

echo "Sent kill signal"
