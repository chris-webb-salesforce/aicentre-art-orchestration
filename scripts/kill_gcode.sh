#!/bin/bash
# Stream Deck script to stop running GCode
# Kills any running test_components.py dexarm process

pkill -f "test_components.py dexarm"

echo "Sent kill signal"
