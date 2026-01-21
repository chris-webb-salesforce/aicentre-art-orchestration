#!/bin/bash
# Webcam to DexArm pipeline
# Captures photo, processes, and draws

cd "$(dirname "$0")/.."

# Run in a new Terminal window so GUI/camera works
osascript -e "tell application \"Terminal\" to do script \"cd '$PWD' && source venv/bin/activate && python scripts/webcam_to_dexarm.py --yes --music music/lounge-jazz-elevator-music-372734.mp3\""
