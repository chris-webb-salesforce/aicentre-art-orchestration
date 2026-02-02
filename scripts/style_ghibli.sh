#!/bin/bash
# Studio Ghibli anime style
cd "$(dirname "$0")/.."
osascript -e "tell application \"Terminal\" to do script \"cd '$PWD' && source venv/bin/activate && python scripts/webcam_to_dexarm.py --yes --style ghibli --music music/lounge-jazz-elevator-music-372734.mp3 --no-logo\""
