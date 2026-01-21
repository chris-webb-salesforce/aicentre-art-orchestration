#!/usr/bin/env python3
"""Send DexArm to home position."""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.hardware.pydexarm import Dexarm

PORT = "/dev/tty.usbmodem3187378532331"

def main():
    print(f"Connecting to DexArm on {PORT}...")
    arm = Dexarm(PORT)

    if not arm.is_open:
        print("ERROR: Failed to connect")
        return 1

    print("Sending home command...")
    arm.go_home()
    print("Done!")

    arm.ser.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())
