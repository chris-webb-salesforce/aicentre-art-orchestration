"""
DexArm controller for drawing operations.

Wraps the existing pydexarm.py with higher-level drawing methods.
"""

import time
import logging
import math
from typing import List, Callable, Optional, Tuple

from .pydexarm import Dexarm

logger = logging.getLogger(__name__)


class DexArmController:
    """
    High-level controller for Rotrics DexArm drawing operations.

    Wraps the low-level Dexarm class with:
    - Drawing-specific methods (pen up/down, execute GCode)
    - Progress tracking for long drawings
    - Error handling with retries
    - Personality animations (practice strokes)
    """

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baud_rate: int = 115200,
        feedrate: int = 2000,
        travel_feedrate: int = 3000,
        z_up: float = 10.0,
        z_down: float = 0.0,
        acceleration: int = 200,
        travel_acceleration: int = 400,
        jerk: float = 5.0
    ):
        """
        Initialize DexArm controller.

        Args:
            port: Serial port for DexArm
            baud_rate: Serial baud rate (115200 default)
            feedrate: Speed for drawing moves (mm/min)
            travel_feedrate: Speed for pen-up moves (mm/min)
            z_up: Z height when pen is up (mm)
            z_down: Z height when pen is down (mm)
            acceleration: Acceleration for drawing moves (mm/s²)
            travel_acceleration: Acceleration for travel moves (mm/s²)
            jerk: Jerk limit for X/Y axes (mm/s) - lower = smoother direction changes
        """
        self.port = port
        self.baud_rate = baud_rate
        self.feedrate = feedrate
        self.travel_feedrate = travel_feedrate
        self.z_up = z_up
        self.z_down = z_down
        self.acceleration = acceleration
        self.travel_acceleration = travel_acceleration
        self.jerk = jerk
        self.arm: Optional[Dexarm] = None
        self._is_initialized = False
        self._pen_is_down = False
        self._current_position = (0.0, 300.0, self.z_up)  # x, y, z

    def initialize(self) -> bool:
        """
        Initialize connection to DexArm.

        Returns:
            True if connection successful.
        """
        try:
            logger.info(f"Connecting to DexArm on {self.port} at {self.baud_rate} baud...")
            self.arm = Dexarm(self.port, self.baud_rate)

            if not self.arm.is_open:
                logger.error("Failed to open DexArm serial port")
                return False

            # Set pen module mode
            self.arm.set_module_kind(0)  # 0 = pen module
            time.sleep(0.5)

            # Set acceleration limits for smooth motion
            logger.info(f"Setting acceleration: {self.acceleration} mm/s² (drawing), {self.travel_acceleration} mm/s² (travel)")
            self.arm.set_acceleration(
                acceleration=self.acceleration,
                travel_acceleration=self.travel_acceleration,
                retract_acceleration=60  # Default for pen lifts
            )
            time.sleep(0.2)

            # Set jerk limits for smoother direction changes (M205)
            logger.info(f"Setting jerk limit: {self.jerk} mm/s")
            self.arm._send_cmd(f"M205 X{self.jerk} Y{self.jerk} Z{self.jerk}\r")
            time.sleep(0.2)

            # Home the arm
            logger.info("Homing DexArm...")
            self.arm.go_home()
            time.sleep(3)

            self._is_initialized = True
            self._pen_is_down = False
            logger.info("DexArm initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize DexArm: {e}")
            return False

    def is_available(self) -> bool:
        """Check if DexArm is initialized and available."""
        return self._is_initialized and self.arm is not None

    def pen_up(self) -> bool:
        """
        Lift pen to travel height.

        Returns:
            True if successful.
        """
        if not self.is_available():
            return False

        try:
            x, y, _ = self._current_position
            self.arm.fast_move_to(x, y, self.z_up, self.travel_feedrate)
            self._current_position = (x, y, self.z_up)
            self._pen_is_down = False
            return True
        except Exception as e:
            logger.error(f"Failed to lift pen: {e}")
            return False

    def pen_down(self) -> bool:
        """
        Lower pen to drawing height.

        Returns:
            True if successful.
        """
        if not self.is_available():
            return False

        try:
            x, y, _ = self._current_position
            self.arm.move_to(x, y, self.z_down, self.feedrate)
            self._current_position = (x, y, self.z_down)
            self._pen_is_down = True
            return True
        except Exception as e:
            logger.error(f"Failed to lower pen: {e}")
            return False

    def move_to(self, x: float, y: float, z: Optional[float] = None, drawing: bool = False) -> bool:
        """
        Move to specified position.

        Args:
            x: X coordinate (mm)
            y: Y coordinate (mm)
            z: Z coordinate (mm), uses current z_up/z_down if None
            drawing: If True, use drawing feedrate; otherwise use travel feedrate

        Returns:
            True if successful.
        """
        if not self.is_available():
            return False

        try:
            if z is None:
                z = self.z_down if drawing else self.z_up

            feedrate = self.feedrate if drawing else self.travel_feedrate

            if drawing:
                self.arm.move_to(x, y, z, feedrate)
            else:
                self.arm.fast_move_to(x, y, z, feedrate)

            self._current_position = (x, y, z)
            return True

        except Exception as e:
            logger.error(f"Failed to move to ({x}, {y}, {z}): {e}")
            return False

    def execute_gcode_line(self, line: str, max_retries: int = 3) -> bool:
        """
        Execute a single GCode line.

        Args:
            line: GCode command string
            max_retries: Number of retries on failure

        Returns:
            True if successful.
        """
        if not self.is_available():
            return False

        # Skip comments and empty lines
        line = line.strip()
        if not line or line.startswith(';'):
            return True

        for attempt in range(max_retries):
            try:
                self.arm._send_cmd(line + '\r')

                # Update internal state based on command
                self._update_state_from_gcode(line)
                return True

            except Exception as e:
                logger.warning(f"GCode execution failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.5)

        logger.error(f"Failed to execute GCode after {max_retries} attempts: {line}")
        return False

    def _update_state_from_gcode(self, line: str):
        """Update internal position state from GCode command."""
        line = line.upper()

        # Parse position from G0/G1 commands
        if line.startswith('G0') or line.startswith('G1'):
            x, y, z = self._current_position

            if 'X' in line:
                try:
                    x = float(line.split('X')[1].split()[0].rstrip('YZFE'))
                except (IndexError, ValueError):
                    pass

            if 'Y' in line:
                try:
                    y = float(line.split('Y')[1].split()[0].rstrip('ZFE'))
                except (IndexError, ValueError):
                    pass

            if 'Z' in line:
                try:
                    z = float(line.split('Z')[1].split()[0].rstrip('FE'))
                except (IndexError, ValueError):
                    pass

            self._current_position = (x, y, z)
            self._pen_is_down = z <= self.z_down

    def stream_gcode(
        self,
        gcode_lines: List[str],
        progress_callback: Optional[Callable[[int, int, Tuple[float, float], Optional[int], Optional[int]], None]] = None
    ) -> bool:
        """
        Stream GCode commands to the DexArm line by line.

        Args:
            gcode_lines: List of GCode command strings
            progress_callback: Optional callback(current_line, total_lines, position, current_contour, total_contours)
                              current_contour and total_contours may be None if not available

        Returns:
            True if all commands executed successfully.
        """
        if not self.is_available():
            logger.error("DexArm not available for streaming")
            return False

        total_lines = len(gcode_lines)
        successful = 0

        # Pre-scan for contour markers to get total count
        total_contours = 0
        for line in gcode_lines:
            if line.strip().startswith(";CONTOUR "):
                total_contours += 1

        current_contour = 0

        logger.info(f"Starting GCode stream: {total_lines} lines, {total_contours} contours")

        try:
            for i, line in enumerate(gcode_lines):
                # Check for contour marker
                stripped = line.strip()
                if stripped.startswith(";CONTOUR "):
                    # Parse contour number (format: ";CONTOUR X/Y")
                    try:
                        parts = stripped.split()[1].split("/")
                        current_contour = int(parts[0])
                    except (IndexError, ValueError):
                        current_contour += 1

                if not self.execute_gcode_line(line):
                    logger.error(f"Failed at line {i + 1}: {line}")
                    return False

                successful += 1

                if progress_callback and (i % 10 == 0 or i == total_lines - 1):
                    x, y, _ = self._current_position
                    progress_callback(
                        i + 1,
                        total_lines,
                        (x, y),
                        current_contour if total_contours > 0 else None,
                        total_contours if total_contours > 0 else None
                    )

            logger.info(f"GCode stream complete: {successful}/{total_lines} lines, {total_contours} contours")
            return True

        except Exception as e:
            logger.error(f"GCode streaming error: {e}")
            return False

        finally:
            # Always lift pen and move to safe position on completion or error
            try:
                self.pen_up()
            except Exception:
                logger.warning("Failed to lift pen during cleanup")

    def get_position(self) -> Tuple[float, float, float]:
        """Get current position (x, y, z)."""
        if self.is_available():
            try:
                result = self.arm.get_current_position()
                if result:
                    x, y, z = result[0], result[1], result[2]
                    self._current_position = (x, y, z)
            except Exception:
                pass
        return self._current_position

    def go_home(self) -> bool:
        """Move DexArm to home position."""
        if not self.is_available():
            return False

        try:
            self.arm.go_home()
            time.sleep(2)
            self._pen_is_down = False
            return True
        except Exception as e:
            logger.error(f"Failed to go home: {e}")
            return False

    def go_to_safe_position(self, x: float = 0, y: float = 300, z: float = 30) -> bool:
        """Move to a safe position above the drawing area."""
        if not self.is_available():
            return False

        try:
            self.pen_up()
            self.arm.fast_move_to(x, y, z, self.travel_feedrate)
            self._current_position = (x, y, z)
            return True
        except Exception as e:
            logger.error(f"Failed to go to safe position: {e}")
            return False

    def perform_idle_dance(
        self,
        safe_z: float = 40.0,
        center_x: float = 0.0,
        center_y: float = 280.0,
        radius_x: float = 40.0,
        radius_y: float = 25.0,
        z_amplitude: float = 15.0,
        speed: int = 1500,
        steps: int = 36,
        stop_event=None
    ) -> bool:
        """
        Perform a smooth figure-8 idle dance at a safe height.

        The arm sways side-to-side in a figure-8 with gentle Z bobbing,
        staying well above the pen contact height.

        Args:
            safe_z: Base Z height for the dance (well above z_down)
            center_x: X center of the dance pattern
            center_y: Y center of the dance pattern
            radius_x: Horizontal extent of the figure-8
            radius_y: Vertical extent of the figure-8
            z_amplitude: How much Z bobs up and down during the dance
            speed: Movement speed (mm/min)
            steps: Number of points per full figure-8 loop
            stop_event: Optional threading.Event to check for early stop

        Returns:
            True if completed (or stopped cleanly).
        """
        if not self.is_available():
            return False

        try:
            # Move to the starting position at safe height
            start_z = safe_z + z_amplitude
            self.arm.fast_move_to(center_x, center_y, start_z, speed)
            self._current_position = (center_x, center_y, start_z)

            # One full figure-8 loop
            for i in range(steps):
                if stop_event and stop_event.is_set():
                    break

                t = (2 * math.pi * i) / steps

                # Figure-8 (lemniscate) parametric form
                px = center_x + radius_x * math.sin(t)
                py = center_y + radius_y * math.sin(t) * math.cos(t)
                # Gentle Z bob — two full bobs per loop, staying above safe_z
                pz = safe_z + z_amplitude * (0.5 + 0.5 * math.sin(2 * t))

                self.arm.fast_move_to(px, py, pz, speed)
                self._current_position = (px, py, pz)

            return True

        except Exception as e:
            logger.error(f"Failed to perform idle dance: {e}")
            return False

    def perform_practice_strokes(
        self,
        height_offset: float = 20.0,
        radius: float = 15.0,
        num_strokes: int = 3,
        speed: int = 1500
    ) -> bool:
        """
        Perform practice strokes in the air (personality animation).

        This runs while the camera is capturing, making the DexArm look
        like it's warming up or practicing.

        Args:
            height_offset: How high above z_up to perform (mm)
            radius: Radius of circular strokes (mm)
            num_strokes: Number of figure-8/circular patterns
            speed: Movement speed

        Returns:
            True if successful.
        """
        if not self.is_available():
            return False

        try:
            logger.info("Performing practice strokes...")

            # Get current position and move up
            x, y, _ = self._current_position
            practice_z = self.z_up + height_offset

            # Move to practice height
            self.arm.fast_move_to(x, y, practice_z, speed)

            # Perform figure-8 patterns
            for _ in range(num_strokes):
                # Figure-8 pattern
                for t in range(0, 360, 30):
                    angle = math.radians(t)
                    # Figure-8 parametric equations
                    px = x + radius * math.sin(angle)
                    py = y + radius * math.sin(angle) * math.cos(angle)
                    self.arm.fast_move_to(px, py, practice_z, speed)

            # Return to original position
            self.arm.fast_move_to(x, y, self.z_up, speed)
            self._current_position = (x, y, self.z_up)

            logger.info("Practice strokes complete")
            return True

        except Exception as e:
            logger.error(f"Failed to perform practice strokes: {e}")
            return False

    def release(self):
        """Release resources and move to safe position."""
        if self.is_available():
            try:
                self.pen_up()
                self.go_home()
            except Exception as e:
                logger.error(f"Error during release: {e}")

        self._is_initialized = False
        logger.info("DexArm released")


if __name__ == "__main__":
    # Test DexArm controller
    logging.basicConfig(level=logging.INFO)

    controller = DexArmController(port="/dev/ttyUSB0")
    if controller.initialize():
        print("DexArm initialized")

        # Test basic movements
        print("Testing pen up/down...")
        controller.move_to(0, 300, controller.z_up)
        time.sleep(1)
        controller.pen_down()
        time.sleep(1)
        controller.pen_up()
        time.sleep(1)

        # Test practice strokes
        print("Testing practice strokes...")
        controller.perform_practice_strokes()

        # Test simple GCode
        print("Testing GCode execution...")
        test_gcode = [
            "G0 Z10",
            ";CONTOUR 1/2",
            "G0 X-20 Y280",
            "G1 Z0",
            "G1 X20 Y280",
            "G1 X20 Y320",
            "G1 X-20 Y320",
            "G1 X-20 Y280",
            "G0 Z10",
            ";CONTOUR 2/2",
            "G0 X0 Y300",
            "G1 Z0",
            "G1 X10 Y310",
            "G0 Z10"
        ]

        def progress(current, total, pos, contour=None, total_contours=None):
            if contour and total_contours:
                print(f"  Contour {contour}/{total_contours} - line {current}/{total} at ({pos[0]:.1f}, {pos[1]:.1f})")
            else:
                print(f"  Progress: {current}/{total} at ({pos[0]:.1f}, {pos[1]:.1f})")

        controller.stream_gcode(test_gcode, progress)

        controller.release()
    else:
        print("Failed to initialize DexArm")
