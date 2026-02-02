"""
MyCobot320 controller for camera positioning and personality movements.
"""

import time
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class MyCobotController:
    """
    Controls the MyCobot320 arm for camera positioning and face tracking.

    The MyCobot acts as the "cameraman" - it holds the camera and adjusts
    position to track and center faces for portrait capture.
    """

    def __init__(
        self,
        port: str = "/dev/ttyAMA0",
        baud_rate: int = 115200,
        home_angles: Optional[List[float]] = None,
        tracking_angles: Optional[List[float]] = None,
        speed: int = 30
    ):
        """
        Initialize MyCobot controller.

        Args:
            port: Serial port for MyCobot (typically /dev/ttyAMA0 on Pi)
            baud_rate: Serial baud rate (115200 default)
            home_angles: Joint angles for home position [J1-J6]
            tracking_angles: Joint angles for initial tracking position
            speed: Movement speed (0-100)
        """
        self.port = port
        self.baud_rate = baud_rate
        self.home_angles = home_angles or [0, 0, 0, 0, 0, 0]
        self.tracking_angles = tracking_angles or [0, 20, -30, 0, 30, 0]
        self.speed = speed
        self.mc = None
        self._is_initialized = False
        self._current_angles = self.home_angles.copy()

    def initialize(self) -> bool:
        """
        Initialize connection to MyCobot.

        Returns:
            True if connection successful, False otherwise.
        """
        try:
            from pymycobot import MyCobot320

            logger.info(f"Connecting to MyCobot320 on {self.port}...")
            self.mc = MyCobot320(self.port, self.baud_rate)
            time.sleep(2)  # Wait for connection to stabilize

            # Enable fresh mode for real-time control
            self.mc.set_fresh_mode(1)
            logger.info("Fresh mode enabled")

            self._is_initialized = True
            logger.info("MyCobot320 initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize MyCobot320: {e}")
            return False

    def is_available(self) -> bool:
        """Check if MyCobot is initialized and available."""
        return self._is_initialized and self.mc is not None

    def go_home(self) -> bool:
        """
        Move to home position.

        Returns:
            True if movement successful.
        """
        if not self.is_available():
            logger.warning("MyCobot not available")
            return False

        try:
            logger.info(f"Moving to home position: {self.home_angles}")
            self.mc.send_angles(self.home_angles, self.speed)
            self._current_angles = self.home_angles.copy()
            time.sleep(2)  # Wait for movement
            return True
        except Exception as e:
            logger.error(f"Failed to move to home: {e}")
            return False

    def go_to_tracking_position(self) -> bool:
        """
        Move to initial tracking position (ready to look for faces).

        Returns:
            True if movement successful.
        """
        if not self.is_available():
            logger.warning("MyCobot not available")
            return False

        try:
            logger.info(f"Moving to tracking position: {self.tracking_angles}")
            self.mc.send_angles(self.tracking_angles, self.speed)
            self._current_angles = self.tracking_angles.copy()
            time.sleep(2)  # Wait for movement
            return True
        except Exception as e:
            logger.error(f"Failed to move to tracking position: {e}")
            return False

    def adjust_for_face(
        self,
        pan_offset: float,
        tilt_offset: float,
        pan_sensitivity: float = 0.05,
        tilt_sensitivity: float = 0.05
    ) -> bool:
        """
        Adjust camera position based on face offset from center.

        The MyCobot will pan (rotate base) and tilt (adjust shoulder/elbow)
        to center the face in the camera frame.

        Args:
            pan_offset: Horizontal offset (-1 to 1, where 0 is centered)
            tilt_offset: Vertical offset (-1 to 1, where 0 is centered)
            pan_sensitivity: Degrees to move per unit offset for pan
            tilt_sensitivity: Degrees to move per unit offset for tilt

        Returns:
            True if adjustment successful.
        """
        if not self.is_available():
            return False

        try:
            # Read actual angles from robot to avoid drift
            actual_angles = self.mc.get_angles()
            if actual_angles and len(actual_angles) == 6:
                self._current_angles = actual_angles

            # Convert offsets to angle adjustments
            # Pan uses joint 0 (base rotation)
            # Tilt uses joint 1 (shoulder) - adjust sign based on your setup
            pan_delta = pan_offset * pan_sensitivity * 100  # Scale to degrees
            tilt_delta = -tilt_offset * tilt_sensitivity * 100  # Negative because camera up = angle down

            # Calculate new angles
            new_angles = self._current_angles.copy()
            new_angles[0] += pan_delta   # Base rotation for pan
            new_angles[1] += tilt_delta  # Shoulder for tilt

            # Clamp to safe ranges (adjust based on your setup)
            new_angles[0] = max(-90, min(90, new_angles[0]))
            new_angles[1] = max(-30, min(60, new_angles[1]))

            # Only move if change is significant
            if abs(pan_delta) > 0.5 or abs(tilt_delta) > 0.5:
                self.mc.send_angles(new_angles, self.speed)
                self._current_angles = new_angles

            return True

        except Exception as e:
            logger.error(f"Failed to adjust for face: {e}")
            return False

    def get_current_angles(self) -> Optional[List[float]]:
        """
        Get current joint angles.

        Returns:
            List of 6 joint angles, or None if unavailable.
        """
        if not self.is_available():
            return None

        try:
            angles = self.mc.get_angles()
            if angles:
                self._current_angles = angles
            return angles
        except Exception as e:
            logger.error(f"Failed to get angles: {e}")
            return self._current_angles

    def perform_curious_tilt(self, angle_range: float = 5.0) -> bool:
        """
        Perform a random curious tilt movement (personality animation).

        This makes the robot look like it's inspecting the drawing.

        Args:
            angle_range: Maximum tilt angle in degrees

        Returns:
            True if movement successful.
        """
        if not self.is_available():
            return False

        try:
            import random

            # Random small adjustments to make it look curious
            new_angles = self._current_angles.copy()
            new_angles[0] += random.uniform(-angle_range, angle_range)  # Pan
            new_angles[4] += random.uniform(-angle_range, angle_range)  # Wrist tilt

            # Clamp to safe ranges
            new_angles[0] = max(-90, min(90, new_angles[0]))
            new_angles[4] = max(-90, min(90, new_angles[4]))

            self.mc.send_angles(new_angles, self.speed // 2)  # Slower for subtle movement
            self._current_angles = new_angles
            return True

        except Exception as e:
            logger.error(f"Failed to perform curious tilt: {e}")
            return False

    def look_at_position(self, x: float, y: float, drawing_bounds: Tuple[float, float, float, float]) -> bool:
        """
        Adjust camera to look at a specific drawing position.

        This is used during drawing to make the MyCobot "watch" where the DexArm is drawing.

        Args:
            x: X coordinate on paper
            y: Y coordinate on paper
            drawing_bounds: Tuple of (x_min, x_max, y_min, y_max)

        Returns:
            True if adjustment successful.
        """
        if not self.is_available():
            return False

        try:
            x_min, x_max, y_min, y_max = drawing_bounds

            # Normalize position to -1 to 1 range
            x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
            y_norm = 2 * (y - y_min) / (y_max - y_min) - 1

            # Adjust angles to look at position
            # This is a simplified mapping - adjust based on actual robot geometry
            new_angles = self.tracking_angles.copy()
            new_angles[0] = x_norm * 20  # Pan based on X
            new_angles[1] = self.tracking_angles[1] - y_norm * 10  # Tilt based on Y

            self.mc.send_angles(new_angles, self.speed)
            self._current_angles = new_angles
            return True

        except Exception as e:
            logger.error(f"Failed to look at position: {e}")
            return False

    def stop(self):
        """Emergency stop - halt all movement."""
        if self.mc is not None:
            try:
                self.mc.stop()
                logger.info("MyCobot stopped")
            except Exception as e:
                logger.error(f"Error stopping MyCobot: {e}")

    def release(self):
        """Release resources."""
        self.stop()
        self._is_initialized = False
        logger.info("MyCobot released")


if __name__ == "__main__":
    # Test MyCobot controller
    logging.basicConfig(level=logging.INFO)

    controller = MyCobotController()
    if controller.initialize():
        print("MyCobot initialized")

        # Test movements
        controller.go_home()
        time.sleep(2)

        controller.go_to_tracking_position()
        time.sleep(2)

        # Test curious tilt
        for _ in range(3):
            controller.perform_curious_tilt()
            time.sleep(1)

        controller.go_home()
        controller.release()
    else:
        print("Failed to initialize MyCobot")
