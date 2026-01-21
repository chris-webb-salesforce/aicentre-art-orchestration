"""
Camera controller for capturing images from USB webcam.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CameraController:
    """Manages USB camera for face tracking and photo capture."""

    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        """
        Initialize camera controller.

        Args:
            index: Camera device index (usually 0 for built-in/first USB camera)
            width: Frame width for regular capture
            height: Frame height for regular capture
        """
        self.index = index
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self._is_open = False

    def initialize(self) -> bool:
        """
        Initialize and open the camera.

        Returns:
            True if camera opened successfully, False otherwise.
        """
        try:
            self.cap = cv2.VideoCapture(self.index)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera at index {self.index}")
                return False

            # Set resolution for tracking
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            # Read a test frame to verify
            ret, _ = self.cap.read()
            if not ret:
                logger.error("Camera opened but cannot read frames")
                self.cap.release()
                return False

            self._is_open = True
            logger.info(f"Camera initialized: index={self.index}, resolution={self.width}x{self.height}")
            return True

        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            return False

    def is_available(self) -> bool:
        """Check if camera is available and working."""
        return self._is_open and self.cap is not None and self.cap.isOpened()

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the camera.

        Returns:
            BGR image as numpy array, or None if capture failed.
        """
        if not self.is_available():
            logger.warning("Camera not available for frame capture")
            return None

        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            return None

        return frame

    def capture_high_res(self, width: int, height: int) -> Optional[np.ndarray]:
        """
        Capture a high-resolution image for the final portrait.

        Temporarily changes camera resolution, captures, then restores.

        Args:
            width: Desired capture width
            height: Desired capture height

        Returns:
            BGR image as numpy array, or None if capture failed.
        """
        if not self.is_available():
            logger.warning("Camera not available for high-res capture")
            return None

        try:
            # Set high resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            # Give camera time to adjust
            # Discard a few frames to let auto-exposure settle
            for _ in range(5):
                self.cap.read()

            # Capture the final frame
            ret, frame = self.cap.read()

            # Restore original resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            if not ret:
                logger.error("Failed to capture high-res frame")
                return None

            actual_h, actual_w = frame.shape[:2]
            logger.info(f"Captured high-res image: {actual_w}x{actual_h}")
            return frame

        except Exception as e:
            logger.error(f"Error during high-res capture: {e}")
            # Try to restore resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            return None

    def get_frame_dimensions(self) -> Tuple[int, int]:
        """
        Get current frame dimensions.

        Returns:
            Tuple of (width, height)
        """
        if self.cap is not None:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (w, h)
        return (self.width, self.height)

    def release(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self._is_open = False
            logger.info("Camera released")

    def __del__(self):
        """Cleanup on destruction."""
        self.release()


if __name__ == "__main__":
    # Test camera
    logging.basicConfig(level=logging.INFO)

    camera = CameraController(index=0, width=640, height=480)
    if camera.initialize():
        print("Camera initialized successfully")

        # Test frame capture
        frame = camera.get_frame()
        if frame is not None:
            print(f"Frame captured: {frame.shape}")
            cv2.imwrite("test_frame.jpg", frame)
            print("Saved test_frame.jpg")

        # Test high-res capture
        hi_res = camera.capture_high_res(1280, 720)
        if hi_res is not None:
            print(f"High-res frame captured: {hi_res.shape}")
            cv2.imwrite("test_hires.jpg", hi_res)
            print("Saved test_hires.jpg")

        camera.release()
    else:
        print("Failed to initialize camera")
