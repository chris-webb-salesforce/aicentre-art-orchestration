"""
Face detection and tracking for the MyCobot camera.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class FaceDetection:
    """Represents a detected face."""
    x: int  # Top-left X
    y: int  # Top-left Y
    width: int
    height: int
    confidence: float
    centroid: Tuple[int, int]  # Center point (cx, cy)


class FaceTracker:
    """
    Detects and tracks faces in camera frames.

    Used to guide the MyCobot to center a face in frame before capture.
    """

    def __init__(self, min_face_size: float = 0.1):
        """
        Initialize face tracker.

        Args:
            min_face_size: Minimum face size as fraction of frame dimension
        """
        self.min_face_size = min_face_size
        self._cascade: Optional[cv2.CascadeClassifier] = None
        self._last_detection: Optional[FaceDetection] = None

    def initialize(self) -> bool:
        """
        Initialize the face detector.

        Returns:
            True if successful.
        """
        try:
            # Load Haar cascade for frontal face detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self._cascade = cv2.CascadeClassifier(cascade_path)

            if self._cascade.empty():
                logger.error("Failed to load Haar cascade")
                return False

            logger.info("Face tracker initialized with Haar cascade")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize face tracker: {e}")
            return False

    def detect_face(self, frame: np.ndarray) -> Optional[FaceDetection]:
        """
        Detect the largest face in a frame.

        Args:
            frame: BGR image from camera

        Returns:
            FaceDetection if found, None otherwise.
        """
        if self._cascade is None:
            logger.warning("Face tracker not initialized")
            return None

        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate minimum face size
            frame_h, frame_w = frame.shape[:2]
            min_size = int(min(frame_w, frame_h) * self.min_face_size)

            # Detect faces
            faces = self._cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(min_size, min_size),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            if len(faces) == 0:
                self._last_detection = None
                return None

            # Find the largest face (assuming it's the main subject)
            largest = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest

            # Calculate centroid
            cx = x + w // 2
            cy = y + h // 2

            # Estimate confidence based on face size (larger = more confident)
            confidence = min(1.0, (w * h) / (frame_w * frame_h * 0.3))

            detection = FaceDetection(
                x=int(x),
                y=int(y),
                width=int(w),
                height=int(h),
                confidence=float(confidence),
                centroid=(int(cx), int(cy))
            )

            self._last_detection = detection
            return detection

        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return None

    def calculate_offset(
        self,
        face: FaceDetection,
        frame_shape: Tuple[int, int, int]
    ) -> Tuple[float, float]:
        """
        Calculate the offset of face center from frame center.

        Args:
            face: Detected face
            frame_shape: Shape of frame (height, width, channels)

        Returns:
            Tuple of (pan_offset, tilt_offset) in range -1 to 1,
            where (0, 0) means face is centered.
        """
        frame_h, frame_w = frame_shape[:2]
        frame_cx = frame_w // 2
        frame_cy = frame_h // 2

        face_cx, face_cy = face.centroid

        # Normalize to -1 to 1 range
        pan_offset = (face_cx - frame_cx) / (frame_w / 2)
        tilt_offset = (face_cy - frame_cy) / (frame_h / 2)

        return (pan_offset, tilt_offset)

    def is_centered(
        self,
        pan_offset: float,
        tilt_offset: float,
        threshold: float = 0.1
    ) -> bool:
        """
        Check if face is centered within threshold.

        Args:
            pan_offset: Horizontal offset (-1 to 1)
            tilt_offset: Vertical offset (-1 to 1)
            threshold: Maximum offset to consider centered

        Returns:
            True if face is centered.
        """
        return abs(pan_offset) < threshold and abs(tilt_offset) < threshold

    def track_until_centered(
        self,
        camera,
        robot,
        config,
        on_frame_callback=None
    ) -> Optional[np.ndarray]:
        """
        Track face and adjust robot until centered, then capture.

        Args:
            camera: CameraController instance
            robot: MyCobotController instance
            config: FaceTrackingConfig with thresholds and timing
            on_frame_callback: Optional callback(frame, face) for visualization

        Returns:
            Captured frame with centered face, or None if timeout.
        """
        if not self._cascade:
            if not self.initialize():
                return None

        start_time = time.time()
        stable_start = None

        logger.info("Starting face tracking...")

        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > config.max_tracking_time:
                logger.warning("Face tracking timeout")
                return None

            # Get frame
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            # Detect face
            face = self.detect_face(frame)

            # Callback for visualization
            if on_frame_callback:
                on_frame_callback(frame, face)

            if face is None:
                stable_start = None
                time.sleep(0.05)
                continue

            # Calculate offset
            pan_offset, tilt_offset = self.calculate_offset(face, frame.shape)

            # Check if centered
            if self.is_centered(pan_offset, tilt_offset, config.center_threshold):
                if stable_start is None:
                    stable_start = time.time()
                    logger.info("Face centered, waiting for stability...")
                elif time.time() - stable_start >= config.stable_duration:
                    logger.info("Face stable! Capturing...")
                    return frame
            else:
                stable_start = None
                # Adjust robot to center face
                robot.adjust_for_face(
                    pan_offset,
                    tilt_offset,
                    config.pan_sensitivity if hasattr(config, 'pan_sensitivity') else 0.05,
                    config.tilt_sensitivity if hasattr(config, 'tilt_sensitivity') else 0.05
                )
                # Give the arm time to reach the new position before next adjustment
                time.sleep(0.3)
                continue

            time.sleep(0.05)  # ~20 FPS tracking

    def draw_detection(
        self,
        frame: np.ndarray,
        face: Optional[FaceDetection],
        show_center: bool = True
    ) -> np.ndarray:
        """
        Draw face detection visualization on frame.

        Args:
            frame: Image to draw on (will be modified)
            face: Face detection (or None)
            show_center: Whether to show frame center crosshair

        Returns:
            Modified frame.
        """
        frame_h, frame_w = frame.shape[:2]

        # Draw frame center
        if show_center:
            cx, cy = frame_w // 2, frame_h // 2
            cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 1)
            cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 1)

        # Draw face detection
        if face:
            # Rectangle around face
            cv2.rectangle(
                frame,
                (face.x, face.y),
                (face.x + face.width, face.y + face.height),
                (255, 0, 0),
                2
            )

            # Face centroid
            cv2.circle(frame, face.centroid, 5, (0, 0, 255), -1)

            # Line from face center to frame center
            frame_center = (frame_w // 2, frame_h // 2)
            cv2.line(frame, face.centroid, frame_center, (255, 255, 0), 1)

            # Confidence text
            cv2.putText(
                frame,
                f"Conf: {face.confidence:.2f}",
                (face.x, face.y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1
            )

        return frame


if __name__ == "__main__":
    # Test face tracker with webcam
    logging.basicConfig(level=logging.INFO)

    tracker = FaceTracker()
    if not tracker.initialize():
        print("Failed to initialize tracker")
        exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera")
        exit(1)

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face = tracker.detect_face(frame)

        if face:
            offset = tracker.calculate_offset(face, frame.shape)
            centered = tracker.is_centered(offset[0], offset[1])
            print(f"Face: offset=({offset[0]:.2f}, {offset[1]:.2f}), centered={centered}")

        # Draw visualization
        frame = tracker.draw_detection(frame, face)
        cv2.imshow("Face Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
