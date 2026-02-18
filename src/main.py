"""
Robot Portrait Drawing System - Main Entry Point

This system orchestrates two robot arms to create artistic portraits:
1. MyCobot320 (Processor): Tracks faces, captures photos
2. Rotrics DexArm (Artist): Draws the portrait

Flow:
1. User presses SPACEBAR
2. MyCobot tracks and centers face in camera
3. Photo is captured and sent to OpenAI for line art conversion
4. Line art is converted to drawing paths
5. DexArm draws the portrait while MyCobot watches curiously

Usage:
    python -m src.main [--mock] [--no-personality]
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Callable

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config, validate_config, Config
from src.hardware.camera_controller import CameraController
from src.hardware.mycobot_controller import MyCobotController
from src.hardware.dexarm_controller import DexArmController
from src.vision.face_tracker import FaceTracker
from src.vision.contour_extractor import ContourExtractor
from src.vision.adaptive_extractor import AdaptiveContourExtractor, AdaptiveExtractorConfig
from src.ai.openai_client import OpenAIClient, MockOpenAIClient
from src.planning.gcode_generator import GCodeGenerator, DrawingBounds
from src.personality import PersonalityDirector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PortraitSystem:
    """
    Main orchestrator for the robot portrait drawing system.

    Coordinates all subsystems through the portrait creation pipeline:
    track face -> capture -> convert to line art -> extract paths -> draw
    """

    def __init__(self, config: Config, use_mock_ai: bool = False,
                 enable_personality: bool = True, remote_mode: bool = False):
        """
        Initialize the portrait system.

        Args:
            config: System configuration
            use_mock_ai: If True, use mock AI client (no OpenAI API calls)
            enable_personality: If True, enable personality animations
            remote_mode: If True, skip camera/MyCobot/face tracker (DexArm only)
        """
        self.config = config
        self.use_mock_ai = use_mock_ai
        self.enable_personality = enable_personality and not remote_mode
        self.remote_mode = remote_mode

        # Initialize components (will be fully setup in initialize())
        self.camera: Optional[CameraController] = None
        self.mycobot: Optional[MyCobotController] = None
        self.dexarm: Optional[DexArmController] = None
        self.face_tracker: Optional[FaceTracker] = None
        self.contour_extractor: Optional[ContourExtractor] = None
        self.openai_client: Optional[OpenAIClient] = None
        self.gcode_generator: Optional[GCodeGenerator] = None
        self.personality: Optional[PersonalityDirector] = None

        self._is_initialized = False
        self._output_dir = Path(config.system.output_dir)

    def initialize(self) -> bool:
        """
        Initialize all subsystems.

        Returns:
            True if all systems initialized successfully.
        """
        logger.info("=" * 60)
        logger.info("Initializing Robot Portrait System")
        logger.info("=" * 60)

        # Create output directory
        self._output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self._output_dir}")

        if not self.remote_mode:
            # Initialize camera
            logger.info("Initializing camera...")
            self.camera = CameraController(
                index=self.config.camera.index,
                width=self.config.camera.width,
                height=self.config.camera.height
            )
            if not self.camera.initialize():
                logger.error("Failed to initialize camera")
                return False

            # Initialize face tracker
            logger.info("Initializing face tracker...")
            self.face_tracker = FaceTracker(
                min_face_size=self.config.face_tracking.min_face_size
            )
            if not self.face_tracker.initialize():
                logger.error("Failed to initialize face tracker")
                return False

            # Initialize MyCobot
            logger.info("Initializing MyCobot320...")
            self.mycobot = MyCobotController(
                port=self.config.mycobot.port,
                baud_rate=self.config.mycobot.baud_rate,
                home_angles=self.config.mycobot.home_angles,
                tracking_angles=self.config.mycobot.tracking_angles,
                speed=self.config.mycobot.speed
            )
            if not self.mycobot.initialize():
                logger.error("Failed to initialize MyCobot")
                return False
        else:
            logger.info("Remote mode: skipping camera, face tracker, and MyCobot")

        # Initialize DexArm
        logger.info("Initializing DexArm...")
        self.dexarm = DexArmController(
            port=self.config.dexarm.port,
            baud_rate=self.config.dexarm.baud_rate,
            feedrate=self.config.dexarm.feedrate,
            travel_feedrate=self.config.dexarm.travel_feedrate,
            z_up=self.config.drawing.z_up,
            z_down=self.config.drawing.z_down,
            acceleration=self.config.dexarm.acceleration,
            travel_acceleration=self.config.dexarm.travel_acceleration
        )
        if not self.dexarm.initialize():
            logger.error("Failed to initialize DexArm")
            return False

        # Initialize OpenAI client
        logger.info("Initializing AI client...")
        if self.use_mock_ai:
            self.openai_client = MockOpenAIClient(
                prompt=self.config.openai.prompt,
                size=self.config.openai.size
            )
        else:
            self.openai_client = OpenAIClient(
                model=self.config.openai.model,
                prompt=self.config.openai.prompt,
                size=self.config.openai.size,
                max_retries=self.config.openai.max_retries,
                retry_delay=self.config.openai.retry_delay
            )
        if not self.openai_client.initialize():
            logger.error("Failed to initialize AI client")
            return False

        # Initialize contour extractor (use adaptive if configured)
        logger.info(f"Initializing contour extractor (method: {self.config.contour.method})...")
        if self.config.contour.method.lower() == "canny":
            # Use original ContourExtractor
            self.contour_extractor = ContourExtractor(
                canny_low=self.config.contour.canny_low,
                canny_high=self.config.contour.canny_high,
                min_area=self.config.contour.min_area,
                simplify_epsilon=self.config.contour.simplify_epsilon,
                blur_kernel=self.config.contour.blur_kernel,
                min_contour_points=self.config.contour.min_contour_points
            )
        else:
            # Use AdaptiveContourExtractor for skeleton/adaptive/hybrid/thinning
            adaptive_config = AdaptiveExtractorConfig(
                method=self.config.contour.method,
                canny_low=self.config.contour.canny_low,
                canny_high=self.config.contour.canny_high,
                min_area=self.config.contour.min_area,
                simplify_epsilon=self.config.contour.simplify_epsilon,
                blur_kernel=self.config.contour.blur_kernel,
                min_contour_points=self.config.contour.min_contour_points,
                thickness_threshold=self.config.contour.thickness_threshold,
                density_threshold=self.config.contour.density_threshold,
                skeleton_simplify=self.config.contour.skeleton_simplify,
                thinning_threshold=self.config.contour.thinning_threshold,
                thinning_cleanup=self.config.contour.thinning_cleanup,
                thinning_cleanup_kernel=self.config.contour.thinning_cleanup_kernel,
                min_straightness=self.config.contour.min_straightness,
                min_length=self.config.contour.min_length,
                merge_distance=self.config.contour.merge_distance,
                merge_enabled=self.config.contour.merge_enabled,
                region_aware=self.config.contour.region_aware,
                detail_simplify_epsilon=self.config.contour.detail_simplify_epsilon,
                detail_min_length=self.config.contour.detail_min_length,
                detail_min_area=self.config.contour.detail_min_area,
                detail_region_padding=self.config.contour.detail_region_padding,
            )
            self.contour_extractor = AdaptiveContourExtractor(adaptive_config)

        # Initialize GCode generator
        logger.info("Initializing GCode generator...")
        bounds = DrawingBounds(
            x_min=self.config.drawing.x_min,
            x_max=self.config.drawing.x_max,
            y_min=self.config.drawing.y_min,
            y_max=self.config.drawing.y_max,
            z_up=self.config.drawing.z_up,
            z_down=self.config.drawing.z_down,
            feedrate=self.config.dexarm.feedrate,
            travel_feedrate=self.config.dexarm.travel_feedrate,
            flip_x=self.config.drawing.flip_x,
            flip_y=self.config.drawing.flip_y
        )
        self.gcode_generator = GCodeGenerator(bounds)

        # Initialize personality director
        if self.enable_personality:
            logger.info("Initializing personality animations...")
            self.personality = PersonalityDirector(
                self.mycobot,
                self.dexarm,
                self.config.personality
            )

        # Move robots to starting positions
        logger.info("Moving robots to starting positions...")
        if self.mycobot:
            self.mycobot.go_home()
        self.dexarm.go_to_safe_position(
            self.config.drawing.safe_position.get('x', 0),
            self.config.drawing.safe_position.get('y', 300),
            self.config.drawing.safe_position.get('z', 30)
        )

        self._is_initialized = True
        logger.info("=" * 60)
        logger.info("System initialized successfully!")
        logger.info("Press SPACEBAR to start portrait capture")
        logger.info("Press 'q' to quit")
        logger.info("=" * 60)
        return True

    def wait_for_trigger(self) -> bool:
        """
        Wait for user to press spacebar to trigger portrait capture.

        Shows a live preview window with face detection overlay.

        Returns:
            True if spacebar pressed, False if 'q' pressed to quit.
        """
        logger.info("Waiting for spacebar trigger...")

        consecutive_failures = 0
        max_failures = 50  # 5 seconds at 0.1s per failure

        while True:
            frame = self.camera.get_frame()
            if frame is None:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    logger.error("Camera failed repeatedly, cannot continue")
                    return False
                time.sleep(0.1)
                continue

            consecutive_failures = 0  # Reset on successful frame

            # Detect and draw face
            face = self.face_tracker.detect_face(frame)
            display_frame = self.face_tracker.draw_detection(frame.copy(), face)

            # Add instructions
            cv2.putText(
                display_frame,
                "Press SPACEBAR to capture portrait",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            cv2.putText(
                display_frame,
                "Press 'q' to quit",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1
            )

            cv2.imshow("Portrait System", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                logger.info("Spacebar pressed - starting portrait capture!")
                return True
            elif key == ord('q'):
                logger.info("Quit requested")
                return False

    def run_portrait_pipeline(self) -> bool:
        """
        Execute the full portrait pipeline.

        Returns:
            True if portrait completed successfully.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self._output_dir / timestamp
        session_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Stage 1: Face tracking and capture
            logger.info("\n" + "=" * 40)
            logger.info("STAGE 1: Face Tracking & Capture")
            logger.info("=" * 40)

            # Start DexArm practice strokes
            if self.personality:
                self.personality.start_capture_mode()

            # Move MyCobot to tracking position
            self.mycobot.go_to_tracking_position()

            # Track face and capture
            captured_image = self._track_and_capture()

            # Stop practice strokes
            if self.personality:
                self.personality.stop_capture_mode()

            if captured_image is None:
                logger.error("Failed to capture face")
                return False

            # Save captured image
            portrait_path = session_dir / "portrait.jpg"
            cv2.imwrite(str(portrait_path), captured_image)
            logger.info(f"Saved portrait to {portrait_path}")

            # Stage 2: Generate line art
            logger.info("\n" + "=" * 40)
            logger.info("STAGE 2: Generating Line Art")
            logger.info("=" * 40)

            lineart_path = session_dir / "lineart.png"
            line_art = self.openai_client.generate_line_art(
                captured_image,
                str(lineart_path)
            )

            if line_art is None:
                logger.error("Failed to generate line art")
                return False

            logger.info(f"Line art generated: {line_art.shape}")

            # Stage 3: Extract contours
            logger.info("\n" + "=" * 40)
            logger.info("STAGE 3: Extracting Contours")
            logger.info("=" * 40)

            contours = self.contour_extractor.extract(line_art)
            if not contours:
                logger.error("No contours extracted from line art")
                return False

            logger.info(f"Extracted {len(contours)} contours")

            # Optimize contour order
            if self.config.path_optimization.enabled:
                start_pos = (
                    (self.config.drawing.x_min + self.config.drawing.x_max) / 2,
                    self.config.drawing.y_min
                )
                contours = self.contour_extractor.optimize_order(contours, start_pos)
                logger.info("Optimized contour order")

            # Stage 4: Generate GCode
            logger.info("\n" + "=" * 40)
            logger.info("STAGE 4: Generating GCode")
            logger.info("=" * 40)

            image_bounds = self.contour_extractor.get_bounds(contours)
            gcode = self.gcode_generator.generate(contours, image_bounds)

            if not gcode:
                logger.error("Failed to generate GCode")
                return False

            # Save GCode
            gcode_path = session_dir / "drawing.gcode"
            self.gcode_generator.save_to_file(gcode, str(gcode_path))
            logger.info(f"Generated {len(gcode)} GCode lines")

            # Estimate time
            est_time = self.gcode_generator.estimate_drawing_time(gcode)
            logger.info(f"Estimated drawing time: {est_time:.1f} seconds")

            # Stage 5: Draw
            logger.info("\n" + "=" * 40)
            logger.info("STAGE 5: Drawing Portrait")
            logger.info("=" * 40)

            # Start curious tilts animation
            if self.personality:
                self.personality.start_drawing_mode()

            # Execute drawing
            success = self._execute_drawing(gcode)

            # Stop curious tilts
            if self.personality:
                self.personality.stop_drawing_mode()

            if not success:
                logger.error("Drawing failed")
                return False

            # Return arm to home position (X0, Y300, Z0)
            self.dexarm.go_home()

            # Done!
            logger.info("\n" + "=" * 40)
            logger.info("PORTRAIT COMPLETE!")
            logger.info(f"Session saved to: {session_dir}")
            logger.info("=" * 40)

            return True

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            # Ensure animations are stopped
            if self.personality:
                self.personality.stop_all()

    def run_pipeline_from_image(
        self,
        image: np.ndarray,
        style_name: str = "minimal",
        status_callback: Optional[Callable] = None
    ) -> bool:
        """
        Run the portrait pipeline from a pre-captured image (for remote submissions).

        Skips face tracking/camera capture. Applies style-specific AI prompt and
        contour extraction settings.

        Args:
            image: BGR image (portrait photo)
            style_name: Art style name (must match a key in config.openai.styles)
            status_callback: Optional callback(status, message, percent) for progress

        Returns:
            True if portrait completed successfully.
        """
        def notify(status, message, percent=0):
            if status_callback:
                status_callback(status, message, percent)
            if status == 'error':
                logger.error(f"[{status}] {message}")
            else:
                logger.info(f"[{status}] {message}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self._output_dir / timestamp
        session_dir.mkdir(parents=True, exist_ok=True)

        # Apply style-specific settings
        style_config = self.config.openai.styles.get(style_name)
        original_prompt = self.openai_client.prompt

        if not style_config:
            logger.warning(f"Style '{style_name}' not found in config, using defaults")
        elif style_config.prompt:
            self.openai_client.prompt = style_config.prompt
            logger.info(f"Using style '{style_name}': {style_config.name}")

        # Apply style-specific contour settings
        original_extractor = self.contour_extractor
        if style_config and style_config.contour:
            sc = style_config.contour
            method = sc.method or self.config.contour.method
            if method.lower() == "canny":
                self.contour_extractor = ContourExtractor(
                    canny_low=sc.canny_low or self.config.contour.canny_low,
                    canny_high=sc.canny_high or self.config.contour.canny_high,
                    min_area=sc.min_area or self.config.contour.min_area,
                    simplify_epsilon=sc.simplify_epsilon if sc.simplify_epsilon is not None else self.config.contour.simplify_epsilon,
                    blur_kernel=sc.blur_kernel or self.config.contour.blur_kernel,
                    min_contour_points=sc.min_contour_points or self.config.contour.min_contour_points,
                )
            else:
                adaptive_config = AdaptiveExtractorConfig(
                    method=method,
                    canny_low=sc.canny_low or self.config.contour.canny_low,
                    canny_high=sc.canny_high or self.config.contour.canny_high,
                    min_area=sc.min_area or self.config.contour.min_area,
                    simplify_epsilon=sc.simplify_epsilon if sc.simplify_epsilon is not None else self.config.contour.simplify_epsilon,
                    blur_kernel=sc.blur_kernel or self.config.contour.blur_kernel,
                    min_contour_points=sc.min_contour_points or self.config.contour.min_contour_points,
                    thickness_threshold=sc.thickness_threshold or self.config.contour.thickness_threshold,
                    density_threshold=sc.density_threshold or self.config.contour.density_threshold,
                    skeleton_simplify=sc.skeleton_simplify or self.config.contour.skeleton_simplify,
                    thinning_threshold=sc.thinning_threshold or self.config.contour.thinning_threshold,
                    thinning_cleanup=sc.thinning_cleanup if sc.thinning_cleanup is not None else self.config.contour.thinning_cleanup,
                    thinning_cleanup_kernel=sc.thinning_cleanup_kernel or self.config.contour.thinning_cleanup_kernel,
                    min_straightness=sc.min_straightness if sc.min_straightness is not None else self.config.contour.min_straightness,
                    min_length=sc.min_length or self.config.contour.min_length,
                    merge_distance=sc.merge_distance or self.config.contour.merge_distance,
                    merge_enabled=sc.merge_enabled if sc.merge_enabled is not None else self.config.contour.merge_enabled,
                    region_aware=sc.region_aware if sc.region_aware is not None else self.config.contour.region_aware,
                    detail_simplify_epsilon=sc.detail_simplify_epsilon or self.config.contour.detail_simplify_epsilon,
                    detail_min_length=sc.detail_min_length or self.config.contour.detail_min_length,
                    detail_min_area=sc.detail_min_area or self.config.contour.detail_min_area,
                    detail_region_padding=sc.detail_region_padding or self.config.contour.detail_region_padding,
                )
                self.contour_extractor = AdaptiveContourExtractor(adaptive_config)

        try:
            # Save portrait
            notify('processing', 'Saving portrait...', 5)
            portrait_path = session_dir / "portrait.jpg"
            cv2.imwrite(str(portrait_path), image)

            # Stage 2: Generate line art
            notify('processing', 'Generating line art... (this may take 30s)', 10)
            lineart_path = session_dir / "lineart.png"
            line_art = self.openai_client.generate_line_art(image, str(lineart_path))

            if line_art is None:
                notify('error', 'Failed to generate line art')
                return False

            # Stage 3: Extract contours
            notify('processing', 'Extracting drawing paths...', 40)
            contours = self.contour_extractor.extract(line_art)
            if not contours:
                notify('error', 'No contours extracted')
                return False

            if self.config.path_optimization.enabled:
                start_pos = (
                    (self.config.drawing.x_min + self.config.drawing.x_max) / 2,
                    self.config.drawing.y_min
                )
                contours = self.contour_extractor.optimize_order(contours, start_pos)

            # Stage 4: Generate GCode
            notify('processing', 'Generating drawing instructions...', 50)
            image_bounds = self.contour_extractor.get_bounds(contours)
            gcode = self.gcode_generator.generate(contours, image_bounds)

            if not gcode:
                notify('error', 'Failed to generate drawing instructions')
                return False

            gcode_path = session_dir / "drawing.gcode"
            self.gcode_generator.save_to_file(gcode, str(gcode_path))

            est_time = self.gcode_generator.estimate_drawing_time(gcode)
            notify('drawing', f'Drawing portrait... (~{est_time:.0f}s)', 55)

            # Stage 5: Draw
            if self.personality:
                self.personality.start_drawing_mode()

            start_time = time.time()

            def progress_cb(current, total, _position, contour=None, total_contours=None):
                if total > 0:
                    pct = 55 + int(45 * current / total)
                    elapsed = time.time() - start_time
                    remaining = (elapsed * total / current - elapsed) if current > 0 else 0
                    msg = f'Drawing... {remaining:.0f}s remaining'
                    if contour and total_contours:
                        msg = f'Drawing line {contour}/{total_contours}'
                    notify('drawing', msg, pct)

            success = self.dexarm.stream_gcode(gcode, progress_cb)

            if self.personality:
                self.personality.stop_drawing_mode()

            if not success:
                notify('error', 'Drawing failed')
                return False

            # Return arm to home position (X0, Y300, Z0)
            self.dexarm.go_home()

            notify('complete', 'Portrait complete!', 100)
            logger.info(f"Portrait complete! Session: {session_dir}")
            return True

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            notify('error', f'Pipeline error: {str(e)}')
            return False

        finally:
            # Restore original settings
            self.openai_client.prompt = original_prompt
            self.contour_extractor = original_extractor
            if self.personality:
                self.personality.stop_all()

    def _track_and_capture(self) -> Optional[np.ndarray]:
        """
        Track face and capture image when centered.

        Returns:
            Captured image or None.
        """
        def on_frame(frame, face):
            display = self.face_tracker.draw_detection(frame.copy(), face)
            cv2.putText(
                display,
                "Tracking face... hold still!",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
            cv2.imshow("Portrait System", display)
            cv2.waitKey(1)

        # Create a simple config-like object for tracking
        class TrackingConfig:
            pass

        tracking_cfg = TrackingConfig()
        tracking_cfg.center_threshold = self.config.face_tracking.center_threshold
        tracking_cfg.stable_duration = self.config.face_tracking.stable_duration
        tracking_cfg.max_tracking_time = self.config.face_tracking.max_tracking_time
        tracking_cfg.pan_sensitivity = self.config.mycobot.pan_sensitivity
        tracking_cfg.tilt_sensitivity = self.config.mycobot.tilt_sensitivity

        # Track until face is centered
        frame = self.face_tracker.track_until_centered(
            self.camera,
            self.mycobot,
            tracking_cfg,
            on_frame_callback=on_frame
        )

        if frame is None:
            return None

        # Capture high-res image
        logger.info("Capturing high-resolution image...")
        hi_res = self.camera.capture_high_res(
            self.config.camera.capture_width,
            self.config.camera.capture_height
        )

        return hi_res if hi_res is not None else frame

    def _execute_drawing(self, gcode: list) -> bool:
        """
        Execute GCode drawing on DexArm.

        Args:
            gcode: List of GCode commands

        Returns:
            True if drawing completed successfully.
        """
        start_time = time.time()

        def progress_callback(current, total, _position, contour=None, total_contours=None):
            elapsed = time.time() - start_time
            if current > 0:
                est_total = elapsed * total / current
                remaining = est_total - elapsed
                if contour and total_contours:
                    logger.info(f"Drawing Line {contour}/{total_contours} ({100*contour/total_contours:.0f}%) ")
                else:
                    logger.info(
                        f"Drawing progress: {current}/{total} ({100*current/total:.1f}%) "
                        f"- {remaining:.0f}s remaining"
                    )

        return self.dexarm.stream_gcode(gcode, progress_callback)

    def shutdown(self):
        """Clean shutdown of all systems."""
        logger.info("Shutting down...")

        # Stop animations
        if self.personality:
            self.personality.stop_all()

        # Close preview window
        cv2.destroyAllWindows()

        # Release hardware
        if self.dexarm:
            self.dexarm.release()

        if self.mycobot:
            self.mycobot.release()

        if self.camera:
            self.camera.release()

        logger.info("Shutdown complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Robot Portrait Drawing System"
    )
    parser.add_argument(
        '--mock',
        action='store_true',
        help='Use mock AI client (no OpenAI API calls)'
    )
    parser.add_argument(
        '--no-personality',
        action='store_true',
        help='Disable personality animations'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file'
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Validate configuration
    errors = validate_config(config)
    if errors:
        logger.error("Configuration errors:")
        for error in errors:
            logger.error(f"  - {error}")
        if not args.mock and "OPENAI_API_KEY" in str(errors):
            logger.error("\nSet OPENAI_API_KEY or use --mock flag for testing")
        sys.exit(1)

    # Create and run system
    system = PortraitSystem(
        config,
        use_mock_ai=args.mock,
        enable_personality=not args.no_personality
    )

    try:
        # Initialize
        if not system.initialize():
            logger.error("Failed to initialize system")
            sys.exit(1)

        # Wait for trigger
        if system.wait_for_trigger():
            # Run portrait pipeline
            system.run_portrait_pipeline()

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")

    finally:
        system.shutdown()


if __name__ == "__main__":
    main()
