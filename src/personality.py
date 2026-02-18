"""
Personality animations for the robot arms.

Adds character and life to the robots through expressive movements:
- MyCobot: Curious head tilts while watching the DexArm draw
- DexArm: Practice strokes in the air while waiting
"""

import threading
import time
import random
import logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class AnimationController:
    """Base class for animation controllers."""

    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self):
        """Start the animation in a background thread."""
        if self._running:
            return

        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._animation_loop, daemon=True)
        self._thread.start()
        logger.info(f"{self.__class__.__name__} animation started")

    def stop(self):
        """Stop the animation."""
        if not self._running:
            return

        self._stop_event.set()
        self._running = False

        if self._thread:
            self._thread.join(timeout=2.0)

        logger.info(f"{self.__class__.__name__} animation stopped")

    def _animation_loop(self):
        """Override in subclass to implement animation."""
        raise NotImplementedError

    @property
    def is_running(self) -> bool:
        return self._running


class CuriousTilts(AnimationController):
    """
    Makes the MyCobot do curious head tilts while watching.

    The arm will make small random adjustments to look like it's
    inspecting or being curious about what the DexArm is drawing.
    """

    def __init__(
        self,
        robot_controller,
        angle_range: float = 5.0,
        interval_min: float = 3.0,
        interval_max: float = 6.0,
        on_position_callback: Optional[Callable] = None
    ):
        """
        Initialize curious tilts animation.

        Args:
            robot_controller: MyCobotController instance
            angle_range: Maximum tilt angle in degrees
            interval_min: Minimum time between tilts (seconds)
            interval_max: Maximum time between tilts (seconds)
            on_position_callback: Optional callback to get current drawing position
        """
        super().__init__()
        self.robot = robot_controller
        self.angle_range = angle_range
        self.interval_min = interval_min
        self.interval_max = interval_max
        self.on_position_callback = on_position_callback

    def _animation_loop(self):
        """Perform random curious tilts at random intervals."""
        while not self._stop_event.is_set():
            try:
                # Random wait between tilts
                wait_time = random.uniform(self.interval_min, self.interval_max)
                if self._stop_event.wait(timeout=wait_time):
                    break

                # Perform curious tilt
                self.robot.perform_curious_tilt(self.angle_range)

                # Small pause after movement
                if self._stop_event.wait(timeout=0.5):
                    break

            except Exception as e:
                logger.warning(f"Curious tilt error: {e}")
                time.sleep(1)


class PracticeStrokes(AnimationController):
    """
    Makes the DexArm do practice strokes in the air.

    While waiting (e.g., during photo capture), the DexArm will
    make artistic movements as if warming up or practicing.
    """

    def __init__(
        self,
        dexarm_controller,
        height_offset: float = 20.0,
        radius: float = 15.0,
        speed: int = 1500,
        num_patterns: int = 2
    ):
        """
        Initialize practice strokes animation.

        Args:
            dexarm_controller: DexArmController instance
            height_offset: Height above paper for practice (mm)
            radius: Radius of stroke patterns (mm)
            speed: Movement speed (mm/min)
            num_patterns: Number of patterns to draw
        """
        super().__init__()
        self.arm = dexarm_controller
        self.height_offset = height_offset
        self.radius = radius
        self.speed = speed
        self.num_patterns = num_patterns

    def _animation_loop(self):
        """Perform practice stroke patterns."""
        while not self._stop_event.is_set():
            try:
                # Perform the practice strokes
                self.arm.perform_practice_strokes(
                    height_offset=self.height_offset,
                    radius=self.radius,
                    num_strokes=self.num_patterns,
                    speed=self.speed
                )

                # Wait before next set (or until stopped)
                if self._stop_event.wait(timeout=3.0):
                    break

            except Exception as e:
                logger.warning(f"Practice strokes error: {e}")
                time.sleep(1)


class IdleDance(AnimationController):
    """
    Makes the DexArm sway in a gentle figure-8 dance while idle.

    Stays at a safe height well above pen contact to avoid damage.
    Used when the system is waiting for a new photo submission.
    """

    def __init__(
        self,
        dexarm_controller,
        safe_z: float = 40.0,
        center_x: float = 0.0,
        center_y: float = 280.0,
        sweep_x: float = 50.0,
        arc_height: float = 20.0,
        speed: int = 1500,
        pause_between: float = 0.5
    ):
        """
        Initialize idle dance animation.

        Args:
            dexarm_controller: DexArmController instance
            safe_z: Base Z height — must be well above z_down
            center_x: X center of the arc
            center_y: Y position (constant)
            sweep_x: How far left/right the arm swings
            arc_height: How much higher at the peak of the arc
            speed: Movement speed (mm/min)
            pause_between: Seconds to pause between loops
        """
        super().__init__()
        self.arm = dexarm_controller
        self.safe_z = safe_z
        self.center_x = center_x
        self.center_y = center_y
        self.sweep_x = sweep_x
        self.arc_height = arc_height
        self.speed = speed
        self.pause_between = pause_between

    def _animation_loop(self):
        """Perform continuous side-to-side arc loops."""
        while not self._stop_event.is_set():
            try:
                self.arm.perform_idle_dance(
                    safe_z=self.safe_z,
                    center_x=self.center_x,
                    center_y=self.center_y,
                    sweep_x=self.sweep_x,
                    arc_height=self.arc_height,
                    speed=self.speed,
                    stop_event=self._stop_event
                )

                if self._stop_event.wait(timeout=self.pause_between):
                    break

            except Exception as e:
                logger.warning(f"Idle dance error: {e}")
                time.sleep(1)


class PersonalityDirector:
    """
    Coordinates personality animations for both robots.

    Manages when each animation should run based on the current
    stage of the portrait drawing process.
    """

    def __init__(
        self,
        mycobot_controller,
        dexarm_controller,
        config=None
    ):
        """
        Initialize personality director.

        Args:
            mycobot_controller: MyCobotController instance
            dexarm_controller: DexArmController instance
            config: PersonalityConfig with animation settings
        """
        self.mycobot = mycobot_controller
        self.dexarm = dexarm_controller
        self.config = config

        # Get settings from config or use defaults
        if config:
            practice_cfg = config.practice_strokes
            curious_cfg = config.curious_tilts
        else:
            practice_cfg = None
            curious_cfg = None

        # Initialize animations
        self.practice_strokes = PracticeStrokes(
            dexarm_controller,
            height_offset=practice_cfg.height_offset if practice_cfg else 20.0,
            radius=practice_cfg.radius if practice_cfg else 15.0,
            speed=practice_cfg.speed if practice_cfg else 1500
        )

        self.curious_tilts = CuriousTilts(
            mycobot_controller,
            angle_range=curious_cfg.angle_range if curious_cfg else 5.0,
            interval_min=curious_cfg.interval_min if curious_cfg else 3.0,
            interval_max=curious_cfg.interval_max if curious_cfg else 6.0
        )

        # Track which animations are enabled
        self._practice_enabled = practice_cfg.enabled if practice_cfg else True
        self._curious_enabled = curious_cfg.enabled if curious_cfg else True

    def start_capture_mode(self):
        """
        Start animations for capture mode.

        While the MyCobot is taking a photo, the DexArm does practice strokes.
        """
        logger.info("Starting capture mode animations")
        if self._practice_enabled:
            self.practice_strokes.start()

    def stop_capture_mode(self):
        """Stop capture mode animations."""
        logger.info("Stopping capture mode animations")
        self.practice_strokes.stop()

    def start_drawing_mode(self, position_callback: Optional[Callable] = None):
        """
        Start animations for drawing mode.

        While the DexArm is drawing, the MyCobot does curious tilts.

        Args:
            position_callback: Optional callback that returns current (x, y) drawing position
        """
        logger.info("Starting drawing mode animations")
        if self._curious_enabled:
            self.curious_tilts.on_position_callback = position_callback
            self.curious_tilts.start()

    def stop_drawing_mode(self):
        """Stop drawing mode animations."""
        logger.info("Stopping drawing mode animations")
        self.curious_tilts.stop()

    def stop_all(self):
        """Stop all animations."""
        self.practice_strokes.stop()
        self.curious_tilts.stop()


if __name__ == "__main__":
    # Test animations (without real hardware)
    logging.basicConfig(level=logging.INFO)

    class MockMyCobot:
        def perform_curious_tilt(self, angle):
            print(f"  MyCobot: curious tilt {angle:.1f}°")

    class MockDexArm:
        def perform_practice_strokes(self, **kwargs):
            print(f"  DexArm: practice strokes (height={kwargs.get('height_offset')}mm)")

    print("Testing personality animations (mock hardware)...")

    mycobot = MockMyCobot()
    dexarm = MockDexArm()

    director = PersonalityDirector(mycobot, dexarm)

    print("\n1. Testing capture mode (DexArm practice strokes)...")
    director.start_capture_mode()
    time.sleep(5)
    director.stop_capture_mode()

    print("\n2. Testing drawing mode (MyCobot curious tilts)...")
    director.start_drawing_mode()
    time.sleep(10)
    director.stop_drawing_mode()

    print("\nDone!")
