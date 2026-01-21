#!/usr/bin/env python3
"""
Component Test Runner

Test each stage of the portrait drawing system independently.

Usage:
    python scripts/test_components.py <component> [options]

Components:
    camera          Test camera connection and capture
    face            Test face detection with live preview
    mycobot         Test MyCobot320 connection and movements
    dexarm          Test DexArm connection and basic drawing
    openai          Test OpenAI API with a sample image
    contours        Test contour extraction from an image
    gcode           Test GCode generation from contours
    personality     Test personality animations
    pipeline        Run full pipeline with mock components

Examples:
    python scripts/test_components.py camera
    python scripts/test_components.py dexarm --port /dev/ttyUSB0
    python scripts/test_components.py dexarm --air-draw circle
    python scripts/test_components.py dexarm --gcode-file output/test.gcode
    python scripts/test_components.py openai --image photo.jpg
    python scripts/test_components.py gcode --image lineart.png
    python scripts/test_components.py pipeline --mock

Air Draw Patterns (dexarm --air-draw):
    square      Simple square outline
    circle      Circle (36 segments)
    figure8     Figure-8 / infinity symbol
    star        5-pointed star
    spiral      Outward spiral (2 rotations)
"""

import sys
import os
import argparse
import time
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, '.env'))

from src.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_camera(args):
    """Test camera connection and display live feed."""
    import cv2
    from src.hardware.camera_controller import CameraController

    config = load_config()

    print("\n" + "=" * 50)
    print("CAMERA TEST")
    print("=" * 50)
    print(f"Camera index: {config.camera.index}")
    print(f"Resolution: {config.camera.width}x{config.camera.height}")
    print("Press 'c' to capture, 'h' for high-res, 'q' to quit")
    print("=" * 50 + "\n")

    camera = CameraController(
        index=config.camera.index,
        width=config.camera.width,
        height=config.camera.height
    )

    if not camera.initialize():
        print("ERROR: Failed to initialize camera")
        return False

    print("SUCCESS: Camera initialized\n")

    capture_count = 0
    while True:
        frame = camera.get_frame()
        if frame is None:
            print("WARNING: Failed to get frame")
            time.sleep(0.1)
            continue

        # Show frame info
        h, w = frame.shape[:2]
        cv2.putText(frame, f"Size: {w}x{h}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "c=capture, h=hi-res, q=quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow("Camera Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            capture_count += 1
            filename = f"test_capture_{capture_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
        elif key == ord('h'):
            print("Capturing high-res...")
            hi_res = camera.capture_high_res(
                config.camera.capture_width,
                config.camera.capture_height
            )
            if hi_res is not None:
                capture_count += 1
                filename = f"test_hires_{capture_count}.jpg"
                cv2.imwrite(filename, hi_res)
                print(f"Saved: {filename} ({hi_res.shape[1]}x{hi_res.shape[0]})")

    camera.release()
    cv2.destroyAllWindows()
    print("\nCamera test complete")
    return True


def test_face(args):
    """Test face detection with live preview."""
    import cv2
    from src.hardware.camera_controller import CameraController
    from src.vision.face_tracker import FaceTracker

    config = load_config()

    print("\n" + "=" * 50)
    print("FACE TRACKING TEST")
    print("=" * 50)
    print("Green box = detected face")
    print("Yellow line = offset from center")
    print("Press 'q' to quit")
    print("=" * 50 + "\n")

    camera = CameraController(
        index=config.camera.index,
        width=config.camera.width,
        height=config.camera.height
    )

    if not camera.initialize():
        print("ERROR: Failed to initialize camera")
        return False

    tracker = FaceTracker(min_face_size=config.face_tracking.min_face_size)
    if not tracker.initialize():
        print("ERROR: Failed to initialize face tracker")
        camera.release()
        return False

    print("SUCCESS: Camera and face tracker initialized\n")

    while True:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue

        face = tracker.detect_face(frame)
        display = tracker.draw_detection(frame.copy(), face)

        if face:
            offset = tracker.calculate_offset(face, frame.shape)
            centered = tracker.is_centered(offset[0], offset[1],
                                          config.face_tracking.center_threshold)
            status = "CENTERED" if centered else f"Offset: ({offset[0]:.2f}, {offset[1]:.2f})"
            color = (0, 255, 0) if centered else (0, 255, 255)
            cv2.putText(display, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(display, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Face Tracking Test", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
    print("\nFace tracking test complete")
    return True


def test_mycobot(args):
    """Test MyCobot320 connection and movements."""
    from src.hardware.mycobot_controller import MyCobotController

    config = load_config()
    port = args.port or config.mycobot.port

    print("\n" + "=" * 50)
    print("MYCOBOT320 TEST")
    print("=" * 50)
    print(f"Port: {port}")
    print(f"Baud rate: {config.mycobot.baud_rate}")
    print("=" * 50 + "\n")

    robot = MyCobotController(
        port=port,
        baud_rate=config.mycobot.baud_rate,
        home_angles=config.mycobot.home_angles,
        tracking_angles=config.mycobot.tracking_angles,
        speed=config.mycobot.speed
    )

    if not robot.initialize():
        print("ERROR: Failed to initialize MyCobot")
        return False

    print("SUCCESS: MyCobot initialized\n")

    tests = [
        ("Moving to HOME position", lambda: robot.go_home()),
        ("Waiting 2 seconds", lambda: time.sleep(2)),
        ("Moving to TRACKING position", lambda: robot.go_to_tracking_position()),
        ("Waiting 2 seconds", lambda: time.sleep(2)),
        ("Performing curious tilt 1", lambda: robot.perform_curious_tilt(5.0)),
        ("Waiting 1 second", lambda: time.sleep(1)),
        ("Performing curious tilt 2", lambda: robot.perform_curious_tilt(5.0)),
        ("Waiting 1 second", lambda: time.sleep(1)),
        ("Performing curious tilt 3", lambda: robot.perform_curious_tilt(5.0)),
        ("Waiting 2 seconds", lambda: time.sleep(2)),
        ("Returning to HOME", lambda: robot.go_home()),
    ]

    for desc, action in tests:
        print(f"  {desc}...", end=" ", flush=True)
        try:
            result = action()
            print("OK" if result is not False else "DONE")
        except Exception as e:
            print(f"FAILED: {e}")

    robot.release()
    print("\nMyCobot test complete")
    return True


def test_dexarm(args):
    """Test DexArm connection and basic drawing."""
    from src.hardware.dexarm_controller import DexArmController

    config = load_config()
    port = args.port or config.dexarm.port

    print("\n" + "=" * 50)
    print("DEXARM TEST")
    print("=" * 50)
    print(f"Port: {port}")
    print(f"Baud rate: {config.dexarm.baud_rate}")
    print(f"Z up: {config.drawing.z_up}, Z down: {config.drawing.z_down}")
    print("=" * 50 + "\n")

    arm = DexArmController(
        port=port,
        baud_rate=config.dexarm.baud_rate,
        feedrate=config.dexarm.feedrate,
        travel_feedrate=config.dexarm.travel_feedrate,
        z_up=config.drawing.z_up,
        z_down=config.drawing.z_down
    )

    if not arm.initialize():
        print("ERROR: Failed to initialize DexArm")
        return False

    print("SUCCESS: DexArm initialized\n")

    if args.gcode_file:
        # Run GCode from file
        if not os.path.exists(args.gcode_file):
            print(f"ERROR: GCode file not found: {args.gcode_file}")
            arm.release()
            return False

        print(f"Loading GCode from: {args.gcode_file}")
        with open(args.gcode_file, 'r') as f:
            gcode_lines = [line.strip() for line in f.readlines()]

        # Count actual commands (not comments/empty)
        commands = [l for l in gcode_lines if l and not l.startswith(';')]
        print(f"Loaded {len(commands)} commands ({len(gcode_lines)} total lines)")

        if not args.yes:
            print("\nWARNING: This will run the GCode file (pen will touch paper!)")
            input("Press ENTER to start or Ctrl+C to cancel...")

        def progress(current, total, pos):
            pct = 100 * current / total if total > 0 else 0
            print(f"  Progress: {current}/{total} ({pct:.1f}%) at ({pos[0]:.1f}, {pos[1]:.1f})")

        success = arm.stream_gcode(gcode_lines, progress)
        if success:
            print("\nGCode execution complete!")
        else:
            print("\nGCode execution failed!")

    elif args.air_draw:
        # Air drawing patterns - arm moves at safe height (no paper contact)
        import math

        pattern = args.air_draw.lower()
        z_base = 30  # Base safe height for air drawing
        z_wave = 10  # Z fluctuation amplitude (+/- mm)
        center_x, center_y = 0, 300
        size = 30  # Pattern size in mm
        feedrate = 10000  # Fast movement

        def z_fluctuate(i, total):
            """Add sinusoidal Z fluctuation for more dynamic movement."""
            return z_base + z_wave * math.sin(4 * math.pi * i / total)

        print(f"Air drawing pattern: {pattern.upper()}")
        print(f"Z height: {z_base}mm +/- {z_wave}mm fluctuation")
        print(f"Center: ({center_x}, {center_y}), Size: {size}mm, Speed: F{feedrate}")
        if not args.yes:
            input("\nPress ENTER to start or Ctrl+C to cancel...")

        gcode = [f"G0 Z{z_base}", f"G0 X{center_x} Y{center_y} Z{z_base}"]

        if pattern == 'square':
            # Simple square with Z fluctuation
            corners = [
                (center_x - size, center_y - size),
                (center_x + size, center_y - size),
                (center_x + size, center_y + size),
                (center_x - size, center_y + size),
                (center_x - size, center_y - size),
            ]
            gcode.append(f"G0 X{corners[0][0]} Y{corners[0][1]} Z{z_base}")
            for i, (x, y) in enumerate(corners[1:], 1):
                z = z_fluctuate(i, len(corners))
                gcode.append(f"G1 X{x} Y{y} Z{z:.2f} F{feedrate}")

        elif pattern == 'circle':
            # Circle approximated with 36 segments
            segments = 36
            for i in range(segments + 1):
                angle = 2 * math.pi * i / segments
                x = center_x + size * math.cos(angle)
                y = center_y + size * math.sin(angle)
                z = z_fluctuate(i, segments)
                cmd = "G0" if i == 0 else "G1"
                gcode.append(f"{cmd} X{x:.2f} Y{y:.2f} Z{z:.2f} F{feedrate}")

        elif pattern == 'figure8':
            # Figure-8 / infinity symbol
            segments = 48
            for i in range(segments + 1):
                t = 2 * math.pi * i / segments
                # Lemniscate of Bernoulli parametric equations
                x = center_x + size * math.cos(t) / (1 + math.sin(t)**2)
                y = center_y + size * math.sin(t) * math.cos(t) / (1 + math.sin(t)**2)
                z = z_fluctuate(i, segments)
                cmd = "G0" if i == 0 else "G1"
                gcode.append(f"{cmd} X{x:.2f} Y{y:.2f} Z{z:.2f} F{feedrate}")

        elif pattern == 'star':
            # 5-pointed star
            points = []
            for i in range(5):
                # Outer points
                angle_out = math.pi / 2 + 2 * math.pi * i / 5
                points.append((
                    center_x + size * math.cos(angle_out),
                    center_y + size * math.sin(angle_out)
                ))
                # Inner points
                angle_in = angle_out + math.pi / 5
                points.append((
                    center_x + size * 0.4 * math.cos(angle_in),
                    center_y + size * 0.4 * math.sin(angle_in)
                ))
            gcode.append(f"G0 X{points[0][0]:.2f} Y{points[0][1]:.2f} Z{z_base}")
            for i, (x, y) in enumerate(points[1:], 1):
                z = z_fluctuate(i, len(points))
                gcode.append(f"G1 X{x:.2f} Y{y:.2f} Z{z:.2f} F{feedrate}")
            z = z_fluctuate(len(points), len(points))
            gcode.append(f"G1 X{points[0][0]:.2f} Y{points[0][1]:.2f} Z{z:.2f} F{feedrate}")

        elif pattern == 'spiral':
            # Spiral outward
            segments = 72
            for i in range(segments + 1):
                angle = 4 * math.pi * i / segments  # 2 full rotations
                r = size * i / segments
                x = center_x + r * math.cos(angle)
                y = center_y + r * math.sin(angle)
                z = z_fluctuate(i, segments)
                cmd = "G0" if i == 0 else "G1"
                gcode.append(f"{cmd} X{x:.2f} Y{y:.2f} Z{z:.2f} F{feedrate}")

        else:
            print(f"ERROR: Unknown pattern '{pattern}'")
            print("Available patterns: square, circle, figure8, star, spiral")
            arm.release()
            return False

        # Return to center
        gcode.append(f"G0 X{center_x} Y{center_y} Z{z_base}")

        print(f"\nExecuting {len(gcode)} commands...")

        def progress(current, total, pos):
            if current % 5 == 0 or current == total:
                print(f"  {current}/{total} at ({pos[0]:.1f}, {pos[1]:.1f})")

        arm.stream_gcode(gcode, progress)
        print(f"\n{pattern.upper()} air drawing complete!")

    elif args.draw_test:
        print("Drawing test square (pen will touch paper!)...")
        if not args.yes:
            input("Press ENTER to start or Ctrl+C to cancel...")

        # Draw a small test square
        test_gcode = [
            f"G0 Z{config.drawing.z_up}",
            "G0 X-20 Y280",
            f"G1 Z{config.drawing.z_down}",
            "G1 X20 Y280",
            "G1 X20 Y320",
            "G1 X-20 Y320",
            "G1 X-20 Y280",
            f"G0 Z{config.drawing.z_up}",
            "G0 X0 Y300",
        ]

        def progress(current, total, pos):
            print(f"  Progress: {current}/{total} at ({pos[0]:.1f}, {pos[1]:.1f})")

        arm.stream_gcode(test_gcode, progress)
        print("Test square complete!")

    else:
        print("Running movement tests (pen stays up)...\n")

        tests = [
            ("Moving to safe position",
             lambda: arm.go_to_safe_position(0, 300, 30)),
            ("Waiting 1 second", lambda: time.sleep(1)),
            ("Moving to corner 1 (pen up)",
             lambda: arm.move_to(-30, 270, config.drawing.z_up)),
            ("Moving to corner 2 (pen up)",
             lambda: arm.move_to(30, 270, config.drawing.z_up)),
            ("Moving to corner 3 (pen up)",
             lambda: arm.move_to(30, 330, config.drawing.z_up)),
            ("Moving to corner 4 (pen up)",
             lambda: arm.move_to(-30, 330, config.drawing.z_up)),
            ("Practice strokes animation",
             lambda: arm.perform_practice_strokes(height_offset=20, radius=15)),
            ("Returning to safe position",
             lambda: arm.go_to_safe_position(0, 300, 30)),
        ]

        for desc, action in tests:
            print(f"  {desc}...", end=" ", flush=True)
            try:
                result = action()
                print("OK" if result is not False else "DONE")
            except Exception as e:
                print(f"FAILED: {e}")

        print("\nTo test actual drawing, run with --draw-test flag")

    arm.release()
    print("\nDexArm test complete")
    return True


def test_openai(args):
    """Test OpenAI API with a sample image."""
    import cv2
    import numpy as np

    print("\n" + "=" * 50)
    print("OPENAI API TEST")
    print("=" * 50)

    if args.mock:
        print("Using MOCK client (no API calls)")
        from src.ai.openai_client import MockOpenAIClient
        client = MockOpenAIClient()
    else:
        print("Using REAL OpenAI API")
        from src.ai.openai_client import OpenAIClient
        config = load_config()
        client = OpenAIClient(
            model=config.openai.model,
            prompt=config.openai.prompt,
            size=config.openai.size,
            max_retries=config.openai.max_retries,
            retry_delay=config.openai.retry_delay
        )

    print("=" * 50 + "\n")

    if not client.initialize():
        print("ERROR: Failed to initialize OpenAI client")
        if not args.mock:
            print("Make sure OPENAI_API_KEY is set, or use --mock")
        return False

    print("SUCCESS: OpenAI client initialized\n")

    # Get or create test image
    if args.image and os.path.exists(args.image):
        print(f"Using provided image: {args.image}")
        image = cv2.imread(args.image)
    else:
        print("Creating test image (simple face)...")
        image = np.ones((400, 400, 3), dtype=np.uint8) * 255
        cv2.circle(image, (200, 150), 80, (100, 100, 100), -1)
        cv2.circle(image, (170, 140), 12, (50, 50, 50), -1)
        cv2.circle(image, (230, 140), 12, (50, 50, 50), -1)
        cv2.ellipse(image, (200, 180), (30, 15), 0, 0, 180, (50, 50, 50), 3)
        cv2.imwrite("test_input.jpg", image)
        print("Saved test_input.jpg")

    print("\nGenerating line art...")
    output_path = "test_lineart.png"
    result = client.generate_line_art(image, output_path)

    if result is not None:
        print(f"SUCCESS: Line art generated ({result.shape[1]}x{result.shape[0]})")
        print(f"Saved to: {output_path}")

        # Show result
        cv2.imshow("Line Art Result", result)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return True
    else:
        print("ERROR: Failed to generate line art")
        return False


def test_contours(args):
    """Test contour extraction from an image."""
    import cv2
    import numpy as np
    from src.vision.contour_extractor import ContourExtractor

    config = load_config()

    print("\n" + "=" * 50)
    print("CONTOUR EXTRACTION TEST")
    print("=" * 50)

    extractor = ContourExtractor(
        canny_low=config.contour.canny_low,
        canny_high=config.contour.canny_high,
        min_area=config.contour.min_area,
        simplify_epsilon=config.contour.simplify_epsilon
    )

    # Get or create test image
    if args.image and os.path.exists(args.image):
        print(f"Using provided image: {args.image}")
        image = cv2.imread(args.image)
    else:
        print("Creating test line art image...")
        image = np.ones((400, 400), dtype=np.uint8) * 255
        cv2.circle(image, (200, 150), 80, 0, 2)
        cv2.circle(image, (170, 140), 15, 0, 2)
        cv2.circle(image, (230, 140), 15, 0, 2)
        cv2.ellipse(image, (200, 200), (40, 20), 0, 0, 180, 0, 2)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.imwrite("test_lineart_input.png", image)
        print("Saved test_lineart_input.png")

    print("=" * 50 + "\n")

    print("Extracting contours...")
    contours = extractor.extract(image)
    print(f"Found {len(contours)} contours\n")

    for i, c in enumerate(contours):
        print(f"  Contour {i+1}: {len(c.points)} points, "
              f"length={c.length:.1f}px, closed={c.is_closed}")

    if contours:
        # Optimize order
        bounds = extractor.get_bounds(contours)
        print(f"\nImage bounds: {bounds}")

        if config.path_optimization.enabled:
            print("Optimizing path order...")
            start = ((bounds[0] + bounds[2]) / 2, bounds[3])
            contours = extractor.optimize_order(contours, start)

        # Visualize
        viz = extractor.visualize(image.copy(), contours)
        cv2.imwrite("test_contours_viz.png", viz)
        print("Saved visualization to test_contours_viz.png")

        cv2.imshow("Contours", viz)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return len(contours) > 0


def test_gcode(args):
    """Test GCode generation from contours."""
    import cv2
    from src.vision.contour_extractor import ContourExtractor
    from src.planning.gcode_generator import GCodeGenerator, DrawingBounds

    config = load_config()

    print("\n" + "=" * 50)
    print("GCODE GENERATION TEST")
    print("=" * 50)
    print(f"Drawing bounds: X({config.drawing.x_min}, {config.drawing.x_max}), "
          f"Y({config.drawing.y_min}, {config.drawing.y_max})")
    print(f"Z heights: up={config.drawing.z_up}, down={config.drawing.z_down}")
    print("=" * 50 + "\n")

    # Get contours from image
    if args.image and os.path.exists(args.image):
        print(f"Loading image: {args.image}")
        image = cv2.imread(args.image)
    else:
        print("Creating test image...")
        import numpy as np
        image = np.ones((400, 400), dtype=np.uint8) * 255
        cv2.circle(image, (200, 150), 80, 0, 2)
        cv2.circle(image, (170, 140), 15, 0, 2)
        cv2.circle(image, (230, 140), 15, 0, 2)
        cv2.ellipse(image, (200, 200), (40, 20), 0, 0, 180, 0, 2)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    extractor = ContourExtractor(
        canny_low=config.contour.canny_low,
        canny_high=config.contour.canny_high,
        min_area=config.contour.min_area,
        simplify_epsilon=config.contour.simplify_epsilon
    )

    print("Extracting contours...")
    contours = extractor.extract(image)
    print(f"Found {len(contours)} contours")

    if not contours:
        print("ERROR: No contours found")
        return False

    # Get bounds and optimize
    image_bounds = extractor.get_bounds(contours)
    if config.path_optimization.enabled:
        contours = extractor.optimize_order(contours)

    # Generate GCode
    bounds = DrawingBounds(
        x_min=config.drawing.x_min,
        x_max=config.drawing.x_max,
        y_min=config.drawing.y_min,
        y_max=config.drawing.y_max,
        z_up=config.drawing.z_up,
        z_down=config.drawing.z_down,
        feedrate=config.dexarm.feedrate,
        travel_feedrate=config.dexarm.travel_feedrate
    )

    generator = GCodeGenerator(bounds)

    print("Generating GCode...")
    gcode = generator.generate(contours, image_bounds)

    print(f"Generated {len(gcode)} lines\n")

    # Show first and last lines
    print("First 15 lines:")
    for line in gcode[:15]:
        print(f"  {line}")

    if len(gcode) > 30:
        print(f"  ... ({len(gcode) - 30} more lines) ...")

    print("\nLast 5 lines:")
    for line in gcode[-5:]:
        print(f"  {line}")

    # Estimate time
    est_time = generator.estimate_drawing_time(gcode)
    print(f"\nEstimated drawing time: {est_time:.1f} seconds")

    # Save
    output_file = "test_output.gcode"
    generator.save_to_file(gcode, output_file)
    print(f"Saved to: {output_file}")

    return True


def test_personality(args):
    """Test personality animations."""
    from src.config import load_config

    config = load_config()

    print("\n" + "=" * 50)
    print("PERSONALITY ANIMATION TEST")
    print("=" * 50)
    print("This will test animations with mock robots")
    print("=" * 50 + "\n")

    # Mock controllers for testing
    class MockMyCobot:
        def perform_curious_tilt(self, angle):
            print(f"    [MyCobot] Curious tilt: {angle:.1f} degrees")
            return True

    class MockDexArm:
        def perform_practice_strokes(self, height_offset, radius, num_strokes, speed):
            print(f"    [DexArm] Practice strokes: height={height_offset}mm, "
                  f"radius={radius}mm, count={num_strokes}")
            return True

    from src.personality import PersonalityDirector

    director = PersonalityDirector(
        MockMyCobot(),
        MockDexArm(),
        config.personality
    )

    print("Testing CAPTURE MODE (DexArm practice strokes)...")
    print("Running for 5 seconds...")
    director.start_capture_mode()
    time.sleep(5)
    director.stop_capture_mode()

    print("\nTesting DRAWING MODE (MyCobot curious tilts)...")
    print("Running for 10 seconds...")
    director.start_drawing_mode()
    time.sleep(10)
    director.stop_drawing_mode()

    print("\nPersonality test complete")
    return True


def test_pipeline(args):
    """Run full pipeline with mock or real components."""
    print("\n" + "=" * 50)
    print("FULL PIPELINE TEST")
    print("=" * 50)

    if args.mock:
        print("Mode: MOCK (no real hardware or API calls)")
    else:
        print("Mode: REAL (requires all hardware connected)")

    print("=" * 50 + "\n")

    # Import main system
    from src.main import PortraitSystem
    config = load_config()

    system = PortraitSystem(
        config,
        use_mock_ai=args.mock,
        enable_personality=not args.no_personality
    )

    if args.mock:
        print("NOTE: Mock mode will use camera but skip robot movements")
        print("      and use local edge detection instead of OpenAI API\n")

    try:
        if not system.initialize():
            print("ERROR: Failed to initialize system")
            return False

        print("\nSystem ready!")
        print("Press SPACEBAR to capture, 'q' to quit\n")

        if system.wait_for_trigger():
            system.run_portrait_pipeline()

    except KeyboardInterrupt:
        print("\nInterrupted")

    finally:
        system.shutdown()

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test components of the Robot Portrait System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        'component',
        choices=['camera', 'face', 'mycobot', 'dexarm', 'openai',
                 'contours', 'gcode', 'personality', 'pipeline'],
        help='Component to test'
    )
    parser.add_argument(
        '--port',
        help='Serial port override (for mycobot/dexarm tests)'
    )
    parser.add_argument(
        '--image',
        help='Image file to use (for openai/contours/gcode tests)'
    )
    parser.add_argument(
        '--mock',
        action='store_true',
        help='Use mock components (no API calls or hardware)'
    )
    parser.add_argument(
        '--draw-test',
        action='store_true',
        help='Actually draw a test pattern (dexarm test only)'
    )
    parser.add_argument(
        '--gcode-file',
        help='GCode file to run (dexarm test only)'
    )
    parser.add_argument(
        '--air-draw',
        choices=['square', 'circle', 'figure8', 'star', 'spiral'],
        help='Air drawing pattern to test (dexarm test only, no paper contact)'
    )
    parser.add_argument(
        '-y', '--yes',
        action='store_true',
        help='Skip confirmation prompts (dexarm test only)'
    )
    parser.add_argument(
        '--no-personality',
        action='store_true',
        help='Disable personality animations (pipeline test only)'
    )

    args = parser.parse_args()

    # Map component names to test functions
    tests = {
        'camera': test_camera,
        'face': test_face,
        'mycobot': test_mycobot,
        'dexarm': test_dexarm,
        'openai': test_openai,
        'contours': test_contours,
        'gcode': test_gcode,
        'personality': test_personality,
        'pipeline': test_pipeline,
    }

    # Run the test
    try:
        success = tests[args.component](args)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
