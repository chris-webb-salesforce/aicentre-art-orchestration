#!/usr/bin/env python3
"""
Webcam to DexArm Pipeline

Captures photo from webcam, processes through OpenAI (or local edge detection),
generates GCode, and sends to DexArm for drawing.

Usage:
    python scripts/webcam_to_dexarm.py                    # Full pipeline with OpenAI
    python scripts/webcam_to_dexarm.py --mock             # Use local edge detection instead of OpenAI
    python scripts/webcam_to_dexarm.py --preview-only     # Preview without drawing
    python scripts/webcam_to_dexarm.py -y                 # Skip confirmations
    python scripts/webcam_to_dexarm.py -d skeleton        # Use skeleton line detection
    python scripts/webcam_to_dexarm.py -d adaptive        # Use smart adaptive detection

Line detection methods (-d/--detection):
    canny    - Traditional edge detection (detects edges of lines)
    skeleton - Skeletonization (finds centerline of thick lines)
    adaptive - Smart auto (uses skeleton for thick areas, canny for thin)
    hybrid   - Runs both methods and merges results
"""

import sys
import os
import argparse
import cv2
import threading

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, '.env'))

from src.config import load_config
from src.hardware.camera_controller import CameraController
from src.vision.face_tracker import FaceTracker
from src.vision.contour_extractor import ContourExtractor
from src.vision.adaptive_extractor import AdaptiveContourExtractor, AdaptiveExtractorConfig
from src.planning.gcode_generator import GCodeGenerator, DrawingBounds


# Configuration
DEXARM_PORT = "/dev/tty.usbmodem3187378532331"


def capture_photo(config, args):
    """Open webcam and auto-capture after face detected for 2 seconds."""
    import time

    camera = CameraController(
        index=args.camera or config.camera.index,
        width=config.camera.width,
        height=config.camera.height
    )

    if not camera.initialize():
        print("ERROR: Failed to initialize camera")
        return None

    tracker = FaceTracker(min_face_size=config.face_tracking.min_face_size)
    tracker.initialize()

    print("\nCamera ready!")
    print("Position your face in the frame")
    print("Auto-capture after 2 seconds of face detection (Q to quit)\n")

    captured_image = None
    last_face = None
    face_detected_start = None
    AUTO_CAPTURE_DELAY = 2.0  # seconds

    while True:
        frame = camera.get_frame()
        if frame is None:
            continue

        # Show face detection
        face = tracker.detect_face(frame)
        if face:
            last_face = face
            if face_detected_start is None:
                face_detected_start = time.time()
        else:
            face_detected_start = None

        display = tracker.draw_detection(frame.copy(), face)

        # Check for auto-capture
        should_capture = False
        if face and face_detected_start:
            elapsed = time.time() - face_detected_start
            remaining = AUTO_CAPTURE_DELAY - elapsed
            if remaining <= 0:
                should_capture = True
            else:
                # Show countdown
                cv2.putText(display, f"Capturing in {remaining:.1f}s...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif face:
            cv2.putText(display, "Face detected - hold still", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, "No face - position yourself", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Webcam - Auto-capture", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') or should_capture:
            if last_face is None:
                print("No face detected! Position your face and try again.")
                continue

            # Capture high-res image
            print("Capturing...")
            captured_image = camera.capture_high_res(
                config.camera.capture_width,
                config.camera.capture_height
            )
            if captured_image is None:
                captured_image = frame.copy()

            # Detect face in high-res image for cropping
            hi_res_face = tracker.detect_face(captured_image)
            if hi_res_face is None:
                # Scale up the face coordinates from preview
                scale_x = captured_image.shape[1] / frame.shape[1]
                scale_y = captured_image.shape[0] / frame.shape[0]
                # Create a simple object with the scaled coordinates
                class ScaledFace:
                    pass
                hi_res_face = ScaledFace()
                hi_res_face.x = int(last_face.x * scale_x)
                hi_res_face.y = int(last_face.y * scale_y)
                hi_res_face.width = int(last_face.width * scale_x)
                hi_res_face.height = int(last_face.height * scale_y)

            # Crop to face with padding
            captured_image = crop_to_face(captured_image, hi_res_face)
            break
        elif key == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

    return captured_image


def crop_to_face(image, face, padding=0.5):
    """Crop image to face region with padding."""
    h, w = image.shape[:2]

    # Get face bounds (FaceDetection dataclass uses .width/.height)
    fx, fy, fw, fh = face.x, face.y, face.width, face.height

    # Add padding around face (percentage of face size)
    pad_x = int(fw * padding)
    pad_y = int(fh * padding)

    # Calculate crop bounds
    x1 = max(0, fx - pad_x)
    y1 = max(0, fy - pad_y)
    x2 = min(w, fx + fw + pad_x)
    y2 = min(h, fy + fh + pad_y)

    # Make it square (use the larger dimension)
    crop_w = x2 - x1
    crop_h = y2 - y1
    size = max(crop_w, crop_h)

    # Center the square crop on the face
    center_x = fx + fw // 2
    center_y = fy + fh // 2

    x1 = max(0, center_x - size // 2)
    y1 = max(0, center_y - size // 2)
    x2 = min(w, x1 + size)
    y2 = min(h, y1 + size)

    # Adjust if we hit image bounds
    if x2 - x1 < size:
        x1 = max(0, x2 - size)
    if y2 - y1 < size:
        y1 = max(0, y2 - size)

    # Crop
    cropped = image[y1:y2, x1:x2]

    print(f"Cropped to face: {cropped.shape[1]}x{cropped.shape[0]} (from {w}x{h})")

    return cropped


def process_image(image, config, args, style=None):
    """Convert photo to line art using OpenAI or local edge detection."""

    if args.mock:
        print("Using local edge detection (mock mode)...")
        # Simple edge detection as fallback
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)
        line_art = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        # Invert so lines are dark on white
        line_art = 255 - line_art
    else:
        from src.ai.openai_client import OpenAIClient

        # Use style prompt if available
        prompt = style.prompt if style else "Transform this face into a minimalist line drawing with black lines on white background."

        # Resolve reference image path
        reference_image = None
        if hasattr(config.openai, 'reference_image') and config.openai.reference_image:
            reference_image = config.openai.reference_image
            if not os.path.isabs(reference_image):
                reference_image = os.path.join(project_root, reference_image)

        client = OpenAIClient(
            model=config.openai.model,
            prompt=prompt,
            size=config.openai.size,
            max_retries=config.openai.max_retries,
            retry_delay=config.openai.retry_delay,
            reference_image=reference_image
        )

        if not client.initialize():
            return None

        line_art = client.generate_line_art(image, "output/line_art.png")
        if line_art is None:
            print("ERROR: Failed to generate line art")
            return None

    # Save for reference
    os.makedirs("output", exist_ok=True)
    cv2.imwrite("output/captured_photo.jpg", image)
    cv2.imwrite("output/line_art.png", line_art)
    print("Saved: output/captured_photo.jpg, output/line_art.png")

    return line_art


def extract_and_generate_gcode(line_art, config, args, style=None):
    """Extract contours and generate GCode. Returns (gcode, contours, drawing_bounds).

    If a style with contour settings is provided, those settings override the base config.
    """

    # Helper to get value with style override
    def get_contour_setting(name, default=None):
        """Get contour setting, preferring style override if available."""
        # Check style contour override first
        if style and style.contour:
            style_val = getattr(style.contour, name, None)
            if style_val is not None:
                return style_val
        # Fall back to base config
        base_val = getattr(config.contour, name, default)
        return base_val if base_val is not None else default

    # Choose extraction method based on --detection argument, then style, then base config
    if getattr(args, 'detection', None):
        detection_method = args.detection
    elif style and style.contour and style.contour.method:
        detection_method = style.contour.method
    else:
        detection_method = config.contour.method

    print(f"Using line detection method: {detection_method}")
    if style:
        print(f"Using contour settings from style: {style.name}")
        if style.contour:
            print(f"  Style overrides: method={style.contour.method}, min_length={style.contour.min_length}, merge_enabled={style.contour.merge_enabled}")
        else:
            print(f"  No contour overrides in style")

    # Debug: show key settings being used
    print(f"  Effective settings: min_length={get_contour_setting('min_length', 10.0)}, merge_enabled={get_contour_setting('merge_enabled', True)}, merge_distance={get_contour_setting('merge_distance', 5.0)}")

    # Always use AdaptiveContourExtractor - it supports all methods and has
    # min_length filtering and merge_nearby_contours
    adaptive_config = AdaptiveExtractorConfig(
        method=detection_method,
        canny_low=get_contour_setting('canny_low', 30),
        canny_high=get_contour_setting('canny_high', 100),
        min_area=get_contour_setting('min_area', 50),
        simplify_epsilon=get_contour_setting('simplify_epsilon', 0.8),
        blur_kernel=get_contour_setting('blur_kernel', 3),
        min_contour_points=get_contour_setting('min_contour_points', 5),
        thickness_threshold=get_contour_setting('thickness_threshold', 3),
        density_threshold=get_contour_setting('density_threshold', 0.3),
        skeleton_simplify=get_contour_setting('skeleton_simplify', 1.0),
        # Speed optimizations
        min_length=get_contour_setting('min_length', 10.0),
        merge_distance=get_contour_setting('merge_distance', 5.0),
        merge_enabled=get_contour_setting('merge_enabled', True),
        # Region-aware processing
        region_aware=get_contour_setting('region_aware', False),
        detail_simplify_epsilon=get_contour_setting('detail_simplify_epsilon', 0.3),
        detail_min_length=get_contour_setting('detail_min_length', 3.0),
        detail_min_area=get_contour_setting('detail_min_area', 10),
        detail_region_padding=get_contour_setting('detail_region_padding', 20)
    )
    extractor = AdaptiveContourExtractor(adaptive_config)

    print("Extracting contours...")
    contours = extractor.extract(line_art)
    print(f"Found {len(contours)} contours")

    if not contours:
        print("ERROR: No contours found in image")
        return None, None, None

    # Get bounds and optimize path
    image_bounds = extractor.get_bounds(contours)
    if config.path_optimization.enabled:
        print("Optimizing path order...")
        contours = extractor.optimize_order(contours)

    # Save visualization
    viz = extractor.visualize(line_art.copy(), contours)
    cv2.imwrite("output/contours_viz.png", viz)
    print("Saved: output/contours_viz.png")

    # Generate GCode
    drawing_bounds = DrawingBounds(
        x_min=config.drawing.x_min,
        x_max=config.drawing.x_max,
        y_min=config.drawing.y_min,
        y_max=config.drawing.y_max,
        z_up=config.drawing.z_up,
        z_down=config.drawing.z_down,
        feedrate=config.dexarm.feedrate,
        travel_feedrate=config.dexarm.travel_feedrate,
        flip_x=config.drawing.flip_x,
        flip_y=config.drawing.flip_y
    )

    generator = GCodeGenerator(drawing_bounds)

    print("Generating GCode...")
    gcode = generator.generate(contours, image_bounds)
    print(f"Generated {len(gcode)} lines")

    # Estimate time
    est_time = generator.estimate_drawing_time(gcode)
    print(f"Estimated drawing time: {est_time:.1f} seconds")

    # Save GCode
    generator.save_to_file(gcode, "output/portrait.gcode")
    # Also save as test_output.gcode for Stream Deck compatibility
    generator.save_to_file(gcode, "test_output.gcode")
    print("Saved: output/portrait.gcode, test_output.gcode")

    return gcode, contours, drawing_bounds


def generate_logo_gcode(config):
    """Generate GCode for the logo. Returns gcode list or None if disabled/not found.

    Supports both .gcode files (loaded directly) and image files (processed for contours).
    """

    # Check if logo is enabled
    if not hasattr(config, 'logo') or not config.logo.enabled:
        print("Logo drawing disabled or not configured")
        return None

    # Resolve path relative to project root
    logo_path = config.logo.path
    if not os.path.isabs(logo_path):
        logo_path = os.path.join(project_root, logo_path)

    if not os.path.exists(logo_path):
        print(f"Logo file not found: {logo_path}")
        return None

    print(f"\nProcessing logo: {logo_path}")

    # If it's a .gcode file, load it directly
    if logo_path.lower().endswith('.gcode'):
        with open(logo_path, 'r') as f:
            gcode = [line.strip() for line in f if line.strip()]
        print(f"Logo GCode loaded: {len(gcode)} lines")
        return gcode

    # Otherwise, process as image
    logo_img = cv2.imread(logo_path)
    if logo_img is None:
        print(f"ERROR: Could not load logo image: {logo_path}")
        return None

    # Extract contours from logo
    extractor = ContourExtractor(
        canny_low=30,  # Lower threshold for logos
        canny_high=100,
        min_area=50,
        simplify_epsilon=1.0
    )

    contours = extractor.extract(logo_img)
    print(f"Logo: found {len(contours)} contours")

    if not contours:
        print("WARNING: No contours found in logo")
        return None

    # Get image bounds and optimize
    image_bounds = extractor.get_bounds(contours)
    contours = extractor.optimize_order(contours)

    # Generate GCode for logo area
    logo_bounds = DrawingBounds(
        x_min=config.logo.x_min,
        x_max=config.logo.x_max,
        y_min=config.logo.y_min,
        y_max=config.logo.y_max,
        z_up=config.drawing.z_up,
        z_down=config.drawing.z_down,
        feedrate=config.dexarm.feedrate,
        travel_feedrate=config.dexarm.travel_feedrate,
        flip_x=config.drawing.flip_x,
        flip_y=config.drawing.flip_y
    )

    generator = GCodeGenerator(logo_bounds)
    gcode = generator.generate(contours, image_bounds)

    print(f"Logo GCode: {len(gcode)} lines")

    return gcode


def play_music(music_file):
    """Start playing music in background, return process handle."""
    import subprocess
    if music_file and os.path.exists(music_file):
        print(f"Playing music: {music_file}")
        # Use afplay on macOS (plays in background)
        return subprocess.Popen(['afplay', music_file],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
    return None


def stop_music(music_process):
    """Stop music playback."""
    if music_process:
        music_process.terminate()
        music_process.wait()


def draw_logo_on_dexarm(config, args):
    """Draw the logo on DexArm. Returns the DexArmController for reuse, or None on failure."""
    from src.hardware.dexarm_controller import DexArmController

    logo_gcode = generate_logo_gcode(config)
    if not logo_gcode:
        return None

    port = args.port or DEXARM_PORT
    print(f"\nConnecting to DexArm on {port} for logo...")

    arm = DexArmController(
        port=port,
        baud_rate=config.dexarm.baud_rate,
        feedrate=config.dexarm.feedrate,
        travel_feedrate=config.dexarm.travel_feedrate,
        z_up=config.drawing.z_up,
        z_down=config.drawing.z_down,
        acceleration=config.dexarm.acceleration,
        travel_acceleration=config.dexarm.travel_acceleration,
        jerk=getattr(config.dexarm, 'jerk', 5.0)
    )

    if not arm.initialize():
        print("ERROR: Failed to initialize DexArm for logo")
        return None

    print("DexArm connected! Drawing logo...")

    def logo_progress(current, total, pos, contour=None, total_contours=None):
        if current % 50 == 0 or current == total:
            pct = 100 * current / total if total > 0 else 0
            if contour and total_contours:
                print(f"  Logo: contour {contour}/{total_contours} ({pct:.0f}%)")
            else:
                print(f"  Logo: {current}/{total} ({pct:.0f}%)")

    arm.stream_gcode(logo_gcode, logo_progress)
    print("Logo complete!")

    return arm


def send_to_dexarm(gcode, config, args, style_name=None, arm=None):
    """Stream GCode to DexArm. Can reuse existing arm connection."""
    from src.hardware.dexarm_controller import DexArmController

    # Reuse existing arm connection or create new one
    if arm is None:
        port = args.port or DEXARM_PORT
        print(f"\nConnecting to DexArm on {port}...")

        arm = DexArmController(
            port=port,
            baud_rate=config.dexarm.baud_rate,
            feedrate=config.dexarm.feedrate,
            travel_feedrate=config.dexarm.travel_feedrate,
            z_up=config.drawing.z_up,
            z_down=config.drawing.z_down,
            acceleration=config.dexarm.acceleration,
            travel_acceleration=config.dexarm.travel_acceleration,
            jerk=getattr(config.dexarm, 'jerk', 5.0)
        )

        if not arm.initialize():
            print("ERROR: Failed to initialize DexArm")
            return False

        print("DexArm connected!")

        if not args.yes:
            print("\nWARNING: This will start drawing (pen will touch paper!)")
            input("Press ENTER to start or Ctrl+C to cancel...")
    else:
        print("\nUsing existing DexArm connection for portrait...")

    last_contour = [0]  # Use list to allow modification in nested function

    def progress(current, total, _pos, contour=None, total_contours=None):
        pct = 100 * current / total if total > 0 else 0

        # Show progress when contour changes or at regular intervals
        contour_changed = contour and contour != last_contour[0]
        at_interval = current % 20 == 0 or current == total

        if contour_changed or at_interval:
            if contour and total_contours:
                contour_pct = 100 * contour / total_contours
                print(f"  Drawing contour {contour}/{total_contours} ({contour_pct:.0f}%) - {pct:.0f}% complete")
                last_contour[0] = contour
            else:
                print(f"  Progress: {pct:.0f}%")

    try:
        success = arm.stream_gcode(gcode, progress)
        if success:
            print("\nDrawing complete!")
        else:
            print("\nDrawing failed!")
    finally:
        arm.release()

    return success


def main():
    parser = argparse.ArgumentParser(
        description="Webcam to DexArm portrait pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--camera', type=int, help='Camera index (default: from config)')
    parser.add_argument('--port', help='DexArm serial port')
    parser.add_argument('--image', help='Path to existing line art PNG (skips capture and AI processing)')
    parser.add_argument('--no-logo', action='store_true', help='Skip drawing the logo')
    parser.add_argument('--mock', action='store_true', help='Use local edge detection instead of OpenAI')
    parser.add_argument('--preview-only', action='store_true', help='Generate GCode but do not draw')
    parser.add_argument('-y', '--yes', action='store_true', help='Skip confirmation prompts')
    parser.add_argument('-s', '--style',
                        choices=['minimal', 'vangogh', 'ghibli', 'picasso', 'sketch', 'caricature', 'geometric', 'contour', 'simple'],
                        default=None,
                        help='Art style (default: minimal). Options: minimal, vangogh, ghibli, picasso, sketch, caricature, geometric, contour')
    parser.add_argument('--music', help='Audio file to play while drawing (mp3, wav, etc.)')
    parser.add_argument('-d', '--detection',
                        choices=['canny', 'skeleton', 'adaptive', 'hybrid'],
                        default=None,
                        help='Line detection method (default: from config). '
                             'canny=edges, skeleton=centerlines, adaptive=smart auto, hybrid=both merged')

    args = parser.parse_args()
    config = load_config()

    # Start music immediately if specified
    music_process = None
    if hasattr(args, 'music') and args.music:
        music_process = play_music(args.music)

    try:
        # If --image provided, skip capture and AI processing
        if args.image:
            print(f"\nLoading line art from: {args.image}")
            line_art = cv2.imread(args.image)
            if line_art is None:
                print(f"ERROR: Could not load image: {args.image}")
                return 1
            print(f"Loaded image: {line_art.shape[1]}x{line_art.shape[0]}")

            # Still allow using style's contour settings with --image
            style_name = args.style or config.openai.default_style
            if hasattr(config.openai, 'styles') and style_name in config.openai.styles:
                style = config.openai.styles[style_name]
                print(f"Using contour settings from style: {style.name}")
            else:
                style = None
                style_name = "custom"

            # Draw logo while we generate GCode (if not preview-only)
            arm = None
            if not args.preview_only:
                if not args.yes:
                    print("\nWARNING: This will start drawing (pen will touch paper!)")
                    input("Press ENTER to start or Ctrl+C to cancel...")
                arm = None if args.no_logo else draw_logo_on_dexarm(config, args)
        else:
            # Full pipeline: capture photo and process with AI
            # Get selected style
            style_name = args.style or config.openai.default_style
            if hasattr(config.openai, 'styles') and style_name in config.openai.styles:
                style = config.openai.styles[style_name]
                print(f"\nArt style: {style.name}")
            else:
                print(f"\nUsing default style: {style_name}")
                style = None

            image = capture_photo(config, args)
            if image is None:
                print("No image captured, exiting.")
                return 1

            # Run OpenAI processing in background thread while logo draws
            line_art_result = [None]  # Use list to allow modification in thread
            processing_error = [None]

            def process_in_background():
                try:
                    line_art_result[0] = process_image(image, config, args, style)
                except Exception as e:
                    processing_error[0] = e

            # Start image processing in background
            processing_thread = threading.Thread(target=process_in_background)
            processing_thread.start()

            # Draw logo while image processes (if not preview-only)
            arm = None
            if not args.preview_only:
                if not args.yes:
                    print("\nWARNING: This will start drawing (pen will touch paper!)")
                    input("Press ENTER to start or Ctrl+C to cancel...")
                arm = None if args.no_logo else draw_logo_on_dexarm(config, args)

            # Wait for image processing to complete
            print("\nWaiting for AI processing to complete...")
            processing_thread.join()

            if processing_error[0]:
                raise processing_error[0]

            line_art = line_art_result[0]
            if line_art is None:
                return 1

        gcode, _, _ = extract_and_generate_gcode(line_art, config, args, style=style)
        if gcode is None:
            return 1

        if args.preview_only:
            print("\nPreview only mode - skipping drawing")
            print("GCode saved to: output/portrait.gcode")
            return 0

        success = send_to_dexarm(
            gcode, config, args,
            style_name=style.name if style else None,
            arm=arm  # Reuse connection from logo drawing
        )

        return 0 if success else 1

    finally:
        # Stop music when script ends
        stop_music(music_process)


if __name__ == "__main__":
    sys.exit(main())
