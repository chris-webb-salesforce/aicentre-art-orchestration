#!/usr/bin/env python3
"""
Compare different line detection methods for robot drawing.

Outputs visualization images for each method to help choose the best approach.

Usage:
    python scripts/compare_line_detection.py <image_path>
    python scripts/compare_line_detection.py <image_path> --output-dir ./comparisons
"""

import sys
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import logging

import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Contour:
    """Represents a drawable path."""
    points: List[Tuple[float, float]]
    is_closed: bool
    area: float
    length: float


def calculate_length(points: List[Tuple[float, float]]) -> float:
    """Calculate total path length."""
    length = 0.0
    for i in range(len(points) - 1):
        dx = points[i + 1][0] - points[i][0]
        dy = points[i + 1][1] - points[i][1]
        length += np.sqrt(dx * dx + dy * dy)
    return length


# =============================================================================
# Method 1: Current approach (Canny + Contours)
# =============================================================================
def detect_canny_contours(
    image: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    simplify_epsilon: float = 0.8
) -> List[Contour]:
    """Current method: Canny edge detection + contour finding."""

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Invert if background is white
    if np.mean(gray) > 127:
        gray = cv2.bitwise_not(gray)

    # Blur and edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)

    # Dilate to connect nearby lines
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours_cv, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contours = []
    for cv_contour in contours_cv:
        area = cv2.contourArea(cv_contour)
        if area < 100 and len(cv_contour) < 20:
            continue

        # Simplify
        simplified = cv2.approxPolyDP(cv_contour, simplify_epsilon, closed=False)
        points = [(float(p[0][0]), float(p[0][1])) for p in simplified]

        if len(points) < 2:
            continue

        length = calculate_length(points)

        # Check if closed
        is_closed = False
        if len(points) > 2:
            first, last = points[0], points[-1]
            dist = np.sqrt((first[0] - last[0])**2 + (first[1] - last[1])**2)
            is_closed = dist < 5

        contours.append(Contour(points=points, is_closed=is_closed, area=area, length=length))

    return contours


# =============================================================================
# Method 2: Skeletonization
# =============================================================================
def detect_skeleton(
    image: np.ndarray,
    simplify_epsilon: float = 1.0
) -> List[Contour]:
    """Skeletonization: reduces lines to single-pixel centerlines."""

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Invert if background is white (we need lines to be white)
    if np.mean(gray) > 127:
        gray = cv2.bitwise_not(gray)

    # Threshold to binary
    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # Skeletonize using morphological operations
    skeleton = np.zeros(binary.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    temp = binary.copy()
    while True:
        eroded = cv2.erode(temp, element)
        dilated = cv2.dilate(eroded, element)
        diff = cv2.subtract(temp, dilated)
        skeleton = cv2.bitwise_or(skeleton, diff)
        temp = eroded.copy()

        if cv2.countNonZero(temp) == 0:
            break

    # Find contours on skeleton
    contours_cv, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    contours = []
    for cv_contour in contours_cv:
        if len(cv_contour) < 5:
            continue

        # Simplify
        simplified = cv2.approxPolyDP(cv_contour, simplify_epsilon, closed=False)
        points = [(float(p[0][0]), float(p[0][1])) for p in simplified]

        if len(points) < 2:
            continue

        area = cv2.contourArea(cv_contour)
        length = calculate_length(points)

        is_closed = False
        if len(points) > 2:
            first, last = points[0], points[-1]
            dist = np.sqrt((first[0] - last[0])**2 + (first[1] - last[1])**2)
            is_closed = dist < 5

        contours.append(Contour(points=points, is_closed=is_closed, area=area, length=length))

    return contours


# =============================================================================
# Method 3: Line Segment Detector (LSD)
# =============================================================================
def detect_lsd(
    image: np.ndarray,
    min_line_length: float = 10.0
) -> List[Contour]:
    """LSD: Detects straight line segments accurately."""

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # LSD works on grayscale directly
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines, _, _, _ = lsd.detect(gray)

    contours = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            if length < min_line_length:
                continue

            points = [(float(x1), float(y1)), (float(x2), float(y2))]
            contours.append(Contour(points=points, is_closed=False, area=0, length=length))

    return contours


# =============================================================================
# Method 4: Hough Lines (Probabilistic)
# =============================================================================
def detect_hough_lines(
    image: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    min_line_length: int = 20,
    max_line_gap: int = 10
) -> List[Contour]:
    """Probabilistic Hough Transform for line detection."""

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Invert if needed
    if np.mean(gray) > 127:
        gray = cv2.bitwise_not(gray)

    # Edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)

    # Hough lines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    contours = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            points = [(float(x1), float(y1)), (float(x2), float(y2))]
            contours.append(Contour(points=points, is_closed=False, area=0, length=length))

    return contours


# =============================================================================
# Method 5: Adaptive Threshold + Contours
# =============================================================================
def detect_adaptive_threshold(
    image: np.ndarray,
    block_size: int = 11,
    c: int = 2,
    simplify_epsilon: float = 0.8
) -> List[Contour]:
    """Adaptive thresholding for varying lighting conditions."""

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Invert if background is white
    if np.mean(gray) > 127:
        gray = cv2.bitwise_not(gray)

    # Adaptive threshold
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, c
    )

    # Find contours
    contours_cv, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contours = []
    for cv_contour in contours_cv:
        area = cv2.contourArea(cv_contour)
        if area < 100 and len(cv_contour) < 20:
            continue

        simplified = cv2.approxPolyDP(cv_contour, simplify_epsilon, closed=False)
        points = [(float(p[0][0]), float(p[0][1])) for p in simplified]

        if len(points) < 2:
            continue

        length = calculate_length(points)

        is_closed = False
        if len(points) > 2:
            first, last = points[0], points[-1]
            dist = np.sqrt((first[0] - last[0])**2 + (first[1] - last[1])**2)
            is_closed = dist < 5

        contours.append(Contour(points=points, is_closed=is_closed, area=area, length=length))

    return contours


# =============================================================================
# Visualization
# =============================================================================
def visualize_contours(
    image: np.ndarray,
    contours: List[Contour],
    title: str = ""
) -> np.ndarray:
    """Draw contours on image with statistics."""

    # Create color image if needed
    if len(image.shape) == 2:
        viz = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        viz = image.copy()

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 255), (255, 128, 0), (0, 128, 255)
    ]

    total_points = 0
    total_length = 0.0

    for i, contour in enumerate(contours):
        color = colors[i % len(colors)]
        points = np.array(contour.points, dtype=np.int32)
        total_points += len(contour.points)
        total_length += contour.length

        # Draw path
        for j in range(len(points) - 1):
            cv2.line(viz, tuple(points[j]), tuple(points[j + 1]), color, 2)

        # Draw start point (green)
        cv2.circle(viz, tuple(points[0]), 4, (0, 255, 0), -1)

        # Draw end point (red)
        if len(points) > 1:
            cv2.circle(viz, tuple(points[-1]), 4, (0, 0, 255), -1)

    # Add statistics overlay
    stats = [
        f"{title}",
        f"Contours: {len(contours)}",
        f"Total points: {total_points}",
        f"Total length: {total_length:.0f}px",
        f"Avg pts/contour: {total_points/max(len(contours),1):.1f}"
    ]

    y_offset = 25
    for stat in stats:
        cv2.putText(viz, stat, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(viz, stat, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 22

    return viz


def create_comparison_grid(images: List[np.ndarray], cols: int = 3) -> np.ndarray:
    """Create a grid of comparison images (titles are embedded in each image)."""

    n = len(images)
    rows = (n + cols - 1) // cols

    # Get max dimensions
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)

    # Create grid
    grid = np.ones((rows * max_h, cols * max_w, 3), dtype=np.uint8) * 255

    for i, img in enumerate(images):
        row = i // cols
        col = i % cols

        # Ensure 3 channels
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Place image
        y1 = row * max_h
        x1 = col * max_w
        grid[y1:y1+img.shape[0], x1:x1+img.shape[1]] = img

    return grid


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Compare line detection methods")
    parser.add_argument("image_path", help="Path to input image (line art)")
    parser.add_argument("--output-dir", "-o", default=".", help="Output directory for visualizations")
    parser.add_argument("--simplify-epsilon", "-e", type=float, default=0.8, help="Simplification epsilon")
    args = parser.parse_args()

    # Load image
    image_path = Path(args.image_path)
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        sys.exit(1)

    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        sys.exit(1)

    logger.info(f"Loaded image: {image_path} ({image.shape[1]}x{image.shape[0]})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import adaptive extractor
    try:
        from src.vision.adaptive_extractor import AdaptiveContourExtractor, AdaptiveExtractorConfig
        has_adaptive = True
    except ImportError:
        has_adaptive = False
        logger.warning("Could not import adaptive extractor")

    # Run all detection methods
    methods = [
        ("1_canny_contours", "Canny + Contours (current)",
         lambda: detect_canny_contours(image, simplify_epsilon=args.simplify_epsilon)),

        ("2_skeleton", "Skeletonization",
         lambda: detect_skeleton(image, simplify_epsilon=args.simplify_epsilon)),

        ("3_lsd", "Line Segment Detector",
         lambda: detect_lsd(image)),

        ("4_hough", "Hough Lines",
         lambda: detect_hough_lines(image)),

        ("5_adaptive_thresh", "Adaptive Threshold",
         lambda: detect_adaptive_threshold(image, simplify_epsilon=args.simplify_epsilon)),
    ]

    # Add smart adaptive methods if available
    if has_adaptive:
        methods.extend([
            ("6_smart_adaptive", "Smart Adaptive (auto)",
             lambda: AdaptiveContourExtractor(AdaptiveExtractorConfig(
                 method="adaptive",
                 simplify_epsilon=args.simplify_epsilon
             )).extract(image)),

            ("7_smart_hybrid", "Smart Hybrid (both)",
             lambda: AdaptiveContourExtractor(AdaptiveExtractorConfig(
                 method="hybrid",
                 simplify_epsilon=args.simplify_epsilon
             )).extract(image)),
        ])

    visualizations = []

    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON OF LINE DETECTION METHODS")
    logger.info("=" * 60)

    for filename, title, detect_func in methods:
        logger.info(f"\nRunning: {title}...")

        try:
            contours = detect_func()

            total_points = sum(len(c.points) for c in contours)
            total_length = sum(c.length for c in contours)

            logger.info(f"  Contours: {len(contours)}")
            logger.info(f"  Total points: {total_points}")
            logger.info(f"  Total length: {total_length:.0f}px")

            # Create visualization
            viz = visualize_contours(image.copy(), contours, title)

            # Save individual file
            output_path = output_dir / f"contours_viz_{filename}.png"
            cv2.imwrite(str(output_path), viz)
            logger.info(f"  Saved: {output_path}")

            visualizations.append(viz)

        except Exception as e:
            logger.error(f"  Failed: {e}")
            import traceback
            traceback.print_exc()

    # Create comparison grid
    if visualizations:
        logger.info("\nCreating comparison grid...")
        grid = create_comparison_grid(visualizations, cols=3)
        grid_path = output_dir / "contours_viz_comparison.png"
        cv2.imwrite(str(grid_path), grid)
        logger.info(f"Saved comparison grid: {grid_path}")

    logger.info("\n" + "=" * 60)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 60)
    logger.info("""
For smoother robot drawing:
- Skeleton: Best for single-stroke paths, reduces parallel edges
- Canny + Contours: Good general purpose, adjustable epsilon
- LSD/Hough: Best for geometric drawings with straight lines

Look for:
- Fewer total points = faster, potentially smoother drawing
- Single paths instead of parallel edges (skeleton wins here)
- Continuous paths vs fragmented segments
""")


if __name__ == "__main__":
    main()
