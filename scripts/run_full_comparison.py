#!/usr/bin/env python3
"""
Comprehensive comparison of line art generation and detection methods.

Compares:
1. Different line art sources (mock edge detection, existing OpenAI output)
2. Different line detection methods (canny, skeleton, adaptive, hybrid)

Usage:
    python scripts/run_full_comparison.py output/captured_photo.jpg
    python scripts/run_full_comparison.py output/line_art.png --is-line-art
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import logging

import cv2
import numpy as np

# Setup project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from src.config import load_config
from src.vision.contour_extractor import ContourExtractor, Contour
from src.vision.adaptive_extractor import AdaptiveContourExtractor, AdaptiveExtractorConfig

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def create_mock_line_art(image: np.ndarray) -> np.ndarray:
    """Create line art using local edge detection (no OpenAI)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    line_art = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    line_art = 255 - line_art  # Invert so lines are dark on white
    return line_art


def extract_with_method(image: np.ndarray, method: str, config) -> Tuple[List[Contour], str]:
    """Extract contours using specified method."""

    if method == "canny":
        extractor = ContourExtractor(
            canny_low=config.contour.canny_low,
            canny_high=config.contour.canny_high,
            min_area=config.contour.min_area,
            simplify_epsilon=config.contour.simplify_epsilon,
            blur_kernel=config.contour.blur_kernel,
            min_contour_points=config.contour.min_contour_points
        )
        contours = extractor.extract(image)
        description = f"Canny Edge Detection (low={config.contour.canny_low}, high={config.contour.canny_high})"
    else:
        adaptive_config = AdaptiveExtractorConfig(
            method=method,
            canny_low=config.contour.canny_low,
            canny_high=config.contour.canny_high,
            min_area=config.contour.min_area,
            simplify_epsilon=config.contour.simplify_epsilon,
            blur_kernel=config.contour.blur_kernel,
            min_contour_points=config.contour.min_contour_points,
            thickness_threshold=config.contour.thickness_threshold,
            density_threshold=config.contour.density_threshold,
            skeleton_simplify=config.contour.skeleton_simplify,
            min_length=config.contour.min_length,
            merge_distance=config.contour.merge_distance,
            merge_enabled=config.contour.merge_enabled,
        )
        extractor = AdaptiveContourExtractor(adaptive_config)
        contours = extractor.extract(image)

        descriptions = {
            "skeleton": "Skeletonization (centerline extraction)",
            "adaptive": "Adaptive (auto-selects per region)",
            "hybrid": "Hybrid (runs both, merges results)"
        }
        description = descriptions.get(method, method)

    return contours, description


def visualize_contours(image: np.ndarray, contours: List[Contour], title: str, subtitle: str = "") -> np.ndarray:
    """Draw contours on a white background with statistics."""

    h, w = image.shape[:2]
    viz = np.ones((h, w, 3), dtype=np.uint8) * 255

    colors = [
        (255, 0, 0), (0, 200, 0), (0, 0, 255),
        (255, 128, 0), (128, 0, 255), (0, 200, 200),
        (200, 0, 128), (0, 128, 200), (128, 128, 0)
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
            cv2.line(viz, tuple(points[j]), tuple(points[j + 1]), color, 1)

    # Add title and stats
    lines = [
        title,
        subtitle if subtitle else "",
        f"Contours: {len(contours)} | Points: {total_points}",
        f"Est. drawing time factor: {total_points / 100:.1f}x"
    ]

    y_offset = 20
    for line in lines:
        if line:
            cv2.putText(viz, line, (8, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
            cv2.putText(viz, line, (8, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
            y_offset += 18

    return viz


def create_comparison_grid(images: List[Tuple[str, np.ndarray]], cols: int = 4) -> np.ndarray:
    """Create a labeled grid of comparison images."""

    if not images:
        return np.ones((100, 100, 3), dtype=np.uint8) * 255

    n = len(images)
    rows = (n + cols - 1) // cols

    # Get max dimensions
    max_h = max(img.shape[0] for _, img in images)
    max_w = max(img.shape[1] for _, img in images)

    # Scale down if images are too large
    scale = 1.0
    if max_w > 400:
        scale = 400 / max_w
        max_w = 400
        max_h = int(max_h * scale)

    # Create grid with padding
    padding = 5
    grid_h = rows * (max_h + padding) + padding
    grid_w = cols * (max_w + padding) + padding
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 240  # Light gray background

    for i, (label, img) in enumerate(images):
        row = i // cols
        col = i % cols

        # Resize if needed
        if scale != 1.0:
            new_h = int(img.shape[0] * scale)
            new_w = int(img.shape[1] * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Ensure 3 channels
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Place image
        y1 = row * (max_h + padding) + padding
        x1 = col * (max_w + padding) + padding

        # Center smaller images
        y_offset = (max_h - img.shape[0]) // 2
        x_offset = (max_w - img.shape[1]) // 2

        grid[y1 + y_offset:y1 + y_offset + img.shape[0],
             x1 + x_offset:x1 + x_offset + img.shape[1]] = img

    return grid


def main():
    parser = argparse.ArgumentParser(description="Compare line art generation methods")
    parser.add_argument("image_path", help="Path to input photo or line art")
    parser.add_argument("--is-line-art", action="store_true", help="Input is already line art (skip mock generation)")
    parser.add_argument("--output-dir", "-o", default="comparisons", help="Output directory")
    args = parser.parse_args()

    # Validate input
    image_path = Path(args.image_path)
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        return 1

    # Load config
    config = load_config()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load input image
    original = cv2.imread(str(image_path))
    if original is None:
        logger.error(f"Failed to load image: {image_path}")
        return 1

    logger.info(f"Loaded: {image_path} ({original.shape[1]}x{original.shape[0]})")

    # =========================================================================
    # Part 1: Generate different line art sources
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PART 1: LINE ART SOURCES")
    logger.info("=" * 60)

    line_arts: Dict[str, np.ndarray] = {}

    if args.is_line_art:
        # Input is already line art
        line_arts["input"] = original
        logger.info(f"Using input as line art: {image_path.name}")
    else:
        # Generate mock line art from photo
        logger.info("\n[Mock] Local edge detection...")
        line_arts["mock"] = create_mock_line_art(original)
        cv2.imwrite(str(output_dir / "lineart_mock.png"), line_arts["mock"])
        logger.info("  Saved: lineart_mock.png")

    # Check for existing OpenAI line art
    existing_line_art = Path("output/line_art.png")
    if existing_line_art.exists() and not args.is_line_art:
        logger.info("\n[Existing] Loading existing OpenAI line art...")
        line_arts["openai"] = cv2.imread(str(existing_line_art))
        logger.info(f"  Loaded: {existing_line_art}")

    # =========================================================================
    # Part 2: Compare detection methods on each line art source
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PART 2: DETECTION METHOD COMPARISON")
    logger.info("=" * 60)

    detection_methods = ["canny", "skeleton", "adaptive", "hybrid"]
    all_comparisons = []

    for source_name, line_art in line_arts.items():
        logger.info(f"\n--- Source: {source_name} ---")

        for method in detection_methods:
            logger.info(f"  [{method}] Extracting contours...")

            try:
                contours, description = extract_with_method(line_art, method, config)

                total_points = sum(len(c.points) for c in contours)
                logger.info(f"    Contours: {len(contours)}, Points: {total_points}")

                # Create visualization
                viz = visualize_contours(
                    line_art,
                    contours,
                    f"{source_name.upper()} + {method.upper()}",
                    description
                )

                # Save individual file
                filename = f"detection_{source_name}_{method}.png"
                cv2.imwrite(str(output_dir / filename), viz)

                all_comparisons.append((f"{source_name}_{method}", viz))

            except Exception as e:
                logger.error(f"    Failed: {e}")
                import traceback
                traceback.print_exc()

    # =========================================================================
    # Part 3: Create comparison grids
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PART 3: CREATING COMPARISON GRIDS")
    logger.info("=" * 60)

    # Grid for each source
    for source_name in line_arts.keys():
        source_comparisons = [(name, img) for name, img in all_comparisons if name.startswith(f"{source_name}_")]
        if source_comparisons:
            grid = create_comparison_grid(source_comparisons, cols=2)
            grid_filename = f"grid_{source_name}_methods.png"
            cv2.imwrite(str(output_dir / grid_filename), grid)
            logger.info(f"Saved: {grid_filename}")

    # Combined grid of all comparisons
    if all_comparisons:
        grid_all = create_comparison_grid(all_comparisons, cols=4)
        cv2.imwrite(str(output_dir / "grid_all_comparisons.png"), grid_all)
        logger.info("Saved: grid_all_comparisons.png")

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"\nOutput directory: {output_dir.absolute()}")
    logger.info(f"Total comparisons generated: {len(all_comparisons)}")
    logger.info("\nFiles created:")
    for f in sorted(output_dir.glob("*.png")):
        logger.info(f"  - {f.name}")

    logger.info("\n" + "=" * 60)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 60)
    logger.info("""
Detection Methods:
  - canny:     Good general purpose, may create parallel edges
  - skeleton:  Best for single-stroke paths, reduces duplicates
  - adaptive:  Auto-selects method per region (smart)
  - hybrid:    Runs both and merges (most detail, most points)

For fastest drawing: skeleton + merge_enabled
For most detail: hybrid
For balanced results: adaptive
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
