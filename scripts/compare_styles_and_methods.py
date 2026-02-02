#!/usr/bin/env python3
"""
Generate line art for each OpenAI style and compare detection methods.

This script:
1. Takes a photo as input
2. Generates line art using each OpenAI style
3. Runs all detection methods on each style's output
4. Creates comparison grids to identify the best algorithm for each style

Usage:
    python scripts/compare_styles_and_methods.py output/captured_photo.jpg
    python scripts/compare_styles_and_methods.py output/captured_photo.jpg --styles minimal vangogh ghibli
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import logging
import time

import cv2
import numpy as np

# Setup project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from dotenv import load_dotenv
load_dotenv(project_root / '.env')

from src.config import load_config
from src.vision.contour_extractor import ContourExtractor, Contour
from src.vision.adaptive_extractor import AdaptiveContourExtractor, AdaptiveExtractorConfig
from src.ai.openai_client import OpenAIClient

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def generate_line_art_for_style(image: np.ndarray, style_name: str, style_config, openai_config, output_path: str) -> np.ndarray:
    """Generate line art using OpenAI with a specific style."""

    client = OpenAIClient(
        model=openai_config.model,
        prompt=style_config.prompt,
        size=openai_config.size,
        max_retries=openai_config.max_retries,
        retry_delay=openai_config.retry_delay
    )

    if not client.initialize():
        logger.error(f"Failed to initialize OpenAI client for style: {style_name}")
        return None

    line_art = client.generate_line_art(image, output_path)
    return line_art


def extract_with_method(image: np.ndarray, method: str, config, style_contour_config=None) -> Tuple[List[Contour], str]:
    """Extract contours using specified method, with optional style-specific overrides."""

    # Helper to get config value with style override
    def get_val(name, default):
        if style_contour_config and hasattr(style_contour_config, name):
            val = getattr(style_contour_config, name)
            if val is not None:
                return val
        return getattr(config.contour, name, default)

    if method == "canny":
        extractor = ContourExtractor(
            canny_low=get_val('canny_low', 30),
            canny_high=get_val('canny_high', 100),
            min_area=get_val('min_area', 50),
            simplify_epsilon=get_val('simplify_epsilon', 0.8),
            blur_kernel=get_val('blur_kernel', 3),
            min_contour_points=get_val('min_contour_points', 5)
        )
        contours = extractor.extract(image)
        description = "Canny Edge Detection"
    else:
        adaptive_config = AdaptiveExtractorConfig(
            method=method,
            canny_low=get_val('canny_low', 30),
            canny_high=get_val('canny_high', 100),
            min_area=get_val('min_area', 50),
            simplify_epsilon=get_val('simplify_epsilon', 0.8),
            blur_kernel=get_val('blur_kernel', 3),
            min_contour_points=get_val('min_contour_points', 5),
            thickness_threshold=get_val('thickness_threshold', 3),
            density_threshold=get_val('density_threshold', 0.3),
            skeleton_simplify=get_val('skeleton_simplify', 1.0),
            min_length=get_val('min_length', 10.0),
            merge_distance=get_val('merge_distance', 5.0),
            merge_enabled=get_val('merge_enabled', True),
        )
        extractor = AdaptiveContourExtractor(adaptive_config)
        contours = extractor.extract(image)

        descriptions = {
            "skeleton": "Skeletonization",
            "adaptive": "Adaptive",
            "hybrid": "Hybrid"
        }
        description = descriptions.get(method, method)

    return contours, description


def visualize_contours(image: np.ndarray, contours: List[Contour], title: str, stats_text: str = "") -> np.ndarray:
    """Draw contours on a white background with statistics."""

    h, w = image.shape[:2]
    viz = np.ones((h, w, 3), dtype=np.uint8) * 255

    colors = [
        (200, 0, 0), (0, 150, 0), (0, 0, 200),
        (200, 100, 0), (100, 0, 200), (0, 150, 150),
    ]

    total_points = 0

    for i, contour in enumerate(contours):
        color = colors[i % len(colors)]
        points = np.array(contour.points, dtype=np.int32)
        total_points += len(contour.points)

        for j in range(len(points) - 1):
            cv2.line(viz, tuple(points[j]), tuple(points[j + 1]), color, 1)

    # Add title and stats
    cv2.putText(viz, title, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(viz, title, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)

    stats = f"C:{len(contours)} P:{total_points}"
    cv2.putText(viz, stats, (5, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
    cv2.putText(viz, stats, (5, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    return viz


def create_style_comparison_grid(style_name: str, line_art: np.ndarray, method_results: List[Tuple[str, np.ndarray, int, int]]) -> np.ndarray:
    """Create a comparison grid for one style showing all detection methods."""

    # method_results is list of (method_name, visualization, contour_count, point_count)

    n = len(method_results) + 1  # +1 for original line art
    cols = 3
    rows = (n + cols - 1) // cols

    # Target size for each cell
    cell_h, cell_w = 300, 300

    # Resize line art
    h, w = line_art.shape[:2]
    scale = min(cell_w / w, cell_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    padding = 5
    grid_h = rows * (cell_h + padding) + padding + 40  # Extra for title
    grid_w = cols * (cell_w + padding) + padding
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 245

    # Add style title
    cv2.putText(grid, f"Style: {style_name.upper()}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Place original line art first
    resized_art = cv2.resize(line_art, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if len(resized_art.shape) == 2:
        resized_art = cv2.cvtColor(resized_art, cv2.COLOR_GRAY2BGR)

    # Add label to original
    labeled_art = resized_art.copy()
    cv2.putText(labeled_art, "ORIGINAL", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(labeled_art, "ORIGINAL", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

    y1 = padding + 40
    x1 = padding
    y_off = (cell_h - new_h) // 2
    x_off = (cell_w - new_w) // 2
    grid[y1 + y_off:y1 + y_off + new_h, x1 + x_off:x1 + x_off + new_w] = labeled_art

    # Place method results
    for idx, (method_name, viz, contours, points) in enumerate(method_results):
        i = idx + 1  # Skip first cell (original)
        row = i // cols
        col = i % cols

        y1 = row * (cell_h + padding) + padding + 40
        x1 = col * (cell_w + padding) + padding

        # Resize visualization
        resized_viz = cv2.resize(viz, (new_w, new_h), interpolation=cv2.INTER_AREA)

        y_off = (cell_h - new_h) // 2
        x_off = (cell_w - new_w) // 2
        grid[y1 + y_off:y1 + y_off + new_h, x1 + x_off:x1 + x_off + new_w] = resized_viz

    return grid


def create_master_comparison(all_results: Dict[str, Dict[str, Tuple[int, int]]]) -> np.ndarray:
    """Create a master comparison table showing best method for each style."""

    styles = list(all_results.keys())
    methods = ["canny", "skeleton", "adaptive", "hybrid"]

    # Calculate cell sizes
    cell_w = 100
    cell_h = 30
    header_h = 40
    row_header_w = 100

    grid_w = row_header_w + len(methods) * cell_w + 20
    grid_h = header_h + len(styles) * cell_h + 60

    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255

    # Title
    cv2.putText(grid, "POINTS BY STYLE & METHOD", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Column headers
    for j, method in enumerate(methods):
        x = row_header_w + j * cell_w + 10
        cv2.putText(grid, method.upper(), (x, header_h + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Draw grid lines
    cv2.line(grid, (0, header_h + 20), (grid_w, header_h + 20), (200, 200, 200), 1)
    cv2.line(grid, (row_header_w - 5, header_h), (row_header_w - 5, grid_h - 40), (200, 200, 200), 1)

    # Data rows
    for i, style in enumerate(styles):
        y = header_h + 25 + i * cell_h

        # Row header (style name)
        cv2.putText(grid, style[:10], (5, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Find min points for this style (best method)
        style_results = all_results[style]
        min_points = min(style_results[m][1] for m in methods if m in style_results)

        # Method values
        for j, method in enumerate(methods):
            if method in style_results:
                contours, points = style_results[method]
                x = row_header_w + j * cell_w + 10

                # Highlight best (lowest points)
                if points == min_points:
                    cv2.rectangle(grid, (x - 5, y + 2), (x + cell_w - 15, y + 24), (200, 255, 200), -1)

                cv2.putText(grid, str(points), (x, y + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Legend
    cv2.rectangle(grid, (10, grid_h - 35), (30, grid_h - 20), (200, 255, 200), -1)
    cv2.putText(grid, "= Fewest points (fastest)", (35, grid_h - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    return grid


def main():
    parser = argparse.ArgumentParser(description="Compare OpenAI styles and detection methods")
    parser.add_argument("image_path", help="Path to input photo")
    parser.add_argument("--styles", nargs="+", default=None,
                        help="Specific styles to test (default: all)")
    parser.add_argument("--output-dir", "-o", default="comparisons", help="Output directory")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip OpenAI generation, use existing line art files")
    args = parser.parse_args()

    # Validate input
    image_path = Path(args.image_path)
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        return 1

    # Load config
    config = load_config()

    # Get available styles
    available_styles = list(config.openai.styles.keys())
    styles_to_test = args.styles if args.styles else available_styles

    # Validate requested styles
    invalid_styles = [s for s in styles_to_test if s not in available_styles]
    if invalid_styles:
        logger.error(f"Invalid styles: {invalid_styles}")
        logger.info(f"Available styles: {available_styles}")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load input image
    original = cv2.imread(str(image_path))
    if original is None:
        logger.error(f"Failed to load image: {image_path}")
        return 1

    logger.info(f"Loaded: {image_path} ({original.shape[1]}x{original.shape[0]})")
    logger.info(f"Testing {len(styles_to_test)} styles: {styles_to_test}")

    detection_methods = ["canny", "skeleton", "adaptive", "hybrid"]

    # Store all results for master comparison
    all_results: Dict[str, Dict[str, Tuple[int, int]]] = {}

    # =========================================================================
    # Process each style
    # =========================================================================
    for style_idx, style_name in enumerate(styles_to_test):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"STYLE {style_idx + 1}/{len(styles_to_test)}: {style_name.upper()}")
        logger.info(f"{'=' * 60}")

        style_config = config.openai.styles[style_name]
        line_art_path = output_dir / f"lineart_{style_name}.png"

        # Generate or load line art
        if args.skip_generation and line_art_path.exists():
            logger.info(f"Loading existing line art: {line_art_path}")
            line_art = cv2.imread(str(line_art_path))
        else:
            logger.info(f"Generating line art with OpenAI ({style_config.name})...")
            line_art = generate_line_art_for_style(
                original, style_name, style_config, config.openai, str(line_art_path)
            )

            if line_art is None:
                logger.error(f"Failed to generate line art for style: {style_name}")
                continue

            cv2.imwrite(str(line_art_path), line_art)
            logger.info(f"Saved: {line_art_path}")

            # Rate limiting - wait between API calls
            if style_idx < len(styles_to_test) - 1:
                logger.info("Waiting 2s before next API call...")
                time.sleep(2)

        # Run detection methods
        method_results = []
        all_results[style_name] = {}

        for method in detection_methods:
            logger.info(f"  [{method}] Extracting contours...")

            try:
                contours, description = extract_with_method(
                    line_art, method, config,
                    style_config.contour if style_config.contour else None
                )

                total_points = sum(len(c.points) for c in contours)
                logger.info(f"    Contours: {len(contours)}, Points: {total_points}")

                # Create visualization
                viz = visualize_contours(
                    line_art, contours,
                    f"{method.upper()}",
                    f"C:{len(contours)} P:{total_points}"
                )

                # Save individual file
                filename = f"detection_{style_name}_{method}.png"
                cv2.imwrite(str(output_dir / filename), viz)

                method_results.append((method, viz, len(contours), total_points))
                all_results[style_name][method] = (len(contours), total_points)

            except Exception as e:
                logger.error(f"    Failed: {e}")
                import traceback
                traceback.print_exc()

        # Create style comparison grid
        if method_results:
            grid = create_style_comparison_grid(style_name, line_art, method_results)
            grid_path = output_dir / f"grid_{style_name}.png"
            cv2.imwrite(str(grid_path), grid)
            logger.info(f"Saved: {grid_path}")

    # =========================================================================
    # Create master comparison
    # =========================================================================
    logger.info(f"\n{'=' * 60}")
    logger.info("CREATING MASTER COMPARISON")
    logger.info(f"{'=' * 60}")

    if all_results:
        master_grid = create_master_comparison(all_results)
        master_path = output_dir / "master_comparison.png"
        cv2.imwrite(str(master_path), master_grid)
        logger.info(f"Saved: {master_path}")

        # Print summary table
        logger.info("\n" + "-" * 70)
        logger.info(f"{'STYLE':<12} {'CANNY':>10} {'SKELETON':>10} {'ADAPTIVE':>10} {'HYBRID':>10} {'BEST':<10}")
        logger.info("-" * 70)

        for style_name, results in all_results.items():
            row = f"{style_name:<12}"
            min_points = float('inf')
            best_method = ""

            for method in detection_methods:
                if method in results:
                    points = results[method][1]
                    row += f" {points:>9}"
                    if points < min_points:
                        min_points = points
                        best_method = method
                else:
                    row += f" {'N/A':>9}"

            row += f" {best_method:<10}"
            logger.info(row)

        logger.info("-" * 70)

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info(f"\n{'=' * 60}")
    logger.info("COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"Output directory: {output_dir.absolute()}")
    logger.info(f"\nFiles created:")
    for f in sorted(output_dir.glob("*.png")):
        logger.info(f"  - {f.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
