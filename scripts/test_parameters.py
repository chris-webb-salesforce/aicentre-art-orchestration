#!/usr/bin/env python3
"""
Parameter testing script for line art extraction.

Tests various parameter combinations on a single line art image to find optimal settings.

Usage:
    python scripts/test_parameters.py comparisons/simple/lineart_simple.png
    python scripts/test_parameters.py comparisons/ghibli/lineart_ghibli.png --method skeleton
    python scripts/test_parameters.py comparisons/minimal/lineart_minimal.png --sweep canny_low
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any
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


# Parameter sweep definitions
PARAMETER_SWEEPS = {
    "canny_low": [10, 20, 30, 50, 80, 100],
    "canny_high": [50, 80, 100, 150, 200, 250],
    "simplify_epsilon": [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0],
    "min_length": [3.0, 5.0, 10.0, 15.0, 25.0, 50.0],
    "min_area": [10, 30, 50, 100, 200, 500],
    "merge_distance": [3.0, 5.0, 8.0, 12.0, 20.0, 30.0],
    "blur_kernel": [1, 3, 5, 7],
}


def extract_contours(image: np.ndarray, method: str, params: Dict[str, Any]) -> List[Contour]:
    """Extract contours with specified parameters."""
    if method == "canny":
        extractor = ContourExtractor(
            canny_low=params.get("canny_low", 30),
            canny_high=params.get("canny_high", 100),
            min_area=params.get("min_area", 50),
            simplify_epsilon=params.get("simplify_epsilon", 0.5),
            blur_kernel=params.get("blur_kernel", 3),
            min_contour_points=params.get("min_contour_points", 3)
        )
        return extractor.extract(image)
    else:
        config = AdaptiveExtractorConfig(
            method=method,
            canny_low=params.get("canny_low", 30),
            canny_high=params.get("canny_high", 100),
            min_area=params.get("min_area", 50),
            simplify_epsilon=params.get("simplify_epsilon", 0.5),
            blur_kernel=params.get("blur_kernel", 3),
            min_contour_points=params.get("min_contour_points", 3),
            thickness_threshold=params.get("thickness_threshold", 5),
            density_threshold=params.get("density_threshold", 0.5),
            skeleton_simplify=params.get("skeleton_simplify", 0.5),
            min_length=params.get("min_length", 5.0),
            merge_distance=params.get("merge_distance", 5.0),
            merge_enabled=params.get("merge_enabled", True),
        )
        extractor = AdaptiveContourExtractor(config)
        return extractor.extract(image)


def visualize_contours(image: np.ndarray, contours: List[Contour], title: str, params_str: str = "") -> np.ndarray:
    """Draw contours on white background with statistics."""
    h, w = image.shape[:2]
    viz = np.ones((h, w, 3), dtype=np.uint8) * 255

    colors = [
        (255, 0, 0), (0, 200, 0), (0, 0, 255),
        (255, 128, 0), (128, 0, 255), (0, 200, 200),
        (200, 0, 128), (0, 128, 200), (128, 128, 0)
    ]

    total_points = 0
    for i, contour in enumerate(contours):
        color = colors[i % len(colors)]
        points = np.array(contour.points, dtype=np.int32)
        total_points += len(contour.points)
        for j in range(len(points) - 1):
            cv2.line(viz, tuple(points[j]), tuple(points[j + 1]), color, 1)

    # Add stats
    lines = [
        title,
        params_str if params_str else "",
        f"Contours: {len(contours)} | Points: {total_points}",
    ]

    y_offset = 20
    for line in lines:
        if line:
            cv2.putText(viz, line, (8, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
            cv2.putText(viz, line, (8, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
            y_offset += 16

    return viz


def create_grid(images: List[tuple], cols: int = 3) -> np.ndarray:
    """Create a grid of labeled images."""
    if not images:
        return np.ones((100, 100, 3), dtype=np.uint8) * 255

    n = len(images)
    rows = (n + cols - 1) // cols

    max_h = max(img.shape[0] for _, img in images)
    max_w = max(img.shape[1] for _, img in images)

    scale = 1.0
    if max_w > 350:
        scale = 350 / max_w
        max_w = 350
        max_h = int(max_h * scale)

    padding = 4
    grid_h = rows * (max_h + padding) + padding
    grid_w = cols * (max_w + padding) + padding
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 240

    for i, (label, img) in enumerate(images):
        row = i // cols
        col = i % cols

        if scale != 1.0:
            new_h = int(img.shape[0] * scale)
            new_w = int(img.shape[1] * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        y1 = row * (max_h + padding) + padding
        x1 = col * (max_w + padding) + padding
        y_offset = (max_h - img.shape[0]) // 2
        x_offset = (max_w - img.shape[1]) // 2

        grid[y1 + y_offset:y1 + y_offset + img.shape[0],
             x1 + x_offset:x1 + x_offset + img.shape[1]] = img

    return grid


def run_parameter_sweep(image: np.ndarray, method: str, param_name: str,
                        base_params: Dict[str, Any], output_dir: Path, style: str = "") -> Dict[str, Any]:
    """Run a sweep over one parameter. Returns best result."""
    if param_name not in PARAMETER_SWEEPS:
        logger.error(f"Unknown parameter: {param_name}")
        logger.info(f"Available: {list(PARAMETER_SWEEPS.keys())}")
        return {}

    values = PARAMETER_SWEEPS[param_name]
    prefix = f"{style}_{method}_" if style else f"{method}_"
    logger.info(f"\nSweeping {param_name}: {values}")

    comparisons = []
    results = []

    for val in values:
        params = base_params.copy()
        params[param_name] = val

        try:
            contours = extract_contours(image, method, params)
            total_points = sum(len(c.points) for c in contours)

            title = f"{style} {method}" if style else method
            viz = visualize_contours(
                image, contours,
                f"{title} | {param_name}={val}",
                f"Points: {total_points}"
            )
            comparisons.append((f"{param_name}={val}", viz))
            results.append({
                "value": val,
                "contours": len(contours),
                "points": total_points
            })

            # Save individual image
            individual_path = output_dir / f"{prefix}{param_name}_{val}.png"
            cv2.imwrite(str(individual_path), viz)

            logger.info(f"  {param_name}={val}: {len(contours)} contours, {total_points} points -> {individual_path.name}")

        except Exception as e:
            logger.error(f"  {param_name}={val}: Failed - {e}")

    # Create and save grid
    grid = create_grid(comparisons, cols=3)
    grid_path = output_dir / f"{prefix}sweep_{param_name}.png"
    cv2.imwrite(str(grid_path), grid)
    logger.info(f"Saved: {grid_path}")

    # Return best result
    if results:
        best = min(results, key=lambda x: x["points"])
        return {"param": param_name, "best_value": best["value"], "points": best["points"]}
    return {}

    # Print summary
    if results:
        logger.info(f"\n{param_name} sweep summary:")
        sorted_results = sorted(results, key=lambda x: x["points"])
        logger.info(f"  Fewest points: {param_name}={sorted_results[0]['value']} ({sorted_results[0]['points']} pts)")
        logger.info(f"  Most points: {param_name}={sorted_results[-1]['value']} ({sorted_results[-1]['points']} pts)")


def run_method_comparison(image: np.ndarray, params: Dict[str, Any], output_dir: Path, style: str = "") -> None:
    """Compare all methods with same parameters."""
    methods = ["canny", "skeleton", "adaptive", "hybrid"]
    comparisons = []

    prefix = f"{style}_" if style else ""
    logger.info(f"\nComparing methods for style '{style}'..." if style else "\nComparing methods...")

    for method in methods:
        try:
            contours = extract_contours(image, method, params)
            total_points = sum(len(c.points) for c in contours)

            title = f"{style.upper()} + {method.upper()}" if style else method.upper()
            viz = visualize_contours(image, contours, title, f"Points: {total_points}")
            comparisons.append((method, viz))

            # Save individual image
            individual_path = output_dir / f"{prefix}{method}.png"
            cv2.imwrite(str(individual_path), viz)

            logger.info(f"  {method}: {len(contours)} contours, {total_points} points -> {individual_path.name}")

        except Exception as e:
            logger.error(f"  {method}: Failed - {e}")

    grid = create_grid(comparisons, cols=2)
    grid_path = output_dir / f"{prefix}method_comparison.png"
    cv2.imwrite(str(grid_path), grid)
    logger.info(f"Saved grid: {grid_path}")


def run_custom_params(image: np.ndarray, method: str, params: Dict[str, Any], output_dir: Path, name: str) -> None:
    """Run extraction with custom parameters and save result."""
    logger.info(f"\nRunning {method} with custom parameters...")
    logger.info(f"  Parameters: {params}")

    try:
        contours = extract_contours(image, method, params)
        total_points = sum(len(c.points) for c in contours)

        viz = visualize_contours(image, contours, f"{method} - {name}", str(params))

        output_path = output_dir / f"custom_{name}.png"
        cv2.imwrite(str(output_path), viz)

        logger.info(f"  Result: {len(contours)} contours, {total_points} points")
        logger.info(f"  Saved: {output_path}")

    except Exception as e:
        logger.error(f"  Failed: {e}")


STYLES = ["simple", "minimal", "vangogh", "ghibli", "picasso", "sketch", "caricature", "geometric", "contour"]


def main():
    parser = argparse.ArgumentParser(description="Test extraction parameters on line art")
    parser.add_argument("image_path", nargs="?", help="Path to line art image")
    parser.add_argument("--method", "-m", default="adaptive",
                        choices=["canny", "skeleton", "adaptive", "hybrid"],
                        help="Detection method to use")
    parser.add_argument("--sweep", "-s", help="Parameter to sweep (e.g., canny_low, simplify_epsilon)")
    parser.add_argument("--compare-methods", "-c", action="store_true",
                        help="Compare all methods with current settings")
    parser.add_argument("--run-all", "-a", action="store_true",
                        help="Run method comparison on all styles")
    parser.add_argument("--sweep-all", action="store_true",
                        help="Run parameter sweeps on all styles and methods")
    parser.add_argument("--style", help="Style name for output filenames (auto-detected from path if not set)")
    parser.add_argument("--output-dir", "-o", help="Output directory (default: same as input)")

    # Custom parameter overrides
    parser.add_argument("--canny-low", type=int, help="Canny low threshold")
    parser.add_argument("--canny-high", type=int, help="Canny high threshold")
    parser.add_argument("--simplify", type=float, help="Simplify epsilon")
    parser.add_argument("--min-length", type=float, help="Minimum contour length")
    parser.add_argument("--min-area", type=int, help="Minimum contour area")
    parser.add_argument("--merge-distance", type=float, help="Merge distance")
    parser.add_argument("--blur", type=int, help="Blur kernel size")
    parser.add_argument("--name", default="test", help="Name for custom output file")

    args = parser.parse_args()

    # Load base config
    config = load_config()

    # Handle --run-all mode
    if args.run_all:
        base_dir = Path("comparisons")
        output_dir = Path(args.output_dir) if args.output_dir else base_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("RUNNING ALL STYLES")
        logger.info("=" * 60)

        base_params = {
            "canny_low": config.contour.canny_low,
            "canny_high": config.contour.canny_high,
            "min_area": config.contour.min_area,
            "simplify_epsilon": config.contour.simplify_epsilon,
            "blur_kernel": config.contour.blur_kernel,
            "min_contour_points": config.contour.min_contour_points,
            "min_length": config.contour.min_length,
            "merge_distance": config.contour.merge_distance,
            "merge_enabled": config.contour.merge_enabled,
        }

        for style in STYLES:
            image_path = base_dir / style / f"lineart_{style}.png"
            if not image_path.exists():
                logger.warning(f"Skipping {style}: {image_path} not found")
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Skipping {style}: failed to load image")
                continue

            logger.info(f"\n--- {style.upper()} ---")
            style_output_dir = base_dir / style
            run_method_comparison(image, base_params, style_output_dir, style)

        logger.info("\n" + "=" * 60)
        logger.info("COMPLETE")
        logger.info("=" * 60)
        return 0

    # Handle --sweep-all mode
    if args.sweep_all:
        base_dir = Path("comparisons")
        methods = ["canny", "skeleton", "adaptive", "hybrid"]
        params_to_sweep = ["simplify_epsilon", "min_length", "min_area", "canny_low", "canny_high"]

        logger.info("=" * 60)
        logger.info("FULL PARAMETER SWEEP - ALL STYLES & METHODS")
        logger.info("=" * 60)
        logger.info(f"Styles: {STYLES}")
        logger.info(f"Methods: {methods}")
        logger.info(f"Parameters: {params_to_sweep}")

        base_params = {
            "canny_low": config.contour.canny_low,
            "canny_high": config.contour.canny_high,
            "min_area": config.contour.min_area,
            "simplify_epsilon": config.contour.simplify_epsilon,
            "blur_kernel": config.contour.blur_kernel,
            "min_contour_points": config.contour.min_contour_points,
            "min_length": config.contour.min_length,
            "merge_distance": config.contour.merge_distance,
            "merge_enabled": config.contour.merge_enabled,
        }

        all_results = []

        for style in STYLES:
            image_path = base_dir / style / f"lineart_{style}.png"
            if not image_path.exists():
                logger.warning(f"Skipping {style}: {image_path} not found")
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Skipping {style}: failed to load image")
                continue

            style_output_dir = base_dir / style / "sweeps"
            style_output_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"\n{'='*60}")
            logger.info(f"STYLE: {style.upper()}")
            logger.info("=" * 60)

            for method in methods:
                logger.info(f"\n--- {style} + {method} ---")

                for param in params_to_sweep:
                    result = run_parameter_sweep(image, method, param, base_params, style_output_dir, style)
                    if result:
                        result["style"] = style
                        result["method"] = method
                        all_results.append(result)

        # Print summary table
        logger.info("\n" + "=" * 60)
        logger.info("BEST PARAMETERS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"{'Style':<12} {'Method':<10} {'Parameter':<18} {'Best Value':<12} {'Points':<10}")
        logger.info("-" * 62)
        for r in all_results:
            logger.info(f"{r['style']:<12} {r['method']:<10} {r['param']:<18} {str(r['best_value']):<12} {r['points']:<10}")

        return 0

    # Single image mode - require image_path
    if not args.image_path:
        parser.error("image_path is required (or use --run-all)")

    # Load image
    image_path = Path(args.image_path)
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        return 1

    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Failed to load: {image_path}")
        return 1

    logger.info(f"Loaded: {image_path} ({image.shape[1]}x{image.shape[0]})")

    # Auto-detect style from path if not provided
    style = args.style
    if not style:
        for s in STYLES:
            if s in str(image_path):
                style = s
                break

    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else image_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    base_params = {
        "canny_low": config.contour.canny_low,
        "canny_high": config.contour.canny_high,
        "min_area": config.contour.min_area,
        "simplify_epsilon": config.contour.simplify_epsilon,
        "blur_kernel": config.contour.blur_kernel,
        "min_contour_points": config.contour.min_contour_points,
        "min_length": config.contour.min_length,
        "merge_distance": config.contour.merge_distance,
        "merge_enabled": config.contour.merge_enabled,
    }

    # Apply any custom overrides
    if args.canny_low:
        base_params["canny_low"] = args.canny_low
    if args.canny_high:
        base_params["canny_high"] = args.canny_high
    if args.simplify:
        base_params["simplify_epsilon"] = args.simplify
    if args.min_length:
        base_params["min_length"] = args.min_length
    if args.min_area:
        base_params["min_area"] = args.min_area
    if args.merge_distance:
        base_params["merge_distance"] = args.merge_distance
    if args.blur:
        base_params["blur_kernel"] = args.blur

    # Run requested operation
    if args.sweep:
        run_parameter_sweep(image, args.method, args.sweep, base_params, output_dir, style or "")
    elif args.compare_methods:
        run_method_comparison(image, base_params, output_dir, style or "")
    else:
        # Run with custom/default parameters
        run_custom_params(image, args.method, base_params, output_dir, args.name)

    return 0


if __name__ == "__main__":
    sys.exit(main())
