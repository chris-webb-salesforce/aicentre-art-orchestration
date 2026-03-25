"""
Fill strategies for solid/filled regions in line art images.

Detects filled blobs (like pupils, solid circles) that would produce chaotic
paths from thinning/skeletonization, and converts them into clean drawable
paths using configurable strategies (outline, spiral, hatching).

This module runs as a preprocessing step before thinning/canny extraction.
"""

import cv2
import numpy as np
from typing import List, Tuple
import logging

from .contour_extractor import Contour

logger = logging.getLogger(__name__)


def detect_filled_regions(
    binary: np.ndarray,
    min_area: int = 150,
    max_area: int = 8000,
    min_solidity: float = 0.75,
    min_compactness: float = 0.12,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Detect solid filled blobs in a binary image.

    Uses contour analysis with interior fill-ratio checking. For each closed
    contour, measures what fraction of the interior pixels are actually filled
    in the binary image. This correctly identifies filled regions (pupils,
    solid circles) even when they share pixels with surrounding line work
    (iris outlines, eyelashes), which connected-component analysis cannot do.

    A compactness filter (4*pi*area / perimeter^2) rejects thin elongated
    contours that happen to have high fill ratios. Circles have compactness
    ~1.0, eyes/pupils ~0.3-0.4, eyebrows ~0.15-0.2, thin lines <0.1.

    Args:
        binary: Binary image (foreground=255, background=0)
        min_area: Minimum contour area in pixels (filters noise/line crossings)
        max_area: Maximum contour area in pixels (avoids treating large regions as fills)
        min_solidity: Minimum fill ratio — fraction of the contour's interior
                      that must be filled with foreground pixels
        min_compactness: Minimum compactness (4*pi*area/perimeter^2). Filters
                         thin elongated contours. 0.12 catches eyes and eyebrows.

    Returns:
        List of (cv2_contour, mask) pairs for each detected filled region.
        The mask is full-image-sized with the contour interior filled to 255.
    """
    h, w = binary.shape

    # Find all contours with hierarchy (RETR_TREE preserves nesting)
    contours_cv, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours_cv or hierarchy is None:
        logger.info("No contours found for fill detection")
        return []

    regions = []
    for i, cnt in enumerate(contours_cv):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        if area > max_area:
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
                logger.debug(f"Skipping large contour: area={area:.0f} > max_area={max_area}, "
                             f"center=({cx:.0f},{cy:.0f})")
            continue

        # Compactness filter: reject thin elongated contours
        # 4*pi*area / perimeter^2 — 1.0 for circle, <0.1 for thin lines
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        compactness = (4 * np.pi * area) / (perimeter * perimeter)
        if compactness < min_compactness:
            continue

        # Create a mask of the contour interior
        interior_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(interior_mask, [cnt], -1, 255, -1)  # fill interior

        # Measure how much of the interior is actually filled in the binary
        interior_pixels = cv2.countNonZero(interior_mask)
        if interior_pixels == 0:
            continue
        filled_pixels = cv2.countNonZero(cv2.bitwise_and(binary, interior_mask))
        fill_ratio = filled_pixels / interior_pixels

        if fill_ratio < min_solidity:
            continue

        regions.append((cnt, interior_mask))

        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
        else:
            x, y, bw, bh = cv2.boundingRect(cnt)
            cx, cy = x + bw / 2, y + bh / 2
        logger.info(
            f"Filled region: area={area:.0f}, fill={fill_ratio:.2f}, "
            f"compact={compactness:.3f}, center=({cx:.0f},{cy:.0f})"
        )

    logger.info(f"Detected {len(regions)} filled regions (min_area={min_area}, "
                f"max_area={max_area}, min_solidity={min_solidity}, "
                f"min_compactness={min_compactness})")
    return regions


def erase_regions_from_image(
    image: np.ndarray,
    regions: List[Tuple[np.ndarray, np.ndarray]],
    dilate_px: int = 2,
) -> np.ndarray:
    """
    Erase detected filled regions from an image by painting them white.

    Dilates the erase mask slightly to remove boundary pixels that would
    otherwise produce a stray outline contour from thinning.

    Args:
        image: Original image (BGR or grayscale)
        regions: List of (cv2_contour, mask) from detect_filled_regions
        dilate_px: Pixels to dilate the erase mask (removes boundary artifacts)

    Returns:
        Copy of image with filled regions painted white.
    """
    if not regions:
        return image

    cleaned = image.copy()
    h, w = image.shape[:2]

    # Build combined erase mask
    erase_mask = np.zeros((h, w), dtype=np.uint8)
    for cv_contour, mask in regions:
        erase_mask = cv2.bitwise_or(erase_mask, mask)

    # Dilate to remove boundary pixels
    if dilate_px > 0:
        kernel = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
        erase_mask = cv2.dilate(erase_mask, kernel, iterations=1)

    # Paint white
    if len(cleaned.shape) == 3:
        cleaned[erase_mask > 0] = [255, 255, 255]
    else:
        cleaned[erase_mask > 0] = 255

    return cleaned


def filled_region_to_contours(
    cv_contour: np.ndarray,
    mask: np.ndarray,
    binary: np.ndarray,
    strategy: str = "spiral",
    spacing: float = 3.0,
    hatch_angle: float = 45.0,
    simplify_epsilon: float = 0.5,
) -> List[Contour]:
    """
    Convert a filled region into drawable contours using the specified strategy.

    Args:
        cv_contour: OpenCV contour of the region boundary
        mask: Binary mask of the filled region (full image size)
        binary: The actual binary image — used to preserve highlights/white
                spots inside filled regions (spiral only draws where ink exists)
        strategy: Fill strategy - "spiral", "outline", or "hatch"
        spacing: Pixel spacing between spiral rings or hatch lines
        hatch_angle: Angle in degrees for hatch lines
        simplify_epsilon: Path simplification factor

    Returns:
        List of Contour objects representing the fill pattern.
    """
    if strategy == "outline":
        return _outline_strategy(cv_contour, simplify_epsilon)
    elif strategy == "spiral":
        # Use actual ink pixels (binary & mask) so highlights are preserved
        ink_mask = cv2.bitwise_and(binary, mask)
        return _spiral_strategy(cv_contour, ink_mask, spacing, simplify_epsilon)
    elif strategy == "hatch":
        ink_mask = cv2.bitwise_and(binary, mask)
        return _hatch_strategy(ink_mask, spacing, hatch_angle, simplify_epsilon)
    else:
        logger.warning(f"Unknown fill strategy '{strategy}', using spiral")
        ink_mask = cv2.bitwise_and(binary, mask)
        return _spiral_strategy(cv_contour, ink_mask, spacing, simplify_epsilon)


def _outline_strategy(
    cv_contour: np.ndarray,
    simplify_epsilon: float = 0.5,
) -> List[Contour]:
    """Draw just the boundary outline of the filled region."""
    simplified = cv2.approxPolyDP(cv_contour, simplify_epsilon, closed=True)
    points = [(float(p[0][0]), float(p[0][1])) for p in simplified]

    if len(points) < 3:
        return []

    # Close the loop
    points.append(points[0])

    length = _calc_length(points)
    return [Contour(points=points, is_closed=True, area=cv2.contourArea(cv_contour), length=length)]


def _spiral_strategy(
    cv_contour: np.ndarray,
    mask: np.ndarray,
    spacing: float = 3.0,
    simplify_epsilon: float = 0.5,
) -> List[Contour]:
    """
    Spiral inward from the boundary to fill the region.

    Repeatedly erodes the mask and traces each ring's contour, connecting
    them into a single continuous path. Produces a natural pen-drawn fill
    that works well for small circular features like pupils.
    """
    current_mask = mask.copy()
    kernel_size = max(3, int(spacing * 2) | 1)  # ensure odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    all_points: List[Tuple[float, float]] = []
    prev_end = None
    max_iterations = 200

    for _ in range(max_iterations):
        # Use pixel count for termination — more reliable than contour area
        # for thin/degenerate shapes after multiple erosions
        if np.count_nonzero(current_mask) < 4:
            break

        contours_cv, _ = cv2.findContours(
            current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours_cv:
            break

        outer = max(contours_cv, key=cv2.contourArea)

        simplified = cv2.approxPolyDP(outer, simplify_epsilon, closed=True)
        ring_points = [(float(p[0][0]), float(p[0][1])) for p in simplified]

        if len(ring_points) < 3:
            break

        # Rotate ring to start at the point closest to the previous ring's end
        # This minimizes the jump between rings for a smoother spiral
        if prev_end is not None:
            dists = [
                (p[0] - prev_end[0]) ** 2 + (p[1] - prev_end[1]) ** 2
                for p in ring_points
            ]
            start_idx = int(np.argmin(dists))
            ring_points = ring_points[start_idx:] + ring_points[:start_idx]

        # Close the ring back to its start
        ring_points.append(ring_points[0])

        all_points.extend(ring_points)
        # Track where the pen ends — the ring closes back to its rotated start,
        # so the next ring will be rotated to start near this point
        prev_end = ring_points[-1]

        # Erode inward for next ring
        current_mask = cv2.erode(current_mask, kernel, iterations=1)

    if len(all_points) < 2:
        return []

    length = _calc_length(all_points)
    area = cv2.contourArea(cv_contour)
    return [Contour(points=all_points, is_closed=False, area=area, length=length)]


def _hatch_strategy(
    mask: np.ndarray,
    spacing: float = 3.0,
    angle: float = 45.0,
    simplify_epsilon: float = 0.5,
) -> List[Contour]:
    """
    Fill the region with parallel lines at the specified angle.

    Rotates the mask, scans horizontal rows at the given spacing,
    then rotates the line coordinates back.
    """
    h, w = mask.shape
    cx, cy = w / 2.0, h / 2.0

    # Rotation matrix and its inverse
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    M_inv = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)

    # Rotate the mask
    rotated = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Find bounding rows of the rotated mask
    rows_with_content = np.where(np.any(rotated > 0, axis=1))[0]
    if len(rows_with_content) == 0:
        return []

    y_min = rows_with_content[0]
    y_max = rows_with_content[-1]

    contours = []
    step = max(1, int(spacing))
    reverse = False  # alternate direction for boustrophedon (zig-zag) path

    for y in range(y_min, y_max + 1, step):
        row = rotated[y, :]
        # Find runs of foreground pixels
        runs = _find_runs(row)

        for x_start, x_end in runs:
            # Transform endpoints back to original coordinates
            p1 = _transform_point(x_start, y, M_inv)
            p2 = _transform_point(x_end, y, M_inv)

            points = [p1, p2] if not reverse else [p2, p1]
            length = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

            if length >= 2:
                contours.append(Contour(
                    points=points,
                    is_closed=False,
                    area=0,
                    length=length,
                ))

        reverse = not reverse

    return contours


def _find_runs(row: np.ndarray) -> List[Tuple[int, int]]:
    """Find contiguous runs of nonzero values in a 1D array."""
    runs = []
    in_run = False
    start = 0

    for i, val in enumerate(row):
        if val > 0 and not in_run:
            start = i
            in_run = True
        elif val == 0 and in_run:
            runs.append((start, i - 1))
            in_run = False

    if in_run:
        runs.append((start, len(row) - 1))

    return runs


def _transform_point(
    x: float, y: float, M: np.ndarray
) -> Tuple[float, float]:
    """Apply a 2x3 affine transform to a point."""
    px = M[0, 0] * x + M[0, 1] * y + M[0, 2]
    py = M[1, 0] * x + M[1, 1] * y + M[1, 2]
    return (px, py)


def _calc_length(points: List[Tuple[float, float]]) -> float:
    """Calculate total path length."""
    length = 0.0
    for i in range(len(points) - 1):
        dx = points[i + 1][0] - points[i][0]
        dy = points[i + 1][1] - points[i][1]
        length += np.sqrt(dx * dx + dy * dy)
    return length
