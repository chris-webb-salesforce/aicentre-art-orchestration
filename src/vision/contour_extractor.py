"""
Contour extraction from line art images.

Converts OpenAI-generated line art PNG to drawable paths.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class Contour:
    """Represents a drawable path extracted from an image."""
    points: List[Tuple[float, float]]  # List of (x, y) coordinates
    is_closed: bool  # Whether the contour forms a closed loop
    area: float  # Area enclosed (0 for open contours)
    length: float  # Total path length


class ContourExtractor:
    """
    Extracts drawable contours from line art images.

    Uses Canny edge detection and contour finding to convert
    raster line art into vector paths suitable for plotting.
    """

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        min_area: int = 100,
        simplify_epsilon: float = 2.0
    ):
        """
        Initialize contour extractor.

        Args:
            canny_low: Lower threshold for Canny edge detection
            canny_high: Upper threshold for Canny edge detection
            min_area: Minimum contour area to keep (filters noise)
            simplify_epsilon: Path simplification factor (higher = fewer points)
        """
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_area = min_area
        self.simplify_epsilon = simplify_epsilon

    def extract(self, image: np.ndarray) -> List[Contour]:
        """
        Extract contours from a line art image.

        Args:
            image: BGR or grayscale image (line art from OpenAI)

        Returns:
            List of Contour objects.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Invert if background is white (lines should be white for contour detection)
        if np.mean(gray) > 127:
            gray = cv2.bitwise_not(gray)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # Dilate edges slightly to connect nearby lines
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Find contours
        contours_cv, hierarchy = cv2.findContours(
            edges,
            cv2.RETR_LIST,  # Get all contours
            cv2.CHAIN_APPROX_SIMPLE  # Compress horizontal/vertical segments
        )

        logger.info(f"Found {len(contours_cv)} raw contours")

        # Convert to our Contour format
        contours = []
        for cv_contour in contours_cv:
            # Calculate area
            area = cv2.contourArea(cv_contour)

            # Filter by minimum area (but keep long thin lines with many points)
            # A contour must have sufficient area OR be a long line (20+ points)
            if area < self.min_area and len(cv_contour) < 20:
                continue

            # Simplify contour
            epsilon = self.simplify_epsilon
            simplified = cv2.approxPolyDP(cv_contour, epsilon, closed=False)

            # Convert to list of points
            points = [(float(p[0][0]), float(p[0][1])) for p in simplified]

            if len(points) < 2:
                continue

            # Calculate path length
            length = 0.0
            for i in range(len(points) - 1):
                dx = points[i + 1][0] - points[i][0]
                dy = points[i + 1][1] - points[i][1]
                length += np.sqrt(dx * dx + dy * dy)

            # Check if closed
            if len(points) > 2:
                first, last = points[0], points[-1]
                dist = np.sqrt((first[0] - last[0])**2 + (first[1] - last[1])**2)
                is_closed = dist < 5  # Close if endpoints within 5 pixels
            else:
                is_closed = False

            contours.append(Contour(
                points=points,
                is_closed=is_closed,
                area=area,
                length=length
            ))

        logger.info(f"Extracted {len(contours)} contours after filtering")
        return contours

    def extract_from_file(self, image_path: str) -> List[Contour]:
        """
        Extract contours from an image file.

        Args:
            image_path: Path to image file

        Returns:
            List of Contour objects.
        """
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return []

        return self.extract(image)

    def optimize_order(
        self,
        contours: List[Contour],
        start_point: Tuple[float, float] = (0, 0)
    ) -> List[Contour]:
        """
        Optimize contour order to minimize pen travel distance.

        Uses a greedy nearest-neighbor algorithm.

        Args:
            contours: List of contours to reorder
            start_point: Starting position of pen

        Returns:
            Reordered list of contours.
        """
        if not contours:
            return []

        remaining = list(contours)
        ordered = []
        current_pos = start_point

        while remaining:
            # Find nearest contour (considering both start and end points)
            best_idx = 0
            best_dist = float('inf')
            best_reversed = False

            for i, contour in enumerate(remaining):
                # Distance to start of contour
                start = contour.points[0]
                dist_to_start = np.sqrt(
                    (start[0] - current_pos[0])**2 +
                    (start[1] - current_pos[1])**2
                )

                # Distance to end of contour (can draw in reverse)
                end = contour.points[-1]
                dist_to_end = np.sqrt(
                    (end[0] - current_pos[0])**2 +
                    (end[1] - current_pos[1])**2
                )

                if dist_to_start < best_dist:
                    best_dist = dist_to_start
                    best_idx = i
                    best_reversed = False

                if dist_to_end < best_dist:
                    best_dist = dist_to_end
                    best_idx = i
                    best_reversed = True

            # Add best contour to ordered list
            contour = remaining.pop(best_idx)

            if best_reversed:
                # Reverse the points
                contour = Contour(
                    points=list(reversed(contour.points)),
                    is_closed=contour.is_closed,
                    area=contour.area,
                    length=contour.length
                )

            ordered.append(contour)
            current_pos = contour.points[-1]

        return ordered

    def get_bounds(self, contours: List[Contour]) -> Tuple[float, float, float, float]:
        """
        Get bounding box of all contours.

        Returns:
            Tuple of (x_min, y_min, x_max, y_max)
        """
        if not contours:
            return (0, 0, 0, 0)

        x_min = float('inf')
        y_min = float('inf')
        x_max = float('-inf')
        y_max = float('-inf')

        for contour in contours:
            for x, y in contour.points:
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

        return (x_min, y_min, x_max, y_max)

    def visualize(
        self,
        image: np.ndarray,
        contours: List[Contour],
        show_numbers: bool = True
    ) -> np.ndarray:
        """
        Draw contours on image for visualization.

        Args:
            image: Image to draw on (will be modified)
            contours: List of contours to draw
            show_numbers: Whether to show contour numbers

        Returns:
            Modified image.
        """
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]

        for i, contour in enumerate(contours):
            color = colors[i % len(colors)]
            points = np.array(contour.points, dtype=np.int32)

            # Draw path
            for j in range(len(points) - 1):
                cv2.line(image, tuple(points[j]), tuple(points[j + 1]), color, 2)

            # Draw start point (green circle)
            cv2.circle(image, tuple(points[0]), 5, (0, 255, 0), -1)

            # Draw end point (red circle)
            cv2.circle(image, tuple(points[-1]), 5, (0, 0, 255), -1)

            # Show number
            if show_numbers:
                cx = int(np.mean([p[0] for p in contour.points]))
                cy = int(np.mean([p[1] for p in contour.points]))
                cv2.putText(
                    image,
                    str(i + 1),
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

        return image


if __name__ == "__main__":
    # Test contour extraction
    logging.basicConfig(level=logging.INFO)

    import sys
    if len(sys.argv) < 2:
        print("Usage: python contour_extractor.py <image_path>")
        print("Creating test image...")

        # Create a test image with some lines
        test_img = np.ones((500, 500), dtype=np.uint8) * 255
        cv2.circle(test_img, (250, 200), 80, 0, 2)  # Head
        cv2.ellipse(test_img, (220, 190), (15, 20), 0, 0, 360, 0, 2)  # Left eye
        cv2.ellipse(test_img, (280, 190), (15, 20), 0, 0, 360, 0, 2)  # Right eye
        cv2.ellipse(test_img, (250, 240), (30, 20), 0, 0, 180, 0, 2)  # Smile

        cv2.imwrite("test_face.png", test_img)
        image_path = "test_face.png"
    else:
        image_path = sys.argv[1]

    extractor = ContourExtractor()
    contours = extractor.extract_from_file(image_path)

    print(f"\nExtracted {len(contours)} contours:")
    for i, c in enumerate(contours):
        print(f"  {i + 1}: {len(c.points)} points, length={c.length:.1f}, closed={c.is_closed}")

    # Optimize order
    optimized = extractor.optimize_order(contours, (250, 400))
    print(f"\nOptimized order (from bottom center):")
    for i, c in enumerate(optimized):
        print(f"  {i + 1}: starts at ({c.points[0][0]:.0f}, {c.points[0][1]:.0f})")

    # Visualize
    image = cv2.imread(image_path)
    if image is not None:
        viz = extractor.visualize(image.copy(), optimized)
        cv2.imwrite("contours_viz.png", viz)
        print("\nSaved visualization to contours_viz.png")
