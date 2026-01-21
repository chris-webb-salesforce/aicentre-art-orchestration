"""
GCode generator for converting contours to DexArm drawing commands.
"""

import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision.contour_extractor import Contour

logger = logging.getLogger(__name__)


@dataclass
class DrawingBounds:
    """Defines the physical drawing area on paper."""
    x_min: float = -40.0
    x_max: float = 40.0
    y_min: float = 260.0
    y_max: float = 340.0
    z_up: float = 10.0
    z_down: float = 0.0
    feedrate: int = 2000
    travel_feedrate: int = 3000


class GCodeGenerator:
    """
    Converts contours to GCode commands for the DexArm.

    Handles:
    - Scaling contours to fit drawing bounds
    - Path optimization to minimize pen lifts
    - GCode header/footer generation
    - Coordinate transformation from pixels to mm
    """

    def __init__(self, bounds: Optional[DrawingBounds] = None):
        """
        Initialize GCode generator.

        Args:
            bounds: Drawing bounds configuration
        """
        self.bounds = bounds or DrawingBounds()
        self._validate_bounds()

    def generate(
        self,
        contours: List[Contour],
        image_bounds: Tuple[float, float, float, float],
        margin: float = 5.0
    ) -> List[str]:
        """
        Generate GCode from contours.

        Args:
            contours: List of contours to draw
            image_bounds: Source image bounds (x_min, y_min, x_max, y_max)
            margin: Margin to leave around drawing (mm)

        Returns:
            List of GCode command strings.
        """
        if not contours:
            logger.warning("No contours to generate GCode from")
            return []

        gcode = []

        # Add header
        gcode.extend(self._generate_header())

        # Calculate transformation
        src_x_min, src_y_min, src_x_max, src_y_max = image_bounds
        scale, offset_x, offset_y = self._calculate_transform(
            src_x_min, src_y_min, src_x_max, src_y_max, margin
        )

        logger.info(f"Transform: scale={scale:.4f}, offset=({offset_x:.2f}, {offset_y:.2f})")

        # Generate drawing commands for each contour
        total_points = sum(len(c.points) for c in contours)
        logger.info(f"Generating GCode for {len(contours)} contours, {total_points} points")

        for i, contour in enumerate(contours):
            gcode.extend(self._contour_to_gcode(contour, scale, offset_x, offset_y))

        # Add footer
        gcode.extend(self._generate_footer())

        logger.info(f"Generated {len(gcode)} GCode lines")
        return gcode

    def _generate_header(self) -> List[str]:
        """Generate GCode header commands."""
        return [
            ";TOOL_PATH_RENDER_METHOD_LINE",
            ";----------- Start Gcode -----------",
            "M2000;custom:line mode",
            "M888 P0;custom:header is write&draw",
            ";-----------------------------------",
            f"G0 Z{self.bounds.z_up}",
            f"G0 F{self.bounds.travel_feedrate}",
            f"G1 F{self.bounds.feedrate}",
        ]

    def _generate_footer(self) -> List[str]:
        """Generate GCode footer commands."""
        safe_x = (self.bounds.x_min + self.bounds.x_max) / 2
        safe_y = self.bounds.y_min + (self.bounds.y_max - self.bounds.y_min) / 2

        return [
            f"G0 Z{self.bounds.z_up}",
            f"G0 X{safe_x:.2f} Y{safe_y:.2f}",
            ";----------- End Gcode -------------",
            ";-----------------------------------",
        ]

    def _calculate_transform(
        self,
        src_x_min: float,
        src_y_min: float,
        src_x_max: float,
        src_y_max: float,
        margin: float
    ) -> Tuple[float, float, float]:
        """
        Calculate transformation from source (pixel) to target (mm) coordinates.

        Returns:
            Tuple of (scale, offset_x, offset_y)
        """
        # Source dimensions
        src_width = src_x_max - src_x_min
        src_height = src_y_max - src_y_min

        if src_width == 0 or src_height == 0:
            return (1.0, 0.0, 0.0)

        # Target dimensions (with margin)
        target_width = (self.bounds.x_max - self.bounds.x_min) - 2 * margin
        target_height = (self.bounds.y_max - self.bounds.y_min) - 2 * margin

        # Calculate scale (fit within bounds, maintaining aspect ratio)
        scale_x = target_width / src_width
        scale_y = target_height / src_height
        scale = min(scale_x, scale_y)

        # Calculate offsets to center in drawing area
        scaled_width = src_width * scale
        scaled_height = src_height * scale

        target_center_x = (self.bounds.x_min + self.bounds.x_max) / 2
        target_center_y = (self.bounds.y_min + self.bounds.y_max) / 2

        src_center_x = (src_x_min + src_x_max) / 2
        src_center_y = (src_y_min + src_y_max) / 2

        offset_x = target_center_x - src_center_x * scale
        offset_y = target_center_y - src_center_y * scale

        return (scale, offset_x, offset_y)

    def _transform_point(
        self,
        x: float,
        y: float,
        scale: float,
        offset_x: float,
        offset_y: float
    ) -> Tuple[float, float]:
        """Transform a point from source to target coordinates."""
        tx = x * scale + offset_x
        ty = y * scale + offset_y

        # Clamp to bounds
        tx = max(self.bounds.x_min, min(self.bounds.x_max, tx))
        ty = max(self.bounds.y_min, min(self.bounds.y_max, ty))

        return (tx, ty)

    def _contour_to_gcode(
        self,
        contour: Contour,
        scale: float,
        offset_x: float,
        offset_y: float
    ) -> List[str]:
        """Convert a single contour to GCode commands."""
        if not contour.points:
            return []

        gcode = []

        # Move to start position (pen up)
        start_x, start_y = self._transform_point(
            contour.points[0][0],
            contour.points[0][1],
            scale, offset_x, offset_y
        )
        gcode.append(f"G0 X{start_x:.2f} Y{start_y:.2f}")

        # Lower pen
        gcode.append(f"G1 Z{self.bounds.z_down}")

        # Draw all points
        for x, y in contour.points[1:]:
            tx, ty = self._transform_point(x, y, scale, offset_x, offset_y)
            gcode.append(f"G1 X{tx:.2f} Y{ty:.2f}")

        # Close the contour if it's closed
        if contour.is_closed and len(contour.points) > 2:
            tx, ty = self._transform_point(
                contour.points[0][0],
                contour.points[0][1],
                scale, offset_x, offset_y
            )
            gcode.append(f"G1 X{tx:.2f} Y{ty:.2f}")

        # Lift pen
        gcode.append(f"G0 Z{self.bounds.z_up}")

        return gcode

    def estimate_drawing_time(self, gcode: List[str]) -> float:
        """
        Estimate drawing time in seconds.

        This is a rough estimate based on feedrate and distance.

        Args:
            gcode: List of GCode commands

        Returns:
            Estimated time in seconds.
        """
        total_distance = 0.0
        current_pos = (0.0, 0.0, self.bounds.z_up)
        current_feedrate = self.bounds.feedrate

        for line in gcode:
            line = line.strip().upper()

            # Skip comments and empty lines
            if not line or line.startswith(';'):
                continue

            # Parse feedrate
            if 'F' in line:
                try:
                    f_idx = line.index('F')
                    f_end = f_idx + 1
                    while f_end < len(line) and (line[f_end].isdigit() or line[f_end] == '.'):
                        f_end += 1
                    current_feedrate = float(line[f_idx + 1:f_end])
                except (ValueError, IndexError):
                    pass

            # Parse movement
            if line.startswith('G0') or line.startswith('G1'):
                new_pos = list(current_pos)

                for axis, idx in [('X', 0), ('Y', 1), ('Z', 2)]:
                    if axis in line:
                        try:
                            a_idx = line.index(axis)
                            a_end = a_idx + 1
                            while a_end < len(line) and (line[a_end].isdigit() or line[a_end] in '.+-'):
                                a_end += 1
                            new_pos[idx] = float(line[a_idx + 1:a_end])
                        except (ValueError, IndexError):
                            pass

                # Calculate distance
                dx = new_pos[0] - current_pos[0]
                dy = new_pos[1] - current_pos[1]
                dz = new_pos[2] - current_pos[2]
                distance = (dx**2 + dy**2 + dz**2) ** 0.5

                total_distance += distance
                current_pos = tuple(new_pos)

        # Convert distance to time (feedrate is mm/min)
        avg_feedrate = (self.bounds.feedrate + self.bounds.travel_feedrate) / 2
        if avg_feedrate <= 0:
            logger.warning("Invalid feedrate, cannot estimate time")
            return 0.0
        time_minutes = total_distance / avg_feedrate
        return time_minutes * 60  # Convert to seconds

    def _validate_bounds(self):
        """Validate that bounds are configured correctly."""
        if self.bounds.x_min >= self.bounds.x_max:
            raise ValueError(f"Invalid X bounds: x_min ({self.bounds.x_min}) must be < x_max ({self.bounds.x_max})")
        if self.bounds.y_min >= self.bounds.y_max:
            raise ValueError(f"Invalid Y bounds: y_min ({self.bounds.y_min}) must be < y_max ({self.bounds.y_max})")
        if self.bounds.z_down >= self.bounds.z_up:
            logger.warning(f"z_down ({self.bounds.z_down}) >= z_up ({self.bounds.z_up}), pen may not lift properly")

    def save_to_file(self, gcode: List[str], filepath: str):
        """Save GCode to a file."""
        with open(filepath, 'w') as f:
            for line in gcode:
                f.write(line + '\n')
        logger.info(f"Saved GCode to {filepath}")


if __name__ == "__main__":
    # Test GCode generation
    logging.basicConfig(level=logging.INFO)

    # Create test contours
    test_contours = [
        # Circle (head)
        Contour(
            points=[(250 + 80 * __import__('math').cos(a), 200 + 80 * __import__('math').sin(a))
                   for a in [i * 3.14159 / 16 for i in range(33)]],
            is_closed=True,
            area=20106,
            length=502
        ),
        # Left eye
        Contour(
            points=[(205, 180), (210, 175), (220, 175), (230, 180), (220, 185), (210, 185), (205, 180)],
            is_closed=True,
            area=150,
            length=60
        ),
        # Right eye
        Contour(
            points=[(270, 180), (275, 175), (285, 175), (295, 180), (285, 185), (275, 185), (270, 180)],
            is_closed=True,
            area=150,
            length=60
        ),
        # Smile
        Contour(
            points=[(220, 230), (230, 250), (250, 260), (270, 250), (280, 230)],
            is_closed=False,
            area=0,
            length=100
        ),
    ]

    # Get bounds from contours
    all_points = [p for c in test_contours for p in c.points]
    x_coords = [p[0] for p in all_points]
    y_coords = [p[1] for p in all_points]
    image_bounds = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

    print(f"Image bounds: {image_bounds}")

    # Generate GCode
    generator = GCodeGenerator()
    gcode = generator.generate(test_contours, image_bounds)

    print(f"\nGenerated {len(gcode)} GCode lines:")
    for line in gcode[:20]:
        print(f"  {line}")
    if len(gcode) > 20:
        print(f"  ... ({len(gcode) - 20} more lines)")

    # Estimate time
    est_time = generator.estimate_drawing_time(gcode)
    print(f"\nEstimated drawing time: {est_time:.1f} seconds ({est_time/60:.1f} minutes)")

    # Save to file
    generator.save_to_file(gcode, "test_output.gcode")
    print("\nSaved to test_output.gcode")
