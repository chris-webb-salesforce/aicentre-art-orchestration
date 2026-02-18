"""
Adaptive contour extraction that intelligently chooses between methods.

Uses skeletonization for thick/dense lines (like hair) and Canny for thin lines.
This produces cleaner paths for robot drawing.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import logging

from .contour_extractor import Contour

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveExtractorConfig:
    """Configuration for adaptive extraction."""
    # Thickness detection
    thickness_threshold: int = 3       # Pixels - lines thicker than this use skeleton
    density_kernel_size: int = 15      # Window size for local density calculation
    density_threshold: float = 0.3     # Density above this triggers skeleton method

    # Canny parameters (for thin lines)
    canny_low: int = 30
    canny_high: int = 100

    # Skeleton parameters (for thick lines)
    skeleton_simplify: float = 1.0

    # General parameters
    simplify_epsilon: float = 0.8
    min_area: int = 50
    min_contour_points: int = 5
    blur_kernel: int = 3

    # Method selection
    method: str = "adaptive"  # "adaptive", "canny", "skeleton", "hybrid"

    # Thinning parameters (Zhang-Suen)
    thinning_threshold: int = 127      # Binary threshold for thinning (0-255)
    thinning_cleanup: bool = True      # Post-thinning morphological cleanup to reconnect gaps
    thinning_cleanup_kernel: int = 2   # Kernel size for cleanup

    # Noise filtering
    min_straightness: float = 0.15     # Min bbox_diagonal / path_length ratio (filters squiggly noise)

    # Speed optimizations
    min_length: float = 10.0           # Minimum contour length in pixels (filters tiny lines)
    merge_distance: float = 5.0        # Merge contours with endpoints within this distance
    merge_enabled: bool = True         # Enable contour merging to reduce pen lifts

    # Region-aware processing (preserves detail in eyes/facial features)
    region_aware: bool = False         # Enable face/eye detection for detail preservation
    detail_simplify_epsilon: float = 0.3   # Lower epsilon for detail regions (more points)
    detail_min_length: float = 3.0     # Keep smaller contours in detail regions
    detail_min_area: int = 10          # Keep smaller areas in detail regions
    detail_region_padding: int = 20    # Pixels to expand around detected features


class AdaptiveContourExtractor:
    """
    Intelligently extracts contours using the best method for each image region.

    Methods:
    - adaptive: Analyzes stroke thickness and uses appropriate method per-region
    - canny: Traditional Canny edge detection (good for thin lines)
    - skeleton: Skeletonization using Zhang-Suen thinning (good for thick lines)
    - thinning: Direct Zhang-Suen thinning pipeline (cleanest single-pixel skeleton)
    - hybrid: Runs both skeleton and canny, merges results intelligently
    """

    def __init__(self, config: AdaptiveExtractorConfig = None):
        self.config = config or AdaptiveExtractorConfig()

    def extract(self, image: np.ndarray) -> List[Contour]:
        """
        Extract contours using the configured method.

        Args:
            image: BGR or grayscale image

        Returns:
            List of Contour objects optimized for drawing
        """
        method = self.config.method.lower()

        # Use region-aware processing if enabled
        if self.config.region_aware:
            contours = self._extract_region_aware(image, method)
        elif method == "adaptive":
            contours = self._extract_adaptive(image)
        elif method == "skeleton":
            contours = self._extract_skeleton(image)
        elif method == "canny":
            contours = self._extract_canny(image)
        elif method == "hybrid":
            contours = self._extract_hybrid(image)
        elif method == "thinning":
            contours = self._extract_thinning(image)
        else:
            logger.warning(f"Unknown method '{method}', using adaptive")
            contours = self._extract_adaptive(image)

        # Merge nearby contours to reduce pen lifts
        if self.config.merge_enabled:
            before_merge = len(contours)
            contours = self.merge_nearby_contours(contours)
            print(f"  Merge: {before_merge} -> {len(contours)} contours (merge_distance={self.config.merge_distance})")
        else:
            print(f"  Merge disabled")

        return contours

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert to grayscale and binary."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Invert if background is white
        if np.mean(gray) > 127:
            gray = cv2.bitwise_not(gray)

        # Create binary image
        _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

        return gray, binary

    def _detect_detail_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect facial feature regions that need more detail (eyes, eyebrows, nose, mouth).

        Uses OpenCV's Haar cascades for face and eye detection, then estimates
        other feature locations relative to detected features.

        Returns:
            List of (x, y, w, h) bounding boxes for detail regions.
        """
        # Convert to grayscale for detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # If image is inverted (white lines on black), invert for detection
        if np.mean(gray) < 127:
            detect_gray = cv2.bitwise_not(gray)
        else:
            detect_gray = gray

        detail_regions = []
        padding = self.config.detail_region_padding
        h, w = gray.shape[:2]

        # Try to load Haar cascades
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
        except Exception as e:
            logger.warning(f"Could not load Haar cascades: {e}")
            # Fall back to center-based estimation
            return self._estimate_detail_regions_from_center(w, h)

        # Detect faces
        faces = face_cascade.detectMultiScale(detect_gray, 1.1, 4, minSize=(50, 50))

        if len(faces) == 0:
            logger.info("No faces detected, using center-based estimation")
            return self._estimate_detail_regions_from_center(w, h)

        # For each face, detect eyes and estimate other features
        for (fx, fy, fw, fh) in faces:
            face_roi = detect_gray[fy:fy+fh, fx:fx+fw]

            # Detect eyes within face region
            eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 3, minSize=(20, 20))

            if len(eyes) >= 2:
                # Use detected eyes
                for (ex, ey, ew, eh) in eyes[:2]:
                    # Convert to full image coordinates and add padding
                    x = max(0, fx + ex - padding)
                    y = max(0, fy + ey - padding)
                    region_w = min(w - x, ew + 2 * padding)
                    region_h = min(h - y, eh + 2 * padding)
                    detail_regions.append((x, y, region_w, region_h))
                    logger.debug(f"Detected eye region: ({x}, {y}, {region_w}, {region_h})")
            else:
                # Estimate eye regions based on face proportions
                # Eyes are typically in upper 1/3 of face, 1/4 from each side
                eye_y = fy + int(fh * 0.25)
                eye_h = int(fh * 0.2)

                # Left eye region
                left_eye_x = fx + int(fw * 0.15)
                eye_w = int(fw * 0.3)
                detail_regions.append((
                    max(0, left_eye_x - padding),
                    max(0, eye_y - padding),
                    min(w - left_eye_x, eye_w + 2 * padding),
                    min(h - eye_y, eye_h + 2 * padding)
                ))

                # Right eye region
                right_eye_x = fx + int(fw * 0.55)
                detail_regions.append((
                    max(0, right_eye_x - padding),
                    max(0, eye_y - padding),
                    min(w - right_eye_x, eye_w + 2 * padding),
                    min(h - eye_y, eye_h + 2 * padding)
                ))

            # Add nose tip region (center of face, lower middle)
            nose_x = fx + int(fw * 0.35)
            nose_y = fy + int(fh * 0.45)
            nose_w = int(fw * 0.3)
            nose_h = int(fh * 0.2)
            detail_regions.append((
                max(0, nose_x - padding // 2),
                max(0, nose_y - padding // 2),
                min(w - nose_x, nose_w + padding),
                min(h - nose_y, nose_h + padding)
            ))

            # Add mouth region
            mouth_x = fx + int(fw * 0.25)
            mouth_y = fy + int(fh * 0.7)
            mouth_w = int(fw * 0.5)
            mouth_h = int(fh * 0.15)
            detail_regions.append((
                max(0, mouth_x - padding // 2),
                max(0, mouth_y - padding // 2),
                min(w - mouth_x, mouth_w + padding),
                min(h - mouth_y, mouth_h + padding)
            ))

        logger.info(f"Detected {len(detail_regions)} detail regions")
        return detail_regions

    def _estimate_detail_regions_from_center(self, w: int, h: int) -> List[Tuple[int, int, int, int]]:
        """
        Estimate detail regions when face detection fails.
        Assumes face is roughly centered in image.
        """
        padding = self.config.detail_region_padding

        # Assume face occupies center 60% of image
        face_x = int(w * 0.2)
        face_y = int(h * 0.15)
        face_w = int(w * 0.6)
        face_h = int(h * 0.7)

        detail_regions = []

        # Eye regions (upper third of estimated face area)
        eye_y = face_y + int(face_h * 0.2)
        eye_h = int(face_h * 0.15)
        eye_w = int(face_w * 0.25)

        # Left eye
        detail_regions.append((
            face_x + int(face_w * 0.1),
            eye_y,
            eye_w + padding,
            eye_h + padding
        ))

        # Right eye
        detail_regions.append((
            face_x + int(face_w * 0.55),
            eye_y,
            eye_w + padding,
            eye_h + padding
        ))

        # Nose
        detail_regions.append((
            face_x + int(face_w * 0.35),
            face_y + int(face_h * 0.4),
            int(face_w * 0.3),
            int(face_h * 0.2)
        ))

        # Mouth
        detail_regions.append((
            face_x + int(face_w * 0.25),
            face_y + int(face_h * 0.65),
            int(face_w * 0.5),
            int(face_h * 0.15)
        ))

        logger.info(f"Using estimated detail regions: {len(detail_regions)} regions")
        return detail_regions

    def _extract_region_aware(self, image: np.ndarray, base_method: str) -> List[Contour]:
        """
        Extract contours with special handling for detail regions.

        Process:
        1. Detect facial feature regions (eyes, nose, mouth)
        2. Extract contours from detail regions with fine settings
        3. Extract contours from rest of image with normal settings
        4. Combine results, preferring detailed versions in overlap areas
        """
        h, w = image.shape[:2]
        gray, binary = self._preprocess(image)

        # Detect detail regions
        detail_regions = self._detect_detail_regions(image)

        if not detail_regions:
            # No detail regions found, use standard extraction
            logger.info("No detail regions found, using standard extraction")
            if base_method == "hybrid":
                return self._extract_hybrid(image)
            elif base_method == "skeleton":
                return self._extract_skeleton(image)
            elif base_method == "thinning":
                return self._extract_thinning(image)
            elif base_method == "adaptive":
                return self._extract_adaptive(image)
            else:
                return self._extract_canny(image)

        # Create mask for detail regions
        detail_mask = np.zeros((h, w), dtype=np.uint8)
        for (x, y, rw, rh) in detail_regions:
            cv2.rectangle(detail_mask, (x, y), (x + rw, y + rh), 255, -1)

        # Save original config values
        orig_simplify = self.config.simplify_epsilon
        orig_min_length = self.config.min_length
        orig_min_area = self.config.min_area
        orig_skeleton_simplify = self.config.skeleton_simplify

        all_contours = []

        # Extract from detail regions with fine settings
        logger.info("Extracting detail regions with fine settings...")
        self.config.simplify_epsilon = self.config.detail_simplify_epsilon
        self.config.min_length = self.config.detail_min_length
        self.config.min_area = self.config.detail_min_area
        self.config.skeleton_simplify = self.config.detail_simplify_epsilon

        # Mask image to only show detail regions
        detail_image = image.copy()
        if len(detail_image.shape) == 3:
            detail_image[detail_mask == 0] = [255, 255, 255]  # White background
        else:
            detail_image[detail_mask == 0] = 255

        # Extract from detail regions
        if base_method == "hybrid":
            detail_contours = self._extract_hybrid(detail_image)
        elif base_method == "skeleton":
            detail_contours = self._extract_skeleton(detail_image)
        elif base_method == "thinning":
            detail_contours = self._extract_thinning(detail_image)
        elif base_method == "adaptive":
            detail_contours = self._extract_adaptive(detail_image)
        else:
            detail_contours = self._extract_canny(detail_image)

        logger.info(f"Detail regions: {len(detail_contours)} contours")
        all_contours.extend(detail_contours)

        # Restore original settings for non-detail regions
        self.config.simplify_epsilon = orig_simplify
        self.config.min_length = orig_min_length
        self.config.min_area = orig_min_area
        self.config.skeleton_simplify = orig_skeleton_simplify

        # Extract from non-detail regions
        logger.info("Extracting non-detail regions with normal settings...")
        non_detail_mask = cv2.bitwise_not(detail_mask)

        non_detail_image = image.copy()
        if len(non_detail_image.shape) == 3:
            non_detail_image[non_detail_mask == 0] = [255, 255, 255]
        else:
            non_detail_image[non_detail_mask == 0] = 255

        if base_method == "hybrid":
            non_detail_contours = self._extract_hybrid(non_detail_image)
        elif base_method == "skeleton":
            non_detail_contours = self._extract_skeleton(non_detail_image)
        elif base_method == "thinning":
            non_detail_contours = self._extract_thinning(non_detail_image)
        elif base_method == "adaptive":
            non_detail_contours = self._extract_adaptive(non_detail_image)
        else:
            non_detail_contours = self._extract_canny(non_detail_image)

        logger.info(f"Non-detail regions: {len(non_detail_contours)} contours")
        all_contours.extend(non_detail_contours)

        logger.info(f"Region-aware extraction: {len(all_contours)} total contours")
        return all_contours

    def _analyze_thickness(self, binary: np.ndarray) -> np.ndarray:
        """
        Analyze local stroke thickness using distance transform.

        Returns a map where higher values indicate thicker strokes.
        """
        # Distance transform gives distance to nearest background pixel
        # For stroke pixels, this approximates half the stroke width
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        return dist_transform

    def _analyze_density(self, binary: np.ndarray) -> np.ndarray:
        """
        Analyze local density of strokes.

        Returns a map where higher values indicate denser areas.
        """
        kernel_size = self.config.density_kernel_size
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)

        # Local density is the average pixel value in the neighborhood
        density = cv2.filter2D(binary.astype(np.float32) / 255.0, -1, kernel)

        return density

    def _extract_adaptive(self, image: np.ndarray) -> List[Contour]:
        """
        Adaptively choose method based on stroke characteristics.

        Strategy:
        1. Analyze the image for thick vs thin regions
        2. Create masks for each region type
        3. Apply skeleton to thick regions, Canny to thin regions
        4. Merge results
        """
        gray, binary = self._preprocess(image)

        # Analyze stroke characteristics
        thickness_map = self._analyze_thickness(binary)
        density_map = self._analyze_density(binary)

        # Determine which regions need which method
        # Thick strokes: use skeleton (thickness > threshold)
        thick_mask = (thickness_map > self.config.thickness_threshold).astype(np.uint8) * 255

        # Dense regions: also use skeleton
        dense_mask = (density_map > self.config.density_threshold).astype(np.uint8) * 255

        # Combine masks - use skeleton where thick OR dense
        skeleton_mask = cv2.bitwise_or(thick_mask, dense_mask)

        # Dilate the skeleton mask to include surrounding areas
        kernel = np.ones((5, 5), np.uint8)
        skeleton_mask = cv2.dilate(skeleton_mask, kernel, iterations=2)

        # Canny mask is the inverse (thin, sparse regions)
        canny_mask = cv2.bitwise_not(skeleton_mask)

        # Calculate what percentage uses each method
        total_stroke_pixels = np.count_nonzero(binary)
        skeleton_pixels = np.count_nonzero(cv2.bitwise_and(binary, skeleton_mask))

        if total_stroke_pixels > 0:
            skeleton_pct = skeleton_pixels / total_stroke_pixels * 100
            logger.info(f"Adaptive extraction: {skeleton_pct:.1f}% skeleton, {100-skeleton_pct:.1f}% canny")

        # Extract from skeleton regions
        skeleton_region = cv2.bitwise_and(binary, skeleton_mask)
        skeleton_contours = self._skeletonize_and_extract(skeleton_region)

        # Extract from canny regions
        canny_region = cv2.bitwise_and(gray, canny_mask)
        canny_contours = self._canny_and_extract(canny_region)

        # Merge results
        all_contours = skeleton_contours + canny_contours

        logger.info(f"Adaptive: {len(skeleton_contours)} skeleton + {len(canny_contours)} canny = {len(all_contours)} total")

        return all_contours

    def _extract_hybrid(self, image: np.ndarray) -> List[Contour]:
        """
        Combine thinning (centerlines) with Canny (fine detail) using skeleton-first
        deduplication to eliminate double walls.

        The problem: Canny edge detection finds BOTH edges of thick strokes, producing
        parallel "double wall" contours alongside the thinning centerline. A thick stroke
        becomes 3 lines (left edge + center + right edge) instead of 1.

        The solution: skeleton-first coverage. Skeleton (thinning) contours are always
        kept — they represent the ground truth centerlines. A coverage map is built
        around each skeleton contour using the local stroke width (from the distance
        transform). Canny contours that mostly fall within this coverage are duplicate
        edges and get removed. Canny contours that extend beyond the coverage capture
        unique thin details (beard hairs, fine lines) and survive.

        Tuning guide:
        - Coverage thickness formula: max(max_half_width * 2 + 2, 3)
          The *2 converts half-width to full stroke width, +2 adds 1px buffer per side.
          Increase the +2 to catch more edges (risks removing thin details near thick
          strokes). Decrease to preserve more detail (risks double walls).
        - Overlap threshold (currently 0.8): fraction of Canny points inside skeleton
          coverage needed to discard that contour. Lower = more aggressive filtering
          (fewer double walls, but loses detail). Higher = more permissive (more detail,
          but double walls may survive). Tested range: 0.5 (too aggressive, only ~11
          Canny survive) to 0.8 (good balance of detail vs double walls).
        - Canny-to-Canny dedup: surviving Canny contours also add to the coverage map
          (processed longest-first), preventing duplicate Canny contours from both
          surviving.
        """
        gray, binary = self._preprocess(image)

        skeleton_contours = self._skeletonize_and_extract(binary)
        canny_contours = self._canny_and_extract(gray)

        logger.info(f"Hybrid: {len(skeleton_contours)} skeleton, {len(canny_contours)} canny candidates")

        h, w = binary.shape
        # Distance transform: value at each foreground pixel = distance to nearest
        # background pixel, approximating local half-stroke-width
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        # Phase 1: Build coverage from ALL skeleton contours (they always survive).
        # Each skeleton contour "claims" a band as wide as the stroke it represents.
        # Any Canny edge running parallel within this band is a duplicate.
        coverage = np.zeros((h, w), dtype=np.uint8)
        for contour in skeleton_contours:
            pts = np.array(contour.points, dtype=np.int32)
            # Find the thickest point along this skeleton contour
            max_half_width = 0
            for pt in pts:
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < w and 0 <= y < h:
                    max_half_width = max(max_half_width, dist_transform[y, x])
            # Coverage band: full stroke width (2 * half_width) + 1px buffer per side
            thickness = max(int(max_half_width * 2) + 2, 3)
            cv2.polylines(coverage, [pts], False, 255, thickness=thickness)

        # Phase 2: Filter Canny contours against skeleton coverage.
        # Process longest Canny contours first — kept contours add to the coverage
        # map, deduplicating shorter Canny contours that overlap with them.
        canny_indexed = sorted(enumerate(canny_contours), key=lambda x: -x[1].length)
        keep_canny = [True] * len(canny_contours)

        for idx, contour in canny_indexed:
            pts = np.array(contour.points, dtype=np.int32)
            # Count how many points fall inside the existing coverage zone
            covered = 0
            for pt in pts:
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < w and 0 <= y < h:
                    if coverage[y, x] > 0:
                        covered += 1
            ratio = covered / max(len(pts), 1)

            # If most of this contour is already covered, it's a duplicate edge
            if ratio > 0.8:
                keep_canny[idx] = False
                continue

            # This Canny contour adds unique detail — keep it, and add its own
            # coverage to prevent duplicate Canny contours from also surviving
            max_hw = 0
            for pt in pts:
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < w and 0 <= y < h:
                    max_hw = max(max_hw, dist_transform[y, x])
            thickness = max(int(max_hw * 2) + 2, 3)
            cv2.polylines(coverage, [pts], False, 255, thickness=thickness)

        filtered_canny = [c for i, c in enumerate(canny_contours) if keep_canny[i]]
        removed = len(canny_contours) - len(filtered_canny)
        logger.info(f"Skeleton coverage filtered {removed} Canny contours, kept {len(filtered_canny)}")

        all_contours = skeleton_contours + filtered_canny
        logger.info(f"Hybrid result: {len(all_contours)} contours ({len(skeleton_contours)} skeleton + {len(filtered_canny)} canny)")

        return all_contours

    def _extract_skeleton(self, image: np.ndarray) -> List[Contour]:
        """Extract using skeletonization (now uses Zhang-Suen thinning)."""
        _, binary = self._preprocess(image)
        return self._skeletonize_and_extract(binary)

    def _extract_canny(self, image: np.ndarray) -> List[Contour]:
        """Extract using only Canny edge detection."""
        gray, _ = self._preprocess(image)
        return self._canny_and_extract(gray)

    def _extract_thinning(self, image: np.ndarray) -> List[Contour]:
        """Extract using Zhang-Suen thinning for clean 1px skeleton."""
        _, binary = self._preprocess(image)
        return self._thinning_and_extract(binary)

    def _zhang_suen_thinning(self, binary: np.ndarray) -> np.ndarray:
        """
        Zhang-Suen thinning algorithm.
        Reduces binary image to 1-pixel wide skeleton while preserving topology.

        Args:
            binary: Binary image (0 or 255, foreground=255)

        Returns:
            Thinned binary image (0 or 255)
        """
        # Try the optimized C++ version first
        try:
            return cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        except (AttributeError, cv2.error):
            pass

        # Pure NumPy fallback
        img = (binary > 0).astype(np.uint8)

        while True:
            # Sub-iteration 1
            markers_1 = self._zhang_suen_iteration(img, step=1)
            img[markers_1] = 0

            # Sub-iteration 2
            markers_2 = self._zhang_suen_iteration(img, step=2)
            img[markers_2] = 0

            # Stop if no pixels were removed
            if not np.any(markers_1) and not np.any(markers_2):
                break

        return (img * 255).astype(np.uint8)

    def _zhang_suen_iteration(self, img: np.ndarray, step: int) -> np.ndarray:
        """
        One sub-iteration of Zhang-Suen thinning.

        Args:
            img: Binary image (0 or 1)
            step: 1 or 2 (the two sub-iterations have different conditions)

        Returns:
            Boolean mask of pixels to remove.
        """
        # Extract 8-neighbors using array slicing
        # P2=N, P3=NE, P4=E, P5=SE, P6=S, P7=SW, P8=W, P9=NW
        p2 = img[0:-2, 1:-1]   # North
        p3 = img[0:-2, 2:]     # NE
        p4 = img[1:-1, 2:]     # East
        p5 = img[2:,   2:]     # SE
        p6 = img[2:,   1:-1]   # South
        p7 = img[2:,   0:-2]   # SW
        p8 = img[1:-1, 0:-2]   # West
        p9 = img[0:-2, 0:-2]   # NW

        center = img[1:-1, 1:-1]

        # B(P1): number of non-zero neighbors (must be 2-6)
        B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        cond_B = (B >= 2) & (B <= 6)

        # A(P1): number of 0->1 transitions in clockwise order (must be exactly 1)
        A = (((p2 == 0) & (p3 == 1)).astype(np.int32) +
             ((p3 == 0) & (p4 == 1)).astype(np.int32) +
             ((p4 == 0) & (p5 == 1)).astype(np.int32) +
             ((p5 == 0) & (p6 == 1)).astype(np.int32) +
             ((p6 == 0) & (p7 == 1)).astype(np.int32) +
             ((p7 == 0) & (p8 == 1)).astype(np.int32) +
             ((p8 == 0) & (p9 == 1)).astype(np.int32) +
             ((p9 == 0) & (p2 == 1)).astype(np.int32))
        cond_A = (A == 1)

        if step == 1:
            cond3 = (p2 * p4 * p6 == 0)
            cond4 = (p4 * p6 * p8 == 0)
        else:
            cond3 = (p2 * p4 * p8 == 0)
            cond4 = (p2 * p6 * p8 == 0)

        # Build result mask (padded with False on borders)
        result = np.zeros(img.shape, dtype=bool)
        result[1:-1, 1:-1] = (center == 1) & cond_B & cond_A & cond3 & cond4

        return result

    def _thinning_and_extract(self, binary: np.ndarray) -> List[Contour]:
        """
        Apply Zhang-Suen thinning and extract contours.

        Pipeline: thin -> optional cleanup -> findContours -> approxPolyDP
        """
        if np.count_nonzero(binary) == 0:
            return []

        # Apply Zhang-Suen thinning
        skeleton = self._zhang_suen_thinning(binary)

        # Post-thinning cleanup: close small gaps from thinning artifacts
        if self.config.thinning_cleanup:
            k = self.config.thinning_cleanup_kernel
            kernel = np.ones((k, k), np.uint8)
            skeleton = cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Find contours on thinned skeleton
        # Use CHAIN_APPROX_SIMPLE to compress redundant points from 1px skeleton tracing
        contours_cv, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Skip area filter for skeleton contours (1px lines have near-zero area)
        return self._process_contours(contours_cv, self.config.skeleton_simplify, skip_area_filter=True)

    def _skeletonize_and_extract(self, binary: np.ndarray) -> List[Contour]:
        """Apply skeletonization and extract contours. Uses Zhang-Suen thinning."""
        return self._thinning_and_extract(binary)

    def _canny_and_extract(self, gray: np.ndarray) -> List[Contour]:
        """Apply Canny edge detection and extract contours."""
        if np.count_nonzero(gray) == 0:
            return []

        # Blur
        kernel_size = (self.config.blur_kernel, self.config.blur_kernel)
        blurred = cv2.GaussianBlur(gray, kernel_size, 0)

        # Canny edge detection
        edges = cv2.Canny(blurred, self.config.canny_low, self.config.canny_high)

        # Dilate to connect nearby edges
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Find contours
        contours_cv, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        return self._process_contours(contours_cv, self.config.simplify_epsilon)

    def _process_contours(self, contours_cv, simplify_epsilon: float, skip_area_filter: bool = False) -> List[Contour]:
        """Convert OpenCV contours to our Contour format with filtering.

        Args:
            skip_area_filter: If True, skip the area-based filter. Used for skeleton/thinning
                contours where 1px lines have near-zero enclosed area by nature.
        """
        contours = []

        for cv_contour in contours_cv:
            area = cv2.contourArea(cv_contour)

            # Filter by area and point count (skip area check for skeleton-derived contours)
            if not skip_area_filter:
                if area < self.config.min_area and len(cv_contour) < self.config.min_contour_points:
                    continue

            # Simplify
            simplified = cv2.approxPolyDP(cv_contour, simplify_epsilon, closed=False)
            points = [(float(p[0][0]), float(p[0][1])) for p in simplified]

            if len(points) < 2:
                continue

            # Calculate length
            length = 0.0
            for i in range(len(points) - 1):
                dx = points[i + 1][0] - points[i][0]
                dy = points[i + 1][1] - points[i][1]
                length += np.sqrt(dx * dx + dy * dy)

            # Filter by minimum length
            if length < self.config.min_length:
                continue

            # Filter squiggly noise: contours with high path length relative to spatial extent
            if self.config.min_straightness > 0 and length > 0:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                bbox_diag = np.sqrt((max(xs) - min(xs))**2 + (max(ys) - min(ys))**2)
                straightness = bbox_diag / length
                if straightness < self.config.min_straightness:
                    continue

            # Check if closed
            is_closed = False
            if len(points) > 2:
                first, last = points[0], points[-1]
                dist = np.sqrt((first[0] - last[0])**2 + (first[1] - last[1])**2)
                is_closed = dist < 5

            contours.append(Contour(
                points=points,
                is_closed=is_closed,
                area=area,
                length=length
            ))

        return contours

    def merge_nearby_contours(self, contours: List[Contour]) -> List[Contour]:
        """
        Merge contours whose endpoints are close together.

        This reduces pen lifts by connecting nearby line segments into longer paths.
        """
        if not self.config.merge_enabled or len(contours) < 2:
            return contours

        merge_dist = self.config.merge_distance
        merged = []
        used = [False] * len(contours)

        for i, contour in enumerate(contours):
            if used[i] or contour.is_closed:
                if not used[i]:
                    merged.append(contour)
                    used[i] = True
                continue

            # Start building a chain from this contour
            chain_points = list(contour.points)
            chain_length = contour.length
            used[i] = True

            # Keep trying to extend the chain
            changed = True
            while changed:
                changed = False
                chain_start = chain_points[0]
                chain_end = chain_points[-1]

                for j, other in enumerate(contours):
                    if used[j] or other.is_closed:
                        continue

                    other_start = other.points[0]
                    other_end = other.points[-1]

                    # Check all four connection possibilities
                    # 1. Our end -> their start
                    dist = np.sqrt((chain_end[0] - other_start[0])**2 +
                                   (chain_end[1] - other_start[1])**2)
                    if dist < merge_dist:
                        chain_points.extend(other.points)
                        chain_length += other.length
                        used[j] = True
                        changed = True
                        continue

                    # 2. Our end -> their end (reverse other)
                    dist = np.sqrt((chain_end[0] - other_end[0])**2 +
                                   (chain_end[1] - other_end[1])**2)
                    if dist < merge_dist:
                        chain_points.extend(reversed(other.points))
                        chain_length += other.length
                        used[j] = True
                        changed = True
                        continue

                    # 3. Their end -> our start (prepend)
                    dist = np.sqrt((chain_start[0] - other_end[0])**2 +
                                   (chain_start[1] - other_end[1])**2)
                    if dist < merge_dist:
                        chain_points = list(other.points) + chain_points
                        chain_length += other.length
                        used[j] = True
                        changed = True
                        continue

                    # 4. Their start -> our start (reverse other, prepend)
                    dist = np.sqrt((chain_start[0] - other_start[0])**2 +
                                   (chain_start[1] - other_start[1])**2)
                    if dist < merge_dist:
                        chain_points = list(reversed(other.points)) + chain_points
                        chain_length += other.length
                        used[j] = True
                        changed = True
                        continue

            # Check if the merged chain is now closed
            first, last = chain_points[0], chain_points[-1]
            is_closed = np.sqrt((first[0] - last[0])**2 + (first[1] - last[1])**2) < 5

            merged.append(Contour(
                points=chain_points,
                is_closed=is_closed,
                area=0,  # Area not meaningful for merged contours
                length=chain_length
            ))

        # Add any remaining unused contours (closed ones)
        for i, contour in enumerate(contours):
            if not used[i]:
                merged.append(contour)

        logger.info(f"Merged {len(contours)} contours into {len(merged)} (saved {len(contours) - len(merged)} pen lifts)")
        return merged

    def optimize_order(
        self,
        contours: List[Contour],
        start_point: Tuple[float, float] = (0, 0)
    ) -> List[Contour]:
        """Optimize contour order to minimize pen travel (greedy nearest-neighbor)."""
        if not contours:
            return []

        remaining = list(contours)
        ordered = []
        current_pos = start_point

        while remaining:
            best_idx = 0
            best_dist = float('inf')
            best_reversed = False

            for i, contour in enumerate(remaining):
                # Distance to start
                start = contour.points[0]
                dist_to_start = np.sqrt(
                    (start[0] - current_pos[0])**2 +
                    (start[1] - current_pos[1])**2
                )

                # Distance to end (can draw in reverse)
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

            contour = remaining.pop(best_idx)

            if best_reversed:
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
        """Get bounding box of all contours."""
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


# Convenience function for quick testing
def extract_adaptive(image: np.ndarray, method: str = "adaptive") -> List[Contour]:
    """Quick extraction with default settings."""
    config = AdaptiveExtractorConfig(method=method)
    extractor = AdaptiveContourExtractor(config)
    return extractor.extract(image)
