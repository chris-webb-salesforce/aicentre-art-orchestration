"""
OpenAI API client for generating line art from portraits.
"""

import os
import base64
import logging
import time
import tempfile
from typing import Optional
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class OpenAIClient:
    """
    Client for OpenAI image generation API.

    Converts portrait photos to single-line artistic renditions.
    """

    def __init__(
        self,
        model: str = "gpt-image-1",
        prompt: str = None,
        size: str = "1024x1024",
        max_retries: int = 3,
        retry_delay: float = 5.0
    ):
        """
        Initialize OpenAI client.

        Args:
            model: OpenAI model to use for image generation
            prompt: Prompt for line art conversion
            size: Output image size
            max_retries: Number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.model = model
        self.prompt = prompt or (
            "Transform this portrait into a minimalist single continuous line drawing. "
            "Use only simple, clean, flowing black lines on a pure white background. "
            "The sketch should be artistic, recognizable, and look like a hand-drawn "
            "portrait with minimal detail. No shading, no filling, just clean single line work."
        )
        self.size = size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client = None

    def initialize(self) -> bool:
        """
        Initialize the OpenAI client.

        Returns:
            True if API key is available and client created.
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY environment variable not set")
            return False

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            return False

    def generate_line_art(
        self,
        image: np.ndarray,
        output_path: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """
        Convert a portrait image to line art using OpenAI.

        Args:
            image: BGR image (portrait photo)
            output_path: Optional path to save the result

        Returns:
            Line art image as numpy array, or None on failure.
        """
        if self._client is None:
            if not self.initialize():
                return None

        # Save image to temporary file (OpenAI needs a file)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, image)

        try:
            return self._generate_from_file(tmp_path, output_path)
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def generate_line_art_from_file(
        self,
        image_path: str,
        output_path: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """
        Convert a portrait image file to line art.

        Args:
            image_path: Path to portrait image
            output_path: Optional path to save the result

        Returns:
            Line art image as numpy array, or None on failure.
        """
        if self._client is None:
            if not self.initialize():
                return None

        return self._generate_from_file(image_path, output_path)

    def _generate_from_file(
        self,
        image_path: str,
        output_path: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """
        Internal method to generate line art from file.

        Args:
            image_path: Path to input image
            output_path: Optional path to save result

        Returns:
            Line art image as numpy array, or None on failure.
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                logger.info(f"Generating line art (attempt {attempt + 1}/{self.max_retries})...")

                with open(image_path, 'rb') as image_file:
                    result = self._client.images.edit(
                        model=self.model,
                        image=image_file,
                        prompt=self.prompt,
                        size=self.size
                    )

                # Extract image data
                image_bytes = self._extract_image_bytes(result)
                if image_bytes is None:
                    raise Exception("No image data in response")

                # Convert to numpy array
                nparr = np.frombuffer(image_bytes, np.uint8)
                line_art = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if line_art is None:
                    raise Exception("Failed to decode image")

                logger.info(f"Line art generated: {line_art.shape}")

                # Save if output path provided
                if output_path:
                    cv2.imwrite(output_path, line_art)
                    logger.info(f"Saved line art to {output_path}")

                return line_art

            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")

                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)

        logger.error(f"Failed to generate line art after {self.max_retries} attempts: {last_error}")
        return None

    def _extract_image_bytes(self, result) -> Optional[bytes]:
        """
        Extract image bytes from OpenAI response.

        Handles both base64 and URL response formats.
        """
        try:
            data = result.data[0]

            # Try base64 first
            if hasattr(data, 'b64_json') and data.b64_json:
                return base64.b64decode(data.b64_json)

            # Try URL
            if hasattr(data, 'url') and data.url:
                import urllib.request
                with urllib.request.urlopen(data.url) as response:
                    return response.read()

            return None

        except Exception as e:
            logger.error(f"Failed to extract image bytes: {e}")
            return None


# Mock client for testing without API calls
class MockOpenAIClient(OpenAIClient):
    """
    Mock OpenAI client for testing.

    Generates a simple line art effect locally without API calls.
    """

    def initialize(self) -> bool:
        logger.info("Mock OpenAI client initialized")
        return True

    def _generate_from_file(
        self,
        image_path: str,
        output_path: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """Generate mock line art using edge detection."""
        logger.info("Generating mock line art (no API call)...")

        image = cv2.imread(image_path)
        if image is None:
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to reduce noise while keeping edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # Detect edges
        edges = cv2.Canny(filtered, 30, 100)

        # Dilate to thicken lines
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Invert (black lines on white background)
        line_art = cv2.bitwise_not(edges)

        # Convert to 3-channel
        line_art = cv2.cvtColor(line_art, cv2.COLOR_GRAY2BGR)

        if output_path:
            cv2.imwrite(output_path, line_art)
            logger.info(f"Saved mock line art to {output_path}")

        return line_art


if __name__ == "__main__":
    # Test the OpenAI client
    logging.basicConfig(level=logging.INFO)

    import sys

    if len(sys.argv) < 2:
        print("Usage: python openai_client.py <image_path> [--mock]")
        print("\nCreating test with mock client...")

        # Create a simple test image
        test_img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        cv2.circle(test_img, (200, 150), 80, (100, 100, 100), -1)  # Head
        cv2.circle(test_img, (170, 140), 10, (50, 50, 50), -1)  # Left eye
        cv2.circle(test_img, (230, 140), 10, (50, 50, 50), -1)  # Right eye
        cv2.ellipse(test_img, (200, 180), (30, 15), 0, 0, 180, (50, 50, 50), 2)  # Smile
        cv2.imwrite("test_portrait.png", test_img)

        client = MockOpenAIClient()
        result = client.generate_line_art_from_file("test_portrait.png", "test_lineart.png")

        if result is not None:
            print(f"Generated line art: {result.shape}")
            print("Saved to test_lineart.png")
        else:
            print("Failed to generate line art")

    else:
        image_path = sys.argv[1]
        use_mock = "--mock" in sys.argv

        if use_mock:
            client = MockOpenAIClient()
        else:
            client = OpenAIClient()
            if not client.initialize():
                print("Failed to initialize OpenAI client. Set OPENAI_API_KEY or use --mock")
                sys.exit(1)

        output_path = Path(image_path).stem + "_lineart.png"
        result = client.generate_line_art_from_file(image_path, output_path)

        if result is not None:
            print(f"Generated line art: {result.shape}")
            print(f"Saved to {output_path}")
        else:
            print("Failed to generate line art")
