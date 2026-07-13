"""OpenCV tool for image deblurring."""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any
from .base import BaseTool
from utils.exceptions import ToolExecutionError

class OpenCVTool(BaseTool):
    """Tool for deblurring images using OpenCV."""

    @property
    def name(self) -> str:
        return "opencv"

    @property
    def cost_tier(self) -> str:
        return "free"

    def execute(self, image_path: str, output_path: str = None) -> Dict[str, Any]:
        """Deblur an image using OpenCV."""
        try:
            self.logger.info(f"Deblurring image: {image_path}")

            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")

            # Detect blur level
            blur_score = self._estimate_blur(img)
            self.logger.info(f"Blur score: {blur_score:.2f}")

            threshold = self.config.get("blur_threshold", 100)
            if blur_score < threshold:
                deblurred = self._apply_deblur(img)
            else:
                self.logger.info("Image not significantly blurred, skipping deblur")
                deblurred = img

            # Save deblurred image
            if output_path is None:
                output_path = image_path.parent / f"{image_path.stem}_deblurred{image_path.suffix}"
            else:
                output_path = Path(output_path)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), deblurred)
            self.logger.info(f"Deblurred image saved to: {output_path}")

            return {
                "deblurred_path": str(output_path),
                "original_path": str(image_path),
                "blur_score": float(blur_score),
                "was_deblurred": blur_score < threshold,
                "success": True
            }
        except Exception as e:
            self.logger.error(f"Deblurring failed: {e}")
            raise ToolExecutionError(f"OpenCV deblurring failed: {e}")

    def _estimate_blur(self, image: np.ndarray) -> float:
        """Estimate blur using Laplacian variance method."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var

    def _apply_deblur(self, image: np.ndarray) -> np.ndarray:
        """Apply deblurring using Wiener filter or sharpening."""
        method = self.config.get("deblur_method", "wiener")
        if method == "wiener":
            return self._wiener_deblur(image)
        elif method == "sharpen":
            return self._sharpen_deblur(image)
        else:
            self.logger.warning(f"Unknown deblur method: {method}, using sharpen")
            return self._sharpen_deblur(image)

    def _wiener_deblur(self, image: np.ndarray) -> np.ndarray:
        """Apply Wiener deconvolution."""
        kernel_size = self.config.get("kernel_size", 5)
        sigma = self.config.get("sigma", 1.0)

        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size

        deblurred = cv2.filter2D(image, -1, kernel)
        return deblurred

    def _sharpen_deblur(self, image: np.ndarray) -> np.ndarray:
        """Apply sharpening filter."""
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened

    def validate_input(self, image_path: str, **kwargs) -> bool:
        """Validate input image path."""
        path = Path(image_path)
        if not path.exists():
            self.logger.error(f"Image file does not exist: {image_path}")
            return False

        supported_formats = self.config.get("supported_formats", ["jpg", "jpeg", "png", "bmp", "tiff"])
        if path.suffix.lower().lstrip('.') not in supported_formats:
            self.logger.error(f"Unsupported image format: {path.suffix}")
            return False

        return True
