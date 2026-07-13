"""Tesseract OCR tool for text extraction."""

import pytesseract
from pathlib import Path
from PIL import Image
from typing import Dict, Any, Optional
from .base import BaseTool
from utils.exceptions import ToolExecutionError


class TesseractTool(BaseTool):
    """Tool for extracting text using Tesseract OCR."""

    @property
    def name(self) -> str:
        return "tesseract"

    @property
    def cost_tier(self) -> str:
        return "free"

    def execute(
        self,
        image_path: str,
        lang: Optional[str] = None,
        psm: Optional[int] = None
    ) -> Dict[str, Any]:
        """Extract text from image using Tesseract OCR."""
        try:
            self.logger.info(f"Extracting text from: {image_path}")

            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            # Load image
            image = Image.open(image_path)

            # Get config
            lang = lang or self.config.get("lang", "eng")
            psm = psm or self.config.get("psm", 3)
            oem = self.config.get("oem", 3)
            timeout = self.config.get("timeout", 30)

            # Build Tesseract config
            custom_config = f"--psm {psm} --oem {oem}"

            # Extract text
            text = pytesseract.image_to_string(
                image,
                lang=lang,
                config=custom_config,
                timeout=timeout
            )

            # Get detailed data for confidence score
            try:
                data = pytesseract.image_to_data(
                    image,
                    lang=lang,
                    config=custom_config,
                    output_type=pytesseract.Output.DICT
                )
                confidences = [int(conf) for conf in data['conf']
                               if conf != '-1' and str(conf).isdigit()]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            except Exception as e:
                self.logger.warning(f"Could not extract confidence scores: {e}")
                avg_confidence = None

            text = text.strip()
            word_count = len(text.split())

            self.logger.info(f"Extracted {word_count} words from {image_path.name}")

            return {
                "text": text,
                "confidence": avg_confidence,
                "word_count": word_count,
                "lang": lang,
                "image_path": str(image_path),
                "success": True
            }
        except pytesseract.TesseractError as e:
            self.logger.error(f"Tesseract OCR failed: {e}")
            raise ToolExecutionError(f"Tesseract OCR failed: {e}")
        except Exception as e:
            self.logger.error(f"Text extraction failed: {e}")
            raise ToolExecutionError(f"Text extraction failed: {e}")

    def extract_from_pdf(self, pdf_path: str, page_num: Optional[int] = None) -> Dict[str, Any]:
        """Extract text from PDF pages converting to images first."""
        try:
            from pdf2image import convert_from_path

            self.logger.info(f"Converting PDF to images: {pdf_path}")

            if page_num is not None:
                images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num)
            else:
                images = convert_from_path(pdf_path)

            results = {}
            for idx, image in enumerate(images, start=page_num or 1):
                self.logger.info(f"Processing page {idx}")

                text = pytesseract.image_to_string(
                    image,
                    lang=self.config.get("lang", "eng"),
                    config=self.config.get("config", "--psm 3 --oem 3")
                )

                results[idx] = {
                    "page": idx,
                    "text": text.strip(),
                    "word_count": len(text.split())
                }

            return {
                "pdf_path": pdf_path,
                "pages": results,
                "total_pages": len(results),
                "success": True
            }
        except Exception as e:
            self.logger.error(f"PDF text extraction failed: {e}")
            raise ToolExecutionError(f"PDF OCR failed: {e}")

    def validate_input(self, image_path: str, **kwargs) -> bool:
        """Validate input file path."""
        path = Path(image_path)
        if not path.exists():
            self.logger.error(f"File does not exist: {image_path}")
            return False

        supported_formats = ["jpg", "jpeg", "png", "bmp", "tiff", "pdf"]
        if path.suffix.lower().lstrip('.') not in supported_formats:
            self.logger.error(f"Unsupported format: {path.suffix}")
            return False

        return True
