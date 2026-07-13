"""Gemma tool for image interpretation via Vertex AI."""

from pathlib import Path
from typing import Dict, Any, List, Optional
from .base import BaseTool
from utils.exceptions import ToolExecutionError

class GemmaTool(BaseTool):
    """Tool for image interpretation using PaliGemma or Gemma via Vertex AI."""

    @property
    def name(self) -> str:
        return "paligemma"

    @property
    def cost_tier(self) -> str:
        return "low"

    def __init__(self, config: Dict[str, Any]):
        """Initialize Gemma/PaliGemma tool."""
        super().__init__(config)

        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel

            project_id = config.get("project_id") or config.get("gcp_project_id")
            location = config.get("location", "us-central1")

            if not project_id:
                raise ValueError("GCP project_id is required for Gemma tool")

            vertexai.init(project=project_id, location=location)

            self.model_name = config.get("model_name", "gemini-2-9b-it")
            self.model = GenerativeModel(self.model_name)
            self.logger.info(f"Gemma vision initialized: {self.model_name} (project: {project_id})")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemma: {e}")
            raise ToolExecutionError(f"Gemma initialization failed: {e}")

    def execute(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Interpret image using Gemma VLM."""
        try:
            from vertexai.generative_models import Image
            self.logger.info(f"Interpreting image: {image_path}")

            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            image = Image.load_from_file(str(image_path))

            if prompt is None:
                prompt = (
                    "Analyze this image in detail. Describe:\n"
                    "1. Main visual elements and their relationships\n"
                    "2. Any text visible in the image\n"
                    "3. Charts, diagrams, or data visualizations\n"
                    "4. Key entities (people, objects, concepts)\n"
                    "5. Contextual information\n\n"
                    "Provide a structured, detailed description."
                )

            interpretation = "Failed to generate interpretation."
            for attempt in range(max_retries):
                try:
                    response = self.model.generate_content(
                        [prompt, image],
                        generation_config={
                            "temperature": 0.5,
                            "max_output_tokens": 1024,
                        }
                    )
                    interpretation = response.text
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    self.logger.warning(f"Retry {attempt + 1}/{max_retries} after error: {e}")

            self.logger.info(f"Successfully interpreted image: {image_path.name}")
            return {
                "interpretation": interpretation,
                "image_path": str(image_path),
                "prompt_used": prompt,
                "model_used": self.model_name,
                "success": True
            }
        except Exception as e:
            self.logger.error(f"Image interpretation failed: {e}")
            raise ToolExecutionError(f"Gemma image interpretation failed: {e}")

    def interpret_multiple_images(
        self,
        image_paths: List[str],
        prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Interpret multiple images."""
        results = []
        for image_path in image_paths:
            try:
                results.append(self.execute(image_path, prompt=prompt))
            except Exception as e:
                self.logger.error(f"Failed to interpret {image_path}: {e}")
                results.append({
                    "image_path": image_path,
                    "error": str(e),
                    "success": False
                })
        return results

    def extract_image_entities(self, image_path: str) -> Dict[str, Any]:
        """Extract entities from image using structured description."""
        prompt = (
            "Extract and list all entities visible in this image. "
            "For each entity, provide:\n"
            "1. Entity type (person, object, concept, text, etc.)\n"
            "2. Entity name or description\n"
            "3. Attributes or properties\n\n"
            "Format as a structured list."
        )
        result = self.execute(image_path, prompt=prompt)
        return {
            "image_path": image_path,
            "entities_text": result.get("interpretation", ""),
            "success": result.get("success", False)
        }

    def validate_input(self, image_path: str, **kwargs) -> bool:
        """Validate input file path and format."""
        path = Path(image_path)
        if not path.exists():
            self.logger.error(f"Image does not exist: {image_path}")
            return False

        supported_formats = ["jpg", "jpeg", "png", "bmp", "gif", "webp"]
        if path.suffix.lower().lstrip('.') not in supported_formats:
            self.logger.error(f"Unsupported image format: {path.suffix}")
            return False

        max_size_mb = self.config.get("max_image_size_mb", 10)
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            self.logger.error(f"Image too large: {size_mb:.2f}MB (max: {max_size_mb}MB)")
            return False

        return True
