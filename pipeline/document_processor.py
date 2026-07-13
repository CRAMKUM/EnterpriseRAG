"""Document processor for parsing and extracting content."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.logger import get_logger
from utils.exceptions import PipelineError
from config.config_loader import get_config_loader
from tools import OpenCVTool, TesseractTool, UnstructuredTool, GemmaTool

logger = get_logger(__name__)

class DocumentProcessor:
    """
    Processes documents through the pipeline.
    """
    def __init__(self):
        """Initialize document processor with tools."""
        self.config_loader = get_config_loader()
        self.pipeline_config = self.config_loader.get_pipeline_config()
        self.tool_config = self.config_loader.get_tool_config()

        # Initialize tools
        self.unstructured_tool = UnstructuredTool(self.tool_config.get("unstructured_io", {}))
        self.opencv_tool = OpenCVTool(self.tool_config.get("opencv", {}))
        self.tesseract_tool = TesseractTool(self.tool_config.get("tesseract", {}))
        self.gemma_tool = GemmaTool(self.tool_config.get("gemma", {}))

        self.max_workers = self.pipeline_config.get("max_workers", 4)
        logger.info("DocumentProcessor initialized")

    def process_document(self, document_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Process single document through full pipeline."""
        try:
            logger.info(f"Processing document: {document_path}")
            document_path = Path(document_path)
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

            parsed_data = self._parse_document(document_path)
            images_data = self._process_images(parsed_data.get("images", []))
            tables_data = parsed_data.get("tables", [])
            pages_data = self._organize_by_pages(parsed_data, images_data, tables_data)

            result = {
                "document_path": str(document_path),
                "pages": pages_data,
                "images": images_data,
                "tables": tables_data,
                "metadata": {
                    "page_count": len(pages_data),
                    "image_count": len(images_data),
                    "table_count": len(tables_data),
                    "success": True
                }
            }

            if output_dir:
                output_file = output_dir / f"{document_path.stem}_processed.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                logger.info(f"Saved processed data to: {output_file}")

            logger.info(f"Successfully processed document: {document_path.name}")
            return result
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            raise PipelineError(f"Document processing failed: {e}")

    def process_batch(self, document_paths: List[str], output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """Process multiple documents in parallel."""
        logger.info(f"Processing batch of {len(document_paths)} documents")
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_doc = {
                executor.submit(self.process_document, doc_path, output_dir): doc_path
                for doc_path in document_paths
            }
            for future in as_completed(future_to_doc):
                doc_path = future_to_doc[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process {doc_path}: {e}")
                    results.append({
                        "document_path": doc_path,
                        "error": str(e),
                        "metadata": {"success": False}
                    })
        return results

    def _parse_document(self, document_path: Path) -> Dict[str, Any]:
        """Parse document with Unstructured.io."""
        logger.info(f"Parsing document: {document_path.name}")
        return self.unstructured_tool.execute(
            str(document_path),
            extract_tables=True,
            extract_images=True
        )

    def _process_images(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process images in parallel."""
        if not images:
            return []

        logger.info(f"Processing {len(images)} images")
        processed_images = []
        max_concurrent = self.pipeline_config.get("parallel_processing", {}).get("max_concurrent_images", 10)

        with ThreadPoolExecutor(max_workers=min(max_concurrent, len(images))) as executor:
            future_to_img = {
                executor.submit(self._process_single_image, img): img
                for img in images
            }
            for future in as_completed(future_to_img):
                try:
                    processed_images.append(future.result())
                except Exception as e:
                    logger.error(f"Image processing failed: {e}")
        return processed_images

    def _process_single_image(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process single image: deblur + interpret."""
        image_path = image_data.get("metadata", {}).get("image_path")
        if not image_path or not Path(image_path).exists():
            return image_data

        try:
            deblur_result = self.opencv_tool.execute(image_path)
            if deblur_result.get("was_deblurred"):
                image_path = deblur_result["deblurred_path"]
        except Exception as e:
            logger.warning(f"Deblurring failed for {image_path}: {e}")

        try:
            interpretation_result = self.gemma_tool.execute(image_path)
            image_data["interpretation"] = interpretation_result.get("interpretation")
            image_data["model_used"] = interpretation_result.get("model_used")
        except Exception as e:
            logger.warning(f"Image interpretation failed for {image_path}: {e}")
            image_data["interpretation"] = None

        return image_data

    def _organize_by_pages(
        self,
        parsed_data: Dict[str, Any],
        images_data: List[Dict[str, Any]],
        tables_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Group document elements by page number."""
        pages = {}

        for element in parsed_data.get("elements", []):
            page_num = element.get("metadata", {}).get("page_number", 0)
            if page_num not in pages:
                pages[page_num] = {
                    "page_number": page_num,
                    "text_elements": [],
                    "images": [],
                    "tables": [],
                    "combined_text": ""
                }
            pages[page_num]["text_elements"].append(element)

        for img in images_data:
            page_num = img.get("metadata", {}).get("page_number", 0)
            if page_num in pages:
                pages[page_num]["images"].append(img)

        for table in tables_data:
            page_num = table.get("metadata", {}).get("page_number", 0)
            if page_num in pages:
                pages[page_num]["tables"].append(table)

        for page_num, page_data in pages.items():
            text_parts = [elem.get("text", "") for elem in page_data["text_elements"]]
            page_data["combined_text"] = "\n".join(text_parts)

        return list(pages.values())

    def chunk_page_content(self, page_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk page text with overlap."""
        doc_config = self.pipeline_config.get("document_processing", {})
        chunk_size = doc_config.get("chunk_token_size", 1200)
        chunk_overlap = doc_config.get("chunk_overlap_token_size", 100)

        text = page_data.get("combined_text", "")
        chunks = []
        words = text.split()
        current_chunk = []
        current_size = 0

        for word in words:
            word_tokens = len(word) // 4
            if current_size + word_tokens > chunk_size and current_chunk:
                chunks.append({
                    "text": " ".join(current_chunk),
                    "page_number": page_data.get("page_number"),
                    "token_estimate": current_size,
                    "images": page_data.get("images", []),
                    "tables": page_data.get("tables", [])
                })
                overlap_words = int(len(current_chunk) * (chunk_overlap / chunk_size))
                current_chunk = current_chunk[-overlap_words:] if overlap_words > 0 else []
                current_size = sum(len(w) // 4 for w in current_chunk)

            current_chunk.append(word)
            current_size += word_tokens

        if current_chunk:
            chunks.append({
                "text": " ".join(current_chunk),
                "page_number": page_data.get("page_number"),
                "token_estimate": current_size,
                "images": page_data.get("images", []),
                "tables": page_data.get("tables", [])
            })

        return chunks
