"""Unstructured.io tool for document parsing."""

from pathlib import Path
from typing import Dict, Any, List, Optional
from .base import BaseTool
from utils.exceptions import ToolExecutionError


class UnstructuredTool(BaseTool):
    """Tool for parsing documents and extracting structure using Unstructured.io."""

    @property
    def name(self) -> str:
        return "unstructured_io"

    @property
    def cost_tier(self) -> str:
        return "low"

    def execute(
        self,
        document_path: str,
        extract_tables: bool = True,
        extract_images: bool = True
    ) -> Dict[str, Any]:
        """Parse document into structured elements."""
        try:
            from unstructured.partition.auto import partition

            self.logger.info(f"Parsing document: {document_path}")

            document_path = Path(document_path)
            if not document_path.exists():
                raise FileNotFoundError(f"Document not found: {document_path}")

            strategy = self.config.get("strategy", "hi_res")
            include_page_breaks = self.config.get("include_page_breaks", True)
            infer_table_structure = self.config.get("infer_table_structure", True)

            elements = partition(
                filename=str(document_path),
                strategy=strategy,
                include_page_breaks=include_page_breaks,
                pdf_infer_table_structure=infer_table_structure if extract_tables else False,
                extract_images_in_pdf=extract_images,
            )

            parsed_elements = []
            tables = []
            images = []

            for element in elements:
                element_dict = {
                    "type": element.category,
                    "text": str(element),
                    "metadata": element.metadata.to_dict() if hasattr(element.metadata, 'to_dict') else {},
                }
                parsed_elements.append(element_dict)

                if extract_tables and element.category == "Table":
                    table_data = self._extract_table_structure(element)
                    if table_data:
                        tables.append(table_data)

                if extract_images and element.category == "Image":
                    images.append(element_dict)

            self.logger.info(f"Parsed {len(parsed_elements)} elements ({len(tables)} tables, {len(images)} images)")

            return {
                "elements": parsed_elements,
                "tables": tables,
                "images": images,
                "metadata": {
                    "document_path": str(document_path),
                    "element_count": len(parsed_elements),
                    "table_count": len(tables),
                    "image_count": len(images)
                },
                "success": True
            }
        except Exception as e:
            self.logger.error(f"Document parsing failed: {e}")
            raise ToolExecutionError(f"Unstructured.io parsing failed: {e}")

    def extract_tables_only(self, document_path: str) -> List[Dict[str, Any]]:
        """Extract tables only."""
        result = self.execute(document_path, extract_tables=True, extract_images=False)
        return result.get("tables", [])

    def extract_text_only(self, document_path: str) -> str:
        """Extract text content only."""
        result = self.execute(document_path, extract_tables=False, extract_images=False)
        elements = result.get("elements", [])

        text_elements = [
            elem["text"] for elem in elements
            if elem["type"] in ["NarrativeText", "Title", "ListItem", "Text"]
        ]
        return "\n".join(text_elements)

    def _extract_table_structure(self, table_element) -> Optional[Dict[str, Any]]:
        """Extract HTML structure of a table."""
        try:
            table_data = {
                "text": str(table_element),
                "metadata": table_element.metadata.to_dict() if hasattr(table_element.metadata, 'to_dict') else {}
            }
            if hasattr(table_element.metadata, 'text_as_html'):
                table_data["html"] = table_element.metadata.text_as_html
            return table_data
        except Exception as e:
            self.logger.warning(f"Could not extract table structure: {e}")
            return None

    def chunk_document(
        self,
        document_path: str,
        chunk_size: int = 1200,
        chunk_overlap: int = 100
    ) -> List[Dict[str, Any]]:
        """Chunk parsed document."""
        try:
            self.logger.info(f"Chunking document: {document_path}")
            result = self.execute(document_path)
            elements = result.get("elements", [])

            chunks = []
            current_chunk = []
            current_size = 0

            for elem in elements:
                text = elem.get("text", "")
                elem_tokens = len(text) // 4
                if current_size + elem_tokens > chunk_size and current_chunk:
                    chunks.append({
                        "text": "\n".join(current_chunk),
                        "token_estimate": current_size,
                        "element_count": len(current_chunk)
                    })
                    overlap_text = current_chunk[-1] if current_chunk else ""
                    current_chunk = [overlap_text] if overlap_text else []
                    current_size = len(overlap_text) // 4

                current_chunk.append(text)
                current_size += elem_tokens

            if current_chunk:
                chunks.append({
                    "text": "\n".join(current_chunk),
                    "token_estimate": current_size,
                    "element_count": len(current_chunk)
                })

            self.logger.info(f"Created {len(chunks)} chunks from document")
            return chunks
        except Exception as e:
            self.logger.error(f"Document chunking failed: {e}")
            raise ToolExecutionError(f"Document chunking failed: {e}")

    def validate_input(self, document_path: str, **kwargs) -> bool:
        """Validate document file exists and has supported extension."""
        path = Path(document_path)
        if not path.exists():
            self.logger.error(f"Document does not exist: {document_path}")
            return False

        supported_formats = ["pdf", "docx", "pptx", "txt", "md", "html", "xml"]
        if path.suffix.lower().lstrip('.') not in supported_formats:
            self.logger.error(f"Unsupported document format: {path.suffix}")
            return False

        return True
