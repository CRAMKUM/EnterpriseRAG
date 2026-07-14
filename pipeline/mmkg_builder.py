"""MMKG Builder for extracting entities and relationships."""

import json
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.logger import get_logger
from utils.exceptions import PipelineError
from config.config_loader import get_config_loader

logger = get_logger(__name__)

class MMKGBuilder:
    """
    Builds MMKG by calling Gemini to extract entities and relationships.
    """
    def __init__(self):
        """Initialize MMKG builder with Gemini config."""
        self.config_loader = get_config_loader()
        self.pipeline_config = self.config_loader.get_pipeline_config()
        self.model_config = self.config_loader.get_model_config()
        self.system_prompts = self.config_loader.get_system_prompts()

        self._init_gemini()
        logger.info("MMKGBuilder initialized")

    def _init_gemini(self):
        """Initialize Vertex GenAI client."""
        try:
            from google import genai
            from google.genai import types

            graph_config = self.config_loader.get_graph_config()
            project_id = graph_config.get("project_id")
            location = graph_config.get("location", "us-central1")

            self.client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location
            )

            extractor_config = self.model_config.get("entity_extractor", {})
            self.model_name = extractor_config.get("name", "gemini-1.5-flash")
            self.model_config_params = extractor_config
            self.types = types
            logger.info(f"Gemini entity extractor initialized: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise PipelineError(f"Gemini initialization failed: {e}")

    def build_mmkg(self, processed_pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build complete MMKG from pages."""
        try:
            logger.info(f"Building MMKG from {len(processed_pages)} pages")
            all_entities = []
            all_relationships = []

            max_workers = self.pipeline_config.get("max_workers", 4)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_page = {
                    executor.submit(self._extract_from_page, page): page
                    for page in processed_pages
                }

                for future in as_completed(future_to_page):
                    page = future_to_page[future]
                    try:
                        result = future.result()
                        all_entities.extend(result.get("entities", []))
                        all_relationships.extend(result.get("relationships", []))
                    except Exception as e:
                        logger.error(f"Failed to extract from page {page.get('page_number')}: {e}")

            unique_entities = self._deduplicate_entities(all_entities)
            cross_modal_rels = self._link_cross_modal_entities(unique_entities, processed_pages)
            all_relationships.extend(cross_modal_rels)

            # Map entity names to their unique entity_id to connect graph nodes and edges
            entity_name_to_id = {e.get("name", "").lower(): e.get("id") for e in unique_entities}
            for rel in all_relationships:
                if not rel.get("cross_modal"):
                    src_name = rel.get("source_entity", "").lower()
                    tgt_name = rel.get("target_entity", "").lower()
                    if src_name in entity_name_to_id:
                        rel["source_entity"] = entity_name_to_id[src_name]
                    if tgt_name in entity_name_to_id:
                        rel["target_entity"] = entity_name_to_id[tgt_name]

            return {
                "entities": unique_entities,
                "relationships": all_relationships,
                "metadata": {
                    "entity_count": len(unique_entities),
                    "relationship_count": len(all_relationships),
                    "page_count": len(processed_pages),
                    "success": True
                }
            }
        except Exception as e:
            logger.error(f"MMKG building failed: {e}")
            raise PipelineError(f"MMKG building failed: {e}")

    def _extract_from_page(self, page_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract entities from single page."""
        page_num = page_data.get("page_number", 0)
        text_content = page_data.get("combined_text", "")

        image_content = ""
        for img in page_data.get("images", []):
            interpretation = img.get("interpretation", "")
            if interpretation:
                image_content += f"\n[IMAGE]: {interpretation}"

        table_content = ""
        for table in page_data.get("tables", []):
            table_text = table.get("text", "")
            if table_text:
                table_content += f"\n[TABLE]: {table_text}"

        combined_content = f"{text_content}{image_content}{table_content}"
        if not combined_content.strip():
            return {"entities": [], "relationships": []}

        return self._call_entity_extractor(combined_content, page_num, page_data)

    def _call_entity_extractor(self, content: str, page_num: int, page_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Call Gemini 3.1 Flash with MegaRAG prompt."""
        try:
            system_instruction = self.system_prompts.get("entity_extractor", "")
            entity_types = [
                "person", "organization", "date", "location",
                "product_or_service", "technology", "metric_or_measurement",
                "document_reference", "concept", "event", "table", "chart", "diagram"
            ]

            tuple_delimiter = "</>"
            record_delimiter = "##"
            completion_delimiter = "</COMPLETE |>"

            extraction_prompt = f"""---Goal---
Given a document page (Page {page_num}) with text content and associated images/tables, identify all entities and pairs of related entities.

---Steps---
1. Identify all entities from the text. For each entity:
    - entity_name: Name of the entity
    - entity_type: One of [{', '.join(entity_types)}]
    - entity_description: Comprehensive description of attributes
    Format: ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)
2. Treat tables/charts as entities.
3. Identify all pairs of related entities:
    - source_entity: Name of source entity
    - target_entity: Name of target entity
    - relationship_description: Why they are related
    - relationship_keywords: High-level keywords
    - relationship_strength: Numeric score 1-10
    Format: ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)
4. Return output as a single list using **{record_delimiter}** as delimiter.
5. When finished, output {completion_delimiter}

---Document Content (Page {page_num})---
{content}
"""

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=extraction_prompt,
                config=self.types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=self.model_config_params.get("temperature", 0.3),
                    max_output_tokens=self.model_config_params.get("max_tokens", 2048)
                )
            )

            return self._parse_megarag_output(
                response.text,
                page_num,
                tuple_delimiter,
                record_delimiter,
                completion_delimiter
            )
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {"entities": [], "relationships": []}

    def _parse_megarag_output(
        self,
        output_text: str,
        page_num: int,
        tuple_delimiter: str,
        record_delimiter: str,
        completion_delimiter: str
    ) -> Dict[str, Any]:
        """Parse MegaRAG-style output."""
        entities = []
        relationships = []

        if completion_delimiter in output_text:
            output_text = output_text.split(completion_delimiter)[0]

        records = output_text.split(record_delimiter)
        for record in records:
            record = record.strip()
            if not record:
                continue

            if record.startswith("('entity'") or record.startswith('("entity"'):
                parts = record.split(tuple_delimiter)
                if len(parts) >= 4:
                    entity_name = parts[1].replace("'", "").replace('"', "").strip()
                    entity_type = parts[2].replace("'", "").replace('"', "").strip()
                    entity_desc = parts[3].replace("'", "").replace('"', "").replace(")", "").strip()

                    entities.append({
                        "id": f"entity_{page_num}_{len(entities)}",
                        "name": entity_name,
                        "type": entity_type,
                        "description": entity_desc,
                        "source_page": page_num,
                        "confidence": 0.8
                    })

            elif record.startswith("('relationship'") or record.startswith('("relationship"'):
                parts = record.split(tuple_delimiter)
                if len(parts) >= 6:
                    source = parts[1].replace("'", "").replace('"', "").strip()
                    target = parts[2].replace("'", "").replace('"', "").strip()
                    desc = parts[3].replace("'", "").replace('"', "").strip()
                    keywords = parts[4].replace("'", "").replace('"', "").strip()
                    strength_str = parts[5].replace("'", "").replace('"', "").replace(")", "").strip()

                    try:
                        strength = float(strength_str)
                    except ValueError:
                        strength = 5.0

                    relationships.append({
                        "source_entity": source,
                        "target_entity": target,
                        "type": keywords.split(',')[0].strip() if keywords else "related_to",
                        "description": desc,
                        "keywords": keywords,
                        "strength": strength,
                        "source_page": page_num,
                        "confidence": strength / 10.0
                    })

        return {"entities": entities, "relationships": relationships}

    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate entities."""
        seen = {}
        unique_entities = []
        for entity in entities:
            # Proactively guarantee attributes dictionary is initialized
            entity.setdefault("attributes", {})
            key = (entity.get("type"), entity.get("name", "").lower())
            if key not in seen:
                seen[key] = entity
                unique_entities.append(entity)
            else:
                existing = seen[key]
                existing_attrs = existing.setdefault("attributes", {})
                new_attrs = entity.get("attributes", {})
                existing_attrs.update(new_attrs)
        logger.info(f"Deduplicated {len(entities)} -> {len(unique_entities)} entities")
        return unique_entities

    def _link_cross_modal_entities(
        self,
        entities: List[Dict[str, Any]],
        pages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Link text entities to page image elements."""
        logger.info("Linking cross-modal entities")
        cross_modal_rels = []
        for page in pages:
            page_num = page.get("page_number")
            page_entities = [e for e in entities if e.get("source_page") == page_num]
            images = page.get("images", [])

            for img in images:
                interpretation = img.get("interpretation", "")
                if not interpretation:
                    continue

                for entity in page_entities:
                    entity_name = entity.get("name", "").lower()
                    if entity_name in interpretation.lower():
                        cross_modal_rels.append({
                            "source_entity": entity.get("id"),
                            "target_entity": f"image_{page_num}_{images.index(img)}",
                            "source_id": entity.get("id"),
                            "target_id": f"image_{page_num}_{images.index(img)}",
                            "type": "appears_in_image",
                            "attributes": {
                                "confidence": 0.7,
                                "image_path": img.get("metadata", {}).get("image_path")
                            },
                            "source_page": page_num,
                            "cross_modal": True
                        })
        return cross_modal_rels

    def refine_entities(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Refinement loop for accuracy."""
        try:
            system_instruction = "You are an expert at knowledge graph construction."
            refinement_prompt = f"""Refine these entities:
            {json.dumps(entities[:50], indent=2)}
            """
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=refinement_prompt,
                config=self.types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.3,
                    max_output_tokens=4096,
                    response_mime_type="application/json"
                )
            )
            refined_data = json.loads(response.text)
            return {
                "entities": refined_data.get("entities", entities),
                "relationships": refined_data.get("relationships", relationships)
            }
        except Exception as e:
            logger.error(f"Refinement failed: {e}")
            return {"entities": entities, "relationships": relationships}
