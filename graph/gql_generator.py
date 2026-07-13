"""Generates GQL (Graph Query Language) queries from natural language."""

from typing import Dict, Any, Optional
from config.config_loader import get_config_loader
from utils.logger import get_logger
from utils.exceptions import GraphError

logger = get_logger(__name__)

class GQLGenerator:
    """
    Generates GQL queries from natural language using Gemini.
    """

    def __init__(self):
        """Initialize GQL generator with Gemini."""
        self.config_loader = get_config_loader()
        self.system_prompts = self.config_loader.get_system_prompts()
        self.model_config = self.config_loader.get_model_config()

        self._init_gemini()
        logger.info("GQLGenerator initialized")

    def _init_gemini(self):
        """Initialize Gemini for GQL generation."""
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

            gql_config = self.model_config.get("orchestrator", {})
            self.model_name = gql_config.get("name", "gemini-1.5-flash")
            self.model_config_params = gql_config
            self.types = types
            logger.info(f"GQL generator initialized: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise GraphError(f"Gemini initialization failed: {e}")

    def generate_gql(
        self,
        natural_language_query: str,
        schema: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate GQL query from natural language.

        Args:
            natural_language_query: User's question
            schema: Optional graph schema description

        Returns:
            Dictionary with GQL query, explanation, and parameters.
        """
        try:
            logger.info(f"Generating GQL for: {natural_language_query[:100]}...")

            system_instruction = self.system_prompts.get("gql_generator", "")

            if schema is None:
                schema = self._get_default_schema()

            prompt = f"""Convert this natural language question to a GQL query for Spanner Graph.

            Question: {natural_language_query}

            Graph Schema:
            {schema}

            Requirements:
            1. Generate efficient GQL query
            2. Use proper WHERE clauses for filtering
            3. Include LIMIT clauses when appropriate
            4. Use joins for multi-hop queries
            5. Return relevant entity and relationship properties

            Output format (JSON):
            {{
                "gql_query": "SELECT ... FROM ...",
                "explanation": "This query ...",
                "parameters": {{"param1": "value1"}}
            }}
            """

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self.types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=self.model_config_params.get("temperature", 0.3),
                    max_output_tokens=1024,
                    response_mime_type="application/json"
                )
            )

            import json
            result = json.loads(response.text)
            logger.info(f"Generated GQL query: {result.get('gql_query', '')[:100]}...")
            return result

        except Exception as e:
            logger.error(f"GQL generation failed: {e}")
            raise GraphError(f"GQL generation failed: {e}")

    def _get_default_schema(self) -> str:
        """Get default graph schema description."""
        return """
Graph Schema:

Tables:
1. Entity (entity_id, entity_type, name, description, source_page, confidence, attributes)
    - entity_id: Primary key (STRING)
    - entity_type: person, organization, date, location, technology, concept, event, etc.
    - name: Entity name
    - description: Detailed description
    - source_page: Page number in source document
    - confidence: Confidence score (0.0-1.0)
    - attributes: JSON with additional properties

2. Relationship (relationship_id, source_entity_id, target_entity_id, relationship_type,
    description, keywords, strength, source_page, confidence, cross_modal, attributes)
    - relationship_id: Primary key (STRING)
    - source_entity_id: Foreign key to Entity
    - target_entity_id: Foreign key to Entity
    - relationship_type: works_for, located_in, related_to, etc.
    - description: Relationship description
    - keywords: Categorization keywords
    - strength: Relationship strength score (0.0-10.0)
    - cross_modal: Boolean indicating cross-modal relationship (text + image)

Example Queries:

1. Find all people:
    SELECT * FROM Entity WHERE entity_type = 'person'
2. Find relationships for entity:
    SELECT r.*, e1.name as source_name, e2.name as target_name
    FROM Relationship r
    JOIN Entity e1 ON r.source_entity_id = e1.entity_id
    JOIN Entity e2 ON r.target_entity_id = e2.entity_id
    WHERE r.source_entity_id = 'entity_123'
"""

    def validate_gql(self, gql_query: str) -> Dict[str, Any]:
        """Validate GQL query syntax."""
        try:
            gql_upper = gql_query.upper()

            if "SELECT" not in gql_upper:
                return {"valid": False, "error": "Query must contain SELECT clause"}

            if "FROM" not in gql_upper:
                return {"valid": False, "error": "Query must contain FROM clause"}

            dangerous_keywords = ["DROP", "DELETE", "TRUNCATE", "ALTER"]
            for keyword in dangerous_keywords:
                if keyword in gql_upper:
                    return {"valid": False, "error": f"Dangerous operation '{keyword}' not allowed"}

            return {"valid": True, "query": gql_query}
        except Exception as e:
            logger.error(f"GQL validation failed: {e}")
            return {"valid": False, "error": str(e)}

    def explain_query(self, gql_query: str) -> str:
        """Generate natural language explanation of GQL query."""
        try:
            prompt = f"""Explain this GQL query in simple, natural language:

{gql_query}

Provide a concise explanation of what this query does, what data it retrieves, and any important filters or joins.
"""
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self.types.GenerateContentConfig(
                    temperature=0.5,
                    max_output_tokens=512
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Query explanation failed: {e}")
            return f"Unable to explain query: {str(e)}"
