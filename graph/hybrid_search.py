"""Hybrid search combining vector similarity and graph traversal."""

from typing import Dict, Any, List, Optional
from utils.logger import get_logger
from utils.exceptions import GraphError
from config.config_loader import get_config_loader

logger = get_logger(__name__)

class HybridSearch:
    """
    Hybrid search that combines semantic keyword search and graph relationships.
    """

    def __init__(self, graph_manager):
        """Initialize hybrid search."""
        self.graph_manager = graph_manager
        self.config_loader = get_config_loader()
        self.tool_config = self.config_loader.get_tool_config()

        self.search_config = self.tool_config.get("hybrid_search", {})
        self.vector_weight = self.search_config.get("vector_weight", 0.5)
        self.graph_weight = self.search_config.get("graph_weight", 0.5)
        self.top_k = self.search_config.get("top_k", 10)
        self.min_similarity = self.search_config.get("min_similarity", 0.7)

        self._init_embedding()
        logger.info("HybridSearch initialized")

    def _init_embedding(self):
        """Initialize embedding model for vector search."""
        try:
            import vertexai
            from vertexai.language_models import TextEmbeddingModel

            graph_config = self.config_loader.get_graph_config()
            model_config = self.config_loader.get_model_config()

            project_id = graph_config.get("project_id")
            location = graph_config.get("location", "us-central1")

            vertexai.init(project=project_id, location=location)

            embedding_config = model_config.get("embedding", {})
            model_name = embedding_config.get("name", "text-embedding-004")

            self.embedding_model = TextEmbeddingModel.from_pretrained(model_name)
            self.embedding_dim = embedding_config.get("dimensions", 768)
            logger.info(f"Embedding model initialized: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise GraphError(f"Embedding initialization failed: {e}")

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text."""
        try:
            embeddings = self.embedding_model.get_embeddings([text])
            return embeddings[0].values
        except Exception as e:
            logger.error(f"Text embedding failed: {e}")
            raise GraphError(f"Embedding failed: {e}")

    def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combined."""
        try:
            logger.info(f"Hybrid search for: {query[:100]}...")
            top_k = top_k or self.top_k

            vector_results = self._vector_search(query, filters, top_k * 2)
            graph_results = self._graph_search(query, filters, top_k * 2)

            combined_results = self._combine_results(
                vector_results,
                graph_results,
                top_k
            )
            logger.info(f"Hybrid search returned {len(combined_results)} results")
            return combined_results

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise GraphError(f"Hybrid search failed: {e}")

    def _vector_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Perform simulated vector similarity search via keyword fallback."""
        try:
            query_embedding = self.embed_text(query)
            logger.warning("Vector search not fully implemented in Spanner - using keyword fallback")

            keywords = query.lower().split()
            results = []

            for keyword in keywords[:3]:
                entities = self.graph_manager.search_entities_by_name(
                    keyword,
                    limit=top_k
                )
                for entity in entities:
                    score = self._calculate_keyword_similarity(query, entity)
                    if score >= self.min_similarity:
                        results.append({
                            **entity,
                            "vector_score": score,
                            "search_type": "vector"
                        })

            seen_ids = set()
            unique_results = []
            for result in sorted(results, key=lambda x: x["vector_score"], reverse=True):
                entity_id = result.get("entity_id")
                if entity_id not in seen_ids:
                    seen_ids.add(entity_id)
                    unique_results.append(result)

            return unique_results[:top_k]
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def _graph_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Perform graph-based relationship search."""
        try:
            keywords = query.lower().split()
            seed_entities = []

            for keyword in keywords[:3]:
                entities = self.graph_manager.search_entities_by_name(
                    keyword,
                    limit=5
                )
                seed_entities.extend(entities)

            if not seed_entities:
                return []

            results = []
            for seed in seed_entities[:5]:
                entity_id = seed.get("entity_id")
                relationships = self.graph_manager.get_entity_relationships(entity_id)

                for rel in relationships:
                    target_id = (
                        rel.get("target_entity_id")
                        if rel.get("source_entity_id") == entity_id
                        else rel.get("source_entity_id")
                    )
                    target_entity = self.graph_manager.get_entity_by_id(target_id)
                    if target_entity:
                        graph_score = rel.get("strength", 5.0) / 10.0
                        results.append({
                            **target_entity,
                            "graph_score": graph_score,
                            "search_type": "graph",
                            "via_relationship": rel
                        })

            results.sort(key=lambda x: x.get("graph_score", 0), reverse=True)
            return results[:top_k]
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []

    def _combine_results(
        self,
        vector_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Combine vector and graph results with weighted scores."""
        combined = {}

        for result in vector_results:
            entity_id = result.get("entity_id")
            combined[entity_id] = {
                **result,
                "vector_score": result.get("vector_score", 0),
                "graph_score": 0
            }

        for result in graph_results:
            entity_id = result.get("entity_id")
            if entity_id in combined:
                combined[entity_id]["graph_score"] = result.get("graph_score", 0)
            else:
                combined[entity_id] = {
                    **result,
                    "vector_score": 0,
                    "graph_score": result.get("graph_score", 0)
                }

        for entity_id, result in combined.items():
            vector_score = result.get("vector_score", 0)
            graph_score = result.get("graph_score", 0)
            hybrid_score = (
                self.vector_weight * vector_score +
                self.graph_weight * graph_score
            )
            result["hybrid_score"] = hybrid_score

        ranked_results = sorted(
            combined.values(),
            key=lambda x: x["hybrid_score"],
            reverse=True
        )
        return ranked_results[:top_k]

    def _calculate_keyword_similarity(self, query: str, entity: Dict[str, Any]) -> float:
        """Calculate simple Jaccard keyword similarity."""
        query_words = set(query.lower().split())
        entity_name = entity.get("name", "").lower()
        entity_desc = entity.get("description", "").lower()
        entity_words = set(entity_name.split() + entity_desc.split())

        if not query_words or not entity_words:
            return 0.0

        intersection = len(query_words & entity_words)
        union = len(query_words | entity_words)
        return intersection / union if union > 0 else 0.0

    def get_graph_context(
        self,
        entity_ids: List[str],
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """Get graph context around entities via BFS."""
        try:
            logger.info(f"Getting graph context for {len(entity_ids)} entities")

            entities = []
            relationships = []
            seen_entities = set()
            seen_relationships = set()

            current_level = entity_ids
            for depth in range(max_depth):
                next_level = []
                for entity_id in current_level:
                    if entity_id in seen_entities:
                        continue

                    entity = self.graph_manager.get_entity_by_id(entity_id)
                    if entity:
                        entities.append(entity)
                        seen_entities.add(entity_id)

                    rels = self.graph_manager.get_entity_relationships(entity_id)
                    for rel in rels:
                        rel_id = rel.get("relationship_id")
                        if rel_id not in seen_relationships:
                            relationships.append(rel)
                            seen_relationships.add(rel_id)

                        source_id = rel.get("source_entity_id")
                        target_id = rel.get("target_entity_id")

                        if source_id != entity_id and source_id not in seen_entities:
                            next_level.append(source_id)
                        if target_id != entity_id and target_id not in seen_entities:
                            next_level.append(target_id)

                current_level = next_level

            logger.info(f"Retrieved {len(entities)} entities and {len(relationships)} relationships")
            return {
                "entities": entities,
                "relationships": relationships,
                "total_entities": len(entities),
                "total_relationships": len(relationships)
            }
        except Exception as e:
            logger.error(f"Failed to get graph context: {e}")
            return {"entities": [], "relationships": []}
