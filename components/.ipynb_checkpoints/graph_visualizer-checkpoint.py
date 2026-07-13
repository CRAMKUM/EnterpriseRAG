"""Graph Visualizer Component - Interactive graph exploration"""

import streamlit as st
import json
from typing import Optional, Dict, List
from utils.logger import get_logger

logger = get_logger(__name__)

class GraphVisualizer:
    """Interactive knowledge graph visualization."""

    def __init__(self, graph_manager):
        """Initialize graph visualizer."""
        self.graph_manager = graph_manager
        logger.info("GraphVisualizer initialized")

    def render(self):
        """Render graph visualization interface."""
        st.markdown("### 🌐 Knowledge Graph Explorer")

        if not self.graph_manager:
            st.warning("Graph database not configured. See deployment guide.")
            return

        col1, col2, col3 = st.columns(3)

        with col1:
            search_type = st.selectbox(
                "Search Type",
                ["Entity", "Relationship", "Neighborhood"]
            )

        with col2:
            if search_type == "Entity":
                entity_types = self._get_entity_types()
                selected_type = st.selectbox("Entity Type", ["All"] + entity_types)
            else:
                selected_type = None

        with col3:
            limit = st.number_input("Limit", min_value=1, max_value=100, value=10)

        search_query = st.text_input(
            "Search graph",
            placeholder="Enter entity name, type, or relationship...",
            key="graph_search"
        )

        if st.button("Search", type="primary", use_container_width=True):
            if search_query:
                self._perform_search(search_type, search_query, selected_type, limit)
            else:
                st.warning("Please enter a search query")

        if "graph_results" in st.session_state and st.session_state.graph_results:
            self._display_results(st.session_state.graph_results)

        st.markdown("---")
        st.markdown("### Graph Statistics")
        self._display_statistics()

    def _get_entity_types(self) -> List[str]:
        """Get all entity types from graph."""
        try:
            stats = self.graph_manager.get_graph_statistics()
            if stats.get("success"):
                entity_types_data = stats.get("entity_types", [])
                return [et["entity_type"] for et in entity_types_data]
            return []
        except Exception as e:
            logger.error(f"Failed to get entity types: {e}")
            return []

    def _perform_search(self, search_type: str, query: str, entity_type: Optional[str], limit: int):
        """Perform graph search."""
        try:
            with st.spinner("Searching graph..."):
                if search_type == "Entity":
                    results = self._search_entities(query, entity_type, limit)
                elif search_type == "Relationship":
                    results = self._search_relationships(query, limit)
                else:
                    results = self._search_neighborhood(query, limit)

            st.session_state.graph_results = {
                "type": search_type,
                "query": query,
                "results": results
            }
        except Exception as e:
            logger.error(f"Search failed: {e}")
            st.error(f"Search failed: {e}")

    def _search_entities(self, query: str, entity_type: Optional[str], limit: int) -> List[Dict]:
        """Search for entities."""
        try:
            gql = f"""
            GRAPH MATCH (e:Entity)
            WHERE e.name CONTAINS '{query}'
            """
            if entity_type and entity_type != "All":
                gql += f" AND e.entity_type = '{entity_type}'"

            gql += f" RETURN e LIMIT {limit}"
            result = self.graph_manager.execute_gql(gql)
            return result
        except Exception as e:
            logger.error(f"Entity search failed: {e}")
            return []

    def _search_relationships(self, query: str, limit: int) -> List[Dict]:
        """Search for relationships."""
        try:
            gql = f"""
            GRAPH MATCH (source:Entity)-[r:RELATIONSHIP]->(target:Entity)
            WHERE r.description CONTAINS '{query}' OR r.keywords CONTAINS '{query}'
            RETURN source, r, target
            LIMIT {limit}
            """
            result = self.graph_manager.execute_gql(gql)
            return result
        except Exception as e:
            logger.error(f"Relationship search failed: {e}")
            return []

    def _search_neighborhood(self, entity_name: str, limit: int) -> Dict:
        """Search for entity neighborhood."""
        try:
            gql = f"""
            GRAPH MATCH (e:Entity {{name: '{entity_name}'}})-[r]-(neighbor:Entity)
            RETURN e, r, neighbor
            LIMIT {limit}
            """
            result = self.graph_manager.execute_gql(gql)
            return {"entities": result, "relationships": result}
        except Exception as e:
            logger.error(f"Neighborhood search failed: {e}")
            return {}

    def _display_results(self, results_data: Dict):
        """Display search results."""
        st.markdown("---")
        st.markdown("### 🔍 Search Results")
        result_type = results_data["type"]
        results = results_data["results"]

        if not results:
            st.info("No results found")
            return

        st.markdown(f"**Query:** `{results_data['query']}`")
        st.markdown(f"**Found:** {len(results) if isinstance(results, list) else 1} {result_type.lower()}(s)")

        if result_type == "Entity":
            self._display_entities(results)
        elif result_type == "Relationship":
            self._display_relationships(results)
        else:
            self._display_neighborhood(results)

    def _display_entities(self, entities: List[Dict]):
        """Display entity results."""
        for idx, entity in enumerate(entities):
            with st.expander(f"🔎 {entity.get('name', 'Unknown')} ({entity.get('entity_type', 'Unknown')})"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown("**Description:**")
                    st.write(entity.get('description', 'No description'))
                with col2:
                    st.markdown("**Properties:**")
                    st.json({
                        "ID": entity.get('entity_id', 'N/A'),
                        "Type": entity.get('entity_type', 'N/A'),
                        "Source": entity.get('source_document', 'N/A')
                    })

    def _display_relationships(self, relationships: List[Dict]):
        """Display relationship results."""
        for idx, rel in enumerate(relationships):
            source = rel.get('source_entity', 'Unknown')
            target = rel.get('target_entity', 'Unknown')
            rel_type = rel.get('relationship_type', 'RELATED')

            with st.expander(f"🔗 {source} → {target}"):
                st.markdown(f"**Type:** `{rel_type}`")
                st.markdown(f"**Description:** {rel.get('description', 'No description')}")
                st.markdown(f"**Keywords:** {rel.get('keywords', 'None')}")
                st.markdown(f"**Strength:** {rel.get('strength', 'N/A')}/10")

    def _display_neighborhood(self, neighborhood: Dict):
        """Display neighborhood results."""
        st.markdown("**Connected Entities:**")
        entities = neighborhood.get("entities", [])
        relationships = neighborhood.get("relationships", [])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Entities")
            if isinstance(entities, list):
                for entity in entities:
                    st.markdown(f"- **{entity.get('name', 'Unknown')}** ({entity.get('entity_type', 'Unknown')})")
        with col2:
            st.markdown("#### Relationships")
            if isinstance(relationships, list):
                for rel in relationships:
                    st.markdown(f"- {rel.get('source_entity', '?')} -> {rel.get('target_entity', '?')}")

    def _display_statistics(self):
        """Display graph statistics."""
        try:
            stats = self.graph_manager.get_graph_statistics()
            if stats.get("success"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Entities", stats.get("total_entities", 0))
                with col2:
                    st.metric("Total Relationships", stats.get("total_relationships", 0))
                with col3:
                    avg_degree = (
                        stats.get("total_relationships", 0) / stats.get("total_entities", 1)
                        if stats.get("total_entities", 0) > 0
                        else 0
                    )
                    st.metric("Avg Connections", f"{avg_degree:.1f}")

                entity_types = stats.get("entity_types", [])
                if entity_types:
                    st.markdown("#### Entity Distribution")
                    import pandas as pd
                    df = pd.DataFrame(entity_types)
                    st.bar_chart(df.set_index("entity_type"))
            else:
                st.error("Failed to load statistics")
        except Exception as e:
            logger.error(f"Failed to display statistics: {e}")
            st.error(f"Failed to load statistics: {e}")
