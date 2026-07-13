"""MMKG Status Component - Display knowledge graph build status"""

import streamlit as st
from utils.logger import get_logger

logger = get_logger(__name__)

class MMKGStatusComponent:
    """Display MMKG build status and statistics."""

    def __init__(self, graph_manager):
        """Initialize MMKG status component."""
        self.graph_manager = graph_manager
        logger.info("MMKGStatusComponent initialized")

    def render(self):
        """Render MMKG status display."""
        st.markdown("### 🧠 Knowledge Graph")

        status = st.session_state.get("mmkg_status", "idle")

        status_config = {
            "idle": {
                "color": "#dadce0",
                "icon": "⚪",
                "text": "No MMKG Built",
                "description": "Upload a document to build knowledge graph"
            },
            "building": {
                "color": "#fbbc04",
                "icon": "🟠",
                "text": "Building MMKG",
                "description": "Extracting entities and relationships..."
            },
            "ready": {
                "color": "#34a853",
                "icon": "🟢",
                "text": "MMKG Ready",
                "description": "Knowledge graph ready for queries"
            }
        }

        config = status_config.get(status, status_config["idle"])

        st.markdown(f"""
        <div style="
            background-color: {config['color']}20;
            border-left: 4px solid {config['color']};
            padding: 12px;
            border-radius: 4px;
            margin-bottom: 12px;
        ">
            <div style="font-size: 0.9rem; font-weight: 600; color: #333;">
                {config['icon']} {config['text']}
            </div>
            <div style="font-size: 0.75rem; color: #5f6368; margin-top: 4px;">
                {config['description']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if status == "building":
            self._display_progress()

        if status == "ready":
            self._display_stats()

        if current_doc := st.session_state.get("current_document"):
            st.caption(f"📄 Current Doc: {current_doc}")

    def _display_progress(self):
        """Display build progress bar and current stage."""
        processing_status = st.session_state.get("processing_status", {})

        current_stage = processing_status.get("progress", 0)
        total_stages = processing_status.get("total_stages", 7)
        stage_description = processing_status.get("stage", "Processing...")

        progress_pct = int((current_stage / total_stages) * 100)

        st.caption(stage_description)
        st.progress(progress_pct / 100)
        st.caption(f"Stage {current_stage} of {total_stages}")

    def _display_stats(self):
        """Display MMKG statistics."""
        stats = st.session_state.get("graph_stats", {})

        if stats:
            col1, col2 = st.columns(2)
            with col1:
                entity_count = stats.get("entity_count", 0)
                st.metric(
                    "Entities",
                    entity_count,
                    delta=None,
                    label_visibility="visible"
                )
            with col2:
                rel_count = stats.get("relationship_count", 0)
                st.metric(
                    "Relations",
                    rel_count,
                    delta=None,
                    label_visibility="visible"
                )
