"""
Enterprise RAG - Streamlit Frontend
Main application with persistent session management in GCS
"""

import streamlit as st

# Page config MUST be first Streamlit command
st.set_page_config(
    page_title="Enterprise RAG - Knowledge Graph",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

import uuid
import warnings
from pathlib import Path

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from config.config_loader import get_config_loader
from agents.orchestrator import Orchestrator
from pipeline.document_processor import DocumentProcessor
from pipeline.mmkg_builder import MMKGBuilder
from graph.spanner_graph import SpannerGraphManager
from graph.gql_generator import GQLGenerator
from graph.hybrid_search import HybridSearch
from utils.logger import get_logger

# Import components
from components.session_manager import SessionManager
from components.file_uploader import FileUploadComponent
from components.chat_interface import ChatInterface
from components.graph_visualizer import GraphVisualizer
from components.mmkg_status import MMKGStatusComponent

logger = get_logger(__name__)

# CSS Styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    color: #1a73e8;
    margin-bottom: 0.5rem;
}
.sub-header {
    font-size: 1.1rem;
    color: #5f6368;
    margin-bottom: 2rem;
}
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s ease;
}
button[kind="primary"] {
    background-color: #1a73e8 !important;
    color: white !important;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "mmkg_status" not in st.session_state:
        st.session_state.mmkg_status = "idle"
    if "current_document" not in st.session_state:
        st.session_state.current_document = None
    if "processing_status" not in st.session_state:
        st.session_state.processing_status = {
            "stage": "",
            "progress": 0,
            "total_stages": 7
        }
    if "graph_stats" not in st.session_state:
        st.session_state.graph_stats = {
            "total_entities": 0,
            "total_relationships": 0,
            "entity_types": []
        }

@st.cache_resource
def get_core_components():
    """Initialize and cache core components."""
    try:
        config_loader = get_config_loader()
        bucket_config = config_loader.get_bucket_config()

        orchestrator = Orchestrator()
        doc_processor = DocumentProcessor()
        mmkg_builder = MMKGBuilder()

        try:
            graph_manager = SpannerGraphManager()
            gql_generator = GQLGenerator()
            hybrid_search = HybridSearch(graph_manager)
        except Exception as e:
            logger.warning(f"Graph components not available: {e}")
            graph_manager = None
            gql_generator = None
            hybrid_search = None

        session_manager = SessionManager(bucket_config)

        return {
            "orchestrator": orchestrator,
            "doc_processor": doc_processor,
            "mmkg_builder": mmkg_builder,
            "graph_manager": graph_manager,
            "gql_generator": gql_generator,
            "hybrid_search": hybrid_search,
            "session_manager": session_manager,
            "bucket_config": bucket_config
        }
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        st.error(f"Initialization failed: {e}")
        return None

def main():
    init_session_state()
    components = get_core_components()
    if not components:
        st.error("Failed to initialize application. Please check logs.")
        return

    if st.session_state.user_id is None:
        st.session_state.user_id = components["session_manager"].get_or_create_user_id()

    if not st.session_state.messages:
        st.session_state.messages = components["session_manager"].load_chat_history(
            st.session_state.user_id
        )

    st.markdown('<h1 class="main-header">🧠 Enterprise RAG</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multimodal Knowledge Graph from Documents</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚙️ Control Panel")
        mmkg_status_component = MMKGStatusComponent(components["graph_manager"])
        mmkg_status_component.render()

        st.markdown("---")
        file_uploader = FileUploadComponent(
            components["bucket_config"],
            components["doc_processor"]
        )
        uploaded_file = file_uploader.render()

        if uploaded_file:
            handle_file_upload(uploaded_file, components)

        st.markdown("---")
        st.markdown("### 👤 Session")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save", use_container_width=True):
                save_session(components["session_manager"])
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                clear_session(components["session_manager"])

        st.caption(f"User: {st.session_state.user_id[:8]}...")
        st.caption(f"Session: {st.session_state.session_id[:8]}...")

    tab1, tab2, tab3 = st.tabs(["💬 Chat", "🗺️ Graph Explorer", "📊 Analytics"])

    with tab1:
        chat_interface = ChatInterface(
            orchestrator=components["orchestrator"],
            graph_manager=components["graph_manager"],
            gql_generator=components["gql_generator"],
            hybrid_search=components["hybrid_search"]
        )
        chat_interface.render()

    with tab2:
        if components["graph_manager"]:
            graph_visualizer = GraphVisualizer(components["graph_manager"])
            graph_visualizer.render()
        else:
            st.warning("Graph database not configured.")

    with tab3:
        st.markdown("### System Analytics")
        if components["graph_manager"]:
            stats = components["graph_manager"].get_graph_statistics()
            if stats.get("success"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Entities", stats.get("total_entities", 0))
                col2.metric("Total Relationships", stats.get("total_relationships", 0))
                avg_degree = (
                    stats.get("total_relationships", 0) / stats.get("total_entities", 1)
                    if stats.get("total_entities", 0) > 0
                    else 0
                )
                col3.metric("Avg Connections", f"{avg_degree:.1f}")

                entity_types = stats.get("entity_types", [])
                if entity_types:
                    import pandas as pd
                    df = pd.DataFrame(entity_types)
                    st.bar_chart(df.set_index("entity_type"))
        else:
            st.info("Graph database not configured")

def handle_file_upload(uploaded_file, components):
    st.session_state.current_document = uploaded_file.name
    st.session_state.mmkg_status = "building"

    update_progress("Uploading document...", 0)
    temp_dir = Path("/tmp/enterprise_rag")
    temp_dir.mkdir(exist_ok=True)
    temp_file = temp_dir / uploaded_file.name

    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        update_progress("Parsing document structure...", 1)
        result = components["doc_processor"].process_document(str(temp_file))

        update_progress("Processing images (deblurring)...", 2)
        update_progress("Interpreting charts and diagrams...", 3)

        update_progress("Extracting entities (MegaRAG)...", 4)
        mmkg_data = components["mmkg_builder"].build_mmkg(result["pages"])

        update_progress("Building relationships...", 5)
        update_progress("Inserting into knowledge graph...", 6)

        if components["graph_manager"]:
            components["graph_manager"].insert_entities(mmkg_data["entities"])
            components["graph_manager"].insert_relationships(mmkg_data["relationships"])

        update_progress("MMKG build complete!", 7)
        st.session_state.mmkg_status = "ready"
        st.session_state.graph_stats = mmkg_data["metadata"]
        st.success(f"MMKG built: {mmkg_data['metadata']['entity_count']} entities, {mmkg_data['metadata']['relationship_count']} relationships")

        st.session_state.processing_status = {
            "stage": "",
            "progress": 0,
            "total_stages": 7
        }
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        st.error(f"Processing failed: {e}")
        st.session_state.mmkg_status = "idle"
    finally:
        if temp_file.exists():
            temp_file.unlink()

def update_progress(stage: str, progress: int):
    st.session_state.processing_status = {
        "stage": stage,
        "progress": progress,
        "total_stages": 7
    }

def save_session(session_manager):
    try:
        session_manager.save_chat_history(
            st.session_state.user_id,
            st.session_state.messages
        )
        session_manager.save_session_state(
            st.session_state.user_id,
            {
                "mmkg_status": st.session_state.mmkg_status,
                "current_document": st.session_state.current_document,
                "graph_stats": st.session_state.graph_stats
            }
        )
        st.success("Session saved to GCS!")
    except Exception as e:
        st.error(f"Failed to save session: {e}")

def clear_session(session_manager):
    st.session_state.messages = []
    st.session_state.mmkg_status = "idle"
    st.session_state.current_document = None
    st.session_state.graph_stats = {
        "total_entities": 0,
        "total_relationships": 0,
        "entity_types": []
    }
    try:
        session_manager.delete_session(st.session_state.user_id)
        st.success("Session cleared!")
    except Exception as e:
        st.warning(f"Failed to clear session: {e}")
    st.rerun()

if __name__ == "__main__":
    main()
