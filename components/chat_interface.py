"""Chat Interface Component - Main chat UI"""

import streamlit as st
from typing import Optional
from utils.logger import get_logger

logger = get_logger(__name__)

class ChatInterface:
    """Main chat interface for interacting with the RAG system."""

    def __init__(self, orchestrator, graph_manager, gql_generator, hybrid_search):
        """Initialize chat interface."""
        self.orchestrator = orchestrator
        self.graph_manager = graph_manager
        self.gql_generator = gql_generator
        self.hybrid_search = hybrid_search
        logger.info("ChatInterface initialized")

    def render(self):
        """Render chat interface."""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                if "tool_calls" in message and message["tool_calls"]:
                    with st.expander("🛠️ Tool Usage", expanded=False):
                        for tool_call in message["tool_calls"]:
                            st.markdown(f"**{tool_call['tool']}**")
                            st.code(tool_call.get('result', 'Processing...'), language="text")

                if "graph_query" in message and message["graph_query"]:
                    with st.expander("💬 Graph Query (GQL)", expanded=False):
                        st.code(message["graph_query"], language="sql")

        if prompt := st.chat_input("Ask a question about your documents..."):
            st.session_state.messages.append({
                "role": "user",
                "content": prompt
            })

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                tool_placeholder = st.expander("🛠️ Tool Usage", expanded=False)

                try:
                    response_data = self._process_query(prompt, tool_placeholder)
                    response_placeholder.markdown(response_data["response"])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_data["response"],
                        "tool_calls": response_data.get("tool_calls", []),
                        "graph_query": response_data.get("graph_query")
                    })
                except Exception as e:
                    logger.error(f"Chat error: {e}")
                    error_msg = f"❌ Error: {str(e)}"
                    response_placeholder.markdown(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

    def _process_query(self, prompt: str, tool_placeholder) -> dict:
        """Process query using orchestrator."""
        try:
            context = self._build_context()
            result = self.orchestrator.process_query(
                query=prompt,
                context=context,
                mmkg_available=(st.session_state.mmkg_status == "ready")
            )

            if result.get("tool_calls"):
                with tool_placeholder:
                    for tool_call in result["tool_calls"]:
                        st.markdown(f"**{tool_call['tool']}**")
                        st.code(tool_call.get('result', 'Processing...'), language="text")

            graph_query = result.get("graph_query")
            if graph_query:
                with tool_placeholder:
                    st.markdown("**Graph Query (GQL)**")
                    st.code(graph_query, language="sql")

            return {
                "response": result.get("response", "No response generated"),
                "tool_calls": result.get("tool_calls", []),
                "graph_query": graph_query
            }
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            raise

    def _build_context(self) -> str:
        """Build context from recent messages."""
        recent_messages = st.session_state.messages[-5:]
        context_parts = []
        for msg in recent_messages:
            role = msg["role"]
            content = msg["content"]
            context_parts.append(f"{role}: {content}")
        return "\n".join(context_parts)
