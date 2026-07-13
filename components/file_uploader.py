"""File Upload Component - Document upload interface"""

import streamlit as st
from typing import Optional
from utils.logger import get_logger

logger = get_logger(__name__)

class FileUploadComponent:
    """Handles document upload interface."""

    def __init__(self, bucket_config: dict, doc_processor):
        """Initialize file uploader."""
        self.bucket_config = bucket_config
        self.doc_processor = doc_processor
        self.upload_bucket = bucket_config.get("upload_bucket")
        logger.info("FileUploadComponent initialized")

    def render(self) -> Optional[st.runtime.uploaded_file_manager.UploadedFile]:
        """Render file upload interface."""
        st.markdown("### Document Upload")

        uploaded_file = st.file_uploader(
            "Upload a document",
            type=["pdf", "docx", "pptx"],
            help="Supported formats: PDF, DOCX, PPTX",
            label_visibility="collapsed"
        )

        if uploaded_file:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.markdown(f"""
                <div style="
                    background-color: #f0f4f8;
                    border-left: 4px solid #1a73e8;
                    padding: 12px;
                    border-radius: 4px;
                    margin-top: 8px;
                ">
                    <div style="font-size: 0.9rem; color: #333;">
                        <strong>📎 {uploaded_file.name}</strong><br>
                        <span style="color: #5f6368;">
                            Size: {file_size_mb:.2f} MB
                        </span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            if file_size_mb > 10:
                st.warning("⚠️ Large file detected. Processing may take several minutes.")

            return uploaded_file
        return None
