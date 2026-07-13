"""Session Manager - Persistent session storage in GCS"""

import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional

import streamlit as st
from google.cloud import storage

from utils.logger import get_logger
from utils.exceptions import SessionError

logger = get_logger(__name__)


class SessionManager:
    """Manages persistent user sessions in GCS."""

    def __init__(self, bucket_config: Dict):
        """Initialize session manager."""
        self.bucket_name = bucket_config.get("bucket_name")
        self.sessions_folder = bucket_config.get("sessions_folder", "sessions/")
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(self.bucket_name)
        logger.info(f"SessionManager initialized with bucket: {self.bucket_name}")

    def get_or_create_user_id(self) -> str:
        """Get or create a persistent user ID using query params."""
        query_params = st.query_params
        user_id = query_params.get("user_id")

        if user_id:
            logger.info(f"Loaded user ID from query params: {user_id[:8]}...")
        else:
            user_id = str(uuid.uuid4())
            logger.info(f"Generated new user ID: {user_id[:8]}...")

        st.query_params["user_id"] = user_id
        return user_id

    def load_chat_history(self, user_id: str) -> List[Dict]:
        """Load chat history from GCS."""
        try:
            blob_path = f"{self.sessions_folder}{user_id}/chat_history.json"
            blob = self.bucket.blob(blob_path)

            if blob.exists():
                content = blob.download_as_string()
                messages = json.loads(content)
                logger.info(f"Loaded {len(messages)} messages for user {user_id[:8]}...")
                return messages
            return []
        except Exception as e:
            logger.error(f"Failed to load chat history: {e}")
            return []

    def save_chat_history(self, user_id: str, messages: List[Dict]) -> None:
        """Save chat history to GCS."""
        try:
            blob_path = f"{self.sessions_folder}{user_id}/chat_history.json"
            blob = self.bucket.blob(blob_path)

            content = json.dumps(messages, indent=2)
            blob.upload_from_string(
                content,
                content_type='application/json'
            )
            logger.info(f"Saved {len(messages)} messages for user {user_id[:8]}...")
        except Exception as e:
            logger.error(f"Failed to save chat history: {e}")
            raise SessionError(f"Failed to save chat history: {e}")

    def save_session_state(self, user_id: str, state_data: Dict) -> None:
        """Save session state to GCS."""
        try:
            blob_path = f"{self.sessions_folder}{user_id}/session_state.json"
            blob = self.bucket.blob(blob_path)

            state_data["last_updated"] = datetime.utcnow().isoformat()
            content = json.dumps(state_data, indent=2)
            blob.upload_from_string(
                content,
                content_type='application/json'
            )
            logger.info(f"Saved session state for user {user_id[:8]}...")
        except Exception as e:
            logger.error(f"Failed to save session state: {e}")
            raise SessionError(f"Failed to save session state: {e}")

    def load_session_state(self, user_id: str) -> Optional[Dict]:
        """Load session state from GCS."""
        try:
            blob_path = f"{self.sessions_folder}{user_id}/session_state.json"
            blob = self.bucket.blob(blob_path)

            if blob.exists():
                content = blob.download_as_string()
                state_data = json.loads(content)
                logger.info(f"Loaded session state for user {user_id[:8]}...")
                return state_data
            return None
        except Exception as e:
            logger.error(f"Failed to load session state: {e}")
            return None

    def delete_session(self, user_id: str) -> None:
        """Delete all session data for a user."""
        try:
            prefix = f"{self.sessions_folder}{user_id}/"
            blobs = self.bucket.list_blobs(prefix=prefix)

            deleted_count = 0
            for blob in blobs:
                blob.delete()
                deleted_count += 1
            logger.info(f"Deleted {deleted_count} session files for user {user_id[:8]}...")
        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            raise SessionError(f"Failed to delete session: {e}")

    def list_user_sessions(self) -> List[Dict]:
        """List all user sessions."""
        try:
            sessions = []
            blobs = self.bucket.list_blobs(prefix=self.sessions_folder)

            seen_users = set()
            for blob in blobs:
                relative_path = blob.name[len(self.sessions_folder):]
                parts = relative_path.split('/')
                if len(parts) >= 1 and parts[0]:
                    user_id = parts[0]
                    if user_id not in seen_users:
                        seen_users.add(user_id)
                        sessions.append({
                            "user_id": user_id,
                            "last_modified": blob.updated
                        })
            return sessions
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []
