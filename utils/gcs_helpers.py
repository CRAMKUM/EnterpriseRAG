"""Google Cloud Storage helper utilities."""

import json
from typing import Any, Dict, Optional
from google.cloud import storage
from .logger import get_logger
from .exceptions import ConfigurationError

logger = get_logger(__name__)


class GCSHelper:
    """Helper class for Google Cloud Storage operations."""

    def __init__(self, project_id: Optional[str] = None):
        """
        Initialize GCS helper.

        Args:
            project_id: GCP project ID (optional, uses default if not provided)
        """
        try:
            self.client = storage.Client(project=project_id)
            logger.info("GCS client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize GCS client: {e}")
            raise ConfigurationError(f"GCS initialization failed: {e}")

    def download_json(self, bucket_name: str, blob_path: str) -> Dict[str, Any]:
        """
        Download and parse JSON file from GCS.

        Args:
            bucket_name: GCS bucket name
            blob_path: Path to blob in bucket

        Returns:
            Parsed JSON content as dictionary
        """
        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            content = blob.download_as_text()
            return json.loads(content)
        except Exception as e:
            logger.error(f"Failed to download JSON from gs://{bucket_name}/{blob_path}: {e}")
            raise ConfigurationError(f"Failed to download config: {e}")

    def upload_file(self, bucket_name: str, source_path: str, destination_blob_path: str) -> str:
        """
        Upload file to GCS.

        Args:
            bucket_name: GCS bucket name
            source_path: Local file path
            destination_blob_path: Destination path in bucket

        Returns:
            GCS URI of uploaded file
        """
        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(destination_blob_path)
            blob.upload_from_filename(source_path)
            uri = f"gs://{bucket_name}/{destination_blob_path}"
            logger.info(f"Uploaded file to {uri}")
            return uri
        except Exception as e:
            logger.error(f"Failed to upload file: {e}")
            raise

    def list_files(self, bucket_name: str, prefix: str = "") -> list:
        """
        List files in GCS bucket with optional prefix.

        Args:
            bucket_name: GCS bucket name
            prefix: Prefix to filter files

        Returns:
            List of blob names
        """
        try:
            bucket = self.client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=prefix)
            return [blob.name for blob in blobs]
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            raise

    def file_exists(self, bucket_name: str, blob_path: str) -> bool:
        """
        Check if file exists in GCS.

        Args:
            bucket_name: GCS bucket name
            blob_path: Path to blob

        Returns:
            True if file exists, False otherwise
        """
        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            return blob.exists()
        except Exception as e:
            logger.error(f"Failed to check file existence: {e}")
            return False
