"""Configuration loader with GCS integration and Streamlit TTL caching."""

import os
import time
from typing import Any, Dict, Optional
from utils.logger import get_logger
from utils.gcs_helpers import GCSHelper
from utils.exceptions import ConfigurationError

logger = get_logger(__name__)

# Try to import streamlit for TTL caching
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    logger.warning("Streamlit not available, using in-memory TTL caching")

class ConfigLoader:
    """
    Loads configuration from GCS bucket with TTL caching.

    Bootstrap: Only requires CONFIG_BUCKET_NAME environment variable.
    Everything else (bucket names, prompts, models) loaded from GCS.
    """
    def __init__(self, ttl_seconds: int = 60):
        """
        Initialize config loader.

        Args:
            ttl_seconds: Time-to-live for cache in seconds (default: 60)
        """
        self.ttl_seconds = ttl_seconds
        self.config_bucket = os.getenv('CONFIG_BUCKET_NAME')
        self._cache = {}
        self._cache_timestamps = {}

        if not self.config_bucket:
            raise ConfigurationError(
                "CONFIG_BUCKET_NAME environment variable is required"
            )

        self.gcs_helper = GCSHelper()
        logger.info(f"ConfigLoader initialized with bucket: {self.config_bucket}")

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache entry is still valid based on TTL."""
        if key not in self._cache_timestamps:
            return False

        elapsed = time.time() - self._cache_timestamps[key]
        return elapsed < self.ttl_seconds

    def _get_from_gcs(self, config_file: str) -> Dict[str, Any]:
        """
        Load configuration file from GCS.

        Args:
            config_file: Config file name (e.g., 'bucket_config.json')

        Returns:
            Configuration dictionary
        """
        try:
            config_path = f"config/{config_file}"
            logger.info(f"Loading config from gs://{self.config_bucket}/{config_path}")
            return self.gcs_helper.download_json(self.config_bucket, config_path)
        except Exception as e:
            logger.error(f"Failed to load {config_file}: {e}")
            raise ConfigurationError(f"Failed to load {config_file}: {e}")

    def _get_cached(self, key: str, config_file: str) -> Dict[str, Any]:
        """
        Get configuration with TTL caching.

        Args:
            key: Cache key
            config_file: Config file name

        Returns:
            Configuration dictionary
        """
        if self._is_cache_valid(key):
            logger.debug(f"Using cached config for {key}")
            return self._cache[key]

        logger.info(f"Cache miss or expired for {key}, reloading from GCS")
        config = self._get_from_gcs(config_file)
        self._cache[key] = config
        self._cache_timestamps[key] = time.time()
        return config

    def get_bucket_config(self) -> Dict[str, str]:
        """
        Get bucket configuration (bucket name and folder structure).
        Bucket name is sourced from graph_config.json for single source of truth.
        Uses internal TTL cache for performance.
        """
        if self._is_cache_valid('bucket_config'):
            return self._cache['bucket_config']

        logger.info("Fetching bucket_config from GCS...")
        bucket_config = self._get_from_gcs('bucket_config.json')

        graph_config = self.get_graph_config()
        bucket_config['bucket_name'] = graph_config['bucket_name']

        self._cache['bucket_config'] = bucket_config
        self._cache_timestamps['bucket_config'] = time.time()
        return bucket_config

    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model configuration (model names, versions, parameters).

        Returns:
            Dictionary with model configurations.
        """
        return self._get_cached('model_config', 'model_config.json')

    def get_system_prompts(self) -> Dict[str, str]:
        """
        Get system prompts for different agents.
        """
        return self._get_cached('system_prompts', 'system_prompts.json')

    def get_user_prompts(self) -> Dict[str, str]:
        """
        Get user prompt templates.
        """
        return self._get_cached('user_prompts', 'user_prompts.json')

    def get_tool_config(self) -> Dict[str, Any]:
        """
        Get tool configuration.
        """
        return self._get_cached('tool_config', 'tool_config.json')

    def get_pipeline_config(self) -> Dict[str, Any]:
        """
        Get pipeline configuration.
        """
        return self._get_cached('pipeline_config', 'pipeline_config.json')

    def get_graph_config(self) -> Dict[str, Any]:
        """
        Get graph configuration.
        """
        return self._get_cached('graph_config', 'graph_config.json')

    def reload_all(self) -> None:
        """Force reload all cached configurations from GCS."""
        logger.info("Force reloading all configurations")
        self._cache.clear()
        self._cache_timestamps.clear()

    def get_config(self, config_name: str) -> Dict[str, Any]:
        """
        Generic method to get any configuration.
        """
        config_file = f"{config_name}.json"
        return self._get_cached(config_name, config_file)

# Singleton instance
_config_loader: Optional[ConfigLoader] = None

def get_config_loader(ttl_seconds: int = 60) -> ConfigLoader:
    """Get singleton ConfigLoader instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(ttl_seconds=ttl_seconds)
    return _config_loader
