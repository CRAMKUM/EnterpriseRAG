"""Base tool class for Enterprise RAG tools."""

import logging
from typing import Dict, Any
from utils.logger import get_logger

class BaseTool:
    """Base class for all tools in the Enterprise RAG system."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize base tool with config."""
        self.config = config
        self.logger = get_logger(self.__class__.__name__)

    @property
    def name(self) -> str:
        """Name of the tool."""
        raise NotImplementedError

    @property
    def cost_tier(self) -> str:
        """Cost tier of the tool (free, low, medium, high)."""
        raise NotImplementedError

    def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """Execute tool logic."""
        raise NotImplementedError

    def validate_input(self, *args, **kwargs) -> bool:
        """Validate input parameters."""
        return True
