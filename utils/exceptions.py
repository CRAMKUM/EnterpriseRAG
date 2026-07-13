"""Custom exceptions for the application."""


class EnterpriseRAGError(Exception):
    """Base exception for Enterprise RAG system."""
    pass


class ConfigurationError(EnterpriseRAGError):
    """Raised when configuration loading or validation fails."""
    pass


class ToolExecutionError(EnterpriseRAGError):
    """Raised when a tool execution fails."""
    pass


class PipelineError(EnterpriseRAGError):
    """Raised when pipeline processing fails."""
    pass


class GraphError(EnterpriseRAGError):
    """Raised when graph operations fail."""
    pass


class GCSError(EnterpriseRAGError):
    """Raised when GCS operations fail."""
    pass


class SessionError(EnterpriseRAGError):
    """Raised when session management operations fail."""
    pass
