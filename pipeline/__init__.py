"""Pipeline for MMKG construction from documents."""
from .document_processor import DocumentProcessor
from .mmkg_builder import MMKGBuilder

__all__ = ['DocumentProcessor', 'MMKGBuilder']
