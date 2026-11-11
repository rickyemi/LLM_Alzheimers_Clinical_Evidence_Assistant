"""AD Clinical Evidence Assistant: governed RAG + agentic AI for Alzheimer's evidence."""
from .config import Settings, load_settings
from .rag import RAGPipeline, RAGResponse, build_pipeline

__version__ = "0.1.0"
__all__ = ["Settings", "load_settings", "RAGPipeline", "RAGResponse", "build_pipeline", "__version__"]
