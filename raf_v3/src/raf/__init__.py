"""Retrieval-based dataset assembly for fair clustering."""
from .config import RAFConfig, SourceSpec
from .data import DataSource, InMemorySource, ParquetSource, SourceManager, load_dataset, load_sources
from .pipeline import RAFPipeline, RAFRunResult

__version__ = "0.3.0"
__all__ = ["RAFConfig", "SourceSpec", "DataSource", "InMemorySource", "ParquetSource",
           "SourceManager", "load_dataset", "load_sources", "RAFPipeline", "RAFRunResult"]
