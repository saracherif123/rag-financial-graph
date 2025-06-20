"""
RAG-Based Property Graph for Financial Data

This package provides tools for building and querying a property graph
for Retrieval Augmented Generation (RAG) in the financial domain.
"""

from .graph_manager import FinancialGraph
from .data_downloader import FinancialDataDownloader

__version__ = "1.0.0"
__author__ = "RAG Financial Graph Team"

__all__ = [
    "FinancialGraph",
    "FinancialDataDownloader"
] 