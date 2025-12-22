"""
Package initialization for translation pipeline.
"""
from .webnlg_loader import WebNLGLoader
from .llm_handler import LLMHandler

__all__ = [
    'WebNLGLoader',
    'LLMHandler'
]
