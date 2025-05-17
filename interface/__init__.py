"""Wopper interface package."""

from .chatgpt_interface import ChatGPTInterface
from .wikidata_interface import WikidataInterface

__all__ = [
    "ChatGPTInterface",
    "WikidataInterface",
]
