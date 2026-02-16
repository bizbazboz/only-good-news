"""
News Feed Parsers Package

Provides parser implementations for various news sources.
"""

from .bbc import BBCNewsParser
from .sky import SkyNewsParser

__all__ = ["BBCNewsParser", "SkyNewsParser"]
