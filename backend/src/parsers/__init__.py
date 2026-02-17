"""
News Feed Parsers Package

Provides parser implementations for various news sources.
"""

from .bbc import BBCNewsParser
from .buzzfeed import BuzzFeedNewsParser
from .cnbc import CNBCNewsParser
from .sky import SkyNewsParser

__all__ = ["BBCNewsParser", "BuzzFeedNewsParser", "CNBCNewsParser", "SkyNewsParser"]
