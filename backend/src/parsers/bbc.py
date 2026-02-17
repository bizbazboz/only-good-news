"""
BBC News RSS Feed Parser
A Python library to fetch and convert BBC news RSS feeds to JSON format.
Uses only built-in Python libraries.
"""

import json
import logging
import re
import urllib.request
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class BBCNewsParser:
    """Parser for BBC News RSS feeds using only built-in libraries."""
    
    def __init__(self, feed_url: str = "https://feeds.bbci.co.uk/news/england/rss.xml"):
        """
        Initialize the BBC News Parser.
        
        Args:
            feed_url: The URL of the BBC RSS feed to parse.
        """
        self.feed_url = feed_url
        self.feed_data = None
        self._thumbnail_hosts = set()
    
    def fetch_feed(self) -> bool:
        """
        Fetch the RSS feed from the BBC.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            with urllib.request.urlopen(self.feed_url, timeout=10) as response:
                xml_content = response.read()
                self.feed_data = ET.fromstring(xml_content)
                self._build_thumbnail_host_allowlist()
                return True
        except Exception as e:
            logger.error(f"Error fetching feed: {e}")
            return False

    def _build_thumbnail_host_allowlist(self):
        """Build allowed thumbnail hosts from actual media fields in this feed."""
        hosts = set()
        parsed_feed_host = urlparse(self.feed_url).hostname
        if parsed_feed_host:
            hosts.add(parsed_feed_host.lower())

        if self.feed_data is None:
            self._thumbnail_hosts = hosts
            return

        channel = self.feed_data.find("channel")
        if channel is not None:
            for item in channel.findall("item")[:30]:
                for path in (
                    ".//{http://search.yahoo.com/mrss/}thumbnail",
                    ".//{http://search.yahoo.com/mrss/}content",
                    "enclosure",
                ):
                    node = item.find(path)
                    if node is None:
                        continue
                    media_url = node.get("url", "")
                    host = urlparse(media_url).hostname
                    if host:
                        hosts.add(host.lower())

        self._thumbnail_hosts = hosts
    
    def _parse_entry(self, item: ET.Element) -> Dict:
        """
        Parse a single RSS item into a dictionary.
        
        Args:
            item: An XML Element object representing an RSS item.
            
        Returns:
            Dictionary containing parsed entry data.
        """
        def get_text(element: ET.Element, tag: str, default: str = "") -> str:
            """Safely extract text from an XML element."""
            child = element.find(tag)
            return child.text if child is not None and child.text else default

        description = get_text(item, "description")
        
        parsed_entry = {
            "title": get_text(item, "title"),
            "link": get_text(item, "link"),
            "description": description,
            "published": get_text(item, "pubDate"),
            "pub_date": get_text(item, "pubDate"),
            "guid": get_text(item, "guid"),
        }

        parsed_entry["thumbnail"] = self._extract_thumbnail(item, description)
        
        return parsed_entry

    def _normalize_thumbnail_url(self, url: str) -> str:
        """Normalize thumbnail URL and keep only valid http(s) links."""
        if not url:
            return ""
        normalized = url.strip()
        if normalized.startswith("//"):
            normalized = "https:" + normalized
        if normalized.startswith("http://") or normalized.startswith("https://"):
            return self._upgrade_bbc_thumbnail_url(normalized)
        return ""

    def _upgrade_bbc_thumbnail_url(self, url: str) -> str:
        """
        Upgrade BBC iChef thumbnails to higher resolution when possible.
        Typical low-res feed URL pattern is /ace/standard/240/... .
        """
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if "ichef.bbci.co.uk" not in host:
            return url

        match = re.search(r"/ace/standard/(\\d+)/", url)
        if not match:
            return url

        try:
            current_size = int(match.group(1))
        except ValueError:
            return url

        if current_size >= 1024:
            return url

        return re.sub(r"/ace/standard/\\d+/", "/ace/standard/1024/", url, count=1)

    def _is_valid_thumbnail_url(self, url: str) -> bool:
        """Validate thumbnail URL against feed-derived host allowlist."""
        if not url:
            return False
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return False
        if not self._thumbnail_hosts:
            return True
        host = parsed.hostname.lower()
        return host in self._thumbnail_hosts

    def _extract_thumbnail(self, item: ET.Element, description: str = "") -> str:
        """Extract thumbnail URL from several common RSS/media patterns."""
        candidates = []

        media_thumbnail = item.find(".//{http://search.yahoo.com/mrss/}thumbnail")
        if media_thumbnail is not None:
            candidates.append(media_thumbnail.get("url", ""))

        media_content = item.find(".//{http://search.yahoo.com/mrss/}content")
        if media_content is not None:
            media_type = (media_content.get("type", "") or "").lower()
            medium = (media_content.get("medium", "") or "").lower()
            if not media_type or media_type.startswith("image/") or medium == "image":
                candidates.append(media_content.get("url", ""))

        enclosure = item.find("enclosure")
        if enclosure is not None:
            enclosure_type = (enclosure.get("type", "") or "").lower()
            if not enclosure_type or enclosure_type.startswith("image/"):
                candidates.append(enclosure.get("url", ""))

        image_url = item.find("image/url")
        if image_url is not None and image_url.text:
            candidates.append(image_url.text)

        image_text = item.find("image")
        if image_text is not None and image_text.text:
            candidates.append(image_text.text)

        if description:
            img_src_match = re.search(r'<img[^>]+src=["\\\']([^"\\\']+)["\\\']', description, flags=re.IGNORECASE)
            if img_src_match:
                candidates.append(img_src_match.group(1))

            srcset_match = re.search(r'srcset=["\\\']([^"\\\']+)["\\\']', description, flags=re.IGNORECASE)
            if srcset_match:
                first_src = srcset_match.group(1).split(",")[0].strip().split(" ")[0]
                if first_src:
                    candidates.append(first_src)

        for candidate in candidates:
            normalized = self._normalize_thumbnail_url(candidate)
            if normalized and self._is_valid_thumbnail_url(normalized):
                return normalized

        return ""
    
    def get_articles(self) -> List[Dict]:
        """
        Get all articles from the feed.
        
        Returns:
            List of dictionaries containing article data.
        """
        if self.feed_data is None:
            if not self.fetch_feed():
                return []
        
        articles = []
        channel = self.feed_data.find("channel")
        if channel is not None:
            for item in channel.findall("item"):
                articles.append(self._parse_entry(item))
        
        return articles
    
    def get_feed_info(self) -> Dict:
        """
        Get metadata about the feed itself.
        
        Returns:
            Dictionary containing feed metadata.
        """
        if self.feed_data is None:
            if not self.fetch_feed():
                return {}
        
        channel = self.feed_data.find("channel")
        if channel is None:
            return {}
        
        def get_text(tag: str, default: str = "") -> str:
            """Safely extract text from channel element."""
            elem = channel.find(tag)
            return elem.text if elem is not None and elem.text else default
        
        return {
            "title": get_text("title"),
            "description": get_text("description"),
            "link": get_text("link"),
            "language": get_text("language"),
            "copyright": get_text("copyright"),
            "last_build_date": get_text("lastBuildDate"),
        }
    
    def to_json(self, indent: int = 2, include_feed_info: bool = True) -> str:
        """
        Convert the RSS feed to JSON string.
        
        Args:
            indent: Number of spaces for JSON indentation (None for compact).
            include_feed_info: Whether to include feed metadata in the output.
            
        Returns:
            JSON string representation of the feed.
        """
        articles = self.get_articles()
        
        result = {
            "articles": articles,
            "article_count": len(articles),
        }
        
        if include_feed_info:
            result["feed_info"] = self.get_feed_info()
        
        return json.dumps(result, indent=indent, ensure_ascii=False)
    
    def to_dict(self, include_feed_info: bool = True) -> Dict:
        """
        Convert the RSS feed to a Python dictionary.
        
        Args:
            include_feed_info: Whether to include feed metadata in the output.
            
        Returns:
            Dictionary representation of the feed.
        """
        articles = self.get_articles()
        
        result = {
            "articles": articles,
            "article_count": len(articles),
        }
        
        if include_feed_info:
            result["feed_info"] = self.get_feed_info()
        
        return result
    
    def save_to_file(self, filename: str, indent: int = 2, 
                     include_feed_info: bool = True) -> bool:
        """
        Save the parsed feed to a JSON file.
        
        Args:
            filename: Path to the output JSON file.
            indent: Number of spaces for JSON indentation.
            include_feed_info: Whether to include feed metadata.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            json_data = self.to_json(indent=indent, include_feed_info=include_feed_info)
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(json_data)
            logger.info(f"Feed saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving feed to file: {e}")
            return False
