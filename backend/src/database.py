#!/usr/bin/env python3
"""
SQLite Database Module

Handles persistent storage of news articles and smart re-indexing.
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


class NewsDatabase:
    """SQLite database for news articles with smart indexing."""
    
    def __init__(self, db_path: str = "news_archive.db"):
        """Initialize database connection."""
        self.db_path = db_path
        self._init_db()
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_db(self):
        """Initialize database schema."""
        try:
            # Ensure the SQLite parent directory exists (e.g. data/news_archive.db).
            db_parent = Path(self.db_path).expanduser().resolve().parent
            db_parent.mkdir(parents=True, exist_ok=True)

            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Create articles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feed_url TEXT NOT NULL,
                    feed_type TEXT NOT NULL,
                    feed_name TEXT NOT NULL,
                    guid TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    thumbnail TEXT,
                    link TEXT NOT NULL,
                    pub_date TEXT,
                    sentiment TEXT,
                    confidence REAL,
                    label TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(feed_url, guid)
                )
            """)
            
            # Create indexes for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_feed_url 
                ON articles(feed_url)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sentiment 
                ON articles(sentiment)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_pub_date 
                ON articles(pub_date DESC)
            """)

            # Backward-compatible migration for existing databases.
            cursor.execute("PRAGMA table_info(articles)")
            existing_columns = {row[1] for row in cursor.fetchall()}
            if "thumbnail" not in existing_columns:
                cursor.execute("ALTER TABLE articles ADD COLUMN thumbnail TEXT")
            
            conn.commit()
            conn.close()
            logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def article_exists(self, feed_url: str, guid: str) -> bool:
        """Check if an article already exists in the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM articles WHERE feed_url = ? AND guid = ? LIMIT 1",
                (feed_url, guid)
            )
            return cursor.fetchone() is not None
    
    def get_feed_article_guids(self, feed_url: str) -> set:
        """Get all article GUIDs for a specific feed."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT guid FROM articles WHERE feed_url = ?",
                (feed_url,)
            )
            return {row['guid'] for row in cursor.fetchall()}
    
    def insert_article(self, article: Dict, feed_url: str, feed_type: str, 
                      feed_name: str, sentiment_info: Optional[Dict] = None):
        """Insert a new article into the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    INSERT INTO articles 
                    (feed_url, feed_type, feed_name, guid, title, description,
                     thumbnail, link, pub_date, sentiment, confidence, label)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feed_url,
                    feed_type,
                    feed_name,
                    article.get('guid', article.get('link', '')),
                    article.get('title', ''),
                    article.get('description', ''),
                    article.get('thumbnail', ''),
                    article.get('link', ''),
                    article.get('pub_date', ''),
                    sentiment_info.get('sentiment') if sentiment_info else None,
                    sentiment_info.get('confidence') if sentiment_info else None,
                    sentiment_info.get('label') if sentiment_info else None
                ))
            except sqlite3.IntegrityError:
                # Article already exists, update it instead
                cursor.execute("""
                    UPDATE articles 
                    SET title = ?, description = ?, thumbnail = ?, link = ?, pub_date = ?,
                        sentiment = ?, confidence = ?, label = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE feed_url = ? AND guid = ?
                """, (
                    article.get('title', ''),
                    article.get('description', ''),
                    article.get('thumbnail', ''),
                    article.get('link', ''),
                    article.get('pub_date', ''),
                    sentiment_info.get('sentiment') if sentiment_info else None,
                    sentiment_info.get('confidence') if sentiment_info else None,
                    sentiment_info.get('label') if sentiment_info else None,
                    feed_url,
                    article.get('guid', article.get('link', ''))
                ))
    
    def bulk_insert_articles(self, articles: List[Dict], feed_url: str, 
                            feed_type: str, feed_name: str, 
                            sentiment_results: List[Dict]):
        """Insert multiple articles with their sentiment analysis."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            for article, sentiment_info in zip(articles, sentiment_results):
                try:
                    cursor.execute("""
                        INSERT INTO articles 
                        (feed_url, feed_type, feed_name, guid, title, description,
                         thumbnail, link, pub_date, sentiment, confidence, label)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        feed_url,
                        feed_type,
                        feed_name,
                        article.get('guid', article.get('link', '')),
                        article.get('title', ''),
                        article.get('description', ''),
                        article.get('thumbnail', ''),
                        article.get('link', ''),
                        article.get('pub_date', ''),
                        sentiment_info['sentiment'],
                        sentiment_info['confidence'],
                        sentiment_info['label']
                    ))
                except sqlite3.IntegrityError:
                    # Skip duplicates
                    pass

    def bulk_upsert_articles(
        self,
        articles: List[Dict],
        feed_url: str,
        feed_type: str,
        feed_name: str,
    ):
        """Insert or update multiple raw articles without sentiment labels."""
        if not articles:
            return

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT INTO articles
                (feed_url, feed_type, feed_name, guid, title, description,
                 thumbnail, link, pub_date, sentiment, confidence, label)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(feed_url, guid) DO UPDATE SET
                    title = excluded.title,
                    description = excluded.description,
                    thumbnail = excluded.thumbnail,
                    link = excluded.link,
                    pub_date = excluded.pub_date,
                    updated_at = CURRENT_TIMESTAMP
                """,
                [
                    (
                        feed_url,
                        feed_type,
                        feed_name,
                        article.get("guid", article.get("link", "")),
                        article.get("title", ""),
                        article.get("description", ""),
                        article.get("thumbnail", ""),
                        article.get("link", ""),
                        article.get("pub_date", ""),
                        None,
                        None,
                        None,
                    )
                    for article in articles
                ],
            )
    
    def get_all_articles(self, min_confidence: float = 0.6) -> Dict:
        """Get positive articles from database filtered by confidence."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT feed_name, feed_type, title, description, thumbnail, link,
                       pub_date, sentiment, confidence, label
                FROM articles 
                WHERE sentiment IS NOT NULL
                ORDER BY pub_date DESC
            """)
            
            rows = cursor.fetchall()
            
            positive_articles = []
            included_count = 0
            
            for row in rows:
                confidence = float(row['confidence'] or 0.0)
                if confidence < min_confidence:
                    # Exclude sub-threshold articles entirely from API response.
                    continue

                article = {
                    'source': row['feed_name'],
                    'title': row['title'],
                    'description': row['description'],
                    'thumbnail': row['thumbnail'],
                    'link': row['link'],
                    'pub_date': row['pub_date'],
                    'sentiment': row['sentiment'],
                    'confidence': confidence,
                    'label': row['label']
                }
                if row['sentiment'] == 'positive':
                    positive_articles.append(article)
                    included_count += 1
            
            return {
                'positive_articles': positive_articles,
                'total': included_count
            }
    
    def get_feed_stats(self) -> List[Dict]:
        """Get statistics for each feed source."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT feed_name, COUNT(*) as count
                FROM articles
                GROUP BY feed_name
                ORDER BY count DESC
            """)
            
            return [
                {'name': row['feed_name'], 'count': row['count']}
                for row in cursor.fetchall()
            ]
    
    def check_for_new_articles(self, feed_url: str, current_guids: List[str]) -> Tuple[bool, List[str]]:
        """
        Check if there are new articles for a feed.
        
        Returns:
            Tuple of (has_new_articles, list_of_new_guids)
        """
        existing_guids = self.get_feed_article_guids(feed_url)
        new_guids = [guid for guid in current_guids if guid not in existing_guids]
        
        return len(new_guids) > 0, new_guids
    
    def reindex_feed(self, feed_url: str):
        """Delete all articles from a feed to prepare for re-indexing."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM articles WHERE feed_url = ?", (feed_url,))
            deleted = cursor.rowcount
            if deleted > 0:
                print(f"    🔄 Clearing {deleted} outdated article(s) for re-indexing")
            return deleted
    
    def clear_feed(self, feed_url: str):
        """Clear all articles for a specific feed."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM articles WHERE feed_url = ?", (feed_url,))
    
    def clear_all(self):
        """Clear all articles from the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM articles")
