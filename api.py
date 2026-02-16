#!/usr/bin/env python3
"""
BBC News Sentiment Analysis - FastAPI Backend

FastAPI application with SQLite persistence and smart re-indexing.
"""

import json
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
from contextlib import asynccontextmanager
import threading

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import NewsDatabase
from parsers import BBCNewsParser, SkyNewsParser
from sentiment_analysis import SentimentAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api.log')
    ]
)
logger = logging.getLogger(__name__)

# Cache remote feed update checks to avoid checking external feeds on every request.
UPDATE_CHECK_INTERVAL_MINUTES = 30
last_update_check: Optional[datetime] = None
update_check_lock = threading.Lock()


# Load config to get allowed origins
with open('config.json', 'r') as f:
    initial_config = json.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # Startup
    logger.info("="*70)
    logger.info("News Sentiment Analysis - FastAPI Backend")
    logger.info("="*70)
    logger.info("Supported sources: BBC News, Sky News")
    logger.info("Database: SQLite with smart re-indexing")
    logger.info("Frontend: Available at /")
    logger.info("Available endpoints:")
    logger.info("  GET  /              (frontend homepage)")
    logger.info("  GET  /api/news      (fetch news from database)")
    logger.info("  GET  /api/config    (get configuration)")
    logger.info("  POST /api/refresh   (force refresh all feeds)")
    logger.info("  POST /api/index     (index all feeds)")
    logger.info("="*70)
    
    yield
    
    # Shutdown - could add cleanup here if needed


# Initialize FastAPI app
app = FastAPI(
    title="News Sentiment Analysis",
    description="AI-powered multi-source news sentiment analysis with SQLite persistence",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware - MUST be added before routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=initial_config.get('allowed_origins', []),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
logger.info("Initializing database...")
db = NewsDatabase()

# Load sentiment analyzer once at startup
logger.info("Loading sentiment analysis model...")
sentiment_analyzer = SentimentAnalyzer()
logger.info("Model loaded successfully")


# Pydantic models
class NewsResponse(BaseModel):
    positive_articles: List[Dict]
    negative_articles: List[Dict]
    neutral_articles: List[Dict]
    feed_sources: List[Dict]
    total: int
    timestamp: str


class RefreshResponse(BaseModel):
    success: bool
    message: str
    data: NewsResponse = None


def load_config() -> Dict:
    """Load configuration from config.json."""
    with open('config.json', 'r') as f:
        return json.load(f)


def contains_banned_keyword(article: Dict, banned_keywords: List[str]) -> bool:
    """Check if article contains any banned keyword."""
    if not banned_keywords:
        return False
    
    # Combine title and description for searching
    text = (article.get('title', '') + ' ' + article.get('description', '')).lower()
    
    # Check each banned keyword
    for keyword in banned_keywords:
        if keyword.lower() in text:
            return True
    
    return False


def process_feed(feed_config: Dict, config: Dict) -> Dict:
    """
    Process a single feed - fetch, check for new articles, and re-index if needed.
    If new articles are detected, delete all articles from this feed and re-analyze everything.
    
    Returns dict with stats about the processing.
    """
    feed_url = feed_config['url']
    feed_name = feed_config['name']
    feed_type = feed_config.get('type', 'bbc_rss')
    
    logger.info(f"Checking {feed_name}...")
    
    # Parse based on feed type
    if feed_type == 'bbc_rss':
        parser = BBCNewsParser(feed_url)
        if not parser.fetch_feed():
            logger.error(f"Failed to fetch {feed_name}")
            return {'feed_name': feed_name, 'new_articles': 0, 'status': 'failed'}
        
        feed_data = parser.to_dict()
        articles = feed_data.get('articles', [])
    elif feed_type == 'sky_rss':
        parser = SkyNewsParser(feed_url)
        if not parser.fetch_feed():
            logger.error(f"Failed to fetch {feed_name}")
            return {'feed_name': feed_name, 'new_articles': 0, 'status': 'failed'}
        
        feed_data = parser.to_dict()
        articles = feed_data.get('articles', [])
    else:
        logger.warning(f"Unknown feed type: {feed_type}")
        return {'feed_name': feed_name, 'new_articles': 0, 'status': 'unknown_type'}
    
    if not articles:
        logger.info(f"No articles found for {feed_name}")
        return {'feed_name': feed_name, 'new_articles': 0, 'status': 'empty'}
    
    # Filter banned keywords
    banned_keywords = config.get('banned_keywords', [])
    if banned_keywords:
        original_count = len(articles)
        articles = [a for a in articles if not contains_banned_keyword(a, banned_keywords)]
        filtered = original_count - len(articles)
        if filtered > 0:
            logger.info(f"Filtered {filtered} article(s) from {feed_name}")

    # Filter by minimum title word count
    min_title_words = config.get('min_title_words', config.get('min_title_length', 4))
    before_len = len(articles)
    articles = [
        a for a in articles
        if len(a.get('title', '').strip().split()) >= min_title_words
    ]
    after_len = len(articles)
    if after_len < before_len:
        logger.info(
            f"Filtered {before_len - after_len} article(s) from {feed_name} "
            f"due to short title (<{min_title_words} words)"
        )
    
    # Check for new articles in the feed
    current_guids = [a.get('guid', a.get('link', '')) for a in articles]
    has_new, new_guids = db.check_for_new_articles(feed_url, current_guids)
    
    if not has_new:
        # No new articles - use cached data
        logger.info(f"No new articles for {feed_name} (cached {len(articles)} total)")
        return {'feed_name': feed_name, 'new_articles': 0, 'status': 'cached'}
    
    # New articles detected - delete all and re-index the entire feed
    logger.info(f"Detected {len(new_guids)} new article(s) for {feed_name}, re-indexing entire feed...")
    db.reindex_feed(feed_url)
    
    # Analyze ALL articles with fresh sentiment analysis (title + description)
    titles = [article.get('title', '') for article in articles]
    descriptions = [article.get('description', '') for article in articles]
    sentiment_results = sentiment_analyzer.analyze_batch(titles, descriptions)
    
    # Insert all articles from this feed
    db.bulk_insert_articles(
        articles, 
        feed_url, 
        feed_type, 
        feed_name, 
        sentiment_results
    )
    
    logger.info(f"Re-indexed {len(articles)} article(s) for {feed_name}")
    
    return {
        'feed_name': feed_name, 
        'new_articles': len(articles), 
        'status': 'updated'
    }


def fetch_and_update_news(config: Dict, force_check: bool = False) -> Dict:
    """Fetch news from all enabled feeds and check for new articles.
    If new articles detected for a feed, delete and re-analyze entire feed.
    Otherwise use cached data."""
    global last_update_check

    interval_minutes = config.get('update_check_interval_minutes', UPDATE_CHECK_INTERVAL_MINUTES)
    now = datetime.now()
    check_updates = force_check or last_update_check is None or (
        now - last_update_check >= timedelta(minutes=interval_minutes)
    )

    feed_stats = []
    feeds_updated = 0

    if check_updates:
        with update_check_lock:
            now = datetime.now()
            check_updates = force_check or last_update_check is None or (
                now - last_update_check >= timedelta(minutes=interval_minutes)
            )

            if check_updates:
                enabled_feeds = [f for f in config['feeds'] if f['enabled']]
                logger.info(
                    f"Checking {len(enabled_feeds)} feed(s) for updates "
                    f"(interval: {interval_minutes} min)..."
                )

                for feed_config in enabled_feeds:
                    stats = process_feed(feed_config, config)
                    feed_stats.append(stats)
                    if stats['status'] == 'updated':
                        feeds_updated += 1

                if feeds_updated > 0:
                    logger.info(f"Re-indexed {feeds_updated} feed(s) with new articles")
                else:
                    logger.info("No new articles found, using cached data")

                last_update_check = now
            else:
                logger.info("Update check already performed by another request; using cached data")
    else:
        minutes_since = (now - last_update_check).total_seconds() / 60 if last_update_check else 0
        minutes_until_next = max(0, interval_minutes - minutes_since)
        logger.info(
            f"Skipping feed update check; next check in about {minutes_until_next:.1f} minute(s)"
        )
    
    # Get all articles from database
    min_confidence = config.get('min_confidence', 0.6)
    result = db.get_all_articles(min_confidence)
    
    # Add feed statistics
    result['feed_sources'] = db.get_feed_stats()
    result['timestamp'] = datetime.now().isoformat()
    result['cached'] = (not check_updates) or feeds_updated == 0
    
    logger.info(f"Response: {len(result['positive_articles'])} positive, " +
                f"{len(result['negative_articles'])} negative, " +
                f"{len(result['neutral_articles'])} neutral")
    
    return result


# Routes

@app.get("/")
async def root():
    """Serve frontend homepage."""
    return FileResponse('frontend/index.html')


@app.get("/api/news", response_model=NewsResponse)
async def get_news():
    """
    Get news articles from database.
    
    Checks all enabled feeds for new articles and indexes them if found.
    Otherwise returns cached data from SQLite.
    """
    try:
        config = load_config()
        data = fetch_and_update_news(config)
        return JSONResponse(content=data)
    except Exception as e:
        logger.exception(f"Error fetching news: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/config")
async def get_config():
    """Get current configuration."""
    try:
        config = load_config()
        return JSONResponse(content=config)
    except Exception as e:
        logger.exception(f"Error loading config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/refresh")
async def refresh_news():
    """
    Force refresh all feeds by clearing cache and re-indexing everything.
    """
    try:
        config = load_config()
        
        logger.info("Force refresh requested - clearing cache...")
        db.clear_all()
        
        data = fetch_and_update_news(config, force_check=True)
        
        return JSONResponse(content={
            'success': True,
            'message': 'All feeds refreshed successfully',
            'data': data
        })
    except Exception as e:
        logger.exception(f"Error during refresh: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                'success': False,
                'message': str(e)
            }
        )


@app.post("/api/index")
async def index_all_feeds():
    """
    Index all feeds - fetch and analyze all articles from enabled feeds.
    This is useful for initial setup or re-indexing everything.
    """
    try:
        config = load_config()
        
        logger.info("Index all feeds requested...")
        # Don't clear data, just fetch and index new articles
        data = fetch_and_update_news(config, force_check=True)
        
        return JSONResponse(content={
            'success': True,
            'message': 'All feeds indexed successfully',
            'data': data
        })
    except Exception as e:
        logger.exception(f"Error during indexing: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                'success': False,
                'message': str(e)
            }
        )


if __name__ == "__main__":
    import uvicorn
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='FastAPI backend for news sentiment analysis')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the API server on (default: 8000)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("Starting FastAPI Backend")
    logger.info("="*70)
    logger.info(f"API Server: http://localhost:{args.port}")
    logger.info(f"Frontend: http://localhost:{args.port}")
    logger.info("Press Ctrl+C to stop the server")
    logger.info("="*70)
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
