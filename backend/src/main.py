#!/usr/bin/env python3
"""
Only Good News API (Appwrite Function Friendly)

Exposes a single endpoint:
  GET /news
"""

import argparse
import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pydantic import BaseModel

from database import NewsDatabase
from parsers import BBCNewsParser, SkyNewsParser
from sentiment_analysis import SentimentAnalyzer

BASE_DIR = Path(__file__).resolve().parent
FUNCTION_DIR = BASE_DIR.parent
CONFIG_PATH = FUNCTION_DIR / "config" / "config.json"
ENV_PATH = FUNCTION_DIR / ".env"

load_dotenv(ENV_PATH)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(FUNCTION_DIR / "api.log"),
    ],
)
logger = logging.getLogger(__name__)

UPDATE_CHECK_INTERVAL_MINUTES = 30
last_update_check: Optional[datetime] = None
update_check_lock = threading.Lock()


def _parse_allowed_origins(raw: str) -> List[str]:
    value = (raw or "").strip()
    if not value:
        return []
    if value.startswith("["):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except json.JSONDecodeError:
            pass
    return [part.strip() for part in value.split(",") if part.strip()]


def _apply_env_overrides(config: Dict) -> Dict:
    cfg = dict(config)

    # Comma-separated or JSON list string.
    if os.getenv("ALLOWED_ORIGINS"):
        cfg["allowed_origins"] = _parse_allowed_origins(os.getenv("ALLOWED_ORIGINS", ""))

    return cfg


def load_config() -> Dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        base = json.load(f)
    return _apply_env_overrides(base)


initial_config = load_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 70)
    logger.info("Only Good News API")
    logger.info("Endpoint: GET /news")
    logger.info("=" * 70)
    yield


app = FastAPI(
    title="Only Good News API",
    description="Positive news feed API",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=initial_config.get("allowed_origins", []),
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'none'; "
        "frame-ancestors 'none'; "
        "base-uri 'none'"
    )
    return response


logger.info("Initializing database...")
db_path = str(FUNCTION_DIR / "data" / "news_archive.db")
db = NewsDatabase(db_path=db_path)

sentiment_model = initial_config.get("sentiment_model")
if sentiment_model:
    logger.info("Loading sentiment model: %s", sentiment_model)
    sentiment_analyzer = SentimentAnalyzer(model_name=sentiment_model)
else:
    sentiment_analyzer = SentimentAnalyzer()


class NewsResponse(BaseModel):
    positive_articles: List[Dict]
    applied_min_confidence: float
    total: int
    timestamp: str
    cached: bool


def contains_banned_keyword(article: Dict, banned_keywords: List[str]) -> bool:
    if not banned_keywords:
        return False
    text = (article.get("title", "") + " " + article.get("description", "")).lower()
    return any(keyword.lower() in text for keyword in banned_keywords)


def normalize_description(description: str) -> str:
    if not isinstance(description, str):
        return ""
    return " ".join(description.strip().lower().split())


def dedupe_articles_by_description(
    articles: List[Dict], seen_descriptions: Optional[set] = None
) -> Tuple[List[Dict], int]:
    if seen_descriptions is None:
        seen_descriptions = set()
    deduped: List[Dict] = []
    removed = 0
    for article in articles:
        norm_desc = normalize_description(article.get("description", ""))
        if not norm_desc:
            deduped.append(article)
            continue
        if norm_desc in seen_descriptions:
            removed += 1
            continue
        seen_descriptions.add(norm_desc)
        deduped.append(article)
    return deduped, removed


def apply_uncertainty(sentiment_info: Dict, unsure_threshold: float) -> Dict:
    confidence = float(sentiment_info.get("confidence", sentiment_info.get("score", 0.0)) or 0.0)
    sentiment = sentiment_info.get("sentiment", "neutral")
    label = (sentiment_info.get("label", "") or "").upper()

    adjusted = dict(sentiment_info)
    adjusted["confidence"] = confidence
    adjusted["score"] = confidence

    if confidence < unsure_threshold:
        adjusted["sentiment"] = "neutral"
        adjusted["label"] = "UNSURE"
        adjusted["is_unsure"] = True
    else:
        adjusted["sentiment"] = sentiment
        adjusted["label"] = label if label else sentiment.upper()
        adjusted["is_unsure"] = False

    return adjusted


def fetch_feed_articles(feed_config: Dict) -> Tuple[str, List[Dict], str]:
    feed_url = feed_config["url"]
    feed_name = feed_config["name"]
    feed_type = feed_config.get("type", "bbc_rss")

    if feed_type == "bbc_rss":
        parser = BBCNewsParser(feed_url)
    elif feed_type == "sky_rss":
        parser = SkyNewsParser(feed_url)
    else:
        return feed_name, [], "unknown_type"

    if not parser.fetch_feed():
        return feed_name, [], "failed"

    articles = parser.to_dict().get("articles", [])
    if not articles:
        return feed_name, [], "empty"
    return feed_name, articles, "ok"


def process_feed(feed_config: Dict, config: Dict, seen_descriptions: Optional[set] = None) -> Dict:
    feed_url = feed_config["url"]
    feed_name = feed_config["name"]
    feed_type = feed_config.get("type", "bbc_rss")

    _, articles, fetch_status = fetch_feed_articles(feed_config)
    if fetch_status != "ok":
        return {"feed_name": feed_name, "new_articles": 0, "status": fetch_status}

    articles, _ = dedupe_articles_by_description(articles, seen_descriptions)

    banned_keywords = config.get("banned_keywords", [])
    if banned_keywords:
        articles = [a for a in articles if not contains_banned_keyword(a, banned_keywords)]

    min_title_words = config.get("min_title_words", 4)
    articles = [a for a in articles if len(a.get("title", "").strip().split()) >= min_title_words]

    current_guids = [a.get("guid", a.get("link", "")) for a in articles]
    has_new, new_guids = db.check_for_new_articles(feed_url, current_guids)
    if not has_new:
        return {"feed_name": feed_name, "new_articles": 0, "status": "cached"}

    db.reindex_feed(feed_url)
    titles = [a.get("title", "") for a in articles]
    descriptions = [a.get("description", "") for a in articles]
    unsure_threshold = config.get("unsure_confidence_threshold", 0.75)
    sentiment_results = [
        apply_uncertainty(result, unsure_threshold)
        for result in sentiment_analyzer.analyze_batch(titles, descriptions)
    ]

    db.bulk_insert_articles(articles, feed_url, feed_type, feed_name, sentiment_results)
    return {"feed_name": feed_name, "new_articles": len(articles), "status": "updated"}


def fetch_and_update_news(config: Dict, force_check: bool = False) -> Dict:
    global last_update_check

    interval_minutes = config.get("update_check_interval_minutes", UPDATE_CHECK_INTERVAL_MINUTES)
    now = datetime.now()
    check_updates = force_check or last_update_check is None or (
        now - last_update_check >= timedelta(minutes=interval_minutes)
    )
    feeds_updated = 0

    if check_updates:
        with update_check_lock:
            now = datetime.now()
            check_updates = force_check or last_update_check is None or (
                now - last_update_check >= timedelta(minutes=interval_minutes)
            )
            if check_updates:
                seen_descriptions = set()
                for feed_config in [f for f in config["feeds"] if f["enabled"]]:
                    stats = process_feed(feed_config, config, seen_descriptions)
                    if stats["status"] == "updated":
                        feeds_updated += 1
                last_update_check = now

    min_confidence = float(config.get("min_confidence", 0.6))
    result = db.get_all_articles(min_confidence)
    result["applied_min_confidence"] = min_confidence
    result["timestamp"] = datetime.now().isoformat()
    result["cached"] = (not check_updates) or feeds_updated == 0
    return result


@app.get("/news", response_model=NewsResponse)
async def get_news():
    try:
        data = fetch_and_update_news(load_config())
        return JSONResponse(content=data)
    except Exception as exc:
        logger.exception("Error fetching news: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Only Good News API server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
