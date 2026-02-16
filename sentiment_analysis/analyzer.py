"""
Sentiment Analyzer using Transformers

Uses a pre-trained transformer model for accurate sentiment analysis.
"""

import logging
from transformers import pipeline
from typing import Dict, List
import re

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Sentiment analyzer using transformer-based models.
    
    Uses DistilBERT fine-tuned on SST-2 for efficient and accurate
    sentiment classification.
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize the sentiment analyzer with the original DistilBERT SST-2 model.
        Args:
            model_name: Name of the pre-trained model to use for sentiment analysis.
        """
        logger.info(f"Loading sentiment analysis model: {model_name}...")
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=-1  # Use CPU (-1), or specify GPU device number
        )
        logger.info("Model loaded successfully")
    
    def analyze_sentiment(self, title: str, description: str = "") -> Dict:
        """
        Analyze the sentiment of a news title and description.
        Args:
            title: The news title (headline).
            description: The news description (optional).
        Returns:
            Dictionary containing:
                - sentiment: 'positive' or 'negative'
                - score: confidence score (0-1)
                - label: raw label from model
                - confidence: same as score (for compatibility)
        """
        text = title.strip()
        if description:
            text = f"{title.strip()}. {description.strip()}"
        if not text or not text.strip():
            return {
                'sentiment': 'neutral',
                'score': 0.0,
                'confidence': 0.0,
                'label': 'NEUTRAL',
                'text': text
            }
        # Run sentiment analysis
        result = self.sentiment_pipeline(text[:512])[0]  # Truncate to model's max length
        label = result['label'].upper()
        score = result['score']
        # siebert/sentiment-roberta-large-english uses POSITIVE/NEGATIVE only
        if label == 'POSITIVE':
            sentiment = 'positive'
        elif label == 'NEGATIVE':
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        return {
            'sentiment': sentiment,
            'score': score,
            'confidence': score,
            'label': label,
            'text': text
        }
    
    def analyze_batch(self, titles: List[str], descriptions: List[str] = None) -> List[Dict]:
        """
        Analyze sentiment for multiple title+description pairs at once (more efficient).
        Args:
            titles: List of news titles.
            descriptions: List of news descriptions (optional, same length as titles).
        Returns:
            List of sentiment analysis dictionaries.
        """
        if not titles:
            return []
        if descriptions is None:
            descriptions = ["" for _ in titles]
        texts = [f"{t.strip()}. {d.strip()}" if d else t.strip() for t, d in zip(titles, descriptions)]
        truncated_texts = [text[:512] for text in texts]
        results = self.sentiment_pipeline(truncated_texts)
        formatted_results = []
        for text, result in zip(texts, results):
            label = result['label'].upper()
            score = result['score']
            if label == 'POSITIVE':
                sentiment = 'positive'
            elif label == 'NEGATIVE':
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            formatted_results.append({
                'sentiment': sentiment,
                'score': score,
                'confidence': score,
                'label': label,
                'text': text
            })
        return formatted_results
    
    def is_positive(self, title: str, description: str = "", confidence_threshold: float = 0.6) -> bool:
        """
        Quick check if title+description is positive with sufficient confidence.
        Args:
            title: News title.
            description: News description (optional).
            confidence_threshold: Minimum confidence required (0-1).
        Returns:
            True if text is positive with confidence >= threshold.
        """
        result = self.analyze_sentiment(title, description)
        return (result['sentiment'] == 'positive' and 
                result['confidence'] >= confidence_threshold)
