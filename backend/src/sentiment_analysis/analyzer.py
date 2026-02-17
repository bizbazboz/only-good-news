"""
Sentiment Analyzer using Transformers

Uses a pre-trained transformer model for accurate sentiment analysis.
"""

import logging
import pickle
import re
from typing import Dict, List

from transformers import pipeline

logger = logging.getLogger(__name__)

DEFAULT_NEWS_MODEL = "pavankrishna/news_sentiment_analysis"
DEFAULT_TRANSFORMER_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


class SentimentAnalyzer:
    """
    Sentiment analyzer using transformer-based models.
    
    Uses DistilBERT fine-tuned on SST-2 for efficient and accurate
    sentiment classification.
    """
    
    def __init__(self, model_name: str = DEFAULT_NEWS_MODEL):
        """
        Initialize the sentiment analyzer.
        Args:
            model_name: Name of the pre-trained model to use for sentiment analysis.
        """
        self.model_name = model_name
        self.sentiment_labels = {0: "positive", 1: "negative", 2: "neutral"}
        self.max_length = 100
        self.padding_type = "post"
        self.trunc_type = "post"
        self.backend = "transformers"
        self.np = None
        self.pad_sequences = None

        logger.info(f"Loading sentiment analysis model: {model_name}...")
        if model_name == DEFAULT_NEWS_MODEL:
            self._load_hf_keras_news_model()
        else:
            self._load_transformers_model(model_name)
        logger.info(f"Model loaded successfully (backend={self.backend})")

    def _load_hf_keras_news_model(self):
        """Load the Hugging Face news sentiment Keras model and tokenizer."""
        try:
            import sys
            import numpy as np
            import keras
            from huggingface_hub import snapshot_download
            from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, LSTM
            from tensorflow.keras.models import load_model
            from tensorflow.keras.preprocessing import text as tf_keras_text
            from tensorflow.keras.preprocessing.sequence import pad_sequences

            @keras.saving.register_keras_serializable(package="Compat")
            class LegacyLSTM(LSTM):
                """Compatibility shim for older H5 configs using `time_major`."""
                @classmethod
                def from_config(cls, config):
                    config = dict(config)
                    config.pop("time_major", None)
                    return cls(**config)

            model_dir = snapshot_download(repo_id=self.model_name)
            model_path = f"{model_dir}/sentiment_analysis_model.h5"
            tokenizer_path = f"{model_dir}/tokenizer.pickle"

            self.keras_model = load_model(
                model_path,
                compile=False,
                custom_objects={
                    "LSTM": LegacyLSTM,
                    "LegacyLSTM": LegacyLSTM,
                    "Embedding": Embedding,
                    "Dense": Dense,
                    "Dropout": Dropout,
                    "Bidirectional": Bidirectional,
                },
            )
            # This tokenizer pickle references old module path:
            # `keras.preprocessing.text.Tokenizer`.
            sys.modules.setdefault("keras.preprocessing.text", tf_keras_text)
            with open(tokenizer_path, "rb") as handle:
                self.keras_tokenizer = pickle.load(handle)

            self.np = np
            self.pad_sequences = pad_sequences
            self.backend = "hf_keras"
        except Exception as exc:
            logger.warning(
                "Failed to load '%s' (%s). Falling back to '%s'.",
                self.model_name,
                exc,
                DEFAULT_TRANSFORMER_MODEL,
            )
            self._load_transformers_model(DEFAULT_TRANSFORMER_MODEL)

    def _load_transformers_model(self, model_name: str):
        """Load a transformers sentiment-analysis pipeline."""
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=-1,
        )
        self.backend = "transformers"

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Lightweight normalization for headline text.
        Uses alphanumeric token filtering for consistency.
        """
        if not isinstance(text, str):
            return ""
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        return re.sub(r"\s+", " ", cleaned).strip()

    def _build_text(self, title: str, description: str = "") -> str:
        title = (title or "").strip()
        description = (description or "").strip()
        if description:
            return f"{title}. {description}".strip()
        return title

    def _keras_predict_single(self, text: str) -> Dict:
        processed = self._normalize_text(text)
        sequence = self.keras_tokenizer.texts_to_sequences([processed])
        padded = self.pad_sequences(
            sequence,
            maxlen=self.max_length,
            padding=self.padding_type,
            truncating=self.trunc_type,
        )
        probabilities = self.keras_model.predict(padded, verbose=0)[0]
        top_idx = int(self.np.argmax(probabilities))
        sentiment = self.sentiment_labels.get(top_idx, "neutral")
        confidence = float(probabilities[top_idx])
        return {
            "sentiment": sentiment,
            "score": confidence,
            "confidence": confidence,
            "label": sentiment.upper(),
        }

    def _keras_predict_batch(self, texts: List[str]) -> List[Dict]:
        processed_texts = [self._normalize_text(text) for text in texts]
        sequences = self.keras_tokenizer.texts_to_sequences(processed_texts)
        padded = self.pad_sequences(
            sequences,
            maxlen=self.max_length,
            padding=self.padding_type,
            truncating=self.trunc_type,
        )
        probabilities = self.keras_model.predict(padded, verbose=0)

        predictions = []
        for probs in probabilities:
            top_idx = int(self.np.argmax(probs))
            sentiment = self.sentiment_labels.get(top_idx, "neutral")
            confidence = float(probs[top_idx])
            predictions.append({
                "sentiment": sentiment,
                "score": confidence,
                "confidence": confidence,
                "label": sentiment.upper(),
            })
        return predictions
    
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
        text = self._build_text(title, description)
        if not text or not text.strip():
            return {
                'sentiment': 'neutral',
                'score': 0.0,
                'confidence': 0.0,
                'label': 'NEUTRAL',
                'text': text
            }
        if self.backend == "hf_keras":
            result = self._keras_predict_single(text)
            sentiment = result["sentiment"]
            label = result["label"]
            score = result["score"]
        else:
            result = self.sentiment_pipeline(text[:512])[0]
            label = result['label'].upper()
            score = result['score']
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
        texts = [self._build_text(t, d) for t, d in zip(titles, descriptions)]
        if self.backend == "hf_keras":
            raw_results = self._keras_predict_batch(texts)
        else:
            truncated_texts = [text[:512] for text in texts]
            raw_results = self.sentiment_pipeline(truncated_texts)
        formatted_results = []
        for text, result in zip(texts, raw_results):
            if self.backend == "hf_keras":
                sentiment = result["sentiment"]
                score = result["score"]
                label = result["label"]
            else:
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
