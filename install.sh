#!/bin/bash
# Installation script for Only Good News
# Installs CPU-only PyTorch to avoid large NVIDIA/CUDA dependencies

set -e  # Exit on error

echo "=================================================="
echo "  Installing Only Good News"
echo "=================================================="
echo ""

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "❌ Python not found: $PYTHON_BIN"
    exit 1
fi

echo "Using Python: $($PYTHON_BIN --version)"
echo "Upgrading pip..."
"$PYTHON_BIN" -m pip install --upgrade pip

# Check if we want CPU-only or GPU support.
# --gpu installs from requirements.txt (default package indexes).
# default installs torch from PyTorch CPU index.
if [ "$1" = "--gpu" ]; then
    echo "Installing with GPU support (default indexes)..."
    "$PYTHON_BIN" -m pip install -r requirements.txt
else
    echo "Installing with CPU-only PyTorch (no NVIDIA dependencies)..."
    echo ""
    
    # Install all dependencies except torch from requirements.txt
    echo "Installing application dependencies..."
    "$PYTHON_BIN" -m pip install \
      "fastapi>=0.129.0" \
      "uvicorn[standard]>=0.40.0" \
      "python-multipart>=0.0.22" \
      "transformers>=5.1.0"
    
    # Install PyTorch CPU-only version
    echo ""
    echo "Installing PyTorch (CPU-only)..."
    "$PYTHON_BIN" -m pip install "torch>=2.10.0" --index-url https://download.pytorch.org/whl/cpu
fi

echo ""
echo "=================================================="
echo "  Downloading Sentiment Analysis Model"
echo "=================================================="
echo ""
echo "Downloading DistilBERT model (~250MB)..."
echo "This is a one-time download."
echo ""

# Download the model by importing and initializing it
"$PYTHON_BIN" << 'PYTHON'
from transformers import pipeline
import sys

try:
    print("Loading sentiment analysis pipeline...")
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1  # CPU
    )
    print("✅ Model downloaded and cached successfully!")
    
    # Test it
    result = sentiment_pipeline("This is great news!")[0]
    print(f"✅ Model test passed: '{result['label']}' with {result['score']:.2%} confidence")
    
except Exception as e:
    print(f"❌ Error downloading model: {e}")
    sys.exit(1)
PYTHON

echo ""
echo "=================================================="
echo "  ✅ Installation Complete!"
echo "=================================================="
echo ""
echo "To start the application:"
echo ""
echo "1. Start the FastAPI backend:"
echo "   python3 api.py --port 8000"
echo ""
echo "2. Open your browser:"
echo "   http://localhost:8000"
