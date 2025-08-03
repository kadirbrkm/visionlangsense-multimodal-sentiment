"""
Multi-Modal Sentiment & Thematic Analysis Package

A comprehensive system for analyzing sentiment, themes, and quality from product images and descriptions.
Built using CLIP-based multi-modal deep learning with explainability features.
"""

__version__ = "1.0.0"
__author__ = "Multi-Modal Sentiment Analysis Team"

from .models.multimodal_sentiment import MultiModalSentimentAnalyzer, SentimentThemeConfig
from .data.dataset import MultiModalProductDataset, DatasetBuilder, create_data_loaders
from .training.trainer import MultiModalTrainer, create_training_config
from .utils.gradcam_explainer import MultiModalGradCAM, TextAttentionVisualizer

__all__ = [
    "MultiModalSentimentAnalyzer",
    "SentimentThemeConfig", 
    "MultiModalProductDataset",
    "DatasetBuilder",
    "create_data_loaders",
    "MultiModalTrainer",
    "create_training_config",
    "MultiModalGradCAM",
    "TextAttentionVisualizer"
]