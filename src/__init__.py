"""
Dating Recommendation System
A comprehensive AI-powered dating recommendation engine
"""

from .data_preprocessing import DataPreprocessor
from .embeddings import AdvancedEmbeddings, InterestEmbeddings
from .recommendation_engine import (
    ContentBasedRecommender,
    CollaborativeFilteringRecommender,
    HybridRecommender,
    RealTimeRecommender,
    create_recommendation_system
)
from .evaluation import RecommenderEvaluator
from .visualization import DataVisualizer

__version__ = "1.0.0"
__author__ = "AI Dating System"

__all__ = [
    'DataPreprocessor',
    'AdvancedEmbeddings',
    'InterestEmbeddings',
    'ContentBasedRecommender',
    'CollaborativeFilteringRecommender',
    'HybridRecommender',
    'RealTimeRecommender',
    'create_recommendation_system',
    'RecommenderEvaluator',
    'DataVisualizer'
]
