"""
Advanced Embeddings Module for Dating Recommendation System
Implements Sentence-BERT embeddings for semantic bio matching
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
import pickle
import os
from pathlib import Path
import hashlib

# Lazy loading for sentence-transformers
_model = None
_model_name = 'all-MiniLM-L6-v2'


def get_embedding_model(model_name: str = 'all-MiniLM-L6-v2'):
    """
    Lazy load the sentence transformer model
    Uses all-MiniLM-L6-v2 for a good balance of speed and quality
    Produces 384-dimensional embeddings
    """
    global _model, _model_name
    
    if _model is None or _model_name != model_name:
        try:
            from sentence_transformers import SentenceTransformer
            print(f"🔄 Loading Sentence-BERT model: {model_name}...")
            _model = SentenceTransformer(model_name)
            _model_name = model_name
            print(f"✅ Model loaded successfully (embedding dim: {_model.get_sentence_embedding_dimension()})")
        except ImportError:
            print("⚠️ sentence-transformers not installed. Using fallback TF-IDF embeddings.")
            return None
    
    return _model


class AdvancedEmbeddings:
    """
    Advanced text embeddings using Sentence-BERT
    Provides semantic understanding of user bios for better matching
    """
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 cache_dir: str = './cache/embeddings'):
        """
        Initialize the embedding generator
        
        Args:
            model_name: Name of the sentence-transformer model
            cache_dir: Directory to cache computed embeddings
        """
        self.model_name = model_name
        self.model = None
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = None
        self.embedding_dim = 384  # Default for MiniLM
        
    def load_model(self):
        """Load the embedding model"""
        self.model = get_embedding_model(self.model_name)
        if self.model is not None:
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        return self.model is not None
    
    def _get_cache_key(self, texts: List[str]) -> str:
        """Generate a cache key based on text content"""
        content = ''.join(texts[:100])  # Use first 100 texts for key
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Load embeddings from cache if available"""
        cache_path = self.cache_dir / f"{cache_key}.npy"
        if cache_path.exists():
            try:
                print(f"📂 Loading embeddings from cache...")
                return np.load(cache_path)
            except Exception as e:
                print(f"⚠️ Cache load failed: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, embeddings: np.ndarray):
        """Save embeddings to cache"""
        cache_path = self.cache_dir / f"{cache_key}.npy"
        try:
            np.save(cache_path, embeddings)
            print(f"💾 Embeddings cached successfully")
        except Exception as e:
            print(f"⚠️ Cache save failed: {e}")
    
    def encode_texts(self, 
                     texts: List[str], 
                     batch_size: int = 64,
                     show_progress: bool = True,
                     use_cache: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            use_cache: Whether to use caching
            
        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(texts)
            cached = self._load_from_cache(cache_key)
            if cached is not None and len(cached) == len(texts):
                self.embeddings = cached
                return cached
        
        # Load model if needed
        if self.model is None:
            if not self.load_model():
                print("⚠️ Using fallback embedding method...")
                return self._fallback_embeddings(texts)
        
        print(f"🔄 Generating embeddings for {len(texts)} texts...")
        
        # Clean texts
        clean_texts = [str(t) if t else '' for t in texts]
        
        # Encode with progress bar
        try:
            embeddings = self.model.encode(
                clean_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
        except Exception as e:
            print(f"⚠️ Encoding failed: {e}")
            return self._fallback_embeddings(texts)
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings = embeddings / norms
        
        # Cache the embeddings
        if use_cache:
            self._save_to_cache(cache_key, embeddings)
        
        self.embeddings = embeddings
        print(f"✅ Generated embeddings: shape {embeddings.shape}")
        
        return embeddings
    
    def _fallback_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Fallback to TF-IDF based embeddings when sentence-transformers unavailable
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        
        print("📊 Using TF-IDF + SVD fallback embeddings...")
        
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Reduce dimensionality to match BERT-like embeddings
        n_components = min(384, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        embeddings = svd.fit_transform(tfidf_matrix)
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms
        
        self.embeddings = embeddings
        self.embedding_dim = embeddings.shape[1]
        
        print(f"✅ Fallback embeddings: shape {embeddings.shape}")
        
        return embeddings
    
    def get_similar_by_embedding(self, 
                                  query_embedding: np.ndarray, 
                                  top_k: int = 10,
                                  exclude_indices: List[int] = None) -> List[Tuple[int, float]]:
        """
        Find most similar items by embedding
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            exclude_indices: Indices to exclude from results
            
        Returns:
            List of (index, similarity_score) tuples
        """
        if self.embeddings is None:
            raise ValueError("No embeddings available. Call encode_texts first.")
        
        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        
        # Compute cosine similarities
        similarities = np.dot(self.embeddings, query_norm)
        
        # Get top indices
        if exclude_indices:
            similarities[exclude_indices] = -1
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(idx, similarities[idx]) for idx in top_indices]
    
    def compute_similarity_matrix(self, 
                                   batch_size: int = 1000) -> np.ndarray:
        """
        Compute pairwise similarity matrix for all embeddings
        Memory-efficient batched computation
        """
        if self.embeddings is None:
            raise ValueError("No embeddings available. Call encode_texts first.")
        
        n = len(self.embeddings)
        print(f"🔄 Computing {n}x{n} similarity matrix...")
        
        # For smaller datasets, compute directly
        if n <= batch_size:
            similarity_matrix = np.dot(self.embeddings, self.embeddings.T)
        else:
            # Batched computation for memory efficiency
            similarity_matrix = np.zeros((n, n), dtype=np.float32)
            
            for i in range(0, n, batch_size):
                end_i = min(i + batch_size, n)
                for j in range(0, n, batch_size):
                    end_j = min(j + batch_size, n)
                    similarity_matrix[i:end_i, j:end_j] = np.dot(
                        self.embeddings[i:end_i],
                        self.embeddings[j:end_j].T
                    )
        
        print(f"✅ Similarity matrix computed: shape {similarity_matrix.shape}")
        
        return similarity_matrix
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text for real-time recommendation"""
        if self.model is None:
            if not self.load_model():
                # Fallback: return zero vector
                return np.zeros(self.embedding_dim)
        
        embedding = self.model.encode([text], convert_to_numpy=True)[0]
        return embedding / (np.linalg.norm(embedding) + 1e-10)


class InterestEmbeddings:
    """
    Create embeddings for user interests using multi-hot encoding
    and optionally semantic embeddings for interest names
    """
    
    def __init__(self):
        self.interest_categories = [
            'music', 'sports', 'travel', 'food', 'movies',
            'reading', 'gaming', 'art', 'outdoors', 'tech',
            'fitness', 'photography', 'dancing', 'cooking', 'pets'
        ]
        self.interest_keywords = {
            'music': ['music', 'concert', 'band', 'guitar', 'piano', 'jazz', 'rock', 'hip hop', 'electronic', 'singing'],
            'sports': ['sports', 'gym', 'fitness', 'running', 'hiking', 'yoga', 'basketball', 'football', 'soccer', 'tennis'],
            'travel': ['travel', 'adventure', 'explore', 'countries', 'backpack', 'trip', 'vacation', 'wanderlust'],
            'food': ['food', 'cooking', 'restaurant', 'cuisine', 'chef', 'baking', 'wine', 'foodie', 'eating'],
            'movies': ['movie', 'film', 'cinema', 'netflix', 'documentary', 'theater', 'watching'],
            'reading': ['book', 'reading', 'literature', 'novel', 'author', 'library', 'stories'],
            'gaming': ['game', 'gaming', 'video game', 'playstation', 'xbox', 'nintendo', 'pc gaming', 'esports'],
            'art': ['art', 'museum', 'painting', 'photography', 'creative', 'drawing', 'design', 'gallery'],
            'outdoors': ['outdoor', 'nature', 'camping', 'beach', 'mountain', 'park', 'hiking', 'fishing'],
            'tech': ['technology', 'programming', 'coding', 'software', 'computer', 'tech', 'startup', 'ai'],
            'fitness': ['fitness', 'workout', 'exercise', 'crossfit', 'weights', 'cardio', 'health'],
            'photography': ['photography', 'photo', 'camera', 'instagram', 'portrait', 'landscape'],
            'dancing': ['dancing', 'dance', 'salsa', 'ballet', 'club', 'choreography'],
            'cooking': ['cooking', 'baking', 'recipes', 'kitchen', 'homemade', 'culinary'],
            'pets': ['pet', 'dog', 'cat', 'animal', 'puppy', 'kitten', 'pets']
        }
        
    def extract_interests(self, texts: List[str]) -> np.ndarray:
        """
        Extract interest vectors from text
        
        Returns:
            numpy array of shape (n_texts, n_interests)
        """
        n = len(texts)
        interest_matrix = np.zeros((n, len(self.interest_categories)))
        
        for i, text in enumerate(texts):
            if not text:
                continue
            text_lower = str(text).lower()
            
            for j, category in enumerate(self.interest_categories):
                keywords = self.interest_keywords.get(category, [category])
                for keyword in keywords:
                    if keyword in text_lower:
                        interest_matrix[i, j] = 1
                        break
        
        return interest_matrix
    
    def compute_jaccard_similarity(self, 
                                    interests1: np.ndarray, 
                                    interests2: np.ndarray) -> float:
        """Compute Jaccard similarity between two interest vectors"""
        intersection = np.sum(np.minimum(interests1, interests2))
        union = np.sum(np.maximum(interests1, interests2))
        
        if union == 0:
            return 0.0
        
        return intersection / union


def create_combined_embeddings(bio_embeddings: np.ndarray,
                                interest_embeddings: np.ndarray,
                                feature_embeddings: np.ndarray,
                                weights: dict = None) -> np.ndarray:
    """
    Combine different embedding types with configurable weights
    
    Args:
        bio_embeddings: Sentence-BERT embeddings for bios
        interest_embeddings: Multi-hot interest vectors
        feature_embeddings: Numerical feature vectors
        weights: Dictionary of weights for each embedding type
        
    Returns:
        Combined embedding matrix
    """
    if weights is None:
        weights = {
            'bio': 0.5,
            'interests': 0.3,
            'features': 0.2
        }
    
    embeddings_list = []
    
    # Scale and weight each embedding type
    if bio_embeddings is not None:
        bio_scaled = bio_embeddings * weights['bio']
        embeddings_list.append(bio_scaled)
    
    if interest_embeddings is not None:
        interest_scaled = interest_embeddings * weights['interests']
        embeddings_list.append(interest_scaled)
    
    if feature_embeddings is not None:
        feature_scaled = feature_embeddings * weights['features']
        embeddings_list.append(feature_scaled)
    
    if not embeddings_list:
        raise ValueError("At least one embedding type must be provided")
    
    # Concatenate all embeddings
    combined = np.hstack(embeddings_list)
    
    # Normalize the combined embeddings
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms[norms == 0] = 1
    combined = combined / norms
    
    return combined
