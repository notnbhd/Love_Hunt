"""
Recommendation Engine for Dating Recommendation System
Implements Content-Based, Collaborative Filtering, and Hybrid approaches
With real-time recommendation capability
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import time
from datetime import datetime
from collections import defaultdict
import pickle
import os


class ContentBasedRecommender:
    """
    Content-Based Filtering using user profile features and bio embeddings
    
    Features used:
    - TF-IDF bio vectors (weight: 0.3)
    - Interest overlap (Jaccard similarity, weight: 0.25)
    - Demographic compatibility (age, weight: 0.15)
    - Lifestyle compatibility (weight: 0.15)
    - Location proximity (weight: 0.15)
    """
    
    def __init__(self, 
                 bio_weight: float = 0.35,
                 interest_weight: float = 0.25,
                 demographic_weight: float = 0.15,
                 lifestyle_weight: float = 0.10,
                 location_weight: float = 0.15):
        
        self.weights = {
            'bio': bio_weight,
            'interest': interest_weight,
            'demographic': demographic_weight,
            'lifestyle': lifestyle_weight,
            'location': location_weight
        }
        
        self.user_features = None
        self.bio_embeddings = None
        self.interest_matrix = None
        self.similarity_matrix = None
        self.df = None
        self.user_id_to_idx = {}
        self.idx_to_user_id = {}
        
    def fit(self, 
            df: pd.DataFrame,
            bio_embeddings: np.ndarray = None,
            interest_matrix: np.ndarray = None):
        """
        Fit the content-based recommender
        
        Args:
            df: DataFrame with user profiles
            bio_embeddings: Pre-computed bio embeddings
            interest_matrix: Pre-computed interest vectors
        """
        print("🔄 Fitting Content-Based Recommender...")
        start_time = time.time()
        
        self.df = df.copy()
        self.bio_embeddings = bio_embeddings
        self.interest_matrix = interest_matrix
        
        # Create user ID mappings
        if 'user_id' in df.columns:
            self.user_id_to_idx = {uid: idx for idx, uid in enumerate(df['user_id'])}
            self.idx_to_user_id = {idx: uid for uid, idx in self.user_id_to_idx.items()}
        else:
            self.user_id_to_idx = {idx: idx for idx in range(len(df))}
            self.idx_to_user_id = self.user_id_to_idx.copy()
        
        # Pre-compute feature vectors for on-demand similarity (memory efficient)
        self._precompute_features()
        
        elapsed = time.time() - start_time
        print(f"✅ Content-Based Recommender fitted in {elapsed:.2f}s")
    
    def _precompute_features(self):
        """Pre-compute feature vectors for efficient on-demand similarity computation"""
        n = len(self.df)
        
        # Pre-compute lifestyle encodings
        lifestyle_cols = [col for col in ['drinks', 'smokes', 'drugs', 'diet'] 
                          if col in self.df.columns]
        if lifestyle_cols:
            lifestyle_data = self.df[lifestyle_cols].fillna('unknown')
            self._lifestyle_encoded = np.zeros((n, len(lifestyle_cols)), dtype=np.int32)
            for idx, col in enumerate(lifestyle_cols):
                self._lifestyle_encoded[:, idx] = pd.Categorical(lifestyle_data[col]).codes
            self._n_lifestyle_cols = len(lifestyle_cols)
        else:
            self._lifestyle_encoded = None
            self._n_lifestyle_cols = 0
        
        # Pre-compute location encodings
        if 'location' in self.df.columns:
            locations = self.df['location'].fillna('unknown').values
            cities = []
            states = []
            for loc in locations:
                loc_str = str(loc).lower()
                parts = loc_str.split(',')
                city = parts[0].strip() if parts else ''
                state = parts[-1].strip() if len(parts) > 1 else ''
                cities.append(city)
                states.append(state)
            self._city_codes = pd.Categorical(cities).codes
            self._state_codes = pd.Categorical(states).codes
            self._valid_city = np.array([c != '' and c != 'unknown' for c in cities])
            self._valid_state = np.array([s != '' and s != 'unknown' for s in states])
        else:
            self._city_codes = None
            self._state_codes = None
        
        # Pre-compute ages
        if 'age' in self.df.columns:
            self._ages = self.df['age'].values.astype(np.float32)
        else:
            self._ages = None
        
        # Pre-compute interest row sums for Jaccard
        if self.interest_matrix is not None:
            self._interest_row_sums = np.sum(self.interest_matrix.astype(np.float32), axis=1)
        else:
            self._interest_row_sums = None
        
        print("   Pre-computed feature vectors for on-demand similarity")
        
    def _compute_user_similarity(self, user_idx: int) -> np.ndarray:
        """Compute similarity scores for a single user against all others (memory efficient)"""
        n = len(self.df)
        combined_sim = np.zeros(n, dtype=np.float32)
        
        # 1. Bio similarity (cosine similarity on embeddings)
        if self.bio_embeddings is not None:
            user_embedding = self.bio_embeddings[user_idx:user_idx+1]
            bio_sim = cosine_similarity(user_embedding, self.bio_embeddings)[0]
            combined_sim += bio_sim.astype(np.float32) * self.weights['bio']
        
        # 2. Interest similarity (Jaccard-like)
        if self.interest_matrix is not None:
            interest_sim = self._compute_user_interest_similarity(user_idx)
            combined_sim += interest_sim * self.weights['interest']
        
        # 3. Demographic similarity
        demo_sim = self._compute_user_demographic_similarity(user_idx)
        combined_sim += demo_sim * self.weights['demographic']
        
        # 4. Lifestyle similarity
        lifestyle_sim = self._compute_user_lifestyle_similarity(user_idx)
        combined_sim += lifestyle_sim * self.weights['lifestyle']
        
        # 5. Location similarity
        location_sim = self._compute_user_location_similarity(user_idx)
        combined_sim += location_sim * self.weights['location']
        
        # Set self-similarity to 0
        combined_sim[user_idx] = 0
        
        return combined_sim
    
    def _compute_user_interest_similarity(self, user_idx: int) -> np.ndarray:
        """Compute Jaccard-like interest similarity for one user vs all"""
        user_interests = self.interest_matrix[user_idx].astype(np.float32)
        user_sum = self._interest_row_sums[user_idx]
        
        # Intersection with all users
        intersection = self.interest_matrix.astype(np.float32) @ user_interests
        
        # Union = sum_user + sum_others - intersection
        union = user_sum + self._interest_row_sums - intersection
        
        with np.errstate(divide='ignore', invalid='ignore'):
            sim = np.where(union > 0, intersection / union, 0)
        
        return sim.astype(np.float32)
    
    def _compute_user_demographic_similarity(self, user_idx: int) -> np.ndarray:
        """Compute demographic similarity for one user vs all"""
        if self._ages is None:
            return np.zeros(len(self.df), dtype=np.float32)
        
        user_age = self._ages[user_idx]
        age_diff = np.abs(self._ages - user_age)
        sim = np.maximum(0, 1 - age_diff / 30)
        
        return sim.astype(np.float32)
    
    def _compute_user_lifestyle_similarity(self, user_idx: int) -> np.ndarray:
        """Compute lifestyle similarity for one user vs all"""
        if self._lifestyle_encoded is None:
            return np.zeros(len(self.df), dtype=np.float32)
        
        user_lifestyle = self._lifestyle_encoded[user_idx]
        # Count matches across all lifestyle columns
        matches = np.sum(self._lifestyle_encoded == user_lifestyle, axis=1)
        sim = matches / self._n_lifestyle_cols
        
        return sim.astype(np.float32)
    
    def _compute_user_location_similarity(self, user_idx: int) -> np.ndarray:
        """Compute location similarity for one user vs all"""
        if self._city_codes is None:
            return np.zeros(len(self.df), dtype=np.float32)
        
        user_city = self._city_codes[user_idx]
        user_state = self._state_codes[user_idx]
        user_valid_city = self._valid_city[user_idx]
        user_valid_state = self._valid_state[user_idx]
        
        same_city = (self._city_codes == user_city) & self._valid_city & user_valid_city
        same_state = (self._state_codes == user_state) & self._valid_state & user_valid_state
        
        # Same city = 1.0, same state (but different city) = 0.5
        sim = same_city.astype(np.float32) * 1.0 + (~same_city & same_state).astype(np.float32) * 0.5
        
        return sim
    
    def recommend(self, 
                  user_id: int, 
                  top_k: int = 10,
                  filters: Dict = None) -> List[Tuple[int, float, Dict]]:
        """
        Get recommendations for a user
        
        Args:
            user_id: Target user ID
            top_k: Number of recommendations
            filters: Optional filters (age_range, location, etc.)
            
        Returns:
            List of (user_id, score, details) tuples
        """
        if user_id not in self.user_id_to_idx:
            return []
        
        idx = self.user_id_to_idx[user_id]
        
        # Compute similarity scores on-demand (memory efficient)
        scores = self._compute_user_similarity(idx).copy()
        
        # Apply filters
        if filters:
            scores = self._apply_filters(scores, filters, idx)
        
        # Apply gender/orientation filter
        scores = self._apply_orientation_filter(scores, idx)
        
        # Get top-k recommendations
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        recommendations = []
        for rec_idx in top_indices:
            if scores[rec_idx] > 0:
                rec_user_id = self.idx_to_user_id[rec_idx]
                details = self._get_match_details(idx, rec_idx)
                recommendations.append((rec_user_id, float(scores[rec_idx]), details))
        
        return recommendations
    
    def _apply_filters(self, scores: np.ndarray, filters: Dict, user_idx: int) -> np.ndarray:
        """Apply user-specified filters"""
        if 'age_range' in filters:
            min_age, max_age = filters['age_range']
            ages = self.df['age'].values
            mask = (ages < min_age) | (ages > max_age)
            scores[mask] = 0
        
        if 'location' in filters and 'location' in self.df.columns:
            target_loc = filters['location'].lower()
            locations = self.df['location'].fillna('').str.lower()
            mask = ~locations.str.contains(target_loc)
            scores[mask] = 0
        
        return scores
    
    def _apply_orientation_filter(self, scores: np.ndarray, user_idx: int) -> np.ndarray:
        """Filter based on sexual orientation compatibility (vectorized)"""
        if 'sex' not in self.df.columns or 'orientation' not in self.df.columns:
            return scores
        
        user_sex = self.df.iloc[user_idx]['sex']
        user_orientation = str(self.df.iloc[user_idx]['orientation']).lower()
        
        candidate_sexes = self.df['sex'].values
        candidate_orientations = self.df['orientation'].fillna('').str.lower().values
        
        # Create compatibility mask (True = compatible)
        compatible = np.ones(len(scores), dtype=bool)
        
        # Determine what sex the user is looking for
        if user_orientation == 'straight':
            # Straight: looking for opposite sex
            if user_sex == 'm':
                user_seeks = 'f'
            else:
                user_seeks = 'm'
        elif user_orientation == 'gay':
            # Gay: looking for same sex
            user_seeks = user_sex
        elif user_orientation == 'bisexual':
            # Bisexual: open to both
            user_seeks = None  # Will match any
        else:
            user_seeks = None  # Default: open to all
        
        for i in range(len(scores)):
            if scores[i] == 0:
                continue
            
            cand_sex = candidate_sexes[i]
            cand_orientation = candidate_orientations[i]
            
            # Check 1: Does candidate's sex match what user is seeking?
            if user_seeks is not None and cand_sex != user_seeks:
                compatible[i] = False
                continue
            
            # Check 2: Is user's sex compatible with what candidate is seeking?
            if cand_orientation == 'straight':
                # Candidate is straight: they want opposite sex
                if cand_sex == 'm' and user_sex != 'f':
                    compatible[i] = False
                elif cand_sex == 'f' and user_sex != 'm':
                    compatible[i] = False
            elif cand_orientation == 'gay':
                # Candidate is gay: they want same sex
                if cand_sex != user_sex:
                    compatible[i] = False
            # Bisexual candidates are compatible with anyone
        
        # Apply the mask
        scores[~compatible] = 0
        
        return scores
    
    def _check_orientation_compatibility(self, 
                                          sex1: str, orientation1: str,
                                          sex2: str, orientation2: str) -> bool:
        """Check if two users are compatible based on orientation"""
        orientation1 = str(orientation1).lower()
        orientation2 = str(orientation2).lower()
        
        # Determine what sex1 is seeking
        if orientation1 == 'straight':
            sex1_seeks = 'f' if sex1 == 'm' else 'm'
        elif orientation1 == 'gay':
            sex1_seeks = sex1  # Same sex
        else:  # bisexual or other
            sex1_seeks = None  # Open to all
        
        # Determine what sex2 is seeking
        if orientation2 == 'straight':
            sex2_seeks = 'f' if sex2 == 'm' else 'm'
        elif orientation2 == 'gay':
            sex2_seeks = sex2  # Same sex
        else:  # bisexual or other
            sex2_seeks = None  # Open to all
        
        # Check mutual compatibility
        # sex1 must match what sex2 is seeking (or sex2 is open)
        if sex2_seeks is not None and sex1 != sex2_seeks:
            return False
        
        # sex2 must match what sex1 is seeking (or sex1 is open)
        if sex1_seeks is not None and sex2 != sex1_seeks:
            return False
        
        return True
    
    def _get_match_details(self, user_idx: int, match_idx: int) -> Dict:
        """Get detailed breakdown of why users match"""
        details = {
            'bio_score': 0,
            'interest_score': 0,
            'demographic_score': 0,
            'lifestyle_score': 0,
            'location_score': 0,
            'common_interests': []
        }
        
        if self.bio_embeddings is not None:
            bio_sim = cosine_similarity(
                [self.bio_embeddings[user_idx]], 
                [self.bio_embeddings[match_idx]]
            )[0][0]
            details['bio_score'] = round(float(bio_sim) * 100, 1)
        
        # Get common interests
        if self.interest_matrix is not None:
            user_interests = self.interest_matrix[user_idx]
            match_interests = self.interest_matrix[match_idx]
            common = np.where((user_interests > 0) & (match_interests > 0))[0]
            interest_names = ['music', 'sports', 'travel', 'food', 'movies',
                            'reading', 'gaming', 'art', 'outdoors', 'tech']
            details['common_interests'] = [interest_names[i] for i in common if i < len(interest_names)]
            
            intersection = np.sum(np.minimum(user_interests, match_interests))
            union = np.sum(np.maximum(user_interests, match_interests))
            if union > 0:
                details['interest_score'] = round(float(intersection / union) * 100, 1)
        
        return details


class CollaborativeFilteringRecommender:
    """
    Collaborative Filtering using Matrix Factorization (SVD) + User-Based KNN
    Combines latent factor model with neighborhood-based approach for better coverage
    """
    
    def __init__(self, n_factors: int = 100, n_neighbors: int = 50, 
                 n_epochs: int = 20, lr: float = 0.01, reg: float = 0.02):
        self.n_factors = n_factors
        self.n_neighbors = n_neighbors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_mean = 0
        
        # For KNN-based predictions
        self.interaction_matrix = None
        self.user_similarity = None
        
        self.user_id_to_idx = {}
        self.idx_to_user_id = {}
        
    def fit(self, interaction_matrix, user_ids: List[int] = None):
        """
        Fit the collaborative filtering model using Truncated SVD (fast) + compute user similarities
        
        Args:
            interaction_matrix: User-User interaction matrix (can be sparse or dense)
            user_ids: List of user IDs
        """
        print("🔄 Fitting Collaborative Filtering Recommender...")
        start_time = time.time()
        
        from scipy.sparse import issparse, csr_matrix
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Convert to sparse if not already
        if not issparse(interaction_matrix):
            interaction_matrix = csr_matrix(interaction_matrix.astype(np.float32))
        else:
            interaction_matrix = csr_matrix(interaction_matrix.astype(np.float32))
        
        self.interaction_matrix = interaction_matrix
        n_users = interaction_matrix.shape[0]
        
        # Create ID mappings
        if user_ids is not None:
            self.user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
            self.idx_to_user_id = {idx: uid for uid, idx in self.user_id_to_idx.items()}
        else:
            self.user_id_to_idx = {i: i for i in range(n_users)}
            self.idx_to_user_id = self.user_id_to_idx.copy()
        
        # Compute mean and biases from sparse matrix
        nnz_per_row = np.diff(interaction_matrix.indptr)
        row_sums = np.array(interaction_matrix.sum(axis=1)).flatten()
        
        nnz_per_col = np.array((interaction_matrix != 0).sum(axis=0)).flatten()
        col_sums = np.array(interaction_matrix.sum(axis=0)).flatten()
        
        total_nnz = interaction_matrix.nnz
        self.global_mean = interaction_matrix.sum() / total_nnz if total_nnz > 0 else 0
        
        # Compute biases
        self.user_bias = np.where(nnz_per_row > 0, row_sums / nnz_per_row - self.global_mean, 0)
        self.item_bias = np.where(nnz_per_col > 0, col_sums / nnz_per_col - self.global_mean, 0)
        
        # Use Truncated SVD for fast factorization
        from sklearn.decomposition import TruncatedSVD
        
        n_components = min(self.n_factors, n_users - 1, 100)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        
        print(f"   Running Truncated SVD with {n_components} factors...")
        self.user_factors = svd.fit_transform(interaction_matrix)
        self.item_factors = svd.components_.T * np.sqrt(svd.singular_values_)
        self.user_factors = self.user_factors / (np.sqrt(svd.singular_values_) + 1e-10)
        
        # Normalize factors
        self.user_factors = self.user_factors.astype(np.float64)
        self.item_factors = self.item_factors.astype(np.float64)
        
        # Compute user-user similarity for KNN (use SVD factors for efficiency)
        print(f"   Computing user similarities for KNN ({self.n_neighbors} neighbors)...")
        self.user_similarity = cosine_similarity(self.user_factors)
        np.fill_diagonal(self.user_similarity, 0)  # No self-similarity
        
        print(f"   Explained variance ratio: {svd.explained_variance_ratio_.sum():.2%}")
        
        elapsed = time.time() - start_time
        print(f"✅ Collaborative Filtering fitted in {elapsed:.2f}s")
    
    def _predict_single(self, user_idx: int, item_idx: int) -> float:
        """Predict interaction score for a user-item pair"""
        pred = self.global_mean
        pred += self.user_bias[user_idx]
        pred += self.item_bias[item_idx]
        pred += np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        return pred
    
    def predict(self, user_id: int) -> np.ndarray:
        """Predict scores for all items using hybrid SVD + KNN approach"""
        if user_id not in self.user_id_to_idx:
            return np.zeros(len(self.item_factors))
        
        user_idx = self.user_id_to_idx[user_id]
        
        # SVD-based predictions
        svd_predictions = (
            self.global_mean + 
            self.user_bias[user_idx] + 
            self.item_bias + 
            self.user_factors[user_idx] @ self.item_factors.T
        )
        
        # KNN-based predictions (weighted average of similar users' interactions)
        similarities = self.user_similarity[user_idx]
        top_k_indices = np.argsort(similarities)[::-1][:self.n_neighbors]
        top_k_sims = similarities[top_k_indices]
        
        # Only use neighbors with positive similarity
        valid_mask = top_k_sims > 0
        if valid_mask.sum() > 0:
            top_k_indices = top_k_indices[valid_mask]
            top_k_sims = top_k_sims[valid_mask]
            
            # Get interactions from similar users
            neighbor_interactions = self.interaction_matrix[top_k_indices].toarray()
            
            # Weighted average
            weights = top_k_sims / (top_k_sims.sum() + 1e-10)
            knn_predictions = np.average(neighbor_interactions, axis=0, weights=weights)
        else:
            knn_predictions = np.zeros_like(svd_predictions)
        
        # Combine SVD and KNN (60% SVD, 40% KNN)
        combined = 0.6 * svd_predictions + 0.4 * knn_predictions
        
        return combined
    
    def recommend(self, user_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """Get top-k recommendations for a user"""
        predictions = self.predict(user_id)
        
        # Exclude self
        if user_id in self.user_id_to_idx:
            predictions[self.user_id_to_idx[user_id]] = -np.inf
        
        top_indices = np.argsort(predictions)[::-1][:top_k]
        
        # Normalize scores to 0-1 range for consistency
        valid_preds = predictions[predictions > -np.inf]
        if len(valid_preds) > 0:
            pred_min = valid_preds.min()
            pred_max = valid_preds.max()
            if pred_max > pred_min:
                predictions = (predictions - pred_min) / (pred_max - pred_min)
        
        return [(self.idx_to_user_id[idx], float(predictions[idx])) 
                for idx in top_indices if predictions[idx] > -np.inf]


class HybridRecommender:
    """
    Hybrid Recommendation System combining Content-Based and Collaborative Filtering
    
    Formula: Final_Score = α × Content_Score + β × Collab_Score
    """
    
    def __init__(self, 
                 content_weight: float = 0.6,
                 collab_weight: float = 0.4):
        self.alpha = content_weight
        self.beta = collab_weight
        
        self.content_recommender = ContentBasedRecommender()
        self.collab_recommender = CollaborativeFilteringRecommender(
            n_factors=100, 
            n_neighbors=50
        )
        
        self.df = None
        self.is_fitted = False
        
    def fit(self, 
            df: pd.DataFrame,
            bio_embeddings: np.ndarray = None,
            interest_matrix: np.ndarray = None,
            interaction_matrix: np.ndarray = None):
        """
        Fit both recommenders
        """
        print("\n" + "="*60)
        print("🚀 FITTING HYBRID RECOMMENDER")
        print("="*60)
        
        self.df = df.copy()
        
        # Fit content-based recommender
        self.content_recommender.fit(df, bio_embeddings, interest_matrix)
        
        # Create interaction matrix if not provided (simulate based on similarity)
        if interaction_matrix is None:
            print("📊 Generating simulated interaction matrix...")
            interaction_matrix = self._simulate_interactions()
        
        # Fit collaborative filtering
        user_ids = df['user_id'].tolist() if 'user_id' in df.columns else list(range(len(df)))
        self.collab_recommender.fit(interaction_matrix, user_ids)
        
        self.is_fitted = True
        
        print("="*60)
        print("✅ HYBRID RECOMMENDER READY")
        print("="*60 + "\n")
    
    def _simulate_interactions(self, density: float = 0.05):
        """
        Simulate user interactions based on content similarity with realistic patterns
        Creates denser interaction matrix for better CF learning
        
        Args:
            density: Approximate fraction of non-zero entries (0.05 = 5%)
            
        Returns sparse matrix for memory efficiency
        """
        n = len(self.df)
        print(f"   Creating interaction matrix for {n} users (target density: {density:.1%})...")
        
        from scipy.sparse import lil_matrix
        interactions = lil_matrix((n, n), dtype=np.float32)
        
        np.random.seed(42)
        
        # Target number of interactions per user (on average)
        interactions_per_user = max(10, int(n * density))
        
        # For each user, create interactions with similar users
        for i in range(n):
            # Compute similarity with all other users
            sim_scores = self.content_recommender._compute_user_similarity(i)
            sim_scores[i] = 0  # No self-interaction
            
            # Create interactions based on similarity (higher sim = higher probability)
            # Use softmax-like probability
            exp_scores = np.exp(sim_scores * 5)  # Temperature scaling
            probs = exp_scores / exp_scores.sum()
            
            # Sample users to interact with
            n_interact = min(interactions_per_user, n - 1)
            interact_indices = np.random.choice(
                n, size=n_interact, replace=False, p=probs
            )
            
            # Generate interaction scores with noise
            for j in interact_indices:
                if i != j:
                    # Base score from similarity + demographic match + randomness
                    base_sim = sim_scores[j]
                    
                    # Add some randomness to simulate real preferences
                    noise = np.random.normal(0, 0.15)
                    
                    # Score: scale to 1-5 range
                    score = np.clip(base_sim * 4 + 1 + noise, 1, 5)
                    
                    interactions[i, j] = score
                    
                    # Make some interactions reciprocal (mutual interest)
                    if np.random.random() < 0.3 + base_sim * 0.4:
                        reciprocal_score = np.clip(score + np.random.normal(0, 0.3), 1, 5)
                        interactions[j, i] = reciprocal_score
            
            if (i + 1) % 500 == 0:
                print(f"      Processed {i+1}/{n} users...")
        
        result = interactions.tocsr()
        actual_density = result.nnz / (n * n)
        print(f"   Created {result.nnz:,} interactions (actual density: {actual_density:.2%})")
        return result
    
    def _compute_pair_similarity(self, idx_i: int, idx_j: int) -> float:
        """Compute similarity between two specific users"""
        sim = 0.0
        
        # Bio similarity
        if self.content_recommender.bio_embeddings is not None:
            emb_i = self.content_recommender.bio_embeddings[idx_i]
            emb_j = self.content_recommender.bio_embeddings[idx_j]
            bio_sim = np.dot(emb_i, emb_j)
            sim += bio_sim * self.content_recommender.weights['bio']
        
        # Interest similarity
        if self.content_recommender.interest_matrix is not None:
            int_i = self.content_recommender.interest_matrix[idx_i]
            int_j = self.content_recommender.interest_matrix[idx_j]
            intersection = np.sum(np.minimum(int_i, int_j))
            union = np.sum(np.maximum(int_i, int_j))
            if union > 0:
                sim += (intersection / union) * self.content_recommender.weights['interest']
        
        # Age similarity
        if self.content_recommender._ages is not None:
            age_diff = abs(self.content_recommender._ages[idx_i] - self.content_recommender._ages[idx_j])
            sim += max(0, 1 - age_diff / 30) * self.content_recommender.weights['demographic']
        
        return sim
    
    def recommend(self, 
                  user_id: int, 
                  top_k: int = 10,
                  filters: Dict = None,
                  return_details: bool = True) -> List[Dict]:
        """
        Get hybrid recommendations
        
        Args:
            user_id: Target user ID
            top_k: Number of recommendations
            filters: Optional filters
            return_details: Whether to include matching details
            
        Returns:
            List of recommendation dictionaries
        """
        if not self.is_fitted:
            raise ValueError("Recommender not fitted. Call fit() first.")
        
        # Get content-based recommendations
        content_recs = self.content_recommender.recommend(user_id, top_k=top_k * 2, filters=filters)
        
        # Get collaborative filtering predictions
        collab_predictions = self.collab_recommender.predict(user_id)
        
        # Normalize collaborative scores
        collab_min = collab_predictions.min()
        collab_max = collab_predictions.max()
        if collab_max > collab_min:
            collab_predictions = (collab_predictions - collab_min) / (collab_max - collab_min)
        
        # Combine scores
        recommendations = []
        seen_ids = set()
        
        for rec_user_id, content_score, details in content_recs:
            if rec_user_id in seen_ids:
                continue
            seen_ids.add(rec_user_id)
            
            # Get collaborative score
            if rec_user_id in self.collab_recommender.user_id_to_idx:
                collab_idx = self.collab_recommender.user_id_to_idx[rec_user_id]
                collab_score = collab_predictions[collab_idx]
            else:
                collab_score = 0.5  # Default for cold start
            
            # Hybrid score
            hybrid_score = self.alpha * content_score + self.beta * collab_score
            
            rec_idx = self.content_recommender.user_id_to_idx.get(rec_user_id, 0)
            user_data = self.df.iloc[rec_idx]
            
            full_bio = str(user_data.get('bio', ''))
            rec = {
                'user_id': rec_user_id,
                'hybrid_score': round(hybrid_score * 100, 1),
                'content_score': round(content_score * 100, 1),
                'collab_score': round(collab_score * 100, 1),
                'age': int(user_data.get('age', 0)),
                'sex': user_data.get('sex', 'unknown'),
                'orientation': user_data.get('orientation', 'unknown'),
                'location': user_data.get('location', 'unknown'),
                'full_bio': full_bio,
                'bio_preview': full_bio[:200] + '...' if len(full_bio) > 200 else full_bio,
            }
            
            if return_details:
                rec['match_details'] = details
            
            recommendations.append(rec)
        
        # Sort by hybrid score and return top-k
        recommendations.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return recommendations[:top_k]
    
    def get_user_profile(self, user_id: int) -> Dict:
        """Get user profile information"""
        if user_id not in self.content_recommender.user_id_to_idx:
            return {}
        
        idx = self.content_recommender.user_id_to_idx[user_id]
        user_data = self.df.iloc[idx]
        
        return {
            'user_id': user_id,
            'age': int(user_data.get('age', 0)),
            'sex': user_data.get('sex', 'unknown'),
            'orientation': user_data.get('orientation', 'unknown'),
            'location': user_data.get('location', 'unknown'),
            'status': user_data.get('status', 'unknown'),
            'bio': str(user_data.get('bio', ''))
        }


class RealTimeRecommender:
    """
    Real-Time Recommendation System
    Provides fast recommendations with caching and context awareness
    """
    
    def __init__(self, hybrid_recommender: HybridRecommender):
        self.recommender = hybrid_recommender
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.user_history = defaultdict(list)
        
    def get_recommendations(self, 
                            user_id: int,
                            top_k: int = 10,
                            context: Dict = None,
                            use_cache: bool = True) -> List[Dict]:
        """
        Get real-time recommendations with context awareness
        
        Args:
            user_id: Target user ID
            top_k: Number of recommendations
            context: Context information (time, location, mood, etc.)
            use_cache: Whether to use caching
            
        Returns:
            List of recommendation dictionaries with timing info
        """
        start_time = time.time()
        
        # Check cache
        cache_key = f"{user_id}_{top_k}"
        if use_cache and cache_key in self.cache:
            cached_time, cached_recs = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                elapsed = (time.time() - start_time) * 1000
                return {
                    'recommendations': cached_recs,
                    'timing_ms': round(elapsed, 2),
                    'from_cache': True
                }
        
        # Get base recommendations
        recommendations = self.recommender.recommend(user_id, top_k=top_k * 2)
        
        # Apply context-aware adjustments
        if context:
            recommendations = self._apply_context(recommendations, context, user_id)
        
        # Apply user history (avoid recently shown)
        recommendations = self._filter_recent(recommendations, user_id)
        
        # Take top-k
        recommendations = recommendations[:top_k]
        
        # Update cache
        self.cache[cache_key] = (time.time(), recommendations)
        
        # Update user history
        self._update_history(user_id, [r['user_id'] for r in recommendations])
        
        elapsed = (time.time() - start_time) * 1000
        
        return {
            'recommendations': recommendations,
            'timing_ms': round(elapsed, 2),
            'from_cache': False
        }
    
    def _apply_context(self, 
                       recommendations: List[Dict], 
                       context: Dict,
                       user_id: int) -> List[Dict]:
        """Apply context-aware adjustments to recommendations"""
        current_hour = context.get('hour', datetime.now().hour)
        day_of_week = context.get('day_of_week', datetime.now().weekday())
        
        for rec in recommendations:
            boost = 0
            
            # Time-based adjustments
            # Evening (6PM-10PM) - boost active users
            if 18 <= current_hour <= 22:
                boost += 0.05
            
            # Weekend boost for nearby users
            if day_of_week >= 5:  # Saturday or Sunday
                if context.get('location') and rec.get('location'):
                    if context['location'].lower() in rec['location'].lower():
                        boost += 0.1
            
            # Mood-based adjustments
            mood = context.get('mood', 'casual')
            if mood == 'serious':
                # Boost similar age profiles
                boost += 0.05
            elif mood == 'adventurous':
                # Boost diverse interests
                if len(rec.get('match_details', {}).get('common_interests', [])) < 3:
                    boost += 0.03  # Some novelty
            
            # Apply boost
            rec['hybrid_score'] = min(100, rec['hybrid_score'] + boost * 100)
            rec['context_adjusted'] = True
        
        # Re-sort
        recommendations.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return recommendations
    
    def _filter_recent(self, 
                       recommendations: List[Dict], 
                       user_id: int) -> List[Dict]:
        """Filter out recently shown recommendations"""
        recent = set(self.user_history.get(user_id, [])[-50:])  # Last 50 shown
        
        return [r for r in recommendations if r['user_id'] not in recent]
    
    def _update_history(self, user_id: int, shown_ids: List[int]):
        """Update user's viewing history"""
        self.user_history[user_id].extend(shown_ids)
        # Keep only last 100
        self.user_history[user_id] = self.user_history[user_id][-100:]
    
    def record_interaction(self, 
                           user_id: int, 
                           target_id: int, 
                           action: str,
                           score: float = None):
        """
        Record user interaction for feedback learning
        
        Args:
            user_id: Acting user
            target_id: Target user
            action: 'like', 'pass', 'superlike', 'match', 'message'
            score: Optional explicit score (1-5) for the interaction
        
        These interactions can be used to:
        1. Retrain Collaborative Filtering with real data
        2. Improve Content-Based by learning user preferences
        3. Build the interaction matrix for hybrid recommendations
        """
        # Determine interaction score based on action
        if score is None:
            action_scores = {
                'pass': 1.0,
                'view': 2.0,
                'like': 4.0,
                'superlike': 5.0,
                'match': 5.0,
                'message': 5.0
            }
            score = action_scores.get(action.lower(), 3.0)
        
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'target_id': target_id,
            'action': action,
            'score': score
        }
        
        # Store interaction
        if not hasattr(self, 'interactions'):
            self.interactions = defaultdict(list)
        
        self.interactions[user_id].append(interaction)
        
        # Also store in a flat list for easy export
        if not hasattr(self, 'all_interactions'):
            self.all_interactions = []
        self.all_interactions.append(interaction)
        
        # Invalidate cache for this user
        for key in list(self.cache.keys()):
            if key.startswith(f"{user_id}_"):
                del self.cache[key]
    
    def get_interaction_matrix(self) -> np.ndarray:
        """
        Build interaction matrix from collected interactions
        Can be used to retrain Collaborative Filtering with real data
        
        Returns:
            Sparse interaction matrix (n_users x n_users)
        """
        if not hasattr(self, 'all_interactions') or len(self.all_interactions) == 0:
            print("⚠️ No interactions recorded yet")
            return None
        
        from scipy.sparse import lil_matrix
        
        n_users = len(self.recommender.df)
        user_id_to_idx = self.recommender.content_recommender.user_id_to_idx
        
        matrix = lil_matrix((n_users, n_users), dtype=np.float32)
        
        for interaction in self.all_interactions:
            user_idx = user_id_to_idx.get(interaction['user_id'])
            target_idx = user_id_to_idx.get(interaction['target_id'])
            
            if user_idx is not None and target_idx is not None:
                matrix[user_idx, target_idx] = interaction['score']
        
        print(f"📊 Built interaction matrix with {matrix.nnz} interactions")
        return matrix.tocsr()
    
    def get_interaction_count(self) -> int:
        """Get total number of recorded interactions"""
        if not hasattr(self, 'all_interactions'):
            return 0
        return len(self.all_interactions)
    
    def can_enable_collaborative(self, min_interactions: int = 1000) -> bool:
        """
        Check if enough interactions exist to enable Collaborative Filtering
        
        Args:
            min_interactions: Minimum interactions needed (default 1000)
        
        Returns:
            True if CF can be meaningfully trained
        """
        count = self.get_interaction_count()
        if count >= min_interactions:
            print(f"✅ {count} interactions collected - CF can be enabled!")
            return True
        else:
            print(f"ℹ️ {count}/{min_interactions} interactions - collect more for CF")
            return False
    
    def retrain_with_interactions(self):
        """
        Retrain the collaborative filtering model with collected interactions
        Call this periodically once enough interactions are collected
        """
        if not self.can_enable_collaborative():
            return False
        
        interaction_matrix = self.get_interaction_matrix()
        if interaction_matrix is None:
            return False
        
        print("🔄 Retraining Collaborative Filtering with real interactions...")
        
        # Get user IDs
        user_ids = self.recommender.df['user_id'].tolist() if 'user_id' in self.recommender.df.columns else list(range(len(self.recommender.df)))
        
        # Retrain CF
        self.recommender.collab_recommender.fit(interaction_matrix, user_ids)
        
        # Update hybrid weights to use CF
        self.recommender.alpha = 0.6  # Content weight
        self.recommender.beta = 0.4   # Collab weight
        
        print("✅ Collaborative Filtering retrained with real user data!")
        return True
    
    def export_interactions(self, filepath: str):
        """Export interactions to CSV for analysis or backup"""
        if not hasattr(self, 'all_interactions') or len(self.all_interactions) == 0:
            print("⚠️ No interactions to export")
            return
        
        import pandas as pd
        df = pd.DataFrame(self.all_interactions)
        df.to_csv(filepath, index=False)
        print(f"📁 Exported {len(df)} interactions to {filepath}")
    
    def import_interactions(self, filepath: str):
        """Import interactions from CSV (e.g., from a database dump)"""
        import pandas as pd
        df = pd.read_csv(filepath)
        
        if not hasattr(self, 'all_interactions'):
            self.all_interactions = []
        
        for _, row in df.iterrows():
            self.all_interactions.append(row.to_dict())
        
        print(f"📁 Imported {len(df)} interactions from {filepath}")
    
    def clear_cache(self):
        """Clear all cached recommendations"""
        self.cache = {}
        print("🗑️ Cache cleared")


def create_recommendation_system(df: pd.DataFrame,
                                   bio_embeddings: np.ndarray = None,
                                   interest_matrix: np.ndarray = None) -> RealTimeRecommender:
    """
    Factory function to create a complete recommendation system
    
    Returns:
        Configured RealTimeRecommender instance
    """
    # Create hybrid recommender
    hybrid = HybridRecommender(content_weight=0.6, collab_weight=0.4)
    
    # Fit the recommender
    hybrid.fit(df, bio_embeddings, interest_matrix)
    
    # Wrap with real-time capabilities
    realtime = RealTimeRecommender(hybrid)
    
    return realtime
