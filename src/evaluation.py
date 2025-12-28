"""
Model Evaluation Module for Dating Recommendation System
Implements RMSE, MAE, Precision@K, Recall@K, NDCG, and other metrics
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


class RecommenderEvaluator:
    """
    Comprehensive evaluation suite for recommendation systems
    
    Metrics implemented:
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - Precision@K
    - Recall@K
    - NDCG@K (Normalized Discounted Cumulative Gain)
    - MRR (Mean Reciprocal Rank)
    - Coverage
    - Diversity
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = {}
        self.ground_truth = None
        
    def create_ground_truth(self, 
                            similarity_matrix: np.ndarray,
                            threshold: float = 0.5,
                            n_relevant_per_user: int = 20) -> Dict[int, List[int]]:
        """
        Create ground truth relevance based on similarity scores
        
        Args:
            similarity_matrix: User-user similarity matrix
            threshold: Minimum similarity to be considered relevant
            n_relevant_per_user: Maximum relevant items per user
            
        Returns:
            Dictionary mapping user_id to list of relevant user_ids
        """
        print("📊 Creating ground truth from similarity matrix...")
        
        n_users = similarity_matrix.shape[0]
        ground_truth = {}
        
        user_ids = self.df['user_id'].values if 'user_id' in self.df.columns else np.arange(n_users)
        
        for i in range(n_users):
            # Get similarity scores for this user
            scores = similarity_matrix[i].copy()
            scores[i] = 0  # Exclude self
            
            # Get indices above threshold
            relevant_mask = scores >= threshold
            relevant_indices = np.where(relevant_mask)[0]
            
            # Sort by score and take top N
            if len(relevant_indices) > 0:
                relevant_scores = scores[relevant_indices]
                sorted_indices = np.argsort(relevant_scores)[::-1][:n_relevant_per_user]
                relevant_user_ids = [int(user_ids[relevant_indices[j]]) for j in sorted_indices]
            else:
                # If no one above threshold, take top N anyway
                top_indices = np.argsort(scores)[::-1][:n_relevant_per_user]
                relevant_user_ids = [int(user_ids[j]) for j in top_indices]
            
            ground_truth[int(user_ids[i])] = relevant_user_ids
        
        self.ground_truth = ground_truth
        print(f"✅ Ground truth created for {len(ground_truth)} users")
        
        return ground_truth
    
    def create_ground_truth_from_recommender(self,
                                              content_recommender,
                                              threshold: float = 0.5,
                                              n_relevant_per_user: int = 20,
                                              sample_users: int = None) -> Dict[int, List[int]]:
        """
        Create ground truth using on-demand similarity computation (memory efficient)
        
        Args:
            content_recommender: ContentBasedRecommender with _compute_user_similarity method
            threshold: Minimum similarity to be considered relevant
            n_relevant_per_user: Maximum relevant items per user
            sample_users: If set, only compute for this many users
            
        Returns:
            Dictionary mapping user_id to list of relevant user_ids
        """
        print("📊 Creating ground truth using on-demand similarity...")
        
        n_users = len(content_recommender.df)
        user_ids = content_recommender.df['user_id'].values if 'user_id' in content_recommender.df.columns else np.arange(n_users)
        
        ground_truth = {}
        
        # Optionally sample users to speed up
        if sample_users and sample_users < n_users:
            indices_to_process = np.random.choice(n_users, sample_users, replace=False)
        else:
            indices_to_process = range(n_users)
        
        for i in indices_to_process:
            # Compute similarity on-demand for this user
            scores = content_recommender._compute_user_similarity(i)
            scores[i] = 0  # Exclude self
            
            # Get indices above threshold
            relevant_mask = scores >= threshold
            relevant_indices = np.where(relevant_mask)[0]
            
            # Sort by score and take top N
            if len(relevant_indices) > 0:
                relevant_scores = scores[relevant_indices]
                sorted_indices = np.argsort(relevant_scores)[::-1][:n_relevant_per_user]
                relevant_user_ids = [int(user_ids[relevant_indices[j]]) for j in sorted_indices]
            else:
                # If no one above threshold, take top N anyway
                top_indices = np.argsort(scores)[::-1][:n_relevant_per_user]
                relevant_user_ids = [int(user_ids[j]) for j in top_indices]
            
            ground_truth[int(user_ids[i])] = relevant_user_ids
        
        self.ground_truth = ground_truth
        print(f"✅ Ground truth created for {len(ground_truth)} users")
        
        return ground_truth
    
    def create_ground_truth_from_collab(self,
                                         collab_recommender,
                                         content_recommender,
                                         n_relevant_per_user: int = 20,
                                         sample_users: int = None,
                                         test_user_ids: List[int] = None,
                                         threshold: float = 0.3) -> Dict[int, List[int]]:
        """
        Create ground truth using content-based similarity (NOT CF predictions)
        This avoids circular evaluation by using independent ground truth
        
        Args:
            collab_recommender: CollaborativeFilteringRecommender (for compatibility)
            content_recommender: ContentBasedRecommender for creating ground truth
            n_relevant_per_user: Maximum relevant items per user
            sample_users: If set, only compute for this many users
            test_user_ids: List of test user IDs to create ground truth for (train/test split)
            threshold: Minimum similarity to be considered relevant
            
        Returns:
            Dictionary mapping user_id to list of relevant user_ids
        """
        print("📊 Creating ground truth from content similarity (for CF evaluation)...")
        
        # Use content-based similarity as ground truth for CF
        # This avoids circular evaluation
        return self.create_ground_truth_from_recommender(
            content_recommender,
            threshold=threshold,
            n_relevant_per_user=n_relevant_per_user,
            sample_users=sample_users,
            test_user_ids=test_user_ids
        )
    
    def create_hybrid_ground_truth(self,
                                    content_recommender,
                                    collab_recommender,
                                    content_weight: float = 0.5,
                                    n_relevant_per_user: int = 20,
                                    sample_users: int = None) -> Dict[int, List[int]]:
        """
        Create hybrid ground truth combining content and collaborative signals
        
        Args:
            content_recommender: ContentBasedRecommender
            collab_recommender: CollaborativeFilteringRecommender
            content_weight: Weight for content-based scores (1 - content_weight for collab)
            n_relevant_per_user: Maximum relevant items per user
            sample_users: If set, only compute for this many users
            
        Returns:
            Dictionary mapping user_id to list of relevant user_ids
        """
        print(f"📊 Creating hybrid ground truth (content: {content_weight:.0%}, collab: {1-content_weight:.0%})...")
        
        n_users = len(content_recommender.df)
        user_ids = content_recommender.df['user_id'].values if 'user_id' in content_recommender.df.columns else np.arange(n_users)
        
        ground_truth = {}
        
        # Optionally sample users to speed up
        if sample_users and sample_users < n_users:
            indices_to_process = np.random.choice(n_users, sample_users, replace=False)
        else:
            indices_to_process = range(n_users)
        
        for i in indices_to_process:
            user_id = int(user_ids[i])
            
            # Get content-based scores
            content_scores = content_recommender._compute_user_similarity(i)
            content_scores[i] = 0
            
            # Normalize content scores
            if content_scores.max() > 0:
                content_scores = content_scores / content_scores.max()
            
            # Get collaborative scores
            if user_id in collab_recommender.user_id_to_idx:
                collab_scores = collab_recommender.predict(user_id)
                # Normalize collaborative scores
                collab_min = collab_scores.min()
                collab_max = collab_scores.max()
                if collab_max > collab_min:
                    collab_scores = (collab_scores - collab_min) / (collab_max - collab_min)
                collab_scores[collab_recommender.user_id_to_idx[user_id]] = 0
            else:
                collab_scores = np.zeros_like(content_scores)
            
            # Combine scores
            combined_scores = content_weight * content_scores + (1 - content_weight) * collab_scores
            
            # Get top N
            top_indices = np.argsort(combined_scores)[::-1][:n_relevant_per_user]
            relevant_user_ids = [int(user_ids[j]) for j in top_indices]
            
            ground_truth[user_id] = relevant_user_ids
        
        self.ground_truth = ground_truth
        print(f"✅ Hybrid ground truth created for {len(ground_truth)} users")
        
        return ground_truth
    
    def calculate_rmse(self, 
                       predictions: Dict[int, List[Tuple[int, float]]],
                       actual_scores: Dict[int, Dict[int, float]]) -> float:
        """
        Calculate Root Mean Squared Error
        
        Args:
            predictions: Dict of user_id -> [(item_id, predicted_score), ...]
            actual_scores: Dict of user_id -> {item_id: actual_score}
            
        Returns:
            RMSE value
        """
        errors = []
        
        for user_id, preds in predictions.items():
            if user_id not in actual_scores:
                continue
            
            user_actual = actual_scores[user_id]
            
            for item_id, pred_score in preds:
                if item_id in user_actual:
                    actual = user_actual[item_id]
                    errors.append((pred_score - actual) ** 2)
        
        if not errors:
            return float('inf')
        
        rmse = np.sqrt(np.mean(errors))
        return round(rmse, 4)
    
    def calculate_mae(self,
                      predictions: Dict[int, List[Tuple[int, float]]],
                      actual_scores: Dict[int, Dict[int, float]]) -> float:
        """
        Calculate Mean Absolute Error
        
        Args:
            predictions: Dict of user_id -> [(item_id, predicted_score), ...]
            actual_scores: Dict of user_id -> {item_id: actual_score}
            
        Returns:
            MAE value
        """
        errors = []
        
        for user_id, preds in predictions.items():
            if user_id not in actual_scores:
                continue
            
            user_actual = actual_scores[user_id]
            
            for item_id, pred_score in preds:
                if item_id in user_actual:
                    actual = user_actual[item_id]
                    errors.append(abs(pred_score - actual))
        
        if not errors:
            return float('inf')
        
        mae = np.mean(errors)
        return round(mae, 4)
    
    def calculate_rmse_from_recommendations(self,
                                              recommendations_with_scores: Dict[int, List[Tuple[int, float]]],
                                              content_recommender = None) -> float:
        """
        Calculate RMSE by comparing predicted scores with actual similarity scores
        
        Args:
            recommendations_with_scores: Dict of user_id -> [(item_id, predicted_score), ...]
            content_recommender: For computing actual similarity scores
            
        Returns:
            RMSE value
        """
        if content_recommender is None:
            return float('nan')
        
        errors = []
        user_id_to_idx = content_recommender.user_id_to_idx
        
        for user_id, recs in recommendations_with_scores.items():
            if user_id not in user_id_to_idx:
                continue
            
            user_idx = user_id_to_idx[user_id]
            # Get actual similarity scores for this user
            actual_scores = content_recommender._compute_user_similarity(user_idx)
            
            for item_id, pred_score in recs:
                if item_id in user_id_to_idx:
                    item_idx = user_id_to_idx[item_id]
                    actual = actual_scores[item_idx]
                    # Normalize predicted score if needed (assume 0-1 scale)
                    pred_normalized = pred_score if pred_score <= 1 else pred_score / 100
                    errors.append((pred_normalized - actual) ** 2)
        
        if not errors:
            return float('nan')
        
        rmse = np.sqrt(np.mean(errors))
        return round(rmse, 4)
    
    def calculate_mae_from_recommendations(self,
                                            recommendations_with_scores: Dict[int, List[Tuple[int, float]]],
                                            content_recommender = None) -> float:
        """
        Calculate MAE by comparing predicted scores with actual similarity scores
        
        Args:
            recommendations_with_scores: Dict of user_id -> [(item_id, predicted_score), ...]
            content_recommender: For computing actual similarity scores
            
        Returns:
            MAE value
        """
        if content_recommender is None:
            return float('nan')
        
        errors = []
        user_id_to_idx = content_recommender.user_id_to_idx
        
        for user_id, recs in recommendations_with_scores.items():
            if user_id not in user_id_to_idx:
                continue
            
            user_idx = user_id_to_idx[user_id]
            # Get actual similarity scores for this user
            actual_scores = content_recommender._compute_user_similarity(user_idx)
            
            for item_id, pred_score in recs:
                if item_id in user_id_to_idx:
                    item_idx = user_id_to_idx[item_id]
                    actual = actual_scores[item_idx]
                    # Normalize predicted score if needed
                    pred_normalized = pred_score if pred_score <= 1 else pred_score / 100
                    errors.append(abs(pred_normalized - actual))
        
        if not errors:
            return float('nan')
        
        mae = np.mean(errors)
        return round(mae, 4)
    
    def calculate_precision_at_k(self,
                                  recommendations: Dict[int, List[int]],
                                  k: int = 10) -> float:
        """
        Calculate Precision@K
        
        Precision@K = (Relevant items in top-K) / K
        
        Args:
            recommendations: Dict of user_id -> [recommended_item_ids]
            k: Number of recommendations to consider
            
        Returns:
            Average Precision@K across all users
        """
        if self.ground_truth is None:
            raise ValueError("Ground truth not set. Call create_ground_truth first.")
        
        precisions = []
        
        for user_id, recs in recommendations.items():
            if user_id not in self.ground_truth:
                continue
            
            relevant = set(self.ground_truth[user_id])
            top_k_recs = set(recs[:k])
            
            if len(top_k_recs) == 0:
                continue
            
            hits = len(top_k_recs & relevant)
            precision = hits / k
            precisions.append(precision)
        
        if not precisions:
            return 0.0
        
        return round(np.mean(precisions), 4)
    
    def calculate_recall_at_k(self,
                               recommendations: Dict[int, List[int]],
                               k: int = 10) -> float:
        """
        Calculate Recall@K
        
        Recall@K = (Relevant items in top-K) / (Total relevant items)
        
        Args:
            recommendations: Dict of user_id -> [recommended_item_ids]
            k: Number of recommendations to consider
            
        Returns:
            Average Recall@K across all users
        """
        if self.ground_truth is None:
            raise ValueError("Ground truth not set. Call create_ground_truth first.")
        
        recalls = []
        
        for user_id, recs in recommendations.items():
            if user_id not in self.ground_truth:
                continue
            
            relevant = set(self.ground_truth[user_id])
            top_k_recs = set(recs[:k])
            
            if len(relevant) == 0:
                continue
            
            hits = len(top_k_recs & relevant)
            recall = hits / len(relevant)
            recalls.append(recall)
        
        if not recalls:
            return 0.0
        
        return round(np.mean(recalls), 4)
    
    def calculate_ndcg_at_k(self,
                            recommendations: Dict[int, List[int]],
                            k: int = 10) -> float:
        """
        Calculate NDCG@K (Normalized Discounted Cumulative Gain)
        
        NDCG considers the position of relevant items in the recommendation list
        
        Args:
            recommendations: Dict of user_id -> [recommended_item_ids]
            k: Number of recommendations to consider
            
        Returns:
            Average NDCG@K across all users
        """
        if self.ground_truth is None:
            raise ValueError("Ground truth not set. Call create_ground_truth first.")
        
        ndcgs = []
        
        for user_id, recs in recommendations.items():
            if user_id not in self.ground_truth:
                continue
            
            relevant = set(self.ground_truth[user_id])
            
            # Calculate DCG
            dcg = 0.0
            for i, item_id in enumerate(recs[:k]):
                if item_id in relevant:
                    dcg += 1.0 / np.log2(i + 2)  # +2 because position is 1-indexed
            
            # Calculate IDCG (ideal DCG)
            n_relevant_in_k = min(len(relevant), k)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant_in_k))
            
            if idcg > 0:
                ndcg = dcg / idcg
            else:
                ndcg = 0.0
            
            ndcgs.append(ndcg)
        
        if not ndcgs:
            return 0.0
        
        return round(np.mean(ndcgs), 4)
    
    def calculate_mrr(self,
                      recommendations: Dict[int, List[int]]) -> float:
        """
        Calculate Mean Reciprocal Rank
        
        MRR = Average of 1/rank of first relevant item
        
        Args:
            recommendations: Dict of user_id -> [recommended_item_ids]
            
        Returns:
            MRR value
        """
        if self.ground_truth is None:
            raise ValueError("Ground truth not set. Call create_ground_truth first.")
        
        reciprocal_ranks = []
        
        for user_id, recs in recommendations.items():
            if user_id not in self.ground_truth:
                continue
            
            relevant = set(self.ground_truth[user_id])
            
            for i, item_id in enumerate(recs):
                if item_id in relevant:
                    reciprocal_ranks.append(1.0 / (i + 1))
                    break
            else:
                reciprocal_ranks.append(0.0)
        
        if not reciprocal_ranks:
            return 0.0
        
        return round(np.mean(reciprocal_ranks), 4)
    
    def calculate_coverage(self,
                           recommendations: Dict[int, List[int]],
                           total_items: int) -> float:
        """
        Calculate Coverage
        
        Coverage = Percentage of items that appear in at least one recommendation list
        
        Args:
            recommendations: Dict of user_id -> [recommended_item_ids]
            total_items: Total number of items in the catalog
            
        Returns:
            Coverage percentage
        """
        recommended_items = set()
        
        for user_id, recs in recommendations.items():
            recommended_items.update(recs)
        
        coverage = len(recommended_items) / total_items
        return round(coverage, 4)
    
    def calculate_diversity(self,
                            recommendations: Dict[int, List[int]],
                            similarity_matrix: np.ndarray,
                            user_id_to_idx: Dict[int, int]) -> float:
        """
        Calculate Intra-list Diversity
        
        Diversity = Average dissimilarity between items in recommendation lists
        
        Args:
            recommendations: Dict of user_id -> [recommended_item_ids]
            similarity_matrix: Item-item similarity matrix
            user_id_to_idx: Mapping from user_id to matrix index
            
        Returns:
            Average diversity score
        """
        diversities = []
        
        for user_id, recs in recommendations.items():
            if len(recs) < 2:
                continue
            
            # Calculate pairwise dissimilarity
            dissimilarities = []
            for i in range(len(recs)):
                for j in range(i + 1, len(recs)):
                    idx_i = user_id_to_idx.get(recs[i])
                    idx_j = user_id_to_idx.get(recs[j])
                    
                    if idx_i is not None and idx_j is not None:
                        sim = similarity_matrix[idx_i, idx_j]
                        dissimilarities.append(1 - sim)
            
            if dissimilarities:
                diversities.append(np.mean(dissimilarities))
        
        if not diversities:
            return 0.0
        
        return round(np.mean(diversities), 4)
    
    def calculate_diversity_on_demand(self,
                                       recommendations: Dict[int, List[int]],
                                       content_recommender) -> float:
        """
        Calculate Intra-list Diversity using on-demand similarity (memory efficient)
        
        Args:
            recommendations: Dict of user_id -> [recommended_item_ids]
            content_recommender: ContentBasedRecommender for on-demand similarity
            
        Returns:
            Average diversity score
        """
        diversities = []
        user_id_to_idx = content_recommender.user_id_to_idx
        
        for user_id, recs in recommendations.items():
            if len(recs) < 2:
                continue
            
            # Calculate pairwise dissimilarity
            dissimilarities = []
            for i in range(len(recs)):
                for j in range(i + 1, len(recs)):
                    idx_i = user_id_to_idx.get(recs[i])
                    idx_j = user_id_to_idx.get(recs[j])
                    
                    if idx_i is not None and idx_j is not None:
                        # Compute similarity on-demand
                        sim = content_recommender._compute_user_similarity(idx_i)[idx_j]
                        dissimilarities.append(1 - sim)
            
            if dissimilarities:
                diversities.append(np.mean(dissimilarities))
        
        if not diversities:
            return 0.0
        
        return round(np.mean(diversities), 4)
    
    def evaluate_model(self,
                       model_name: str,
                       get_recommendations_fn,
                       test_users: List[int],
                       k_values: List[int] = [5, 10, 20],
                       content_recommender = None,
                       similarity_matrix: np.ndarray = None,
                       user_id_to_idx: Dict[int, int] = None) -> Dict:
        """
        Comprehensive model evaluation
        
        Args:
            model_name: Name of the model being evaluated
            get_recommendations_fn: Function that takes user_id and returns recommendations
            test_users: List of user IDs to test on
            k_values: List of K values for @K metrics
            content_recommender: For on-demand diversity calculation (preferred)
            similarity_matrix: For diversity calculation (legacy, uses more memory)
            user_id_to_idx: For diversity calculation (legacy)
            
        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\n📊 Evaluating {model_name}...")
        
        # Get recommendations for all test users (with and without scores)
        recommendations = {}
        recommendations_with_scores = {}
        
        for user_id in test_users:
            recs = get_recommendations_fn(user_id)
            if isinstance(recs, list):
                if len(recs) > 0 and isinstance(recs[0], dict):
                    recommendations[user_id] = [r['user_id'] for r in recs]
                    # Extract scores if available
                    score_key = 'hybrid_score' if 'hybrid_score' in recs[0] else 'score'
                    recommendations_with_scores[user_id] = [
                        (r['user_id'], r.get(score_key, r.get('content_score', 0)) / 100) 
                        for r in recs
                    ]
                elif len(recs) > 0 and isinstance(recs[0], tuple):
                    recommendations[user_id] = [r[0] for r in recs]
                    recommendations_with_scores[user_id] = [(r[0], r[1]) for r in recs]
                else:
                    recommendations[user_id] = recs
                    recommendations_with_scores[user_id] = [(r, 0.5) for r in recs]
        
        results = {'model': model_name}
        
        # Calculate metrics for each K
        for k in k_values:
            results[f'precision@{k}'] = self.calculate_precision_at_k(recommendations, k)
            results[f'recall@{k}'] = self.calculate_recall_at_k(recommendations, k)
            results[f'ndcg@{k}'] = self.calculate_ndcg_at_k(recommendations, k)
        
        # Calculate MRR
        results['mrr'] = self.calculate_mrr(recommendations)
        
        # Calculate Coverage
        total_items = len(self.df)
        results['coverage'] = self.calculate_coverage(recommendations, total_items)
        
        # Calculate Diversity (using on-demand computation if available)
        if content_recommender is not None:
            results['diversity'] = self.calculate_diversity_on_demand(
                recommendations, content_recommender
            )
        elif similarity_matrix is not None and user_id_to_idx is not None:
            results['diversity'] = self.calculate_diversity(
                recommendations, similarity_matrix, user_id_to_idx
            )
        
        # Calculate RMSE and MAE
        if content_recommender is not None:
            results['rmse'] = self.calculate_rmse_from_recommendations(
                recommendations_with_scores, content_recommender
            )
            results['mae'] = self.calculate_mae_from_recommendations(
                recommendations_with_scores, content_recommender
            )
        
        self.results[model_name] = results
        
        print(f"✅ {model_name} evaluation complete")
        
        return results
    
    def compare_models(self) -> pd.DataFrame:
        """
        Create comparison table of all evaluated models
        
        Returns:
            DataFrame with model comparison
        """
        if not self.results:
            raise ValueError("No models evaluated yet.")
        
        df = pd.DataFrame(self.results).T
        
        # Reorder columns
        metric_order = ['model']
        for k in [5, 10, 20]:
            metric_order.extend([f'precision@{k}', f'recall@{k}', f'ndcg@{k}'])
        metric_order.extend(['mrr', 'coverage', 'diversity', 'rmse', 'mae'])
        
        available_cols = [c for c in metric_order if c in df.columns]
        df = df[available_cols]
        
        return df
    
    def plot_metrics_comparison(self, 
                                 save_path: str = None,
                                 figsize: Tuple[int, int] = (14, 8)):
        """
        Plot comparison of metrics across models
        """
        if not self.results:
            raise ValueError("No models evaluated yet.")
        
        comparison_df = self.compare_models()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Model Evaluation Comparison', fontsize=14, fontweight='bold')
        
        models = list(self.results.keys())
        x = np.arange(len(models))
        width = 0.25
        
        # Precision@K comparison
        ax = axes[0, 0]
        for i, k in enumerate([5, 10, 20]):
            col = f'precision@{k}'
            if col in comparison_df.columns:
                values = [self.results[m].get(col, 0) for m in models]
                ax.bar(x + i*width, values, width, label=f'@{k}')
        ax.set_ylabel('Precision')
        ax.set_title('Precision@K')
        ax.set_xticks(x + width)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Recall@K comparison
        ax = axes[0, 1]
        for i, k in enumerate([5, 10, 20]):
            col = f'recall@{k}'
            if col in comparison_df.columns:
                values = [self.results[m].get(col, 0) for m in models]
                ax.bar(x + i*width, values, width, label=f'@{k}')
        ax.set_ylabel('Recall')
        ax.set_title('Recall@K')
        ax.set_xticks(x + width)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # NDCG@K comparison
        ax = axes[1, 0]
        for i, k in enumerate([5, 10, 20]):
            col = f'ndcg@{k}'
            if col in comparison_df.columns:
                values = [self.results[m].get(col, 0) for m in models]
                ax.bar(x + i*width, values, width, label=f'@{k}')
        ax.set_ylabel('NDCG')
        ax.set_title('NDCG@K')
        ax.set_xticks(x + width)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Other metrics
        ax = axes[1, 1]
        metrics = ['mrr', 'coverage', 'diversity']
        available_metrics = [m for m in metrics if m in comparison_df.columns]
        
        for i, metric in enumerate(available_metrics):
            values = [self.results[m].get(metric, 0) for m in models]
            ax.bar(x + i*width, values, width, label=metric.upper())
        ax.set_ylabel('Score')
        ax.set_title('Other Metrics')
        ax.set_xticks(x + width)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Saved comparison plot to {save_path}")
        
        return fig
    
    def generate_report(self) -> str:
        """
        Generate a text report of evaluation results
        """
        if not self.results:
            return "No models evaluated yet."
        
        report = []
        report.append("=" * 60)
        report.append("RECOMMENDATION SYSTEM EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        comparison_df = self.compare_models()
        
        for model_name, metrics in self.results.items():
            report.append(f"\n📊 {model_name}")
            report.append("-" * 40)
            
            for metric, value in metrics.items():
                if metric != 'model':
                    report.append(f"  {metric}: {value}")
        
        report.append("\n" + "=" * 60)
        report.append("COMPARISON TABLE")
        report.append("=" * 60)
        report.append("")
        report.append(comparison_df.to_string())
        
        # Best model analysis
        report.append("\n" + "=" * 60)
        report.append("BEST MODEL ANALYSIS")
        report.append("=" * 60)
        
        key_metrics = ['precision@10', 'recall@10', 'ndcg@10']
        for metric in key_metrics:
            if metric in comparison_df.columns:
                best_model = comparison_df[metric].idxmax()
                best_value = comparison_df[metric].max()
                report.append(f"  Best {metric}: {best_model} ({best_value})")
        
        return "\n".join(report)


def create_train_test_split(df: pd.DataFrame, 
                             test_size: float = 0.2,
                             random_state: int = 42) -> Tuple[List[int], List[int]]:
    """
    Create train/test split of user IDs
    
    Args:
        df: DataFrame with user_id column
        test_size: Fraction of users for testing
        random_state: Random seed
        
    Returns:
        (train_user_ids, test_user_ids)
    """
    user_ids = df['user_id'].values if 'user_id' in df.columns else np.arange(len(df))
    
    train_ids, test_ids = train_test_split(
        user_ids, 
        test_size=test_size, 
        random_state=random_state
    )
    
    return list(train_ids), list(test_ids)


def simulate_rating_scores(similarity_matrix: np.ndarray,
                            noise_level: float = 0.2) -> Dict[int, Dict[int, float]]:
    """
    Simulate rating scores based on similarity for RMSE/MAE evaluation
    
    Args:
        similarity_matrix: User-user similarity matrix
        noise_level: Standard deviation of noise to add
        
    Returns:
        Dictionary of user_id -> {item_id: score}
    """
    n = similarity_matrix.shape[0]
    
    # Scale similarities to 1-5 rating range
    min_sim = similarity_matrix.min()
    max_sim = similarity_matrix.max()
    
    if max_sim > min_sim:
        scaled = (similarity_matrix - min_sim) / (max_sim - min_sim) * 4 + 1
    else:
        scaled = np.ones_like(similarity_matrix) * 3
    
    # Add noise
    noise = np.random.normal(0, noise_level, scaled.shape)
    noisy_scores = np.clip(scaled + noise, 1, 5)
    
    # Convert to dictionary format
    scores = {}
    for i in range(n):
        scores[i] = {j: float(noisy_scores[i, j]) for j in range(n) if i != j}
    
    return scores
