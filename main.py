"""
Main Entry Point for Dating Recommendation System
Provides CLI interface for running different components
"""

import os
import sys
import argparse
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))


def run_preprocessing(data_path: str, output_path: str = None, sample_size: int = None):
    """Run data preprocessing pipeline"""
    from src.data_preprocessing import DataPreprocessor
    
    print("="*60)
    print("DATA PREPROCESSING PIPELINE")
    print("="*60)
    
    preprocessor = DataPreprocessor(data_path)
    df = preprocessor.preprocess_all(sample_size=sample_size)
    
    if output_path:
        preprocessor.save_clean_data(output_path)
    
    # Print report
    report = preprocessor.get_report()
    print("\n📋 Preprocessing Report:")
    for key, value in report.items():
        print(f"  {key}: {value}")
    
    return df, preprocessor


def run_embeddings(df, cache_dir: str = './cache'):
    """Generate embeddings for the dataset"""
    from src.embeddings import AdvancedEmbeddings, InterestEmbeddings
    
    print("\n" + "="*60)
    print("GENERATING EMBEDDINGS")
    print("="*60)
    
    # Bio embeddings
    embedder = AdvancedEmbeddings(cache_dir=cache_dir)
    bio_embeddings = embedder.encode_texts(
        df['bio'].fillna('').tolist(),
        batch_size=64
    )
    
    # Interest embeddings
    interest_embedder = InterestEmbeddings()
    interest_matrix = interest_embedder.extract_interests(df['bio'].fillna('').tolist())
    
    print(f"✅ Bio embeddings shape: {bio_embeddings.shape}")
    print(f"✅ Interest matrix shape: {interest_matrix.shape}")
    
    return bio_embeddings, interest_matrix, embedder


def run_training(df, bio_embeddings, interest_matrix):
    """Train recommendation models"""
    from src.recommendation_engine import HybridRecommender, RealTimeRecommender
    
    print("\n" + "="*60)
    print("TRAINING RECOMMENDATION MODELS")
    print("="*60)
    
    # Create and fit hybrid recommender
    hybrid = HybridRecommender(content_weight=0.6, collab_weight=0.4)
    hybrid.fit(df, bio_embeddings, interest_matrix)
    
    # Wrap with real-time capabilities
    realtime = RealTimeRecommender(hybrid)
    
    return realtime, hybrid


def run_evaluation(df, realtime_recommender, hybrid_recommender, output_dir: str = './output'):
    """Evaluate the recommendation system with Content-Based and Hybrid models"""
    from src.evaluation import RecommenderEvaluator
    import os
    
    print("\n" + "="*60)
    print("EVALUATING MODELS")
    print("="*60)
    
    # Use sample of users for faster evaluation
    test_ids = df['user_id'].sample(n=100, random_state=42).tolist()
    
    all_results = {}
    
    # ============================================
    # 1. Evaluate Content-Based
    # ============================================
    print("\n" + "-"*50)
    print("📊 Evaluating Content-Based Model")
    print("-"*50)
    
    evaluator_cb = RecommenderEvaluator(df)
    evaluator_cb.create_ground_truth_from_recommender(
        hybrid_recommender.content_recommender,
        threshold=0.3,
        n_relevant_per_user=20
    )
    
    def get_content_recs(user_id):
        recs = hybrid_recommender.content_recommender.recommend(user_id, top_k=20)
        return [(r[0], r[1]) for r in recs]
    
    content_results = evaluator_cb.evaluate_model(
        "Content-Based",
        get_content_recs,
        test_ids,
        k_values=[5, 10, 20],
        content_recommender=hybrid_recommender.content_recommender
    )
    all_results['Content-Based'] = content_results
    
    # ============================================
    # 2. Evaluate Hybrid with hybrid ground truth
    # ============================================
    print("\n" + "-"*50)
    print("📊 Evaluating Hybrid Model")
    print("-"*50)
    
    evaluator_hybrid = RecommenderEvaluator(df)
    evaluator_hybrid.create_hybrid_ground_truth(
        hybrid_recommender.content_recommender,
        hybrid_recommender.collab_recommender,
        content_weight=0.6,
        n_relevant_per_user=20
    )
    
    def get_hybrid_recs(user_id):
        recs = hybrid_recommender.recommend(user_id, top_k=20)
        return [(r['user_id'], r['hybrid_score']/100) for r in recs]
    
    hybrid_results = evaluator_hybrid.evaluate_model(
        "Hybrid",
        get_hybrid_recs,
        test_ids,
        k_values=[5, 10, 20],
        content_recommender=hybrid_recommender.content_recommender
    )
    all_results['Hybrid'] = hybrid_results
    
    # ============================================
    # Print and Save Results
    # ============================================
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Create combined results DataFrame
    import pandas as pd
    results_df = pd.DataFrame(all_results).T
    print(results_df.to_string())
    
    # Generate comprehensive report
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("DATING RECOMMENDATION SYSTEM - EVALUATION REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Test Users: {len(test_ids)}")
    report_lines.append(f"Total Users in Dataset: {len(df)}")
    report_lines.append("")
    
    report_lines.append("=" * 70)
    report_lines.append("EVALUATION METHODOLOGY")
    report_lines.append("=" * 70)
    report_lines.append("- Content-Based: Ground truth from content similarity (threshold=0.3)")
    report_lines.append("- Hybrid: Ground truth combining content and collaborative signals (60/40)")
    report_lines.append("- Collaborative Filtering: Not evaluated (circular evaluation issue)")
    report_lines.append("=" * 70)
    report_lines.append("")
    
    for model_name, results in all_results.items():
        report_lines.append(f"\n{'='*50}")
        report_lines.append(f"📊 {model_name}")
        report_lines.append("="*50)
        for metric, value in results.items():
            if metric != 'model':
                report_lines.append(f"  {metric:15s}: {value}")
    
    report_lines.append("\n" + "="*70)
    report_lines.append("COMPARISON TABLE")
    report_lines.append("="*70)
    report_lines.append(results_df.to_string())
    
    report_lines.append("\n" + "="*70)
    report_lines.append("METRIC DEFINITIONS")
    report_lines.append("="*70)
    report_lines.append("  Precision@K  : Fraction of recommended items that are relevant")
    report_lines.append("  Recall@K     : Fraction of relevant items that are recommended")
    report_lines.append("  NDCG@K       : Normalized Discounted Cumulative Gain (ranking quality)")
    report_lines.append("  MRR          : Mean Reciprocal Rank (position of first relevant item)")
    report_lines.append("  Coverage     : Fraction of items ever recommended")
    report_lines.append("  Diversity    : Average dissimilarity within recommendation lists")
    report_lines.append("  RMSE         : Root Mean Squared Error (lower is better)")
    report_lines.append("  MAE          : Mean Absolute Error (lower is better)")
    report_lines.append("="*70)
    
    report = "\n".join(report_lines)
    print("\n" + report)
    
    # Save to file
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'evaluation_results.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Evaluation results saved to: {report_path}")
    
    # Create a combined evaluator for visualization
    combined_evaluator = RecommenderEvaluator(df)
    combined_evaluator.results = all_results
    
    return combined_evaluator


def run_visualizations(df, output_dir: str = './visualizations'):
    """Generate all visualizations"""
    from src.visualization import DataVisualizer
    
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    visualizer = DataVisualizer(df)
    figures = visualizer.generate_all_visualizations(output_dir=output_dir, show_plots=False)
    
    return figures


def run_demo(recommender, df):
    """Run a quick demo of the recommendation system"""
    print("\n" + "="*60)
    print("DEMO: REAL-TIME RECOMMENDATIONS")
    print("="*60)
    
    # Select a random user
    import random
    test_user_id = random.choice(df['user_id'].tolist())
    
    print(f"\n👤 Getting recommendations for User {test_user_id}")
    
    # Get user profile
    user_profile = recommender.recommender.get_user_profile(test_user_id)
    print(f"   Age: {user_profile.get('age')}")
    print(f"   Sex: {user_profile.get('sex')}")
    print(f"   Location: {user_profile.get('location')}")
    
    # Get recommendations
    context = {
        'hour': 20,  # Evening
        'day_of_week': 5,  # Saturday
        'mood': 'casual'
    }
    
    start_time = time.time()
    result = recommender.get_recommendations(
        user_id=test_user_id,
        top_k=5,
        context=context
    )
    elapsed = (time.time() - start_time) * 1000
    
    print(f"\n⏱️  Response time: {result['timing_ms']:.2f}ms (total: {elapsed:.2f}ms)")
    print(f"📦 From cache: {result['from_cache']}")
    
    print("\n💕 Top 5 Recommendations:")
    print("-" * 60)
    
    for i, rec in enumerate(result['recommendations'], 1):
        print(f"\n#{i} Match Score: {rec['hybrid_score']:.1f}%")
        print(f"   User ID: {rec['user_id']}")
        print(f"   Age: {rec.get('age')}, Sex: {rec.get('sex')}")
        print(f"   Location: {rec.get('location')}")
        print(f"   Content Score: {rec['content_score']:.1f}%")
        print(f"   Collab Score: {rec['collab_score']:.1f}%")
        if rec.get('match_details', {}).get('common_interests'):
            print(f"   Common Interests: {', '.join(rec['match_details']['common_interests'])}")


def run_webapp():
    """Launch the Streamlit web application"""
    import subprocess
    
    print("\n" + "="*60)
    print("LAUNCHING WEB APPLICATION")
    print("="*60)
    
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'app.py')
    subprocess.run(['streamlit', 'run', app_path])


def save_models(df, bio_embeddings, interest_matrix, hybrid_recommender, cache_dir='./cache'):
    """Save all trained models for instant webapp loading"""
    import pickle
    import numpy as np
    
    print("\n" + "="*60)
    print("💾 SAVING MODELS FOR INSTANT LOADING")
    print("="*60)
    
    model_dir = os.path.join(cache_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Save processed dataframe
    df_path = os.path.join(model_dir, 'processed_df.pkl')
    df.to_pickle(df_path)
    print(f"   ✅ Saved: {df_path}")
    
    # Save embeddings
    bio_path = os.path.join(model_dir, 'bio_embeddings.npy')
    np.save(bio_path, bio_embeddings)
    print(f"   ✅ Saved: {bio_path}")
    
    interest_path = os.path.join(model_dir, 'interest_matrix.npy')
    np.save(interest_path, interest_matrix)
    print(f"   ✅ Saved: {interest_path}")
    
    # Save hybrid recommender
    hybrid_path = os.path.join(model_dir, 'hybrid_recommender.pkl')
    with open(hybrid_path, 'wb') as f:
        pickle.dump(hybrid_recommender, f)
    print(f"   ✅ Saved: {hybrid_path}")
    
    # Save metadata
    metadata = {
        'n_users': len(df),
        'embedding_dim': bio_embeddings.shape[1],
        'n_interests': interest_matrix.shape[1],
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    meta_path = os.path.join(model_dir, 'metadata.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"   ✅ Saved: {meta_path}")
    
    print("\n🎉 Models saved! Webapp will now load instantly.")


def main():
    """Main entry point with CLI interface"""
    parser = argparse.ArgumentParser(
        description="AI Dating Recommendation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --full                    # Run full pipeline
  python main.py --webapp                  # Launch web app
  python main.py --preprocess              # Run preprocessing only
  python main.py --demo --sample 5000      # Run demo with 5000 samples
        """
    )
    
    parser.add_argument('--data', type=str, default='./data/okcupid_profiles.csv',
                       help='Path to the OkCupid profiles CSV file')
    parser.add_argument('--sample', type=int, default=None,
                       help='Sample size for faster processing')
    parser.add_argument('--output', type=str, default='./output',
                       help='Output directory for results')
    parser.add_argument('--cache', type=str, default='./cache',
                       help='Cache directory for embeddings')
    
    # Action flags
    parser.add_argument('--full', action='store_true',
                       help='Run full pipeline (preprocess, train, evaluate, visualize)')
    parser.add_argument('--preprocess', action='store_true',
                       help='Run preprocessing only')
    parser.add_argument('--train', action='store_true',
                       help='Train models only (requires preprocessed data)')
    parser.add_argument('--evaluate', action='store_true',
                       help='Run evaluation only')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations only')
    parser.add_argument('--demo', action='store_true',
                       help='Run quick demo')
    parser.add_argument('--webapp', action='store_true',
                       help='Launch Streamlit web application')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)
    
    print("\n" + "="*60)
    print("💘 AI DATING RECOMMENDATION SYSTEM")
    print("="*60)
    print(f"Data path: {args.data}")
    print(f"Sample size: {args.sample if args.sample else 'Full dataset'}")
    print(f"Output dir: {args.output}")
    
    # Launch webapp if requested
    if args.webapp:
        run_webapp()
        return
    
    # Default to full pipeline if no specific action
    if not any([args.preprocess, args.train, args.evaluate, args.visualize, args.demo]):
        args.full = True
    
    # Run pipeline
    df = None
    bio_embeddings = None
    interest_matrix = None
    realtime_recommender = None
    hybrid_recommender = None
    
    if args.full or args.preprocess:
        df, preprocessor = run_preprocessing(
            args.data, 
            output_path=os.path.join(args.output, 'cleaned_data.csv'),
            sample_size=args.sample
        )
    
    if (args.full or args.train) and df is not None:
        bio_embeddings, interest_matrix, embedder = run_embeddings(df, args.cache)
        realtime_recommender, hybrid_recommender = run_training(df, bio_embeddings, interest_matrix)
        
        # Save models for instant webapp loading
        save_models(df, bio_embeddings, interest_matrix, hybrid_recommender, args.cache)
    
    if (args.full or args.evaluate) and realtime_recommender is not None:
        evaluator = run_evaluation(df, realtime_recommender, hybrid_recommender, output_dir=args.output)
        
        # Save evaluation plots
        from src.visualization import create_evaluation_plots
        create_evaluation_plots(
            evaluator.results,
            save_path=os.path.join(args.output, 'evaluation_results.png')
        )
    
    if (args.full or args.visualize) and df is not None:
        viz_dir = os.path.join(args.output, 'visualizations')
        run_visualizations(df, output_dir=viz_dir)
    
    if args.demo and realtime_recommender is not None:
        run_demo(realtime_recommender, df)
    elif args.demo and df is None:
        print("⚠️ Running demo requires preprocessing first. Adding --preprocess")
        df, _ = run_preprocessing(args.data, sample_size=args.sample or 5000)
        bio_embeddings, interest_matrix, _ = run_embeddings(df, args.cache)
        realtime_recommender, hybrid_recommender = run_training(df, bio_embeddings, interest_matrix)
        run_demo(realtime_recommender, df)
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {args.output}")
    print("To launch the web app, run: python main.py --webapp")
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {args.output}")
    print("To launch the web app, run: python main.py --webapp")


if __name__ == "__main__":
    main()
