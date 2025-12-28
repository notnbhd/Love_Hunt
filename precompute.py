"""
Pre-compute Script for Dating Recommendation System
Run this ONCE to generate all models and cached data.
After this, the webapp will load instantly.
"""

import os
import sys
import pickle
import time
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Configuration
DATA_PATH = './data/okcupid_profiles.csv'
CACHE_DIR = './cache'
MODEL_DIR = './cache/models'
SAMPLE_SIZE = 20000  # Set to None for full dataset (slower)

def main():
    start_total = time.time()
    
    print("="*60)
    print("🚀 PRE-COMPUTING ALL MODELS (One-time Setup)")
    print("="*60)
    
    # Create directories
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # =========================================================================
    # STEP 1: Data Preprocessing
    # =========================================================================
    print("\n📊 STEP 1: Data Preprocessing...")
    start = time.time()
    
    from src.data_preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor(DATA_PATH)
    df = preprocessor.preprocess_all(sample_size=SAMPLE_SIZE)
    
    # Save processed dataframe
    df_path = os.path.join(MODEL_DIR, 'processed_df.pkl')
    df.to_pickle(df_path)
    print(f"   ✅ Saved processed data: {df_path} ({time.time()-start:.1f}s)")
    
    # =========================================================================
    # STEP 2: Generate Embeddings
    # =========================================================================
    print("\n🧠 STEP 2: Generating Embeddings...")
    start = time.time()
    
    from src.embeddings import AdvancedEmbeddings, InterestEmbeddings
    
    # Bio embeddings (uses caching internally)
    embedder = AdvancedEmbeddings(cache_dir=CACHE_DIR)
    bio_embeddings = embedder.encode_texts(
        df['bio'].fillna('').tolist(),
        batch_size=64,
        show_progress=True
    )
    
    # Save bio embeddings explicitly
    bio_path = os.path.join(MODEL_DIR, 'bio_embeddings.npy')
    np.save(bio_path, bio_embeddings)
    print(f"   ✅ Saved bio embeddings: {bio_path}")
    
    # Interest matrix
    interest_embedder = InterestEmbeddings()
    interest_matrix = interest_embedder.extract_interests(df['bio'].fillna('').tolist())
    
    interest_path = os.path.join(MODEL_DIR, 'interest_matrix.npy')
    np.save(interest_path, interest_matrix)
    print(f"   ✅ Saved interest matrix: {interest_path} ({time.time()-start:.1f}s)")
    
    # =========================================================================
    # STEP 3: Train Recommendation Models
    # =========================================================================
    print("\n🔧 STEP 3: Training Recommendation Models...")
    start = time.time()
    
    from src.recommendation_engine import HybridRecommender, RealTimeRecommender
    
    hybrid = HybridRecommender(content_weight=0.6, collab_weight=0.4)
    hybrid.fit(df, bio_embeddings, interest_matrix)
    
    # Save the trained hybrid recommender
    hybrid_path = os.path.join(MODEL_DIR, 'hybrid_recommender.pkl')
    with open(hybrid_path, 'wb') as f:
        pickle.dump(hybrid, f)
    print(f"   ✅ Saved hybrid recommender: {hybrid_path} ({time.time()-start:.1f}s)")
    
    # =========================================================================
    # STEP 4: Save Metadata
    # =========================================================================
    metadata = {
        'sample_size': SAMPLE_SIZE,
        'n_users': len(df),
        'embedding_dim': bio_embeddings.shape[1],
        'n_interests': interest_matrix.shape[1],
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    meta_path = os.path.join(MODEL_DIR, 'metadata.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    # =========================================================================
    # DONE
    # =========================================================================
    total_time = time.time() - start_total
    
    print("\n" + "="*60)
    print("✅ PRE-COMPUTATION COMPLETE!")
    print("="*60)
    print(f"\n📁 All models saved to: {MODEL_DIR}")
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    print(f"\n🎉 You can now run the webapp instantly using:")
    print(f"   Double-click 'run_webapp.cmd'")
    print(f"   OR run: streamlit run src/app.py")
    

if __name__ == '__main__':
    main()
