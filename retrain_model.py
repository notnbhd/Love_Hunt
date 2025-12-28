import pickle
import os
import tempfile
import shutil
from src.recommendation_engine import RealTimeRecommender

MODEL_PATH = './cache/models/hybrid_recommender.pkl'

# Load the recommender
print(f"Loading model from {MODEL_PATH}...")
with open(MODEL_PATH, 'rb') as f:
    hybrid = pickle.load(f)

recommender = RealTimeRecommender(hybrid)

# Load your interactions
interactions_file = './cache/user_interactions.csv'
if os.path.exists(interactions_file):
    recommender.import_interactions(interactions_file)
else:
    print(f"⚠️ No interactions file found at {interactions_file}")

# Check if enough interactions
print(f"Total interactions: {recommender.get_interaction_count()}")

# Retrain with real user data
if recommender.can_enable_collaborative():
    success = recommender.retrain_with_interactions()
    
    if success:
        print("✅ Model retrained with real user feedback!")
        
        # Save the updated model SAFELY using temp file
        # This prevents corruption if the save fails
        temp_fd, temp_path = tempfile.mkstemp(suffix='.pkl')
        try:
            with os.fdopen(temp_fd, 'wb') as f:
                pickle.dump(recommender.recommender, f)
            
            # Atomic replace - only after successful write
            shutil.move(temp_path, MODEL_PATH)
            print(f"✅ Model saved to {MODEL_PATH}")
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise
    else:
        print("❌ Retraining failed")
else:
    print("❌ Not enough interactions yet to enable collaborative filtering.")
    print(f"   Need at least 1000 interactions, currently have: {recommender.get_interaction_count()}")
