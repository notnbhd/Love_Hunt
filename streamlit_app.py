"""
Streamlit Cloud Entry Point
This script handles model initialization and runs the main app
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Check if models exist, if not run precompute
MODEL_DIR = './cache/models'
REQUIRED_FILES = [
    os.path.join(MODEL_DIR, 'processed_df.pkl'),
    os.path.join(MODEL_DIR, 'bio_embeddings.npy'),
    os.path.join(MODEL_DIR, 'hybrid_recommender.pkl')
]

def check_and_setup():
    """Check if models exist and have content"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    for f in REQUIRED_FILES:
        if not os.path.exists(f):
            return False
        # Check if file has content
        if os.path.getsize(f) == 0:
            return False
    return True

if not check_and_setup():
    import streamlit as st
    st.set_page_config(page_title="Love Hunt 💕", page_icon="💕", layout="centered")
    st.title("🔄 First-time Setup")
    st.info("Setting up recommendation models... This may take a few minutes on first run.")
    
    with st.spinner("Running precompute script..."):
        try:
            # Run precompute
            exec(open('precompute.py').read())
            st.success("✅ Setup complete! Please refresh the page.")
            st.rerun()
        except Exception as e:
            st.error(f"Setup failed: {e}")
            st.stop()
else:
    # Run the main app
    exec(open('src/app.py').read())
