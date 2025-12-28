# Love Hunt 💕 - AI Dating Recommendation System

A cutting-edge dating recommendation web app powered by hybrid machine learning algorithms, combining Content-Based Filtering with Collaborative Filtering for intelligent matchmaking.

## 🚀 Features

- **Hybrid Recommendation Engine**: Combines content similarity with collaborative filtering
- **Semantic Bio Matching**: Uses Sentence-BERT embeddings for deep understanding of user bios
- **Real-time Feedback Learning**: Adapts recommendations based on your likes/passes
- **Comprehensive Profiles**: 20+ profile attributes for accurate matching
- **OLED Dark Mode UI**: Beautiful, modern interface optimized for all screens

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **ML/AI**: scikit-learn, sentence-transformers, scipy
- **Data**: pandas, numpy

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/dating-recommendation.git
cd dating-recommendation

# Install dependencies
pip install -r requirements.txt

# Run the precompute script (first time only)
python precompute.py

# Launch the app
streamlit run src/app.py
```

## 🌐 Deployment

This app is deployed on Streamlit Community Cloud.

**Live Demo**: [Coming Soon]

## 📊 How It Works

1. **Content-Based Filtering**: Matches users based on profile similarity (bio, interests, demographics)
2. **Collaborative Filtering**: Learns from user interactions to find patterns
3. **Hybrid Scoring**: `Final_Score = 0.6 × Content_Score + 0.4 × Collab_Score`

## 📁 Project Structure

```
dating/
├── src/
│   ├── app.py                 # Main Streamlit app
│   ├── recommendation_engine.py
│   ├── embeddings.py
│   └── data_preprocessing.py
├── data/
│   └── okcupid_profiles.csv
├── cache/models/              # Pre-trained models
├── requirements.txt
└── README.md
```

## 📝 License

MIT License
