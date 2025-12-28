"""
Dating Recommendation System - Streamlined Web App
Clean, OLED-optimized dark mode interface
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Page config
st.set_page_config(
    page_title="Love Hunt <3",
    page_icon="",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# OLED Dark Mode CSS
st.markdown("""
<style>
    /* Base OLED Dark Theme */
    .stApp {
        background-color: #000000;
    }
    
    [data-testid="stHeader"] {
        background-color: #000000;
    }
    
    [data-testid="stSidebar"] {
        background-color: #000000;
        border-right: 1px solid #1a1a1a;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Typography */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin: 1rem 0 0.5rem 0;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        font-size: 0.95rem;
        color: #666666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Profile Card */
    .profile-card {
        background: #121212;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid #1a1a1a;
    }
    
    .profile-name {
        font-size: 1.8rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 0.3rem;
    }
    
    .profile-meta {
        font-size: 1rem;
        color: #888888;
        margin-bottom: 1rem;
    }
    
    .profile-bio {
        font-size: 1rem;
        color: #cccccc;
        line-height: 1.6;
        margin: 1.5rem 0;
        padding: 1rem;
        background: #0a0a0a;
        border-radius: 12px;
        border-left: 3px solid #FF6B9D;
    }
    
    .match-score {
        display: inline-block;
        background: linear-gradient(135deg, #FF6B9D 0%, #C44569 100%);
        color: #ffffff;
        font-size: 1rem;
        font-weight: 600;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        margin-bottom: 1rem;
    }
    
    /* Profile Details */
    .detail-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.8rem;
        margin: 1rem 0;
    }
    
    .detail-item {
        background: #0a0a0a;
        padding: 0.8rem 1rem;
        border-radius: 10px;
        border: 1px solid #1a1a1a;
    }
    
    .detail-label {
        font-size: 0.75rem;
        color: #666666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .detail-value {
        font-size: 1rem;
        color: #ffffff;
        margin-top: 0.2rem;
        word-wrap: break-word;
        overflow-wrap: break-word;
        white-space: normal;
    }
    
    /* Action Buttons */
    .action-container {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-top: 1.5rem;
        padding: 1rem 0;
    }
    
    /* Primary Button - Orange accent */
    .stButton > button[kind="primary"],
    div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #FF8C42 0%, #E55B13 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 2rem !important;
        font-weight: 600 !important;
    }
    
    /* Secondary buttons */
    .stButton > button {
        background: #121212 !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
        border-radius: 12px !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        background: #1a1a1a !important;
        border-color: #FF6B9D !important;
    }
    
    /* Form inputs */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > div {
        background-color: #121212 !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
        border-radius: 10px !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #FF6B9D !important;
        box-shadow: 0 0 0 1px #FF6B9D !important;
    }
    
    .stNumberInput > div > div > input {
        background-color: #121212 !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }
    
    /* Select box */
    [data-baseweb="select"] {
        background-color: #121212 !important;
    }
    
    [data-baseweb="select"] > div {
        background-color: #121212 !important;
        border-color: #333333 !important;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background: #121212;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #1a1a1a;
    }
    
    .stRadio label {
        color: #ffffff !important;
    }
    
    /* Profile selector */
    .profile-option {
        background: #121212;
        border: 1px solid #1a1a1a;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .profile-option:hover {
        border-color: #FF6B9D;
        background: #1a1a1a;
    }
    
    .profile-option.selected {
        border-color: #FF6B9D;
        background: #1a0a12;
    }
    
    /* Success/Info messages */
    .stSuccess {
        background-color: #0a1a0a !important;
        border: 1px solid #2a5a2a !important;
    }
    
    .stInfo {
        background-color: #0a0a1a !important;
        border: 1px solid #2a2a5a !important;
    }
    
    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 3rem;
        color: #666666;
    }
    
    .empty-state-title {
        font-size: 1.2rem;
        color: #888888;
        margin-bottom: 0.5rem;
    }
    
    /* Loading */
    .loading-text {
        text-align: center;
        color: #FF6B9D;
        font-size: 1.1rem;
        padding: 2rem;
    }
    
    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #1a1a1a;
        margin: 1.5rem 0;
    }
    
    /* Labels */
    .stTextInput label, .stTextArea label, .stSelectbox label, .stNumberInput label {
        color: #888888 !important;
    }
    
    /* Counter */
    .counter {
        text-align: center;
        color: #666666;
        font-size: 0.85rem;
        margin-top: 1rem;
    }
    
    /* Welcome screen */
    .welcome-box {
        background: #121212;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        border: 1px solid #1a1a1a;
        margin: 1rem 0;
    }
    
    .welcome-title {
        font-size: 1.5rem;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .welcome-desc {
        color: #888888;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# Path for persisting interactions
INTERACTIONS_FILE = './cache/user_interactions.csv'

@st.cache_resource
def load_system():
    """Load recommendation system"""
    import pickle
    MODEL_DIR = './cache/models'
    
    precomputed = [
        os.path.join(MODEL_DIR, 'processed_df.pkl'),
        os.path.join(MODEL_DIR, 'hybrid_recommender.pkl'),
        os.path.join(MODEL_DIR, 'bio_embeddings.npy')
    ]
    
    if all(os.path.exists(f) for f in precomputed):
        try:
            from recommendation_engine import RealTimeRecommender
            from embeddings import AdvancedEmbeddings
            
            df = pd.read_pickle(os.path.join(MODEL_DIR, 'processed_df.pkl'))
            
            with open(os.path.join(MODEL_DIR, 'hybrid_recommender.pkl'), 'rb') as f:
                hybrid = pickle.load(f)
            
            recommender = RealTimeRecommender(hybrid)
            embedder = AdvancedEmbeddings(cache_dir='./cache')
            
            # Load persisted interactions if they exist
            if os.path.exists(INTERACTIONS_FILE):
                try:
                    recommender.import_interactions(INTERACTIONS_FILE)
                except Exception as e:
                    print(f"Warning: Could not load interactions: {e}")
            
            return df, recommender, embedder
        except Exception as e:
            st.error(f"Failed to load: {e}")
            return None, None, None
    else:
        st.error("Models not found. Run precompute.py first.")
        return None, None, None


def init_session():
    """Initialize session state"""
    defaults = {
        'stage': 'profile_select',
        'profile_type': None,
        'selected_user_id': None,
        'new_profile': None,
        'current_rec_index': 0,
        'recommendations': [],
        'interactions': [],
        'system_loaded': False,
        # Real-time feedback tracking
        'liked_profiles': [],      # Profiles the user liked/superliked
        'passed_profiles': [],     # Profiles the user passed on
        'custom_user_id': None,    # Temporary ID for custom profiles
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_recommendations(df, recommender, embedder, user_id=None, custom_profile=None, num_recs=None):
    """Get recommendations for user"""
    from sklearn.metrics.pairwise import cosine_similarity
    
    if custom_profile and embedder:
        custom_bio = custom_profile.get('bio', '')
        custom_embedding = embedder.encode_texts([custom_bio], show_progress=False, use_cache=False)
        
        bio_embeddings = recommender.recommender.content_recommender.bio_embeddings
        
        if bio_embeddings is not None:
            similarities = cosine_similarity(custom_embedding, bio_embeddings)[0]
            
            age_filter = np.abs(df['age'].values - custom_profile.get('age', 30)) <= 15
            
            user_sex = custom_profile.get('sex', 'm')
            user_orientation = custom_profile.get('orientation', 'straight')
            
            if user_orientation == 'straight':
                sex_filter = df['sex'].values == ('f' if user_sex == 'm' else 'm')
            elif user_orientation == 'gay':
                sex_filter = df['sex'].values == user_sex
            else:
                sex_filter = np.ones(len(df), dtype=bool)
            
            combined = age_filter & sex_filter
            filtered_sim = similarities.copy()
            filtered_sim[~combined] = -1
            
            # Get all matching indices (unlimited if num_recs is None)
            sorted_indices = np.argsort(filtered_sim)[::-1]
            top_indices = sorted_indices if num_recs is None else sorted_indices[:num_recs]
            
            recs = []
            for idx in top_indices:
                if filtered_sim[idx] < 0:
                    continue
                user_data = df.iloc[idx]
                rec_dict = {
                    'user_id': int(user_data.get('user_id', idx)),
                    'age': int(user_data.get('age', 0)),
                    'sex': user_data.get('sex', '?'),
                    'location': user_data.get('location', 'Unknown'),
                    'orientation': user_data.get('orientation', 'Unknown'),
                    'status': user_data.get('status', 'Unknown'),
                    'score': float(filtered_sim[idx] * 100),
                    'bio': str(user_data.get('bio', ''))[:500],
                    'education': user_data.get('education', 'Unknown'),
                    'body_type': user_data.get('body_type', 'Unknown'),
                    'drinks': user_data.get('drinks', 'Unknown'),
                    'smokes': user_data.get('smokes', 'Unknown'),
                    'diet': user_data.get('diet', 'Unknown'),
                    'drugs': user_data.get('drugs', 'Unknown'),
                    'pets': user_data.get('pets', 'Unknown'),
                    'religion': user_data.get('religion', 'Unknown'),
                    'sign': user_data.get('sign', 'Unknown'),
                    'job': user_data.get('job', 'Unknown'),
                    'income': user_data.get('income', 'Unknown'),
                    'offspring': user_data.get('offspring', 'Unknown'),
                }
                # Add interest fields
                for col in user_data.index:
                    if col.startswith('interest_'):
                        rec_dict[col] = user_data.get(col, False)
                recs.append(rec_dict)
            return recs
    else:
        # For unlimited matching, use a large top_k value
        effective_num_recs = num_recs if num_recs is not None else len(df)
        result = recommender.get_recommendations(user_id=user_id, top_k=effective_num_recs)
        recs = []
        for rec in result['recommendations']:
            user_data = df[df['user_id'] == rec['user_id']].iloc[0] if 'user_id' in df.columns else df.iloc[rec['user_id']]
            rec_dict = {
                'user_id': rec['user_id'],
                'age': int(user_data.get('age', 0)),
                'sex': user_data.get('sex', '?'),
                'location': user_data.get('location', 'Unknown'),
                'orientation': user_data.get('orientation', 'Unknown'),
                'status': user_data.get('status', 'Unknown'),
                'score': rec['hybrid_score'],
                'bio': str(user_data.get('bio', ''))[:500],
                'education': user_data.get('education', 'Unknown'),
                'body_type': user_data.get('body_type', 'Unknown'),
                'drinks': user_data.get('drinks', 'Unknown'),
                'smokes': user_data.get('smokes', 'Unknown'),
                'diet': user_data.get('diet', 'Unknown'),
                'drugs': user_data.get('drugs', 'Unknown'),
                'pets': user_data.get('pets', 'Unknown'),
                'religion': user_data.get('religion', 'Unknown'),
                'sign': user_data.get('sign', 'Unknown'),
                'job': user_data.get('job', 'Unknown'),
                'income': user_data.get('income', 'Unknown'),
                'offspring': user_data.get('offspring', 'Unknown'),
            }
            # Add interest fields
            for col in user_data.index:
                if col.startswith('interest_'):
                    rec_dict[col] = user_data.get(col, False)
            recs.append(rec_dict)
        return recs
    
    return []


def render_profile_select(df):
    """Profile selection screen"""
    st.markdown('<h1 class="main-title">Love Hunt <3</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Find your perfect match</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="welcome-box">
            <div class="welcome-title">Choose Existing</div>
            <div class="welcome-desc">Browse from our database</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Select Profile", key="btn_existing", use_container_width=True):
            st.session_state['profile_type'] = 'existing'
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="welcome-box">
            <div class="welcome-title">Create New</div>
            <div class="welcome-desc">Make your own profile</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Create Profile", key="btn_new", use_container_width=True):
            st.session_state['profile_type'] = 'new'
            st.rerun()
    
    if st.session_state.get('profile_type') == 'existing':
        st.markdown("---")
        st.markdown("### Select a Profile")
        
        options = []
        for idx, row in df.head(100).iterrows():
            sex_display = "M" if row.get('sex') == 'm' else "F" if row.get('sex') == 'f' else "?"
            loc = str(row.get('location', 'Unknown'))[:25]
            options.append(f"{sex_display}, {int(row.get('age', 0))} - {loc}")
        
        selected = st.selectbox("Choose profile:", options, key="profile_dropdown")
        selected_idx = options.index(selected)
        selected_user = df.iloc[selected_idx]
        
        st.markdown('<div class="profile-card">', unsafe_allow_html=True)
        
        sex_display = "Male" if selected_user.get('sex') == 'm' else "Female" if selected_user.get('sex') == 'f' else "Other"
        st.markdown(f'<div class="profile-name">{sex_display}, {int(selected_user.get("age", 0))}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="profile-meta">{selected_user.get("location", "Unknown")}</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="detail-grid">
            <div class="detail-item">
                <div class="detail-label">Orientation</div>
                <div class="detail-value">{str(selected_user.get('orientation', 'Unknown')).title()}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Status</div>
                <div class="detail-value">{str(selected_user.get('status', 'Unknown')).title()}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Body Type</div>
                <div class="detail-value">{str(selected_user.get('body_type', 'Unknown')).title()}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Education</div>
                <div class="detail-value">{str(selected_user.get('education', 'Unknown')).title()}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Drinks</div>
                <div class="detail-value">{str(selected_user.get('drinks', 'Unknown')).title()}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Smokes</div>
                <div class="detail-value">{str(selected_user.get('smokes', 'Unknown')).title()}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show interests if available
        interests = []
        for col in selected_user.index:
            if col.startswith('interest_') and selected_user.get(col):
                interest_name = col.replace('interest_', '').replace('_', ' ').title()
                interests.append(interest_name)
        if interests:
            st.markdown(f'<div class="detail-label" style="margin-top:1rem;">Interests</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="detail-value">{", ".join(interests)}</div>', unsafe_allow_html=True)
        
        bio = str(selected_user.get('bio', ''))[:300]
        if bio:
            st.markdown(f'<div class="profile-bio">{bio}...</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("Confirm Selection", type="primary", use_container_width=True):
            st.session_state['selected_user_id'] = int(selected_user['user_id'])
            st.session_state['stage'] = 'profile_confirm'
            st.rerun()
    
    elif st.session_state.get('profile_type') == 'new':
        st.markdown("---")
        st.markdown("### Create Your Profile")
        
        # Basic Information
        st.markdown("#### Basic Information")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            age = st.number_input("Age", min_value=18, max_value=100, value=25)
            sex = st.selectbox("Gender", ["m", "f"], format_func=lambda x: "Male" if x == "m" else "Female")
        
        with c2:
            orientation = st.selectbox("Orientation", ["straight", "gay", "bisexual"])
            status = st.selectbox("Relationship Status", ["single", "available", "seeing someone", "married", "unknown"])
        
        with c3:
            body_type = st.selectbox("Body Type", ["average", "fit", "athletic", "thin", "curvy", "full figured", "a little extra", "overweight", "jacked", "rather not say"])
            education = st.selectbox("Education", ["graduated from college/university", "working on college/university", "graduated from masters program", "working on masters program", "graduated from ph.d program", "working on ph.d program", "graduated from high school", "working on high school", "dropped out of high school", "dropped out of college/university", "graduated from two-year college", "working on two-year college", "graduated from law school", "working on law school", "graduated from med school", "working on med school", "rather not say"])
        
        location = st.text_input("Location", "san francisco, california")
        
        # Lifestyle
        st.markdown("#### Lifestyle")
        l1, l2, l3, l4 = st.columns(4)
        
        with l1:
            drinks = st.selectbox("Drinks", ["socially", "often", "rarely", "not at all", "very often", "desperately"])
        
        with l2:
            smokes = st.selectbox("Smokes", ["no", "sometimes", "when drinking", "yes", "trying to quit"])
        
        with l3:
            drugs = st.selectbox("Drugs", ["never", "sometimes", "often"])
        
        with l4:
            diet = st.selectbox("Diet", ["anything", "vegetarian", "vegan", "kosher", "halal", "other"])
        
        # Personal Details
        st.markdown("#### Personal Details")
        p1, p2, p3 = st.columns(3)
        
        with p1:
            religion = st.selectbox("Religion", ["agnosticism", "atheism", "christianity", "catholicism", "judaism", "buddhism", "hinduism", "islam", "other", "rather not say"])
            pets = st.selectbox("Pets", ["has dogs", "has cats", "has other pets", "likes dogs", "likes cats", "dislikes dogs", "dislikes cats", "no pets"])
        
        with p2:
            sign = st.selectbox("Zodiac Sign", ["aries", "taurus", "gemini", "cancer", "leo", "virgo", "libra", "scorpio", "sagittarius", "capricorn", "aquarius", "pisces", "rather not say"])
            offspring = st.selectbox("Children", ["doesn't have kids", "has a kid", "has kids", "doesn't want kids", "might want kids", "wants kids"])
        
        with p3:
            job = st.text_input("Job/Occupation", "")
            income = st.selectbox("Income", ["rather not say", "less than $20,000", "$20,000-$40,000", "$40,000-$60,000", "$60,000-$80,000", "$80,000-$100,000", "$100,000-$150,000", "$150,000-$250,000", "$250,000+"])
        
        # Interests
        st.markdown("#### Interests")
        st.markdown("<small style='color: #888;'>Select all that apply</small>", unsafe_allow_html=True)
        
        interest_options = ["music", "sports", "travel", "food", "movies", "reading", "gaming", "art", "outdoors", "tech", "fitness", "photography", "cooking", "hiking", "dancing", "yoga", "meditation", "writing", "fashion", "animals"]
        
        # Display interests in a grid
        int_cols = st.columns(5)
        selected_interests = {}
        for i, interest in enumerate(interest_options):
            with int_cols[i % 5]:
                selected_interests[f"interest_{interest}"] = st.checkbox(interest.title(), key=f"int_{interest}")
        
        # Bio
        st.markdown("#### About You")
        bio = st.text_area("Bio", "Tell us about yourself, your hobbies, what you're looking for...", height=150)
        
        if st.button("Confirm Profile", type="primary", use_container_width=True):
            # Generate a temporary user ID for custom profiles (negative to avoid conflicts)
            import random
            st.session_state['custom_user_id'] = -random.randint(100000, 999999)
            
            profile_data = {
                'age': age,
                'sex': sex,
                'orientation': orientation,
                'location': location,
                'status': status,
                'body_type': body_type,
                'education': education,
                'drinks': drinks,
                'smokes': smokes,
                'drugs': drugs,
                'diet': diet,
                'religion': religion,
                'pets': pets,
                'sign': sign,
                'offspring': offspring,
                'job': job if job else 'Unknown',
                'income': income,
                'bio': bio
            }
            # Add interests
            profile_data.update(selected_interests)
            
            st.session_state['new_profile'] = profile_data
            st.session_state['stage'] = 'profile_confirm'
            st.rerun()


def render_profile_confirm(df):
    """Profile confirmation screen"""
    st.markdown('<h1 class="main-title">Confirm Profile</h1>', unsafe_allow_html=True)
    
    is_new = st.session_state.get('new_profile') is not None
    
    if is_new:
        profile = st.session_state['new_profile']
    else:
        user_id = st.session_state['selected_user_id']
        profile_row = df[df['user_id'] == user_id].iloc[0]
        profile = {
            'age': int(profile_row.get('age', 0)),
            'sex': profile_row.get('sex', '?'),
            'orientation': profile_row.get('orientation', 'Unknown'),
            'location': profile_row.get('location', 'Unknown'),
            'status': profile_row.get('status', 'Unknown'),
            'bio': str(profile_row.get('bio', ''))[:400],
            'education': profile_row.get('education', 'Unknown'),
            'body_type': profile_row.get('body_type', 'Unknown'),
            'drinks': profile_row.get('drinks', 'Unknown'),
            'smokes': profile_row.get('smokes', 'Unknown'),
            'diet': profile_row.get('diet', 'Unknown'),
            'drugs': profile_row.get('drugs', 'Unknown'),
            'pets': profile_row.get('pets', 'Unknown'),
            'religion': profile_row.get('religion', 'Unknown'),
            'sign': profile_row.get('sign', 'Unknown'),
            'job': profile_row.get('job', 'Unknown'),
            'income': profile_row.get('income', 'Unknown'),
            'offspring': profile_row.get('offspring', 'Unknown'),
        }
        # Add interest fields
        for col in profile_row.index:
            if col.startswith('interest_'):
                profile[col] = profile_row.get(col, False)
    
    st.markdown('<div class="profile-card">', unsafe_allow_html=True)
    
    sex_display = "Male" if profile['sex'] == 'm' else "Female" if profile['sex'] == 'f' else "Other"
    st.markdown(f'<div class="profile-name">{sex_display}, {profile["age"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="profile-meta">{profile["location"]}</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="detail-grid">
        <div class="detail-item">
            <div class="detail-label">Orientation</div>
            <div class="detail-value">{str(profile.get('orientation', 'Unknown')).title()}</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Status</div>
            <div class="detail-value">{str(profile.get('status', 'Unknown')).title()}</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Body Type</div>
            <div class="detail-value">{str(profile.get('body_type', 'Unknown')).title()}</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Education</div>
            <div class="detail-value">{str(profile.get('education', 'Unknown')).title()}</div>
        </div>
    </div>
    
    <div class="detail-grid" style="margin-top: 0.5rem;">
        <div class="detail-item">
            <div class="detail-label">Drinks</div>
            <div class="detail-value">{str(profile.get('drinks', 'Unknown')).title()}</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Smokes</div>
            <div class="detail-value">{str(profile.get('smokes', 'Unknown')).title()}</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Diet</div>
            <div class="detail-value">{str(profile.get('diet', 'Unknown')).title()}</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Drugs</div>
            <div class="detail-value">{str(profile.get('drugs', 'Unknown')).title()}</div>
        </div>
    </div>
    
    <div class="detail-grid" style="margin-top: 0.5rem;">
        <div class="detail-item">
            <div class="detail-label">Religion</div>
            <div class="detail-value">{str(profile.get('religion', 'Unknown')).title()}</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Zodiac Sign</div>
            <div class="detail-value">{str(profile.get('sign', 'Unknown')).title()}</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Pets</div>
            <div class="detail-value">{str(profile.get('pets', 'Unknown')).title()}</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Children</div>
            <div class="detail-value">{str(profile.get('offspring', 'Unknown')).title()}</div>
        </div>
    </div>
    
    <div class="detail-grid" style="margin-top: 0.5rem;">
        <div class="detail-item">
            <div class="detail-label">Job</div>
            <div class="detail-value">{str(profile.get('job', 'Unknown')).title()}</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Income</div>
            <div class="detail-value">{str(profile.get('income', 'Unknown'))}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show interests if available
    interests = []
    for k in profile.keys():
        if k.startswith('interest_') and profile[k]:
            interest_name = k.replace('interest_', '').replace('_', ' ').title()
            interests.append(interest_name)
    if interests:
        st.markdown(f'<div class="detail-label" style="margin-top:1rem;">Interests</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="detail-value">{", ".join(interests)}</div>', unsafe_allow_html=True)
    
    if profile.get('bio'):
        st.markdown(f'<div class="profile-bio">{profile["bio"]}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Back", use_container_width=True):
            st.session_state['stage'] = 'profile_select'
            st.session_state['profile_type'] = None
            st.session_state['new_profile'] = None
            st.session_state['selected_user_id'] = None
            st.rerun()
    
    with col2:
        if st.button("Start Matching", type="primary", use_container_width=True):
            st.session_state['stage'] = 'recommendations'
            st.session_state['current_rec_index'] = 0
            st.session_state['recommendations'] = []
            st.rerun()


def render_recommendations(df, recommender, embedder):
    """Recommendations screen - one profile at a time"""
    
    if not st.session_state.get('recommendations'):
        with st.spinner("Finding matches..."):
            if st.session_state.get('new_profile'):
                recs = get_recommendations(
                    df, recommender, embedder,
                    custom_profile=st.session_state['new_profile']
                )
            else:
                recs = get_recommendations(
                    df, recommender, embedder,
                    user_id=st.session_state['selected_user_id']
                )
            st.session_state['recommendations'] = recs
    
    recs = st.session_state['recommendations']
    idx = st.session_state['current_rec_index']
    
    st.markdown('<h1 class="main-title">People</h1>', unsafe_allow_html=True)
    
    if not recs or idx >= len(recs):
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-title">No more profiles</div>
            <p>You've seen all available matches</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Start Over", type="primary", use_container_width=True):
            st.session_state['stage'] = 'profile_select'
            st.session_state['profile_type'] = None
            st.session_state['new_profile'] = None
            st.session_state['selected_user_id'] = None
            st.session_state['current_rec_index'] = 0
            st.session_state['recommendations'] = []
            st.session_state['liked_profiles'] = []
            st.session_state['passed_profiles'] = []
            st.session_state['custom_user_id'] = None
            st.rerun()
        return
    
    profile = recs[idx]
    
    st.markdown('<div class="profile-card">', unsafe_allow_html=True)
    
    st.markdown(f'<span class="match-score">{profile["score"]:.0f}% Match</span>', unsafe_allow_html=True)
    
    sex_display = "Male" if profile['sex'] == 'm' else "Female" if profile['sex'] == 'f' else "Other"
    st.markdown(f'<div class="profile-name">{sex_display}, {profile["age"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="profile-meta">{profile["location"]}</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="detail-grid">
        <div class="detail-item">
            <div class="detail-label">Orientation</div>
            <div class="detail-value">{str(profile.get('orientation', 'Unknown')).title()}</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Status</div>
            <div class="detail-value">{str(profile.get('status', 'Unknown')).title()}</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Body Type</div>
            <div class="detail-value">{str(profile.get('body_type', 'Unknown')).title()}</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Education</div>
            <div class="detail-value">{str(profile.get('education', 'Unknown')).title()}</div>
        </div>
    </div>
    
    <div class="detail-grid" style="margin-top: 0.5rem;">
        <div class="detail-item">
            <div class="detail-label">Drinks</div>
            <div class="detail-value">{str(profile.get('drinks', 'Unknown')).title()}</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Smokes</div>
            <div class="detail-value">{str(profile.get('smokes', 'Unknown')).title()}</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Diet</div>
            <div class="detail-value">{str(profile.get('diet', 'Unknown')).title()}</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Drugs</div>
            <div class="detail-value">{str(profile.get('drugs', 'Unknown')).title()}</div>
        </div>
    </div>
    
    <div class="detail-grid" style="margin-top: 0.5rem;">
        <div class="detail-item">
            <div class="detail-label">Religion</div>
            <div class="detail-value">{str(profile.get('religion', 'Unknown')).title()}</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Zodiac Sign</div>
            <div class="detail-value">{str(profile.get('sign', 'Unknown')).title()}</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Pets</div>
            <div class="detail-value">{str(profile.get('pets', 'Unknown')).title()}</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Children</div>
            <div class="detail-value">{str(profile.get('offspring', 'Unknown')).title()}</div>
        </div>
    </div>
    
    <div class="detail-grid" style="margin-top: 0.5rem;">
        <div class="detail-item">
            <div class="detail-label">Job</div>
            <div class="detail-value">{str(profile.get('job', 'Unknown')).title()}</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Income</div>
            <div class="detail-value">{str(profile.get('income', 'Unknown'))}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Show interests if available
    interests = []
    for k in profile.keys():
        if k.startswith('interest_') and profile[k]:
            interest_name = k.replace('interest_', '').replace('_', ' ').title()
            if profile[k]:
                interests.append(interest_name)
    if interests:
        st.markdown(f'<div class="detail-label" style="margin-top:1rem;">Interests</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="detail-value">{", ".join(interests)}</div>', unsafe_allow_html=True)
    
    if profile.get('bio'):
        bio_text = profile['bio'][:400] + "..." if len(profile['bio']) > 400 else profile['bio']
        st.markdown(f'<div class="profile-bio">{bio_text}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    def next_profile(action):
        # Store in session state for UI tracking
        st.session_state['interactions'].append({
            'user_id': profile['user_id'],
            'action': action,
            'timestamp': datetime.now().isoformat()
        })
        
        # Track liked/passed profiles for real-time re-ranking
        if action in ['like', 'superlike']:
            st.session_state['liked_profiles'].append(profile)
        elif action == 'pass':
            st.session_state['passed_profiles'].append(profile)
        
        # Get the user ID - works for both existing and custom profiles
        current_user_id = st.session_state.get('selected_user_id')
        if current_user_id is None:
            current_user_id = st.session_state.get('custom_user_id')
        
        # Record interaction with RealTimeRecommender for feedback learning
        if current_user_id is not None:
            recommender.record_interaction(
                user_id=current_user_id,
                target_id=profile['user_id'],
                action=action
            )
            
            # Auto-save interactions to disk for persistence
            try:
                recommender.export_interactions(INTERACTIONS_FILE)
            except Exception as e:
                pass  # Silently handle save errors to not disrupt UX
        
        # Real-time re-ranking: boost similar profiles to liked ones, demote similar to passed
        current_idx = st.session_state['current_rec_index']
        remaining_recs = st.session_state['recommendations'][current_idx + 1:]
        
        if remaining_recs and (action in ['like', 'superlike'] or action == 'pass'):
            # Adjust scores based on feedback
            liked_profiles = st.session_state.get('liked_profiles', [])
            passed_profiles = st.session_state.get('passed_profiles', [])
            
            for rec in remaining_recs:
                # Use original score if available, otherwise current score
                if '_original_score' not in rec:
                    rec['_original_score'] = rec.get('score', 50)
                
                boost = 0
                
                # Boost profiles similar to liked ones (only last liked profile to avoid accumulation)
                if liked_profiles:
                    liked = liked_profiles[-1]  # Only use most recent like
                    # Check common interests
                    liked_interests = set(k for k in liked.keys() if k.startswith('interest_') and liked.get(k))
                    rec_interests = set(k for k in rec.keys() if k.startswith('interest_') and rec.get(k))
                    common = len(liked_interests & rec_interests)
                    if common > 0:
                        boost += min(common * 1.5, 8)  # Cap interest boost at 8
                    
                    # Similar age bonus
                    if abs(liked.get('age', 0) - rec.get('age', 0)) <= 5:
                        boost += 2
                    
                    # Same location bonus
                    if liked.get('location', '').lower() == rec.get('location', '').lower():
                        boost += 3
                
                # Penalize profiles similar to passed ones (only last passed)
                if passed_profiles:
                    passed = passed_profiles[-1]  # Only use most recent pass
                    passed_interests = set(k for k in passed.keys() if k.startswith('interest_') and passed.get(k))
                    rec_interests = set(k for k in rec.keys() if k.startswith('interest_') and rec.get(k))
                    common = len(passed_interests & rec_interests)
                    if common > 2:  # Only penalize if very similar
                        boost -= min(common * 0.5, 5)  # Cap penalty at 5
                
                # Apply boost to ORIGINAL score and cap at 100
                rec['score'] = min(100, max(0, rec['_original_score'] + boost))
            
            # Re-sort remaining recommendations by adjusted score
            remaining_recs.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            # Update the recommendations list
            st.session_state['recommendations'] = (
                st.session_state['recommendations'][:current_idx + 1] + remaining_recs
            )
        
        st.session_state['current_rec_index'] += 1
    
    with col1:
        if st.button("PASS", key="btn_pass", use_container_width=True):
            next_profile('pass')
            st.rerun()
    
    with col2:
        if st.button("LIKE", key="btn_like", use_container_width=True, type="primary"):
            next_profile('like')
            st.rerun()
    
    with col3:
        if st.button("SUPERLIKE", key="btn_super", use_container_width=True):
            next_profile('superlike')
            st.rerun()
    
    st.markdown(f'<div class="counter">{idx + 1} of {len(recs)}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    if st.button("Change Profile", use_container_width=True):
        st.session_state['stage'] = 'profile_select'
        st.session_state['profile_type'] = None
        st.session_state['new_profile'] = None
        st.session_state['selected_user_id'] = None
        st.session_state['current_rec_index'] = 0
        st.session_state['recommendations'] = []
        st.session_state['liked_profiles'] = []
        st.session_state['passed_profiles'] = []
        st.session_state['custom_user_id'] = None
        st.rerun()


def main():
    """Main entry point"""
    init_session()
    
    if not st.session_state.get('system_loaded'):
        df, recommender, embedder = load_system()
        st.session_state['df'] = df
        st.session_state['recommender'] = recommender
        st.session_state['embedder'] = embedder
        st.session_state['system_loaded'] = True
    else:
        df = st.session_state.get('df')
        recommender = st.session_state.get('recommender')
        embedder = st.session_state.get('embedder')
    
    if df is None:
        st.error("Failed to load system. Please run precompute.py first.")
        return
    
    stage = st.session_state.get('stage', 'profile_select')
    
    if stage == 'profile_select':
        render_profile_select(df)
    elif stage == 'profile_confirm':
        render_profile_confirm(df)
    elif stage == 'recommendations':
        render_recommendations(df, recommender, embedder)


if __name__ == "__main__":
    main()
