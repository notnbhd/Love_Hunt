"""
Data Preprocessing Module for Dating Recommendation System
Handles all 5 data cleaning tasks: missing values, normalization, duplicates, outliers, vectorization
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.impute import SimpleImputer
import re
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Comprehensive data preprocessing class implementing all 5 required tasks:
    1. Handle Missing Values
    2. Data Normalization
    3. Remove Duplicates
    4. Handle Outliers
    5. Vectorization (TF-IDF + embeddings ready)
    """
    
    def __init__(self, data_path: str = None):
        self.data_path = data_path
        self.df = None
        self.df_clean = None
        self.scalers = {}
        self.encoders = {}
        self.vectorizers = {}
        self.preprocessing_report = {}
        
        # Define column categories
        self.numeric_cols = ['age', 'height', 'income']
        self.categorical_cols = ['status', 'sex', 'orientation', 'body_type', 'diet', 
                                  'drinks', 'drugs', 'education', 'ethnicity', 'job',
                                  'offspring', 'pets', 'religion', 'sign', 'smokes']
        self.text_cols = ['essay0', 'essay1', 'essay2', 'essay3', 'essay4', 
                          'essay5', 'essay6', 'essay7', 'essay8', 'essay9']
        
    def load_data(self, sample_size: int = None) -> pd.DataFrame:
        """Load the OkCupid dataset"""
        print("📂 Loading data...")
        self.df = pd.read_csv(self.data_path)
        
        if sample_size and sample_size < len(self.df):
            self.df = self.df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        # Add user_id if not present
        if 'user_id' not in self.df.columns:
            self.df['user_id'] = range(len(self.df))
        
        self.preprocessing_report['original_shape'] = self.df.shape
        print(f"✅ Loaded {len(self.df)} profiles with {len(self.df.columns)} features")
        return self.df
    
    # =========================================================================
    # TASK 2.1: Handle Missing Values
    # =========================================================================
    def handle_missing_values(self) -> pd.DataFrame:
        """
        Handle missing values using appropriate strategies:
        - Numerical: Median imputation (robust to outliers)
        - Categorical: Mode or 'Unknown' category
        - Text: Empty string or placeholder
        """
        print("\n🔧 Task 2.1: Handling Missing Values...")
        
        # Document before state
        missing_before = self.df.isnull().sum()
        missing_pct = (missing_before / len(self.df) * 100).round(2)
        
        self.preprocessing_report['missing_before'] = missing_before[missing_before > 0].to_dict()
        print(f"   Columns with missing values: {len(missing_before[missing_before > 0])}")
        
        df = self.df.copy()
        
        # Handle numeric columns - use median (robust to skewness)
        for col in self.numeric_cols:
            if col in df.columns and df[col].isnull().any():
                # Handle special case for income (-1 means not reported)
                if col == 'income':
                    df[col] = df[col].replace(-1, np.nan)
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"   → {col}: Imputed with median ({median_val:.2f})")
        
        # Handle categorical columns - use mode or 'Unknown'
        for col in self.categorical_cols:
            if col in df.columns and df[col].isnull().any():
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col].fillna(mode_val[0], inplace=True)
                else:
                    df[col].fillna('unknown', inplace=True)
                print(f"   → {col}: Imputed with mode")
        
        # Handle text columns - fill with placeholder
        for col in self.text_cols:
            if col in df.columns:
                df[col] = df[col].fillna('')
                # Clean HTML entities
                df[col] = df[col].apply(lambda x: re.sub(r'&[a-z]+;', ' ', str(x)))
                df[col] = df[col].apply(lambda x: re.sub(r'<[^>]+>', '', str(x)))
        
        # Handle location
        if 'location' in df.columns:
            df['location'].fillna('unknown', inplace=True)
        
        # Document after state
        missing_after = df.isnull().sum()
        self.preprocessing_report['missing_after'] = missing_after[missing_after > 0].to_dict()
        
        print(f"   ✅ After handling: {len(missing_after[missing_after > 0])} columns with missing values")
        
        self.df = df
        return df
    
    # =========================================================================
    # TASK 2.2: Data Normalization
    # =========================================================================
    def normalize_data(self) -> pd.DataFrame:
        """
        Normalize numerical features and encode categorical variables:
        - StandardScaler for age, height (normally distributed)
        - MinMaxScaler for income (skewed distribution)
        - LabelEncoder for ordinal categorical
        - One-hot encoding for nominal categorical with low cardinality
        """
        print("\n🔧 Task 2.2: Data Normalization...")
        
        df = self.df.copy()
        
        # Normalize numerical columns
        # Age - use StandardScaler (approximately normal distribution)
        if 'age' in df.columns:
            self.scalers['age'] = StandardScaler()
            df['age_normalized'] = self.scalers['age'].fit_transform(df[['age']])
            print(f"   → age: StandardScaler applied (mean={df['age'].mean():.1f}, std={df['age'].std():.1f})")
        
        # Height - use StandardScaler
        if 'height' in df.columns:
            self.scalers['height'] = StandardScaler()
            df['height_normalized'] = self.scalers['height'].fit_transform(df[['height']])
            print(f"   → height: StandardScaler applied")
        
        # Income - use MinMaxScaler (highly skewed)
        if 'income' in df.columns:
            self.scalers['income'] = MinMaxScaler()
            df['income_normalized'] = self.scalers['income'].fit_transform(df[['income']])
            print(f"   → income: MinMaxScaler applied")
        
        # Encode categorical columns
        encoded_cols = []
        for col in ['sex', 'orientation', 'status']:
            if col in df.columns:
                self.encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col].astype(str))
                encoded_cols.append(col)
        
        print(f"   → Encoded {len(encoded_cols)} categorical columns with LabelEncoder")
        
        # One-hot encode key categorical features with limited categories
        for col in ['drinks', 'smokes', 'drugs']:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                # Limit to avoid too many columns
                if len(dummies.columns) <= 5:
                    df = pd.concat([df, dummies], axis=1)
                    print(f"   → {col}: One-hot encoded ({len(dummies.columns)} categories)")
        
        # Extract location features
        if 'location' in df.columns:
            df['city'] = df['location'].apply(lambda x: str(x).split(',')[0].strip() if pd.notna(x) else 'unknown')
            df['state'] = df['location'].apply(lambda x: str(x).split(',')[-1].strip() if ',' in str(x) else 'unknown')
        
        self.preprocessing_report['normalized_columns'] = list(self.scalers.keys())
        self.preprocessing_report['encoded_columns'] = list(self.encoders.keys())
        
        self.df = df
        print(f"   ✅ Normalization complete")
        return df
    
    # =========================================================================
    # TASK 2.3: Remove Duplicates
    # =========================================================================
    def remove_duplicates(self) -> pd.DataFrame:
        """
        Remove exact and near-duplicate profiles
        """
        print("\n🔧 Task 2.3: Removing Duplicates...")
        
        df = self.df.copy()
        original_count = len(df)
        
        # Check for exact duplicates (excluding user_id)
        cols_to_check = [c for c in df.columns if c != 'user_id']
        exact_duplicates = df.duplicated(subset=cols_to_check, keep='first').sum()
        
        # Remove exact duplicates
        df = df.drop_duplicates(subset=cols_to_check, keep='first')
        
        # Check for near-duplicates based on key profile features
        key_cols = ['age', 'sex', 'location', 'essay0']
        key_cols = [c for c in key_cols if c in df.columns]
        
        if key_cols:
            near_duplicates = df.duplicated(subset=key_cols, keep='first').sum()
            df = df.drop_duplicates(subset=key_cols, keep='first')
        else:
            near_duplicates = 0
        
        # Reset index
        df = df.reset_index(drop=True)
        
        # Reassign user_ids
        df['user_id'] = range(len(df))
        
        final_count = len(df)
        total_removed = original_count - final_count
        
        self.preprocessing_report['duplicates'] = {
            'exact_duplicates': exact_duplicates,
            'near_duplicates': near_duplicates,
            'total_removed': total_removed,
            'original_count': original_count,
            'final_count': final_count
        }
        
        print(f"   → Exact duplicates found: {exact_duplicates}")
        print(f"   → Near duplicates found: {near_duplicates}")
        print(f"   ✅ Removed {total_removed} duplicates ({original_count} → {final_count} profiles)")
        
        self.df = df
        return df
    
    # =========================================================================
    # TASK 2.4: Handle Outliers
    # =========================================================================
    def handle_outliers(self, method: str = 'iqr') -> pd.DataFrame:
        """
        Detect and handle outliers using IQR method
        """
        print("\n🔧 Task 2.4: Handling Outliers...")
        
        df = self.df.copy()
        outlier_report = {}
        
        for col in self.numeric_cols:
            if col not in df.columns:
                continue
                
            # Calculate IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            num_outliers = outliers_mask.sum()
            
            if num_outliers > 0:
                # Cap outliers (winsorization)
                df[col] = np.clip(df[col], lower_bound, upper_bound)
                
                outlier_report[col] = {
                    'count': num_outliers,
                    'percentage': round(num_outliers / len(df) * 100, 2),
                    'lower_bound': round(lower_bound, 2),
                    'upper_bound': round(upper_bound, 2)
                }
                print(f"   → {col}: {num_outliers} outliers ({outlier_report[col]['percentage']}%) capped")
        
        self.preprocessing_report['outliers'] = outlier_report
        print(f"   ✅ Outliers handled using {method.upper()} method")
        
        self.df = df
        return df
    
    # =========================================================================
    # TASK 2.5: Vectorization
    # =========================================================================
    def create_bio_text(self) -> pd.Series:
        """Combine all essay columns into a single bio text"""
        df = self.df.copy()
        
        # Combine all essays into one bio
        bio_cols = [c for c in self.text_cols if c in df.columns]
        df['bio'] = df[bio_cols].apply(
            lambda x: ' '.join([str(v) for v in x if pd.notna(v) and str(v).strip()]), 
            axis=1
        )
        
        # Clean the bio text
        df['bio'] = df['bio'].apply(self._clean_text)
        
        self.df = df
        return df['bio']
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ''
        
        # Remove HTML tags and entities
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'&[a-z]+;', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Convert to lowercase
        text = text.lower()
        
        return text
    
    def vectorize_text_tfidf(self, max_features: int = 200) -> np.ndarray:
        """
        Create TF-IDF vectors from bio text
        """
        print("\n🔧 Task 2.5: Text Vectorization (TF-IDF)...")
        
        # Ensure bio column exists
        if 'bio' not in self.df.columns:
            self.create_bio_text()
        
        # Create TF-IDF vectorizer
        self.vectorizers['tfidf'] = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.95
        )
        
        # Fit and transform
        tfidf_matrix = self.vectorizers['tfidf'].fit_transform(self.df['bio'])
        
        self.preprocessing_report['tfidf'] = {
            'num_features': tfidf_matrix.shape[1],
            'vocabulary_size': len(self.vectorizers['tfidf'].vocabulary_)
        }
        
        print(f"   → Created TF-IDF matrix: {tfidf_matrix.shape}")
        print(f"   ✅ TF-IDF vectorization complete")
        
        return tfidf_matrix
    
    def extract_interests(self) -> pd.DataFrame:
        """Extract and encode user interests from essays"""
        print("\n🔧 Extracting Interests...")
        
        # Define interest categories
        interest_keywords = {
            'music': ['music', 'concert', 'band', 'guitar', 'piano', 'jazz', 'rock', 'hip hop', 'electronic'],
            'sports': ['sports', 'gym', 'fitness', 'running', 'hiking', 'yoga', 'basketball', 'football', 'soccer'],
            'travel': ['travel', 'adventure', 'explore', 'countries', 'backpack', 'trip', 'vacation'],
            'food': ['food', 'cooking', 'restaurant', 'cuisine', 'chef', 'baking', 'wine'],
            'movies': ['movie', 'film', 'cinema', 'netflix', 'documentary'],
            'reading': ['book', 'reading', 'literature', 'novel', 'author'],
            'gaming': ['game', 'gaming', 'video game', 'playstation', 'xbox', 'nintendo'],
            'art': ['art', 'museum', 'painting', 'photography', 'creative'],
            'outdoors': ['outdoor', 'nature', 'camping', 'beach', 'mountain', 'park'],
            'tech': ['technology', 'programming', 'coding', 'software', 'computer', 'tech']
        }
        
        # Create interest columns
        bio = self.df['bio'].fillna('').str.lower()
        
        for interest, keywords in interest_keywords.items():
            pattern = '|'.join(keywords)
            self.df[f'interest_{interest}'] = bio.str.contains(pattern, regex=True).astype(int)
        
        interest_cols = [c for c in self.df.columns if c.startswith('interest_')]
        print(f"   → Extracted {len(interest_cols)} interest categories")
        print(f"   ✅ Interest extraction complete")
        
        return self.df[interest_cols]
    
    def create_feature_matrix(self) -> np.ndarray:
        """Create combined feature matrix for similarity computation"""
        print("\n🔧 Creating Feature Matrix...")
        
        features = []
        
        # Numerical features (normalized)
        num_features = []
        for col in ['age_normalized', 'height_normalized', 'income_normalized']:
            if col in self.df.columns:
                num_features.append(self.df[col].values.reshape(-1, 1))
        
        if num_features:
            features.append(np.hstack(num_features))
        
        # Encoded categorical features
        cat_features = []
        for col in self.df.columns:
            if col.endswith('_encoded'):
                cat_features.append(self.df[col].values.reshape(-1, 1))
        
        if cat_features:
            features.append(np.hstack(cat_features))
        
        # Interest features
        interest_cols = [c for c in self.df.columns if c.startswith('interest_')]
        if interest_cols:
            features.append(self.df[interest_cols].values)
        
        # Combine all features
        feature_matrix = np.hstack(features) if features else np.array([])
        
        print(f"   → Feature matrix shape: {feature_matrix.shape}")
        
        return feature_matrix
    
    # =========================================================================
    # Main Preprocessing Pipeline
    # =========================================================================
    def preprocess_all(self, sample_size: int = None) -> pd.DataFrame:
        """Run all preprocessing steps"""
        print("\n" + "="*60)
        print("🚀 STARTING DATA PREPROCESSING PIPELINE")
        print("="*60)
        
        # Load data
        self.load_data(sample_size=sample_size)
        
        # Task 2.1: Handle missing values
        self.handle_missing_values()
        
        # Task 2.2: Normalize data
        self.normalize_data()
        
        # Task 2.3: Remove duplicates
        self.remove_duplicates()
        
        # Task 2.4: Handle outliers
        self.handle_outliers()
        
        # Task 2.5: Vectorization
        self.create_bio_text()
        self.extract_interests()
        
        self.df_clean = self.df.copy()
        
        print("\n" + "="*60)
        print("✅ PREPROCESSING COMPLETE")
        print(f"   Final dataset: {len(self.df_clean)} profiles")
        print("="*60 + "\n")
        
        return self.df_clean
    
    def get_report(self) -> dict:
        """Get preprocessing report"""
        return self.preprocessing_report
    
    def save_clean_data(self, output_path: str):
        """Save cleaned dataset"""
        if self.df_clean is not None:
            self.df_clean.to_csv(output_path, index=False)
            print(f"💾 Saved cleaned data to {output_path}")


# Utility functions
def get_age_group(age: int) -> str:
    """Convert age to age group"""
    if age < 25:
        return '18-24'
    elif age < 35:
        return '25-34'
    elif age < 45:
        return '35-44'
    elif age < 55:
        return '45-54'
    else:
        return '55+'


def calculate_compatibility_score(user1: dict, user2: dict) -> float:
    """Calculate basic compatibility between two users"""
    score = 0.0
    
    # Age compatibility (prefer similar age)
    age_diff = abs(user1.get('age', 30) - user2.get('age', 30))
    score += max(0, 1 - age_diff / 20) * 0.2
    
    # Orientation compatibility
    if user1.get('orientation') == user2.get('orientation'):
        score += 0.1
    
    # Location compatibility
    if user1.get('location') == user2.get('location'):
        score += 0.2
    
    return score
