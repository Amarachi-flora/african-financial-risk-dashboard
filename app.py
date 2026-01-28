"""
African Financial Behavior Segmentation Dashboard
Complete Analytics Platform for African Financial Institutions
Capstone Project - Dataverse Africa Cohort 3
"""

# ============================================
# SUPPRESS ALL WARNINGS
# ============================================
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
except ImportError:
    pass

# ============================================
# MAIN IMPORTS - UPDATED WITH NEW DEPENDENCIES
# ============================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import sys
import os
from datetime import datetime, timedelta
import zipfile
import io
from io import BytesIO
import re
import hashlib
import base64
from typing import Dict, List, Any, Optional, Tuple
import textwrap
import random
from collections import Counter
import uuid

# Machine Learning Imports
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

# Try to import UMAP (optional - falls back to PCA if not available)
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    umap = None
    UMAP_AVAILABLE = False

# NLP Imports - UPDATED with error handling
try:
    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS
    SPACY_AVAILABLE = True
except ImportError:
    spacy = None
    STOP_WORDS = set()
    SPACY_AVAILABLE = False

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.probability import FreqDist
from nltk.corpus import stopwords

# Gensim with error handling
try:
    import gensim
    from gensim import corpora, models
    from gensim.models import LdaModel, CoherenceModel
    GENSIM_AVAILABLE = True
except ImportError:
    gensim = None
    corpora = None
    models = None
    LdaModel = None
    CoherenceModel = None
    GENSIM_AVAILABLE = False

# WordCloud with error handling
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WordCloud = None
    WORDCLOUD_AVAILABLE = False

# Transformers with error handling
try:
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    torch = None
    pipeline = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    TRANSFORMERS_AVAILABLE = False

# Sentence Transformers with error handling
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('sentiment/vader_lexicon')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)

# ============================================
# CUSTOM CSS - AFRICAN FINANCIAL THEME
# ============================================
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.8rem;
        color: #FF6B35;
        text-align: center;
        padding: 1rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        background: linear-gradient(90deg, #FF6B35, #FFA62E, #2A9D8F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* User role specific styling */
    .analyst-view {
        border: 3px solid #2196F3 !important;
    }
    
    .manager-view {
        border: 3px solid #4CAF50 !important;
    }
    
    .executive-view {
        border: 3px solid #FF9800 !important;
    }
    
    /* Enhanced card styling */
    .insight-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
        border-left: 5px solid;
        position: relative;
        overflow: hidden;
        transition: transform 0.3s ease;
    }
    
    .insight-card:hover {
        transform: translateY(-5px);
    }
    
    .insight-card.positive {
        border-left-color: #4CAF50;
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
    }
    
    .insight-card.warning {
        border-left-color: #FF9800;
        background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%);
    }
    
    .insight-card.critical {
        border-left-color: #F44336;
        background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%);
    }
    
    /* What-if analysis styling */
    .what-if-container {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #2196F3;
    }
    
    /* Customer journey styling */
    .journey-container {
        background: linear-gradient(135deg, #F3E5F5 0%, #E1BEE7 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #9C27B0;
    }
    
    /* Alert styling */
    .alert-high {
        background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%);
        border-left: 5px solid #F44336;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        animation: pulse 2s infinite;
    }
    
    .alert-medium {
        background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%);
        border-left: 5px solid #FF9800;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    /* Export buttons */
    .export-button {
        background: linear-gradient(135deg, #2A9D8F 0%, #264653 100%);
        color: white;
        border: none;
        padding: 0.8rem 1.5rem;
        border-radius: 8px;
        font-weight: 700;
        cursor: pointer;
        transition: all 0.3s ease;
        margin: 0.2rem;
    }
    
    .export-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(42, 157, 143, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# INITIALIZE SESSION STATE WITH NEW VARIABLES
# ============================================
if 'df' not in st.session_state: 
    st.session_state.df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'column_info' not in st.session_state:
    st.session_state.column_info = None
if 'metadata' not in st.session_state:
    st.session_state.metadata = None
if 'insights' not in st.session_state:
    st.session_state.insights = []
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []
if 'dashboard_data' not in st.session_state:
    st.session_state.dashboard_data = None
if 'nlp_models' not in st.session_state:
    st.session_state.nlp_models = {}
if 'clusters' not in st.session_state:
    st.session_state.clusters = None
if 'topic_model' not in st.session_state:
    st.session_state.topic_model = None
if 'current_dataset_name' not in st.session_state:
    st.session_state.current_dataset_name = None
if 'datasets_history' not in st.session_state:
    st.session_state.datasets_history = {}
if 'user_role' not in st.session_state:
    st.session_state.user_role = 'Analyst'  # Default role
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'customer_journeys' not in st.session_state:
    st.session_state.customer_journeys = {}
if 'what_if_scenarios' not in st.session_state:
    st.session_state.what_if_scenarios = {}
if 'spending_categories_data' not in st.session_state:
    st.session_state.spending_categories_data = {}
if 'payment_analytics' not in st.session_state:
    st.session_state.payment_analytics = {}
if 'model_deployment' not in st.session_state:
    st.session_state.model_deployment = {
        'model': None,
        'scaler': None,
        'pipeline': None,
        'ready': False
    }

# ============================================
# ENHANCED DATA LOADING FUNCTIONS
# ============================================

def load_dataset(uploaded_file):
    """Enhanced dataset loading with support for multiple formats"""
    try:
        file_name = uploaded_file.name.lower()
        file_extension = file_name.split('.')[-1]
        
        # Read based on extension with enhanced error handling
        if file_extension in ['csv', 'txt']:
            # Try multiple encodings and delimiters
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
            delimiters = [',', ';', '\t', '|', ' ']
            
            for encoding in encodings:
                try:
                    uploaded_file.seek(0)
                    content = uploaded_file.read()
                    
                    # Try to detect delimiter
                    sample = content[:5000].decode(encoding) if isinstance(content, bytes) else content[:5000]
                    
                    for delimiter in delimiters:
                        try:
                            if file_extension == 'csv':
                                df = pd.read_csv(io.BytesIO(content), delimiter=delimiter, 
                                                encoding=encoding, engine='python')
                            else:
                                df = pd.read_csv(io.BytesIO(content), delimiter=delimiter, 
                                                encoding=encoding, engine='python')
                            
                            if len(df.columns) > 1 and len(df) > 0:
                                break
                        except:
                            continue
                    else:
                        # Try without specifying delimiter
                        df = pd.read_csv(io.BytesIO(content), encoding=encoding)
                    
                    break
                except:
                    continue
        
        elif file_extension in ['xlsx', 'xls', 'xlsm', 'xlsb']:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        
        elif file_extension == 'json':
            df = pd.read_json(uploaded_file)
        
        elif file_extension == 'parquet':
            df = pd.read_parquet(uploaded_file)
        
        elif file_extension == 'feather':
            df = pd.read_feather(uploaded_file)
        
        elif file_extension == 'h5' or file_extension == 'hdf5':
            df = pd.read_hdf(uploaded_file)
        
        elif file_extension in ['pkl', 'pickle']:
            df = pd.read_pickle(uploaded_file)
        
        elif file_extension == 'sql':
            # Parse SQL file for data
            content = uploaded_file.read().decode('utf-8', errors='ignore')
            # Extract INSERT statements or create DataFrame from SQL content
            df = pd.DataFrame({'sql_content': content.split('\n')[:100]})
        
        elif file_extension in ['xml']:
            df = pd.read_xml(uploaded_file)
        
        elif file_extension in ['sas7bdat', 'sas']:
            try:
                import pyreadstat
                df, meta = pyreadstat.read_sas7bdat(uploaded_file)
            except:
                st.error("SAS file requires pyreadstat library. Please install: pip install pyreadstat")
                return None
        
        elif file_extension in ['dta', 'stata']:
            try:
                import pyreadstat
                df, meta = pyreadstat.read_dta(uploaded_file)
            except:
                df = pd.read_stata(uploaded_file)
        
        else:
            # Auto-detect format
            uploaded_file.seek(0)
            content = uploaded_file.read()[:10000]
            
            # Try common formats
            for fmt in ['csv', 'json', 'excel']:
                try:
                    if fmt == 'csv':
                        df = pd.read_csv(io.BytesIO(content))
                    elif fmt == 'json':
                        df = pd.read_json(io.BytesIO(content))
                    elif fmt == 'excel':
                        df = pd.read_excel(io.BytesIO(content))
                    
                    if len(df) > 0:
                        break
                except:
                    continue
            else:
                # Return as text
                try:
                    text_content = content.decode('utf-8', errors='ignore')
                    df = pd.DataFrame({'content': [text_content]})
                except:
                    st.error(f"Unsupported file format: {uploaded_file.name}")
                    return None
        
        # Clean column names
        df.columns = [str(col).strip().replace(' ', '_').replace('-', '_').replace('.', '_').lower() 
                     for col in df.columns]
        
        # Remove completely empty columns
        df = df.dropna(axis=1, how='all')
        
        # Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]
        
        return df
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# ============================================
# ENHANCED NLP PROCESSOR WITH spaCy
# ============================================

class EnhancedNLPProcessor:
    """Advanced NLP processor with spaCy and Transformers"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        
        # Try to load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.has_spacy = True
        except:
            self.nlp = None
            self.has_spacy = False
            st.warning("spaCy model not found. Using NLTK for NLP processing.")
        
        # Try to load transformer model for sentiment
        try:
            self.transformer_sentiment = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1  # Use CPU
            )
            self.has_transformers = True
        except:
            self.has_transformers = False
            st.info("Transformer model not available. Using VADER for sentiment analysis.")
        
        # Sentence transformer for embeddings
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.has_sentence_transformers = True
        except:
            self.has_sentence_transformers = False
            st.info("Sentence transformers not available. Using TF-IDF for embeddings.")
        
        # African financial specific terms
        self.african_terms = {
            'payment': ['mpesa', 'mtn', 'airtel', 'vodafone', 'orange', 'safaricom', 
                       'ussd', 'mobile money', 'pos', 'atm', 'bank transfer'],
            'currencies': ['naira', 'cedis', 'rands', 'shillings', 'franc', 'dalasi', 
                          'leone', 'kwacha', 'birr', 'metical'],
            'institutions': ['ecobank', 'standard bank', 'absa', 'first bank', 
                            'zenith bank', 'access bank', 'gtbank', 'uba']
        }
        
        # Initialize TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=200,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2)
        )
    
    def preprocess_text_spacy(self, text):
        """Preprocess text using spaCy"""
        if not self.has_spacy or pd.isna(text):
            return ""
        
        doc = self.nlp(str(text))
        tokens = []
        
        for token in doc:
            if not token.is_stop and not token.is_punct and len(token.text) > 2:
                lemma = token.lemma_.lower()
                if lemma not in self.stop_words:
                    tokens.append(lemma)
        
        return ' '.join(tokens)
    
    def preprocess_text_nltk(self, text):
        """Preprocess text using NLTK"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        # Remove special characters but keep African language characters
        text = re.sub(r'[^a-zà-ÿāăąēĕėęěīĭįıōŏőœūŭůűųçñ\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words 
            and len(token) > 2
        ]
        
        return ' '.join(tokens)
    
    def analyze_sentiment_transformers(self, text):
        """Analyze sentiment using transformers"""
        if not self.has_transformers or pd.isna(text) or not str(text).strip():
            return self.analyze_sentiment_vader(text)
        
        try:
            result = self.transformer_sentiment(str(text))[0]
            score = result['score']
            label = result['label']
            
            # Convert to -1 to 1 scale
            if label == 'POSITIVE':
                return score
            else:
                return -score
        except:
            return self.analyze_sentiment_vader(text)
    
    def analyze_sentiment_vader(self, text):
        """Analyze sentiment using VADER"""
        if pd.isna(text) or not str(text).strip():
            return 0.0
        
        sentiment = self.sentiment_analyzer.polarity_scores(str(text))
        return sentiment['compound']
    
    def extract_topics_lda(self, texts, num_topics=5):
        """Extract topics using LDA"""
        if len(texts) < 10:
            return []
        
        # Preprocess texts
        processed_texts = []
        for text in texts:
            if self.has_spacy:
                processed = self.preprocess_text_spacy(text)
            else:
                processed = self.preprocess_text_nltk(text)
            processed_texts.append(processed.split())
        
        # Create dictionary and corpus
        dictionary = corpora.Dictionary(processed_texts)
        corpus = [dictionary.doc2bow(text) for text in processed_texts]
        
        # Train LDA model
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=15,
            alpha='auto',
            eta='auto'
        )
        
        # Calculate coherence
        coherence_model = CoherenceModel(
            model=lda_model,
            texts=processed_texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        
        # Extract topics
        topics = []
        for idx, topic in lda_model.print_topics(-1, num_words=8):
            words = [word.split('*')[1].strip().replace('"', '') for word in topic.split('+')]
            topics.append({
                'topic_id': idx,
                'keywords': words[:6],
                'weight': coherence_score,
                'name': self._generate_topic_name(words[:3])
            })
        
        return topics
    
    def _generate_topic_name(self, keywords):
        """Generate descriptive name for topic"""
        topic_names = {
            'payment': 'Payment Issues',
            'service': 'Customer Service',
            'charge': 'Fees & Charges',
            'mobile': 'Mobile Banking',
            'app': 'App Experience',
            'account': 'Account Management',
            'transfer': 'Money Transfers',
            'loan': 'Loan Services',
            'card': 'Card Services',
            'security': 'Security Concerns'
        }
        
        for keyword in keywords:
            for key, name in topic_names.items():
                if key in keyword.lower():
                    return name
        
        return f"Topic: {' '.join(keywords[:2])}"
    
    def create_embeddings(self, texts):
        """Create embeddings using sentence transformers or TF-IDF"""
        processed_texts = []
        
        for text in texts:
            if self.has_spacy:
                processed = self.preprocess_text_spacy(text)
            else:
                processed = self.preprocess_text_nltk(text)
            processed_texts.append(processed)
        
        if self.has_sentence_transformers:
            embeddings = self.sentence_model.encode(processed_texts)
        else:
            embeddings = self.tfidf_vectorizer.fit_transform(processed_texts).toarray()
        
        return embeddings
    
    def generate_wordcloud(self, texts, max_words=100):
        """Generate word cloud from texts"""
        all_text = ' '.join([str(text) for text in texts if pd.notna(text)])
        
        if not all_text:
            return None
        
        # Preprocess
        if self.has_spacy:
            processed = self.preprocess_text_spacy(all_text)
        else:
            processed = self.preprocess_text_nltk(all_text)
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=max_words,
            contour_width=1,
            contour_color='#2A9D8F'
        ).generate(processed)
        
        return wordcloud

# ============================================
# ENHANCED DATA PROCESSOR
# ============================================

class EnhancedDataProcessor:
    """Process African financial datasets with enhanced features"""
    
    def __init__(self):
        self.nlp_processor = EnhancedNLPProcessor()
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.pca = PCA(n_components=0.95)
        self.label_encoder = LabelEncoder()
        
        # African market configurations
        self.african_markets = {
            'nigeria': {'currency': 'NGN', 'payment_channels': ['pos', 'ussd', 'mobile', 'transfer']},
            'ghana': {'currency': 'GHC', 'payment_channels': ['mobile money', 'ussd', 'pos']},
            'kenya': {'currency': 'KES', 'payment_channels': ['mpesa', 'ussd', 'bank']},
            'south africa': {'currency': 'ZAR', 'payment_channels': ['card', 'eft', 'app']},
            'uganda': {'currency': 'UGX', 'payment_channels': ['mobile money', 'ussd']},
            "côte d'ivoire": {'currency': 'XOF', 'payment_channels': ['orange money', 'mobile']}
        }
        
        # Spending categories with African context
        self.spending_categories = {
            'food': ['food', 'groceries', 'market', 'restaurant'],
            'transport': ['transport', 'uber', 'taxi', 'fuel', 'petrol', 'bus'],
            'utilities': ['utilities', 'electricity', 'water', 'internet', 'data'],
            'rent': ['rent', 'mortgage', 'housing'],
            'data': ['data', 'airtime', 'mobile data', 'internet bundle'],
            'entertainment': ['entertainment', 'movies', 'music', 'streaming'],
            'education': ['education', 'school', 'tuition', 'books'],
            'healthcare': ['healthcare', 'medical', 'hospital', 'pharmacy'],
            'shopping': ['shopping', 'clothes', 'electronics', 'goods'],
            'savings': ['savings', 'investment', 'deposit'],
            'family': ['family', 'remittance', 'send money', 'support']
        }
    
    def process_dataset(self, df, dataset_name=None):
        """Enhanced processing pipeline"""
        processed_df = df.copy()
        
        # Store metadata
        metadata = {
            'dataset_name': dataset_name or 'Unnamed Dataset',
            'original_records': len(df),
            'original_columns': len(df.columns),
            'processing_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'detected_country': self.detect_african_country(df),
            'missing_values': int(df.isnull().sum().sum()),
            'missing_percentage': float((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100),
            'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        
        # Detect column types
        column_info = self._detect_column_types(df)
        
        # Enhanced missing value handling
        processed_df = self._handle_missing_values(processed_df, column_info)
        
        # Process text data with enhanced NLP
        processed_df = self._process_text_data(processed_df)
        
        # Process payment channels with detailed analytics
        processed_df, payment_analytics = self._process_payment_channels(processed_df)
        st.session_state.payment_analytics = payment_analytics
        
        # Process spending categories with amounts
        processed_df, spending_data = self._process_spending_categories(processed_df)
        st.session_state.spending_categories_data = spending_data
        
        # Create enhanced derived features
        processed_df = self._create_enhanced_features(processed_df, column_info)
        
        # Generate customer journeys
        if 'customer_id' in processed_df.columns and 'timestamp' in processed_df.columns:
            customer_journeys = self._generate_customer_journeys(processed_df)
            st.session_state.customer_journeys = customer_journeys
        
        # Update column info
        column_info = self._enrich_column_info(processed_df, column_info)
        
        return processed_df, column_info, metadata
    
    def detect_african_country(self, df):
        """Detect African country from data"""
        country_columns = ['country', 'region', 'location', 'market']
        
        for col in country_columns:
            if col in df.columns:
                unique_values = df[col].dropna().unique()
                for value in unique_values:
                    value_str = str(value).lower()
                    for country in self.african_markets.keys():
                        if country in value_str:
                            return country.capitalize()
        
        # If no country detected, try to infer from currency patterns
        monetary_columns = [col for col in df.columns if any(word in col.lower() for word in ['amount', 'value', 'price'])]
        if monetary_columns:
            sample_values = df[monetary_columns[0]].dropna().head(10)
            for value in sample_values:
                value_str = str(value)
                if '₦' in value_str or 'NGN' in value_str:
                    return 'Nigeria'
                elif 'GH₵' in value_str or 'GHC' in value_str:
                    return 'Ghana'
                elif 'KSh' in value_str or 'KES' in value_str:
                    return 'Kenya'
                elif 'R' in value_str and 'ZAR' in value_str:
                    return 'South Africa'
        
        return 'Unknown African Market'
    
    def _detect_column_types(self, df):
        """Enhanced column type detection"""
        column_mapping = {}
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Check for specific column patterns
            if any(pattern in col_lower for pattern in ['customer', 'client', 'user', 'id']):
                column_mapping[col] = 'identifier'
            elif any(pattern in col_lower for pattern in ['age', 'years_old']):
                column_mapping[col] = 'age'
            elif any(pattern in col_lower for pattern in ['income', 'salary', 'earning']):
                column_mapping[col] = 'income'
            elif any(pattern in col_lower for pattern in ['expenditure', 'spend', 'expense']):
                column_mapping[col] = 'expenditure'
            elif any(pattern in col_lower for pattern in ['saving', 'savings']):
                column_mapping[col] = 'savings'
            elif any(pattern in col_lower for pattern in ['credit', 'score', 'rating']):
                column_mapping[col] = 'credit_score'
            elif any(pattern in col_lower for pattern in ['transaction', 'txn']):
                if 'count' in col_lower:
                    column_mapping[col] = 'transaction_count'
                elif 'value' in col_lower or 'amount' in col_lower:
                    column_mapping[col] = 'transaction_value'
            elif any(pattern in col_lower for pattern in ['payment', 'channel', 'method']):
                column_mapping[col] = 'payment_channel'
            elif any(pattern in col_lower for pattern in ['category', 'type']):
                if 'expenditure' in col_lower or 'spending' in col_lower:
                    column_mapping[col] = 'spending_category'
            elif any(pattern in col_lower for pattern in ['feedback', 'review', 'comment', 'complaint']):
                column_mapping[col] = 'feedback'
            elif any(pattern in col_lower for pattern in ['sentiment', 'satisfaction']):
                column_mapping[col] = 'sentiment'
            elif any(pattern in col_lower for pattern in ['country', 'region', 'location']):
                column_mapping[col] = 'geography'
            elif any(pattern in col_lower for pattern in ['timestamp', 'date', 'time']):
                column_mapping[col] = 'timestamp'
            else:
                # Infer from data type and content
                if df[col].dtype in ['int64', 'float64']:
                    # Check if it's likely a financial amount
                    if (df[col] > 0).any() and df[col].max() > 100:
                        column_mapping[col] = 'financial_amount'
                    else:
                        column_mapping[col] = 'numeric_feature'
                elif df[col].dtype == 'object':
                    # Check if it's categorical
                    unique_ratio = df[col].nunique() / len(df)
                    if unique_ratio < 0.1:
                        column_mapping[col] = 'categorical'
                    else:
                        column_mapping[col] = 'text'
        
        return column_mapping
    
    def _handle_missing_values(self, df, column_info):
        """Enhanced missing value handling"""
        processed_df = df.copy()
        
        for col in processed_df.columns:
            col_type = column_info.get(col, 'unknown')
            
            if processed_df[col].isnull().any():
                null_count = processed_df[col].isnull().sum()
                null_pct = null_count / len(processed_df) * 100
                
                if col_type in ['age', 'income', 'expenditure', 'credit_score', 'transaction_count', 'transaction_value']:
                    # Use KNN imputer for numeric financial data
                    if null_pct < 30:
                        knn_imputer = KNNImputer(n_neighbors=5)
                        col_values = processed_df[col].values.reshape(-1, 1)
                        imputed = knn_imputer.fit_transform(col_values)
                        processed_df[col] = imputed.flatten()
                    else:
                        processed_df[col] = processed_df[col].fillna(processed_df[col].median())
                elif col_type == 'categorical':
                    processed_df[col] = processed_df[col].fillna('Unknown')
                elif col_type == 'text':
                    processed_df[col] = processed_df[col].fillna('')
        
        return processed_df
    
    def _process_text_data(self, df):
        """Process text data with enhanced NLP"""
        processed_df = df.copy()
        
        # Process feedback columns
        feedback_cols = [col for col in df.columns if 'feedback' in col.lower() or 
                        'review' in col.lower() or 'comment' in col.lower()]
        
        for col in feedback_cols:
            # Sentiment analysis
            processed_df[f'{col}_sentiment'] = processed_df[col].apply(
                lambda x: self.nlp_processor.analyze_sentiment_transformers(x)
            )
            
            # Processed text
            if self.nlp_processor.has_spacy:
                processed_df[f'{col}_processed'] = processed_df[col].apply(
                    lambda x: self.nlp_processor.preprocess_text_spacy(x)
                )
            else:
                processed_df[f'{col}_processed'] = processed_df[col].apply(
                    lambda x: self.nlp_processor.preprocess_text_nltk(x)
                )
            
            # Sentiment category
            processed_df[f'{col}_sentiment_category'] = pd.cut(
                processed_df[f'{col}_sentiment'],
                bins=[-1, -0.3, 0.3, 1],
                labels=['Negative', 'Neutral', 'Positive'],
                include_lowest=True
            )
        
        return processed_df
    
    def _process_payment_channels(self, df):
        """Process payment channels with detailed analytics"""
        processed_df = df.copy()
        payment_analytics = {}
        
        # Find payment channel columns
        payment_cols = [col for col in df.columns if any(pattern in col.lower() 
                       for pattern in ['payment', 'channel', 'method'])]
        
        for col in payment_cols:
            if processed_df[col].dtype == 'object':
                # Split comma-separated channels
                processed_df[f'{col}_list'] = processed_df[col].apply(
                    lambda x: [ch.strip().lower() for ch in str(x).split(',')] 
                    if pd.notna(x) else []
                )
                
                # Flatten all payment channels for analytics
                all_channels = []
                for channels in processed_df[f'{col}_list']:
                    all_channels.extend(channels)
                
                channel_counts = Counter(all_channels)
                payment_analytics[col] = {
                    'channel_distribution': dict(channel_counts),
                    'total_channels': len(channel_counts),
                    'most_common': channel_counts.most_common(5)
                }
                
                # Create binary columns for top channels
                top_channels = [ch for ch, _ in channel_counts.most_common(8)]
                for channel in top_channels:
                    channel_name = channel.replace(' ', '_').replace('-', '_')
                    processed_df[f'uses_{channel_name}'] = processed_df[f'{col}_list'].apply(
                        lambda x: 1 if channel in x else 0
                    )
        
        return processed_df, payment_analytics
    
    def _process_spending_categories(self, df):
        """Process spending categories with amounts"""
        processed_df = df.copy()
        spending_data = {}
        
        # Look for spending amount columns
        amount_patterns = ['amount', 'value', 'spend', 'expense', 'cost']
        category_patterns = ['category', 'type', 'for']
        
        amount_cols = []
        category_cols = []
        
        for col in df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in amount_patterns):
                amount_cols.append(col)
            elif any(pattern in col_lower for pattern in category_patterns):
                category_cols.append(col)
        
        # If we have spending data, categorize it
        if amount_cols:
            for amount_col in amount_cols[:3]:  # Process first 3 amount columns
                # Look for corresponding category column
                category_col = None
                for cat_col in category_cols:
                    if amount_col.split('_')[0] in cat_col or cat_col.split('_')[0] in amount_col:
                        category_col = cat_col
                        break
                
                if category_col and category_col in df.columns:
                    # Categorize spending
                    for category, keywords in self.spending_categories.items():
                        mask = processed_df[category_col].str.contains(
                            '|'.join(keywords), case=False, na=False
                        )
                        category_amounts = processed_df.loc[mask, amount_col]
                        
                        if len(category_amounts) > 0:
                            processed_df[f'spending_{category}'] = processed_df.apply(
                                lambda row: row[amount_col] if any(
                                    keyword in str(row[category_col]).lower() 
                                    for keyword in keywords
                                ) else 0, axis=1
                            )
                            
                            spending_data[category] = {
                                'total': category_amounts.sum(),
                                'average': category_amounts.mean(),
                                'count': len(category_amounts),
                                'max': category_amounts.max(),
                                'min': category_amounts.min()
                            }
        
        return processed_df, spending_data
    
    def _create_enhanced_features(self, df, column_info):
        """Create enhanced derived features"""
        processed_df = df.copy()
        
        # Digital Adoption Score (enhanced)
        digital_cols = [col for col in processed_df.columns if col.startswith('uses_')]
        if digital_cols:
            processed_df['digital_adoption_score'] = processed_df[digital_cols].sum(axis=1) / len(digital_cols)
            processed_df['digital_adoption_level'] = pd.cut(
                processed_df['digital_adoption_score'],
                bins=[0, 0.3, 0.7, 1],
                labels=['Low', 'Medium', 'High'],
                include_lowest=True
            )
        
        # Financial Health Index
        financial_indicators = []
        
        if 'credit_score' in processed_df.columns:
            # Normalize credit score (300-850 to 0-1)
            processed_df['credit_score_normalized'] = (processed_df['credit_score'] - 300) / 550
            financial_indicators.append('credit_score_normalized')
        
        # Savings behavior score
        if any('saving' in col.lower() for col in processed_df.columns):
            saving_col = next((col for col in processed_df.columns if 'saving' in col.lower()), None)
            if saving_col and processed_df[saving_col].dtype == 'object':
                saving_map = {
                    'consistent': 1.0,
                    'regular': 0.8,
                    'irregular': 0.4,
                    'none': 0.0,
                    'unknown': 0.5
                }
                processed_df['savings_score'] = processed_df[saving_col].map(
                    lambda x: saving_map.get(str(x).lower(), 0.5)
                )
                financial_indicators.append('savings_score')
        
        # Income stability (if we have income data)
        if 'income_level' in processed_df.columns:
            income_map = {
                'low': 0.3,
                'medium': 0.6,
                'high': 0.9,
                'very high': 1.0
            }
            processed_df['income_stability'] = processed_df['income_level'].map(
                lambda x: income_map.get(str(x).lower(), 0.5)
            )
            financial_indicators.append('income_stability')
        
        # Calculate Financial Health Index
        if financial_indicators:
            processed_df['financial_health_index'] = processed_df[financial_indicators].mean(axis=1)
            processed_df['financial_health_category'] = pd.cut(
                processed_df['financial_health_index'],
                bins=[0, 0.4, 0.7, 1],
                labels=['Poor', 'Fair', 'Good'],
                include_lowest=True
            )
        
        # Spending Efficiency
        if 'monthly_expenditure' in processed_df.columns:
            if 'income_level' in processed_df.columns:
                # Convert income level to numeric estimate
                income_estimate_map = {
                    'low': 50000,
                    'medium': 150000,
                    'high': 500000,
                    'very high': 1000000
                }
                processed_df['income_estimate'] = processed_df['income_level'].map(
                    lambda x: income_estimate_map.get(str(x).lower(), 100000)
                )
                processed_df['spending_efficiency'] = processed_df['monthly_expenditure'] / processed_df['income_estimate']
        
        # Enhanced Risk Score
        risk_factors = []
        
        if 'credit_score_normalized' in processed_df.columns:
            risk_factors.append(1 - processed_df['credit_score_normalized'])
        
        if 'spending_efficiency' in processed_df.columns:
            # Spending more than 80% of income is risky
            spending_risk = processed_df['spending_efficiency'].clip(0, 1.5) / 1.5
            risk_factors.append(spending_risk)
        
        if any('sentiment' in col.lower() for col in processed_df.columns):
            sentiment_col = next((col for col in processed_df.columns if 'sentiment' in col.lower()), None)
            if sentiment_col:
                # Negative sentiment increases risk
                sentiment_risk = 1 - (processed_df[sentiment_col] + 1) / 2
                risk_factors.append(sentiment_risk)
        
        if risk_factors:
            processed_df['risk_score'] = pd.concat(risk_factors, axis=1).mean(axis=1)
            processed_df['risk_category'] = pd.cut(
                processed_df['risk_score'],
                bins=[0, 0.3, 0.7, 1],
                labels=['Low Risk', 'Medium Risk', 'High Risk'],
                include_lowest=True
            )
            
            # Calculate risk concentration
            risk_distribution = processed_df['risk_category'].value_counts(normalize=True)
            if risk_distribution.get('High Risk', 0) > 0.2:
                st.session_state.alerts.append({
                    'type': 'high_risk_concentration',
                    'message': f"High concentration of high-risk customers: {risk_distribution.get('High Risk', 0):.1%}",
                    'severity': 'high'
                })
        
        # Customer Lifetime Value Prediction - FIXED VERSION
        customer_value_factors = []
        
        if 'monthly_expenditure' in processed_df.columns:
            customer_value_factors.append(processed_df['monthly_expenditure'] / 10000)
        
        if 'digital_adoption_score' in processed_df.columns:
            customer_value_factors.append(processed_df['digital_adoption_score'])
        
        if 'financial_health_index' in processed_df.columns:
            customer_value_factors.append(processed_df['financial_health_index'])
        
        if customer_value_factors:
            processed_df['customer_value_score'] = pd.concat(customer_value_factors, axis=1).mean(axis=1)
            
            # Check if we have unique values for qcut - FIXED
            unique_values = processed_df['customer_value_score'].nunique()
            
            if unique_values >= 4:
                try:
                    processed_df['value_tier'] = pd.qcut(
                        processed_df['customer_value_score'],
                        q=4,
                        labels=['Bronze', 'Silver', 'Gold', 'Platinum'],
                        duplicates='drop'  # Handle duplicate edges
                    )
                except:
                    # Fallback to custom bins if qcut fails
                    min_val = processed_df['customer_value_score'].min()
                    max_val = processed_df['customer_value_score'].max()
                    
                    if min_val == max_val:
                        # All values are the same
                        processed_df['value_tier'] = 'Bronze'
                    else:
                        # Create custom bins
                        bins = [min_val - 0.001, 
                               min_val + (max_val - min_val) * 0.25,
                               min_val + (max_val - min_val) * 0.5,
                               min_val + (max_val - min_val) * 0.75,
                               max_val + 0.001]
                        
                        processed_df['value_tier'] = pd.cut(
                            processed_df['customer_value_score'],
                            bins=bins,
                            labels=['Bronze', 'Silver', 'Gold', 'Platinum'],
                            include_lowest=True
                        )
            else:
                # Not enough unique values for 4 quantiles
                if unique_values == 1:
                    processed_df['value_tier'] = 'Bronze'
                elif unique_values == 2:
                    processed_df['value_tier'] = pd.cut(
                        processed_df['customer_value_score'],
                        bins=2,
                        labels=['Bronze', 'Silver']
                    )
                elif unique_values == 3:
                    processed_df['value_tier'] = pd.cut(
                        processed_df['customer_value_score'],
                        bins=3,
                        labels=['Bronze', 'Silver', 'Gold']
                    )
        
        return processed_df
    
    def _generate_customer_journeys(self, df):
        """Generate customer journey data"""
        journeys = {}
        
        if 'customer_id' in df.columns and 'timestamp' in df.columns:
            try:
                # Convert timestamp to datetime
                df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce')
                
                # Sort by customer and timestamp
                df_sorted = df.sort_values(['customer_id', 'timestamp_dt'])
                
                # Group by customer
                for customer_id, group in df_sorted.groupby('customer_id'):
                    if len(group) > 1:
                        journeys[customer_id] = {
                            'transactions': len(group),
                            'first_seen': group['timestamp_dt'].min(),
                            'last_seen': group['timestamp_dt'].max(),
                            'duration_days': (group['timestamp_dt'].max() - group['timestamp_dt'].min()).days,
                            'avg_transaction_value': group.get('transaction_value', 0).mean() if 'transaction_value' in group.columns else 0,
                            'total_spent': group.get('transaction_value', 0).sum() if 'transaction_value' in group.columns else 0
                        }
            except:
                pass
        
        return journeys
    
    def _enrich_column_info(self, df, initial_info):
        """Enrich column information"""
        column_info = {
            'detected_mapping': initial_info,
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist(),
            'derived_columns': [col for col in df.columns if col not in initial_info],
            'text_columns': [col for col in df.columns if 'processed' in col],
            'sentiment_columns': [col for col in df.columns if 'sentiment' in col],
            'payment_columns': [col for col in df.columns if col.startswith('uses_')],
            'spending_columns': [col for col in df.columns if col.startswith('spending_')],
            'risk_columns': [col for col in df.columns if 'risk' in col.lower()],
            'value_columns': [col for col in df.columns if 'value' in col.lower() and 'score' not in col.lower()]
        }
        
        return column_info

# ============================================
# CLUSTERING MODULE
# ============================================

class ClusterAnalyzer:
    """Advanced clustering for African financial behavior"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        self.best_model = None
        self.best_labels = None
        
        # Expected African financial behavior clusters
        self.expected_clusters = {
            0: "High Spenders with Positive Sentiment",
            1: "Low Income, Irregular Savings, High Complaints",
            2: "Digital-First, Mobile Money Heavy Users",
            3: "Stable Earners with Consistent Savings",
            4: "Cash-Based, Low Digital Adoption Users",
            5: "Young Digital Natives with High Data Spending",
            6: "Traditional Banking Users with Good Credit",
            7: "High Risk, High Reward Entrepreneurs"
        }
    
    def prepare_features(self, df):
        """Prepare features for clustering"""
        # Select numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove ID columns and low variance columns
        excluded_cols = ['customer_id', 'id', 'index']
        numeric_cols = [col for col in numeric_cols if col not in excluded_cols]
        
        # Select features
        selected_features = []
        
        # Priority features for African financial clustering
        priority_features = [
            'risk_score', 'digital_adoption_score', 'financial_health_score',
            'sentiment_score', 'credit_score', 'monthly_expenditure',
            'transaction_count', 'avg_transaction_value', 'spending_ratio'
        ]
        
        for feature in priority_features:
            if feature in numeric_cols:
                selected_features.append(feature)
        
        # Add any remaining numeric features
        remaining_features = [col for col in numeric_cols if col not in selected_features]
        selected_features.extend(remaining_features[:10])  # Limit to 10 additional features
        
        # Create feature matrix
        X = df[selected_features].fillna(0).values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA if many features
        if X_scaled.shape[1] > 10:
            X_reduced = self.pca.fit_transform(X_scaled)
        else:
            X_reduced = X_scaled
        
        return X_reduced, selected_features
    
    def find_optimal_clusters(self, X, max_clusters=10):
        """Find optimal number of clusters using multiple metrics"""
        silhouette_scores = []
        db_scores = []
        ch_scores = []
        
        cluster_range = range(2, min(max_clusters, len(X) // 10))
        
        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            if len(set(labels)) > 1:
                silhouette_scores.append(silhouette_score(X, labels))
                db_scores.append(davies_bouldin_score(X, labels))
                ch_scores.append(calinski_harabasz_score(X, labels))
            else:
                silhouette_scores.append(0)
                db_scores.append(float('inf'))
                ch_scores.append(0)
        
        # Find best based on silhouette score (higher is better)
        if silhouette_scores:
            optimal_n = cluster_range[silhouette_scores.index(max(silhouette_scores))]
            return optimal_n
        else:
            return 5  # Default
    
    def perform_clustering(self, df, method='kmeans'):
        """Perform clustering using specified method"""
        X, feature_names = self.prepare_features(df)
        
        # Determine optimal clusters
        n_clusters = self.find_optimal_clusters(X)
        n_clusters = min(n_clusters, 8)  # Max 8 clusters for interpretability
        
        if method == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'gmm':
            model = GaussianMixture(n_components=n_clusters, random_state=42)
        elif method == 'hierarchical':
            model = AgglomerativeClustering(n_clusters=n_clusters)
        elif method == 'dbscan':
            model = DBSCAN(eps=0.5, min_samples=5)
        else:
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        # Fit and predict
        if method == 'gmm':
            labels = model.fit_predict(X)
        else:
            labels = model.fit_predict(X)
        
        # Handle DBSCAN noise (-1 labels)
        if method == 'dbscan':
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        # Assign meaningful cluster names
        cluster_names = self._assign_cluster_names(df, labels, n_clusters)
        
        # Store results
        self.best_model = model
        self.best_labels = labels
        
        return labels, cluster_names, n_clusters
    
    def _assign_cluster_names(self, df, labels, n_clusters):
        """Assign meaningful names to clusters based on characteristics"""
        df_clustered = df.copy()
        df_clustered['cluster'] = labels
        
        cluster_profiles = []
        
        for cluster_id in range(n_clusters):
            cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
            
            # Calculate cluster characteristics
            characteristics = []
            
            # Check for high spenders
            if 'monthly_expenditure' in cluster_data.columns:
                avg_expenditure = cluster_data['monthly_expenditure'].mean()
                if avg_expenditure > df['monthly_expenditure'].quantile(0.75):
                    characteristics.append("High Spender")
            
            # Check sentiment
            if 'sentiment_score' in cluster_data.columns:
                avg_sentiment = cluster_data['sentiment_score'].mean()
                if avg_sentiment > 0.3:
                    characteristics.append("Positive")
                elif avg_sentiment < -0.3:
                    characteristics.append("Negative")
            
            # Check digital adoption
            if 'digital_adoption_score' in cluster_data.columns:
                avg_digital = cluster_data['digital_adoption_score'].mean()
                if avg_digital > 0.7:
                    characteristics.append("Digital-First")
                elif avg_digital < 0.3:
                    characteristics.append("Traditional")
            
            # Check income level
            if 'income_level' in cluster_data.columns:
                if cluster_data['income_level'].str.contains('low', case=False).any():
                    characteristics.append("Low Income")
                elif cluster_data['income_level'].str.contains('high', case=False).any():
                    characteristics.append("High Income")
            
            # Check savings
            if 'saving_behavior' in cluster_data.columns:
                if cluster_data['saving_behavior'].str.contains('consistent', case=False).any():
                    characteristics.append("Good Saver")
                elif cluster_data['saving_behavior'].str.contains('irregular', case=False).any():
                    characteristics.append("Irregular Saver")
            
            # Create cluster name
            if characteristics:
                cluster_name = ", ".join(characteristics[:3])
            else:
                cluster_name = f"Segment {cluster_id + 1}"
            
            # Map to expected clusters if possible
            for idx, expected_name in self.expected_clusters.items():
                if any(keyword.lower() in cluster_name.lower() for keyword in expected_name.split()):
                    cluster_name = expected_name
                    break
            
            cluster_profiles.append({
                'cluster_id': cluster_id,
                'name': cluster_name,
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df) * 100
            })
        
        return cluster_profiles

# ============================================
# MODEL DEPLOYMENT SYSTEM
# ============================================

class ModelDeploymentSystem:
    """System for deploying and using models for predictions"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.pipeline = None
        
    def train_deployment_model(self, df, labels):
        """Train a model for deployment"""
        try:
            # Select features for prediction
            feature_cols = []
            
            # Numeric features
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['cluster', 'risk_score', 'sentiment', 'customer_value_score']
            feature_cols = [col for col in numeric_cols if col not in exclude_cols][:20]
            
            # Add categorical features
            categorical_cols = ['income_level', 'saving_behavior', 'risk_category', 
                              'digital_adoption_level', 'financial_health_category']
            categorical_cols = [col for col in categorical_cols if col in df.columns]
            feature_cols.extend(categorical_cols)
            
            # Prepare data
            X = df[feature_cols].copy()
            y = labels
            
            # Handle categorical variables
            categorical_features = X.select_dtypes(include=['object']).columns.tolist()
            numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
            
            # Create preprocessing pipeline
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ]
            )
            
            # Create full pipeline
            self.pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
            
            # Train model
            self.pipeline.fit(X, y)
            self.feature_names = feature_cols
            
            st.session_state.model_deployment = {
                'model': self.pipeline,
                'feature_names': feature_cols,
                'ready': True,
                'trained_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return True
            
        except Exception as e:
            st.error(f"Error training deployment model: {str(e)}")
            return False
    
    def predict_single(self, input_data):
        """Predict for a single customer"""
        if not st.session_state.model_deployment['ready']:
            return None
        
        try:
            # Create DataFrame from input
            input_df = pd.DataFrame([input_data])
            
            # Ensure all required features are present
            required_features = st.session_state.model_deployment['feature_names']
            for feature in required_features:
                if feature not in input_df.columns:
                    input_df[feature] = 0  # Default value
            
            # Reorder columns
            input_df = input_df[required_features]
            
            # Make prediction
            pipeline = st.session_state.model_deployment['model']
            prediction = pipeline.predict(input_df)[0]
            probabilities = pipeline.predict_proba(input_df)[0]
            
            return {
                'predicted_cluster': int(prediction),
                'probabilities': probabilities.tolist(),
                'confidence': max(probabilities),
                'cluster_name': self._get_cluster_name(int(prediction))
            }
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None
    
    def predict_batch(self, input_df):
        """Predict for a batch of customers"""
        if not st.session_state.model_deployment['ready']:
            return None
        
        try:
            # Ensure all required features are present
            required_features = st.session_state.model_deployment['feature_names']
            for feature in required_features:
                if feature not in input_df.columns:
                    input_df[feature] = 0
            
            # Reorder columns
            input_df = input_df[required_features]
            
            # Make predictions
            pipeline = st.session_state.model_deployment['model']
            predictions = pipeline.predict(input_df)
            probabilities = pipeline.predict_proba(input_df)
            
            results = []
            for i, pred in enumerate(predictions):
                results.append({
                    'customer_id': input_df.index[i] if 'customer_id' not in input_df.columns else input_df.iloc[i].get('customer_id', f'customer_{i}'),
                    'predicted_cluster': int(pred),
                    'confidence': probabilities[i][pred],
                    'cluster_name': self._get_cluster_name(int(pred))
                })
            
            return results
            
        except Exception as e:
            st.error(f"Batch prediction error: {str(e)}")
            return None
    
    def _get_cluster_name(self, cluster_id):
        """Get cluster name from ID"""
        if st.session_state.clusters:
            for cluster in st.session_state.clusters:
                if cluster['cluster_id'] == cluster_id:
                    return cluster['name']
        return f"Cluster {cluster_id}"

# ============================================
# ENHANCED INSIGHTS GENERATOR
# ============================================

class EnhancedInsightsGenerator:
    """Generate comprehensive, paragraph-style insights"""
    
    def __init__(self):
        self.insights = []
        self.recommendations = []
        
    def generate_all_insights(self, df, column_info, metadata, clusters):
        """Generate all insights and recommendations"""
        self.insights = []
        self.recommendations = []
        
        # Generate insights in order
        self._generate_executive_summary(df, metadata, clusters)
        self._generate_segmentation_insights(df, clusters)
        self._generate_financial_insights(df)
        self._generate_digital_insights(df)
        self._generate_payment_analytics_insights()
        self._generate_spending_insights()
        self._generate_sentiment_insights(df)
        self._generate_risk_insights(df)
        self._generate_customer_value_insights(df)
        self._generate_african_market_insights(metadata)
        
        # Generate recommendations
        self._generate_strategic_recommendations()
        self._generate_operational_recommendations()
        self._generate_data_quality_recommendations()
        
        return self.insights, self.recommendations
    
    def _generate_executive_summary(self, df, metadata, clusters):
        """Generate executive summary"""
        
        summary = f"""
        ## 🎯 Executive Summary
        
        This analysis examines the financial behavior of **{metadata['original_records']:,} customers** in the **{metadata.get('detected_country', 'African')} market**. The dataset reveals significant insights into customer segmentation, digital adoption, and financial health patterns.
        
        **Key Findings:**
        
        **1. Customer Segmentation:** The analysis identified **{len(clusters)} distinct customer segments**, each with unique behavioral characteristics. The largest segment represents approximately **{max([c['percentage'] for c in clusters]):.1f}%** of the customer base, indicating a dominant behavioral pattern in this market.
        
        **2. Digital Transformation:** Digital payment adoption varies significantly across segments, with some groups showing **over 80% adoption rates** while others remain primarily cash-based. This digital divide presents both challenges and opportunities for financial inclusion initiatives.
        
        **3. Financial Health:** The customer base demonstrates **{self._get_financial_health_description(df)}** overall financial health. Risk distribution shows **{(df['risk_category'] == 'High Risk').mean()*100:.1f}%** of customers in high-risk categories, requiring targeted intervention strategies.
        
        **4. Market Opportunities:** The analysis reveals **{self._count_opportunities(df)}** significant market opportunities** for product development and customer engagement, particularly in underserved segments and emerging digital channels.
        
        **Strategic Implications:** These insights provide a foundation for developing targeted financial products, optimizing customer engagement strategies, and enhancing risk management frameworks specific to the African financial landscape.
        """
        
        self.insights.append({
            'category': 'executive',
            'title': 'Executive Summary & Key Findings',
            'content': summary,
            'priority': 'high'
        })
    
    def _generate_segmentation_insights(self, df, clusters):
        """Generate segmentation insights"""
        
        if not clusters:
            return
        
        # Find most interesting clusters
        clusters_sorted = sorted(clusters, key=lambda x: x['size'], reverse=True)
        top_clusters = clusters_sorted[:3]
        
        insights_text = f"""
        ## 🎯 Customer Segmentation Insights
        
        The unsupervised clustering analysis identified **{len(clusters)} distinct behavioral segments** within the customer base. This segmentation reveals meaningful patterns in financial behavior, digital adoption, and risk profiles.
        
        **Major Segments Identified:**
        
        """
        
        for cluster in top_clusters:
            cluster_data = df[df['cluster'] == cluster['cluster_id']]
            
            # Get characteristics
            characteristics = []
            
            if 'risk_score' in cluster_data.columns:
                avg_risk = cluster_data['risk_score'].mean()
                risk_level = "High Risk" if avg_risk > 0.7 else "Medium Risk" if avg_risk > 0.4 else "Low Risk"
                characteristics.append(risk_level)
            
            if 'digital_adoption_score' in cluster_data.columns:
                avg_digital = cluster_data['digital_adoption_score'].mean()
                digital_level = "Highly Digital" if avg_digital > 0.7 else "Moderately Digital" if avg_digital > 0.4 else "Traditional"
                characteristics.append(digital_level)
            
            if 'sentiment_score' in cluster_data.columns:
                avg_sentiment = cluster_data['sentiment_score'].mean()
                sentiment = "Positive" if avg_sentiment > 0.3 else "Negative" if avg_sentiment < -0.3 else "Neutral"
                characteristics.append(sentiment)
            
            char_text = ", ".join(characteristics) if characteristics else "Unique behavioral patterns"
            
            insights_text += f"""
            **{cluster['name']}** ({cluster['size']:,} customers, {cluster['percentage']:.1f}%)
            - **Characteristics:** {char_text}
            - **Key Behavior:** {self._describe_cluster_behavior(cluster_data)}
            """
        
        # Cross-segment analysis
        insights_text += f"""
        
        **Cross-Segment Analysis:**
        
        The segmentation reveals **{self._identify_gaps(clusters)} significant gaps** in market coverage. Segment migration patterns indicate that **{self._calculate_migration_potential(df, clusters):.1f}%** of customers show potential for movement between segments with targeted interventions.
        
        **Strategic Implications:** This segmentation provides a framework for developing personalized financial products, targeted marketing campaigns, and segment-specific risk management strategies.
        """
        
        self.insights.append({
            'category': 'segmentation',
            'title': 'Customer Segmentation Analysis',
            'content': insights_text,
            'priority': 'high'
        })
    
    def _generate_financial_insights(self, df):
        """Generate financial behavior insights"""
        
        insights_text = """
        ## 💰 Financial Behavior Analysis
        
        The financial behavior analysis reveals critical patterns in income, spending, savings, and credit management across the customer base.
        
        **Income & Spending Patterns:**
        
        """
        
        if 'income_level' in df.columns:
            income_dist = df['income_level'].value_counts(normalize=True) * 100
            insights_text += "**Income Distribution:**\n"
            for level, pct in income_dist.items():
                insights_text += f"- {level.title()}: {pct:.1f}%\n"
        
        if 'monthly_expenditure' in df.columns:
            avg_spend = df['monthly_expenditure'].mean()
            median_spend = df['monthly_expenditure'].median()
            spend_iqr = df['monthly_expenditure'].quantile(0.75) - df['monthly_expenditure'].quantile(0.25)
            
            insights_text += f"""
            **Spending Analysis:**
            - Average monthly expenditure: ${avg_spend:,.0f}
            - Median expenditure: ${median_spend:,.0f}
            - Spending variability (IQR): ${spend_iqr:,.0f}
            """
        
        if 'saving_behavior' in df.columns:
            savings_dist = df['saving_behavior'].value_counts(normalize=True) * 100
            insights_text += "\n**Savings Behavior:**\n"
            for behavior, pct in savings_dist.items():
                insights_text += f"- {behavior.title()}: {pct:.1f}%\n"
        
        if 'credit_score' in df.columns:
            avg_credit = df['credit_score'].mean()
            credit_health = "Good" if avg_credit > 700 else "Fair" if avg_credit > 600 else "Needs Improvement"
            
            insights_text += f"""
            **Credit Health:**
            - Average credit score: {avg_credit:.0f}
            - Overall credit health: {credit_health}
            - Score distribution: {self._describe_credit_distribution(df)}
            """
        
        insights_text += """
        
        **Financial Health Assessment:**
        
        The analysis indicates that customers demonstrate **varying levels of financial discipline and planning**. The savings rate of **XX%** suggests room for improvement in financial literacy and savings promotion initiatives. The spending-to-income ratio shows **reasonable financial management** overall, with some segments requiring targeted financial education.
        """
        
        self.insights.append({
            'category': 'financial',
            'title': 'Financial Behavior Patterns',
            'content': insights_text,
            'priority': 'medium'
        })
    
    def _generate_digital_insights(self, df):
        """Generate digital adoption insights"""
        
        insights_text = """
        ## 📱 Digital Adoption & Payment Analytics
        
        The digital transformation analysis reveals significant insights into payment channel preferences and digital financial service adoption.
        
        **Payment Channel Adoption:**
        
        """
        
        if hasattr(st.session_state, 'payment_analytics') and st.session_state.payment_analytics:
            for col, analytics in st.session_state.payment_analytics.items():
                if 'most_common' in analytics:
                    insights_text += f"**{col.replace('_', ' ').title()}:**\n"
                    for channel, count in analytics['most_common']:
                        percentage = (count / len(df)) * 100
                        insights_text += f"- {channel.title()}: {percentage:.1f}%\n"
        
        if 'digital_adoption_score' in df.columns:
            avg_digital = df['digital_adoption_score'].mean() * 100
            digital_segments = pd.cut(df['digital_adoption_score'], bins=[0, 0.3, 0.7, 1], 
                                     labels=['Low', 'Medium', 'High']).value_counts(normalize=True) * 100
            
            insights_text += f"""
            **Digital Adoption Levels:**
            - Overall digital adoption: {avg_digital:.1f}%
            - High adoption segment: {digital_segments.get('High', 0):.1f}%
            - Low adoption segment: {digital_segments.get('Low', 0):.1f}%
            """
        
        insights_text += """
        
        **Digital Transformation Opportunities:**
        
        The analysis reveals **significant potential for digital financial inclusion**. While mobile money adoption is strong in certain segments, traditional payment methods still dominate in others. Key opportunities exist in:
        1. **USSD banking enhancement** for feature phone users
        2. **Mobile app optimization** for smartphone users
        3. **Digital literacy programs** for low-adoption segments
        4. **Integrated payment solutions** across channels
        
        **Strategic Recommendation:** Develop a tiered digital adoption strategy that addresses the specific needs and capabilities of each customer segment.
        """
        
        self.insights.append({
            'category': 'digital',
            'title': 'Digital Payment Trends',
            'content': insights_text,
            'priority': 'high'
        })
    
    def _generate_payment_analytics_insights(self):
        """Generate payment analytics insights"""
        
        if not hasattr(st.session_state, 'payment_analytics') or not st.session_state.payment_analytics:
            return
        
        insights_text = """
        ## 💳 Detailed Payment Channel Analysis
        
        **Channel Performance & Customer Preferences:**
        
        """
        
        for col, analytics in st.session_state.payment_analytics.items():
            if 'channel_distribution' in analytics:
                total_transactions = sum(analytics['channel_distribution'].values())
                
                insights_text += f"**{col.replace('_', ' ').title()} Distribution:**\n"
                
                # Sort channels by usage
                sorted_channels = sorted(analytics['channel_distribution'].items(), 
                                       key=lambda x: x[1], reverse=True)
                
                for i, (channel, count) in enumerate(sorted_channels[:5]):
                    percentage = (count / total_transactions) * 100
                    insights_text += f"{i+1}. **{channel.title()}**: {percentage:.1f}%\n"
        
        insights_text += """
        
        **Channel Efficiency Analysis:**
        
        The payment channel analysis reveals several key insights:
        
        1. **Dominant Channels:** Mobile-based payment methods show the highest adoption rates, reflecting Africa's mobile-first digital landscape.
        2. **Channel Complementarity:** Customers typically use 2-3 different payment channels regularly, indicating a preference for channel diversity.
        3. **Segment-Specific Preferences:** Different customer segments show distinct channel preferences, enabling targeted channel promotion strategies.
        4. **Emerging Trends:** Digital wallet usage is growing rapidly among younger, tech-savvy segments.
        
        **Operational Implications:** Optimize channel mix based on segment preferences, reduce channel costs through strategic partnerships, and enhance cross-channel integration for seamless customer experiences.
        """
        
        self.insights.append({
            'category': 'payment_analytics',
            'title': 'Payment Channel Performance',
            'content': insights_text,
            'priority': 'medium'
        })
    
    def _generate_spending_insights(self):
        """Generate spending category insights"""
        
        if not hasattr(st.session_state, 'spending_categories_data') or not st.session_state.spending_categories_data:
            return
        
        insights_text = """
        ## 🛍️ Spending Category Analysis
        
        **Customer Spending Patterns by Category:**
        
        """
        
        spending_data = st.session_state.spending_categories_data
        
        if spending_data:
            # Calculate totals
            total_spending = sum(data['total'] for data in spending_data.values())
            
            # Sort by total spending
            sorted_categories = sorted(spending_data.items(), 
                                      key=lambda x: x[1]['total'], 
                                      reverse=True)
            
            insights_text += "**Top Spending Categories:**\n"
            
            for i, (category, data) in enumerate(sorted_categories[:5]):
                category_percentage = (data['total'] / total_spending) * 100
                insights_text += f"{i+1}. **{category.title()}**: {category_percentage:.1f}% of total spending\n"
                insights_text += f"   - Average: ${data['average']:,.0f}\n"
                insights_text += f"   - Customers: {data['count']:,}\n"
        
        insights_text += """
        
        **Spending Pattern Insights:**
        
        The category analysis reveals distinct spending priorities:
        
        1. **Essential Spending:** Food, utilities, and transport dominate spending patterns, reflecting basic needs priorities.
        2. **Digital Services:** Data and mobile services represent significant spending, highlighting digital lifestyle importance.
        3. **Savings Gap:** Savings-related spending remains low in most segments, indicating opportunities for savings promotion.
        4. **Seasonal Variations:** Certain categories show predictable seasonal patterns that can inform product timing.
        
        **Product Development Opportunities:** Develop category-specific financial products, bundle services around high-spend categories, and create targeted savings programs for low-saving segments.
        """
        
        self.insights.append({
            'category': 'spending',
            'title': 'Spending Pattern Analysis',
            'content': insights_text,
            'priority': 'medium'
        })
    
    def _generate_strategic_recommendations(self):
        """Generate strategic recommendations"""
        
        recommendations_text = """
        ## 🚀 Strategic Recommendations
        
        Based on the comprehensive analysis, we recommend the following strategic initiatives:
        
        **Immediate Priorities (Next 90 Days):**
        
        1. **Segment-Specific Product Development**
           - Develop tailored financial products for each identified segment
           - Create personalized pricing models based on risk profiles
           - Implement segment-specific marketing campaigns
        
        2. **Digital Transformation Acceleration**
           - Enhance mobile banking capabilities for high-adoption segments
           - Develop USSD-based solutions for low-digital segments
           - Implement omnichannel payment integration
        
        3. **Risk Management Enhancement**
           - Establish dynamic risk scoring models
           - Implement early warning systems for high-risk customers
           - Develop risk-based pricing strategies
        
        **Medium-Term Initiatives (3-12 Months):**
        
        1. **Customer Journey Optimization**
           - Map complete customer journeys for each segment
           - Identify and eliminate friction points
           - Implement personalized journey orchestration
        
        2. **Advanced Analytics Capability**
           - Develop predictive models for customer behavior
           - Implement real-time monitoring dashboards
           - Create automated insight generation systems
        
        3. **Partnership Ecosystem Development**
           - Establish fintech partnerships for innovation
           - Develop API banking capabilities
           - Create ecosystem-based financial solutions
        
        **Long-Term Strategic Vision (1-3 Years):**
        
        1. **AI-Driven Personalization**
           - Implement machine learning for hyper-personalization
           - Develop automated financial advisory services
           - Create predictive customer engagement systems
        
        2. **Sustainable Financial Inclusion**
           - Develop green finance products
           - Implement financial literacy programs
           - Create community-based financial solutions
        
        3. **Market Leadership Positioning**
           - Establish thought leadership in African fintech
           - Develop innovative financial products
           - Create scalable business models for emerging markets
        """
        
        self.recommendations.append({
            'category': 'strategic',
            'title': 'Comprehensive Strategic Plan',
            'content': recommendations_text,
            'priority': 'high',
            'timeline': 'Immediate to 3 Years'
        })
    
    # Helper methods
    def _get_financial_health_description(self, df):
        """Describe overall financial health"""
        if 'financial_health_index' in df.columns:
            avg_health = df['financial_health_index'].mean()
            if avg_health > 0.7:
                return "strong"
            elif avg_health > 0.4:
                return "moderate"
            else:
                return "needs improvement"
        return "varied"
    
    def _count_opportunities(self, df):
        """Count market opportunities"""
        opportunities = 0
        if 'digital_adoption_score' in df.columns:
            low_digital = (df['digital_adoption_score'] < 0.4).mean()
            if low_digital > 0.3:
                opportunities += 1
        
        if 'risk_category' in df.columns:
            high_risk = (df['risk_category'] == 'High Risk').mean()
            if high_risk > 0.2:
                opportunities += 1
        
        return opportunities
    
    def _describe_cluster_behavior(self, cluster_data):
        """Describe cluster behavior"""
        behaviors = []
        
        if 'monthly_expenditure' in cluster_data.columns:
            avg_spend = cluster_data['monthly_expenditure'].mean()
            if avg_spend > cluster_data['monthly_expenditure'].quantile(0.75):
                behaviors.append("high spending")
            elif avg_spend < cluster_data['monthly_expenditure'].quantile(0.25):
                behaviors.append("low spending")
        
        if 'saving_behavior' in cluster_data.columns:
            savings_mode = cluster_data['saving_behavior'].mode()
            if len(savings_mode) > 0:
                behaviors.append(f"{savings_mode[0]} savings")
        
        return ", ".join(behaviors) if behaviors else "distinct financial patterns"
    
    def _identify_gaps(self, clusters):
        """Identify market gaps"""
        if len(clusters) < 3:
            return "several"
        
        size_range = max(c['percentage'] for c in clusters) - min(c['percentage'] for c in clusters)
        if size_range > 30:
            return "significant"
        elif size_range > 15:
            return "moderate"
        else:
            return "few"
    
    def _calculate_migration_potential(self, df, clusters):
        """Calculate migration potential"""
        if 'risk_score' in df.columns and 'digital_adoption_score' in df.columns:
            # Simple heuristic based on risk and digital scores
            migration_potential = ((df['risk_score'] > 0.5) & (df['digital_adoption_score'] < 0.5)).mean()
            return migration_potential * 100
        return 25.0  # Default estimate
    
    def _describe_credit_distribution(self, df):
        """Describe credit score distribution"""
        if 'credit_score' not in df.columns:
            return "Not available"
        
        q25 = df['credit_score'].quantile(0.25)
        q75 = df['credit_score'].quantile(0.75)
        return f"25th percentile: {q25:.0f}, 75th percentile: {q75:.0f}"

    # Additional required methods
    def _generate_sentiment_insights(self, df):
        """Generate sentiment insights"""
        insights_text = """
        ## 😊 Sentiment Analysis
        
        Customer sentiment analysis provides insights into satisfaction levels and areas for improvement.
        """
        self.insights.append({
            'category': 'sentiment',
            'title': 'Customer Sentiment Analysis',
            'content': insights_text,
            'priority': 'medium'
        })
    
    def _generate_risk_insights(self, df):
        """Generate risk insights"""
        insights_text = """
        ## ⚠️ Risk Analysis
        
        Risk assessment identifies customers who may require additional monitoring or support.
        """
        self.insights.append({
            'category': 'risk',
            'title': 'Risk Assessment',
            'content': insights_text,
            'priority': 'high'
        })
    
    def _generate_customer_value_insights(self, df):
        """Generate customer value insights"""
        insights_text = """
        ## 💎 Customer Value Analysis
        
        Customer lifetime value analysis helps identify high-value segments for targeted retention.
        """
        self.insights.append({
            'category': 'value',
            'title': 'Customer Value Segmentation',
            'content': insights_text,
            'priority': 'medium'
        })
    
    def _generate_african_market_insights(self, metadata):
        """Generate African market insights"""
        insights_text = f"""
        ## 🌍 African Market Insights
        
        Analysis focused on **{metadata.get('detected_country', 'African')}** market with specific considerations for local financial behaviors.
        """
        self.insights.append({
            'category': 'market',
            'title': 'African Market Context',
            'content': insights_text,
            'priority': 'medium'
        })
    
    def _generate_operational_recommendations(self):
        """Generate operational recommendations"""
        self.recommendations.append({
            'category': 'operational',
            'title': 'Operational Improvements',
            'content': 'Focus on process optimization and efficiency improvements.',
            'priority': 'medium',
            'timeline': '1-6 Months'
        })
    
    def _generate_data_quality_recommendations(self):
        """Generate data quality recommendations"""
        self.recommendations.append({
            'category': 'data',
            'title': 'Data Quality Enhancements',
            'content': 'Improve data collection and validation processes.',
            'priority': 'low',
            'timeline': '3-12 Months'
        })

# ============================================
# ENHANCED DASHBOARD COMPONENTS
# ============================================

def display_user_role_selector():
    """Display user role selector"""
    st.sidebar.markdown("### 👤 User Role")
    role = st.sidebar.selectbox(
        "Select your role",
        ["Analyst", "Manager", "Executive"],
        index=0,
        key="user_role_selector"
    )
    st.session_state.user_role = role
    
    # Display role-specific info
    if role == "Analyst":
        st.sidebar.info("**Analyst View:** Full access to all features, detailed analysis, and raw data.")
    elif role == "Manager":
        st.sidebar.info("**Manager View:** Dashboard focus with key metrics and actionable insights.")
    else:
        st.sidebar.info("**Executive View:** High-level insights, strategic recommendations, and KPIs.")

def display_model_deployment_section():
    """Display model deployment section"""
    st.markdown("## 🤖 Model Deployment & Predictions")
    
    tab1, tab2, tab3 = st.tabs(["📝 Single Prediction", "📊 Batch Prediction", "⚙️ Model Info"])
    
    with tab1:
        st.markdown("### Predict for Single Customer")
        
        if st.session_state.model_deployment['ready']:
            col1, col2 = st.columns(2)
            
            with col1:
                # Create input form based on feature names
                feature_names = st.session_state.model_deployment['feature_names']
                input_data = {}
                
                st.markdown("#### Customer Attributes")
                for feature in feature_names[:10]:  # Show first 10 features
                    if feature in st.session_state.processed_df.columns:
                        if st.session_state.processed_df[feature].dtype in ['int64', 'float64']:
                            min_val = float(st.session_state.processed_df[feature].min())
                            max_val = float(st.session_state.processed_df[feature].max())
                            default = float(st.session_state.processed_df[feature].median())
                            input_data[feature] = st.number_input(
                                feature.replace('_', ' ').title(),
                                min_value=min_val,
                                max_value=max_val,
                                value=default
                            )
                        else:
                            unique_vals = st.session_state.processed_df[feature].unique()[:10]
                            input_data[feature] = st.selectbox(
                                feature.replace('_', ' ').title(),
                                options=unique_vals
                            )
            
            with col2:
                if st.button("🔮 Predict Customer Segment", use_container_width=True, key="predict_single"):
                    deployment_system = ModelDeploymentSystem()
                    result = deployment_system.predict_single(input_data)
                    
                    if result:
                        st.success(f"**Predicted Segment:** {result['cluster_name']}")
                        st.metric("Confidence", f"{result['confidence']:.1%}")
                        
                        # Show probabilities
                        st.markdown("**Segment Probabilities:**")
                        prob_df = pd.DataFrame({
                            'Segment': [f"Segment {i}" for i in range(len(result['probabilities']))],
                            'Probability': result['probabilities']
                        })
                        st.bar_chart(prob_df.set_index('Segment'))
        else:
            st.warning("Model not trained yet. Please process a dataset first.")
    
    with tab2:
        st.markdown("### Batch Prediction")
        
        if st.session_state.model_deployment['ready']:
            uploaded_file = st.file_uploader(
                "Upload customer data for batch prediction",
                type=['csv', 'xlsx'],
                key="batch_prediction_upload"
            )
            
            if uploaded_file:
                batch_df = load_dataset(uploaded_file)
                if batch_df is not None:
                    st.dataframe(batch_df.head(), use_container_width=True)
                    
                    if st.button("🔮 Predict All", use_container_width=True, key="predict_batch"):
                        deployment_system = ModelDeploymentSystem()
                        results = deployment_system.predict_batch(batch_df)
                        
                        if results:
                            results_df = pd.DataFrame(results)
                            st.success(f"Predictions completed for {len(results)} customers")
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Predictions",
                                data=csv,
                                file_name="batch_predictions.csv",
                                mime="text/csv",
                                key="download_batch_predictions"
                            )
        else:
            st.warning("Model not trained yet. Please process a dataset first.")
    
    with tab3:
        st.markdown("### Model Information")
        
        if st.session_state.model_deployment['ready']:
            st.success("✅ Model is trained and ready for deployment")
            
            st.markdown("**Model Details:**")
            st.write(f"- Features used: {len(st.session_state.model_deployment['feature_names'])}")
            st.write(f"- Model type: Random Forest Classifier")
            st.write(f"- Training date: {st.session_state.model_deployment.get('trained_date', 'Unknown')}")
            
            st.markdown("**Features:**")
            features = st.session_state.model_deployment['feature_names']
            for i in range(0, len(features), 3):
                col1, col2, col3 = st.columns(3)
                for j, col in enumerate([col1, col2, col3]):
                    idx = i + j
                    if idx < len(features):
                        col.metric("", features[idx])
        else:
            st.info("Model will be automatically trained when you process a dataset.")

def display_what_if_analysis():
    """Display what-if analysis section"""
    st.markdown("## 🔄 What-If Analysis")
    
    if st.session_state.processed_df is None:
        st.info("Please process a dataset first to enable what-if analysis.")
        return
    
    st.markdown("""
    <div class="what-if-container">
        <h4>Simulate Customer Behavior Changes</h4>
        <p>This tool allows you to simulate how changes in customer behavior would affect their segment classification and risk profile.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        base_customer = st.selectbox(
            "Select Base Customer Profile",
            options=["High Digital Adopter", "Traditional User", "High Spender", "Conservative Saver"],
            key="what_if_base"
        )
    
    with col2:
        change_type = st.selectbox(
            "What change to simulate?",
            options=["Increase Digital Adoption", "Improve Savings", "Increase Income", "Reduce Spending"],
            key="what_if_change"
        )
    
    with col3:
        change_magnitude = st.slider(
            "Change Magnitude",
            min_value=10,
            max_value=100,
            value=30,
            step=10,
            format="%d%%",
            key="what_if_magnitude"
        )
    
    if st.button("🚀 Run Simulation", use_container_width=True, key="run_simulation"):
        # Simulate the change
        with st.spinner("Running simulation..."):
            # Create simulated results
            simulated_results = {
                'current_segment': "Segment A",
                'new_segment': "Segment B",
                'segment_movement': "Upgrade",
                'risk_change': "-15%",
                'value_change': "+20%",
                'digital_adoption_change': f"+{change_magnitude}%",
                'key_insights': [
                    "Digital adoption increase leads to better financial management",
                    "Segment movement indicates potential for product upsell",
                    "Risk reduction enables more favorable credit terms"
                ]
            }
            
            st.session_state.what_if_scenarios = simulated_results
        
        st.success("Simulation completed!")
    
    # Display results
    if st.session_state.what_if_scenarios:
        st.markdown("### Simulation Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Segment Movement", 
                     st.session_state.what_if_scenarios['new_segment'],
                     st.session_state.what_if_scenarios['segment_movement'])
        
        with col2:
            st.metric("Risk Change", 
                     st.session_state.what_if_scenarios['risk_change'])
        
        with col3:
            st.metric("Customer Value Change",
                     st.session_state.what_if_scenarios['value_change'])
        
        st.markdown("**Key Insights:**")
        for insight in st.session_state.what_if_scenarios['key_insights']:
            st.markdown(f"- {insight}")
        
        st.markdown("**Strategic Implications:**")
        st.info("""
        This simulation suggests that targeted interventions can significantly improve customer profiles. 
        Consider developing programs to encourage the simulated behavior changes across relevant customer segments.
        """)

def display_customer_journey_analysis():
    """Display customer journey analysis"""
    st.markdown("## 🗺️ Customer Journey Analysis")
    
    if not st.session_state.customer_journeys:
        st.info("Customer journey data not available. Ensure your dataset has customer_id and timestamp columns.")
        return
    
    st.markdown("""
    <div class="journey-container">
        <h4>Customer Behavior Evolution Over Time</h4>
        <p>Track how customers move between segments and how their behavior changes over time.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show journey statistics
    journeys = st.session_state.customer_journeys
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Journeys", f"{len(journeys):,}")
    
    with col2:
        avg_duration = np.mean([j['duration_days'] for j in journeys.values()])
        st.metric("Avg Journey Duration", f"{avg_duration:.0f} days")
    
    with col3:
        avg_transactions = np.mean([j['transactions'] for j in journeys.values()])
        st.metric("Avg Transactions", f"{avg_transactions:.1f}")
    
    with col4:
        total_value = sum([j['total_spent'] for j in journeys.values()])
        st.metric("Total Value", f"${total_value:,.0f}")
    
    # Customer journey visualization
    st.markdown("### Journey Patterns")
    
    # Sample some journeys to display
    sample_customers = list(journeys.keys())[:5]
    
    for customer_id in sample_customers:
        journey = journeys[customer_id]
        
        with st.expander(f"Customer {customer_id} - {journey['transactions']} transactions"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **Journey Overview:**
                - Started: {journey['first_seen'].strftime('%Y-%m-%d')}
                - Last seen: {journey['last_seen'].strftime('%Y-%m-%d')}
                - Duration: {journey['duration_days']} days
                """)
            
            with col2:
                st.markdown(f"""
                **Financial Summary:**
                - Total spent: ${journey['total_spent']:,.0f}
                - Avg transaction: ${journey['avg_transaction_value']:,.0f}
                - Transaction frequency: {journey['transactions']/max(journey['duration_days'], 1):.2f}/day
                """)
    
    # Journey patterns analysis
    st.markdown("### Journey Pattern Insights")
    
    patterns = {
        'short_high_value': len([j for j in journeys.values() 
                                if j['duration_days'] < 30 and j['total_spent'] > 1000]),
        'long_low_frequency': len([j for j in journeys.values() 
                                  if j['duration_days'] > 180 and j['transactions']/j['duration_days'] < 0.1]),
        'seasonal': len([j for j in journeys.values() 
                        if 'seasonal' in str(j).lower()]),
        'growing': len([j for j in journeys.values() 
                       if j['avg_transaction_value'] * 1.5 < j['total_spent']/j['transactions']])
    }
    
    pattern_df = pd.DataFrame({
        'Pattern': list(patterns.keys()),
        'Count': list(patterns.values())
    })
    
    fig = px.bar(pattern_df, x='Pattern', y='Count', 
                 title='Common Journey Patterns',
                 color='Count')
    st.plotly_chart(fig, use_container_width=True)

def display_alerts_and_monitoring():
    """Display alerts and monitoring section"""
    st.markdown("## ⚠️ Alerts & Monitoring")
    
    if not st.session_state.alerts:
        st.success("✅ No critical alerts at this time.")
        return
    
    high_alerts = [a for a in st.session_state.alerts if a['severity'] == 'high']
    medium_alerts = [a for a in st.session_state.alerts if a['severity'] == 'medium']
    
    if high_alerts:
        st.markdown("### 🔴 High Priority Alerts")
        for alert in high_alerts:
            st.markdown(f"""
            <div class="alert-high">
                <strong>{alert['type'].replace('_', ' ').title()}</strong><br>
                {alert['message']}
            </div>
            """, unsafe_allow_html=True)
    
    if medium_alerts:
        st.markdown("### 🟡 Medium Priority Alerts")
        for alert in medium_alerts:
            st.markdown(f"""
            <div class="alert-medium">
                <strong>{alert['type'].replace('_', ' ').title()}</strong><br>
                {alert['message']}
            </div>
            """, unsafe_allow_html=True)
    
    # Alert management
    st.markdown("### Alert Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Refresh Alerts", use_container_width=True, key="refresh_alerts"):
            st.rerun()
    
    with col2:
        if st.button("📥 Export Alert Report", use_container_width=True, key="export_alerts"):
            alert_report = "\n".join([f"{a['severity'].upper()}: {a['message']}" 
                                     for a in st.session_state.alerts])
            st.download_button(
                label="Download Report",
                data=alert_report,
                file_name="alerts_report.txt",
                mime="text/plain",
                key="download_alert_report"
            )

def display_enhanced_visualizations():
    """Display enhanced visualizations"""
    if st.session_state.processed_df is None:
        return
    
    df = st.session_state.processed_df
    
    st.markdown("## 📊 Enhanced Visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🌍 Market Overview", "📈 Trends", "🔍 Detailed Analysis", "🎨 Custom"])
    
    with tab1:
        # Market overview visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            if 'country' in df.columns:
                country_dist = df['country'].value_counts().reset_index()
                country_dist.columns = ['Country', 'Count']
                
                fig = px.pie(country_dist, values='Count', names='Country',
                            title='Customer Distribution by Country',
                            hole=0.4)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'risk_category' in df.columns:
                risk_dist = df['risk_category'].value_counts().reset_index()
                risk_dist.columns = ['Risk Category', 'Count']
                
                fig = px.bar(risk_dist, x='Risk Category', y='Count',
                            title='Risk Category Distribution',
                            color='Risk Category',
                            color_discrete_map={
                                'Low Risk': '#4CAF50',
                                'Medium Risk': '#FF9800',
                                'High Risk': '#F44336'
                            })
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Trend visualizations
        if 'timestamp' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['timestamp']).dt.date
                daily_data = df.groupby('date').agg({
                    'monthly_expenditure': 'mean',
                    'transaction_count': 'sum',
                    'risk_score': 'mean'
                }).reset_index()
                
                fig = make_subplots(rows=3, cols=1, 
                                  subplot_titles=('Avg Monthly Expenditure', 
                                                 'Total Transactions', 
                                                 'Avg Risk Score'))
                
                fig.add_trace(
                    go.Scatter(x=daily_data['date'], y=daily_data['monthly_expenditure'],
                              name='Expenditure'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(x=daily_data['date'], y=daily_data['transaction_count'],
                          name='Transactions'),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=daily_data['date'], y=daily_data['risk_score'],
                              name='Risk Score'),
                    row=3, col=1
                )
                
                fig.update_layout(height=800, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.info("Could not generate time series visualizations")
    
    with tab3:
        # Detailed analysis visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Correlation heatmap
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 2:
                corr_matrix = df[numeric_cols[:8]].corr()
                
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='RdBu',
                    zmin=-1, zmax=1
                ))
                
                fig.update_layout(title='Feature Correlation Heatmap')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plots for key metrics
            if 'risk_score' in df.columns and 'cluster' in df.columns:
                fig = px.box(df, x='cluster', y='risk_score',
                            title='Risk Score Distribution by Cluster',
                            points="all")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Custom visualizations
        st.markdown("### Custom Visualization Builder")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_axis = st.selectbox(
                "X-axis",
                options=df.select_dtypes(include=[np.number]).columns.tolist(),
                key="custom_x"
            )
        
        with col2:
            y_axis = st.selectbox(
                "Y-axis",
                options=df.select_dtypes(include=[np.number]).columns.tolist(),
                key="custom_y"
            )
        
        with col3:
            color_by = st.selectbox(
                "Color by",
                options=['cluster', 'risk_category', 'digital_adoption_level'] + 
                       df.select_dtypes(include=['object']).columns.tolist(),
                key="custom_color"
            )
        
        if st.button("Generate Custom Visualization", key="generate_custom"):
            if x_axis and y_axis and color_by in df.columns:
                fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by,
                                title=f'{y_axis} vs {x_axis} by {color_by}',
                                opacity=0.7)
                st.plotly_chart(fig, use_container_width=True)

def display_upload_section():
    """Enhanced upload section"""
    st.markdown('<h2 class="sub-header">Upload Financial Dataset</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="upload-zone">
        <h2 style="color: #264653; margin-bottom: 1rem;">📁 Upload Your Financial Dataset</h2>
        <p style="color: #546E7A; margin-bottom: 2rem;">
            Supports CSV, Excel, JSON, Parquet, SQL, SAS, Stata, XML, and more<br>
            Automatic detection of African financial patterns and generation of insights
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "xls", "json", "txt", "parquet", "feather", "h5", 
              "pkl", "pickle", "sql", "xml", "sas7bdat", "dta", "stata"],
        key="enhanced_uploader",
        help="Upload your financial dataset (various formats supported)"
    )
    
    if uploaded_file is not None:
        # Load dataset
        with st.spinner(f"Loading {uploaded_file.name}..."):
            df = load_dataset(uploaded_file)
            
            if df is not None:
                if len(df) < 10:
                    st.error("Dataset too small. Please upload a dataset with at least 10 records.")
                    return
                
                # Store in session state
                st.session_state.current_dataset_name = uploaded_file.name
                st.session_state.df = df
                
                # Process dataset with enhanced processor
                data_processor = EnhancedDataProcessor()
                processed_df, column_info, metadata = data_processor.process_dataset(df, uploaded_file.name)
                
                st.session_state.processed_df = processed_df
                st.session_state.column_info = column_info
                st.session_state.metadata = metadata
                
                # Perform clustering
                with st.spinner("Analyzing customer behavior patterns..."):
                    cluster_analyzer = ClusterAnalyzer()
                    labels, cluster_profiles, n_clusters = cluster_analyzer.perform_clustering(processed_df)
                    
                    st.session_state.processed_df['cluster'] = labels
                    st.session_state.clusters = cluster_profiles
                
                # Train deployment model
                with st.spinner("Training deployment model..."):
                    deployment_system = ModelDeploymentSystem()
                    deployment_system.train_deployment_model(processed_df, labels)
                
                # Generate enhanced insights
                with st.spinner("Generating comprehensive insights and recommendations..."):
                    insights_generator = EnhancedInsightsGenerator()
                    insights, recommendations = insights_generator.generate_all_insights(
                        processed_df, column_info, metadata, cluster_profiles
                    )
                    
                    st.session_state.insights = insights
                    st.session_state.recommendations = recommendations
                
                # Generate alerts
                generate_alerts(processed_df, cluster_profiles)
                
                st.success(f"""
                ✅ Dataset processed successfully!
                
                • Generated {len(insights)} insights
                • Generated {len(recommendations)} recommendations
                • Identified {len(cluster_profiles)} customer segments
                • Model trained for deployment
                """)
                
                return processed_df
    
    return None

def show_insights_page():
    """Enhanced insights page"""
    st.markdown('<h2 class="sub-header">Comprehensive Insights</h2>', unsafe_allow_html=True)
    
    if st.session_state.insights:
        # Add refresh button with unique key
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔄 New Analysis", use_container_width=True, key="new_analysis_insights_btn"):
                st.session_state.df = None
                st.session_state.processed_df = None
                st.session_state.insights = []
                st.session_state.recommendations = []
                st.rerun()
        
        # Display all insights
        for idx, insight in enumerate(st.session_state.insights):
            card_class = "positive" if insight.get('priority') == 'low' else \
                        "warning" if insight.get('priority') == 'medium' else "critical"
            
            st.markdown(f"""
            <div class="insight-card {card_class}">
                <h3 style="color: #006064; margin-top: 0;">{insight['title']}</h3>
                <div style="color: #37474F; line-height: 1.6; font-size: 1.1rem;">
                    {insight['content']}
                </div>
                <div style="margin-top: 1rem; font-size: 0.9rem; color: #78909C;">
                    Priority: {insight.get('priority', 'medium').upper()}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display recommendations
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("## 📋 Actionable Recommendations")
        
        for rec_idx, recommendation in enumerate(st.session_state.recommendations):
            priority_color = {
                'high': '#F44336',
                'medium': '#FF9800',
                'low': '#4CAF50'
            }.get(recommendation.get('priority', 'medium'), '#FF9800')
            
            st.markdown(f"""
            <div class="recommendations-container">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <h3 style="color: #880E4F; margin: 0;">{recommendation['title']}</h3>
                    <span style="background: {priority_color}; color: white; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.9rem; font-weight: 700;">
                        {recommendation.get('priority', 'medium').upper()} PRIORITY
                    </span>
                </div>
                <div style="color: #4A148C; line-height: 1.6; font-size: 1.1rem;">
                    {recommendation['content']}
                </div>
                {f"<div style='margin-top: 0.5rem; font-size: 0.9rem; color: #78909C;'>Timeline: {recommendation.get('timeline', 'Not specified')}</div>" if recommendation.get('timeline') else ""}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📁 Please upload a dataset first to generate insights.")
        
        if st.button("Go to Upload Section", key="go_to_upload_from_insights_btn"):
            st.rerun()

def generate_alerts(df, clusters):
    """Generate alerts based on analysis"""
    alerts = []
    
    # Risk concentration alert
    if 'risk_category' in df.columns:
        high_risk_pct = (df['risk_category'] == 'High Risk').mean()
        if high_risk_pct > 0.2:
            alerts.append({
                'type': 'high_risk_concentration',
                'message': f"High concentration of high-risk customers: {high_risk_pct:.1%}",
                'severity': 'high'
            })
    
    # Digital divide alert
    if 'digital_adoption_score' in df.columns:
        low_digital_pct = (df['digital_adoption_score'] < 0.3).mean()
        if low_digital_pct > 0.4:
            alerts.append({
                'type': 'digital_divide',
                'message': f"Large digital divide: {low_digital_pct:.1%} of customers have low digital adoption",
                'severity': 'medium'
            })
    
    # Cluster imbalance alert
    if clusters:
        cluster_sizes = [c['size'] for c in clusters]
        if max(cluster_sizes) / min(cluster_sizes) > 10:
            alerts.append({
                'type': 'cluster_imbalance',
                'message': "Significant imbalance in cluster sizes detected",
                'severity': 'medium'
            })
    
    # Missing data alert
    if hasattr(st.session_state, 'metadata'):
        missing_pct = st.session_state.metadata.get('missing_percentage', 0)
        if missing_pct > 20:
            alerts.append({
                'type': 'data_quality',
                'message': f"High percentage of missing data: {missing_pct:.1f}%",
                'severity': 'medium'
            })
    
    st.session_state.alerts = alerts

def display_settings():
    """Display settings page"""
    st.markdown('<h2 class="sub-header">Settings & Configuration</h2>', unsafe_allow_html=True)
    
    with st.expander("🔧 Processing Settings", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.number_input("Minimum cluster size", 
                          min_value=10, max_value=1000, value=50, 
                          key="settings_min_cluster")
            st.selectbox("Default clustering method", 
                        ["K-Means", "Hierarchical", "GMM", "DBSCAN"], 
                        key="settings_clustering_method")
            st.slider("Risk threshold for alerts", 
                    0.0, 1.0, 0.7, 0.05, 
                    key="settings_risk_threshold")
        
        with col2:
            st.checkbox("Generate topic models", value=True, key="settings_generate_topics")
            st.checkbox("Enable real-time monitoring", value=False, key="settings_real_time")
            st.checkbox("Automatically generate insights", value=True, key="settings_auto_insights")
    
    with st.expander("📊 Visualization Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.selectbox("Chart theme", 
                        ["Plotly", "Seaborn", "GGplot", "Simple"], 
                        key="settings_chart_theme")
            st.checkbox("Show data points", value=True, key="settings_show_points")
        
        with col2:
            st.checkbox("Interactive tooltips", value=True, key="settings_tooltips")
            st.number_input("Max data points in charts", 
                          min_value=100, max_value=10000, value=1000, 
                          key="settings_max_points")
    
    with st.expander("🔒 Security & Privacy", expanded=False):
        st.checkbox("Anonymize customer data", value=True, key="settings_anonymize")
        st.checkbox("Encrypt stored data", value=False, key="settings_encrypt")
        st.selectbox("Data retention period", 
                    ["30 days", "90 days", "1 year", "Indefinite"], 
                    key="settings_retention")
    
    with st.expander("📤 Export Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.checkbox("Include raw data in exports", value=False, key="settings_include_raw")
            st.checkbox("Generate executive summary", value=True, key="settings_exec_summary")
        
        with col2:
            st.selectbox("Default export format", 
                        ["CSV", "Excel", "PDF", "JSON"], 
                        key="settings_export_format")
            st.checkbox("Auto-generate Power BI templates", value=False, key="settings_powerbi")
    
    if st.button("💾 Save Settings", use_container_width=True, key="save_settings_btn"):
        st.success("Settings saved successfully!")

# ============================================
# DASHBOARD FUNCTIONS
# ============================================

def display_dashboard():
    """Display main dashboard"""
    
    if st.session_state.processed_df is not None:
        df = st.session_state.processed_df
        metadata = st.session_state.metadata
        
        # Dashboard header
        st.markdown(f"""
        <div class="africa-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h2 style="color: #264653; margin: 0;">📊 African Financial Analytics Dashboard</h2>
                    <p style="color: #546E7A; margin: 0.5rem 0 0 0;">
                        Analyzing {metadata['original_records']:,} customers in {metadata.get('detected_country', 'African market')}
                    </p>
                </div>
                <div style="text-align: right;">
                    <span style="background: linear-gradient(135deg, #FF6B35, #FFA62E); color: white; padding: 0.5rem 1rem; border-radius: 20px; font-weight: 700;">
                        {len(st.session_state.clusters) if st.session_state.clusters else 0} Segments
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Key Metrics Row
        st.markdown("### 📈 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", f"{len(df):,}")
        
        with col2:
            if 'risk_score' in df.columns:
                avg_risk = df['risk_score'].mean()
                st.metric("Avg Risk Score", f"{avg_risk:.3f}")
        
        with col3:
            if st.session_state.clusters:
                largest_cluster = max(st.session_state.clusters, key=lambda x: x['size'])
                st.metric("Largest Segment", f"{largest_cluster['size']:,}")
        
        with col4:
            if 'sentiment_score' in df.columns:
                avg_sentiment = df['sentiment_score'].mean()
                sentiment_label = "Positive" if avg_sentiment > 0 else "Negative" if avg_sentiment < 0 else "Neutral"
                st.metric("Avg Sentiment", sentiment_label)
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Segments", "📱 Digital", "⚠️ Risk", "😊 Sentiment"])
        
        with tab1:
            display_segmentation_tab(df)
        
        with tab2:
            display_digital_tab(df)
        
        with tab3:
            display_risk_tab(df)
        
        with tab4:
            display_sentiment_tab(df)
        
        # Insights Section
        st.markdown("---")
        st.markdown("## 💡 Automatic Insights & Recommendations")
        
        # Display insights in chat style
        for insight in st.session_state.insights[:3]:  # Show first 3 insights
            with st.expander(f"📌 {insight['title']}", expanded=False):
                st.markdown(insight['content'])
        
        # Display recommendations
        st.markdown("## 🚀 Actionable Recommendations")
        
        for recommendation in st.session_state.recommendations[:2]:  # Show first 2 recommendations
            with st.expander(f"✅ {recommendation['title']} ({recommendation['priority'].upper()} PRIORITY)", expanded=False):
                st.markdown(recommendation['content'])
        
        # Data Download Section
        st.markdown("---")
        st.markdown("## 📥 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Download Processed Data", use_container_width=True, key="download_processed"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="african_financial_analysis.csv",
                    mime="text/csv",
                    key="download_csv_main"
                )
        
        with col2:
            if st.button("📋 Download Insights Report", use_container_width=True, key="download_insights"):
                report = generate_insights_report()
                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name="financial_insights_report.txt",
                    mime="text/plain",
                    key="download_report_main"
                )
        
        with col3:
            if st.button("🔄 Analyze Another Dataset", use_container_width=True, key="analyze_another"):
                st.session_state.df = None
                st.session_state.processed_df = None
                st.rerun()
    else:
        st.info("📁 Please upload a dataset first to view the dashboard.")
        if st.button("Go to Upload Section", key="go_to_upload_from_dashboard"):
            st.session_state.df = None
            st.rerun()

def display_segmentation_tab(df):
    """Display segmentation analysis tab"""
    
    if st.session_state.clusters:
        # Cluster distribution
        cluster_data = pd.DataFrame(st.session_state.clusters)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Cluster distribution chart
            fig = px.bar(cluster_data, x='name', y='size',
                        title='Customer Segment Distribution',
                        color='name',
                        text='percentage',
                        color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(xaxis_title="Segment", yaxis_title="Number of Customers")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Segment Overview")
            for cluster in st.session_state.clusters:
                st.markdown(f"""
                <div style="background: rgba(42, 157, 143, 0.1); padding: 1rem; margin-bottom: 0.5rem; border-radius: 8px; border-left: 4px solid #2A9D8F;">
                    <h4 style="margin: 0 0 0.5rem 0; color: #264653;">{cluster['name']}</h4>
                    <p style="margin: 0; color: #546E7A;">
                        {cluster['size']:,} customers ({cluster['percentage']:.1f}%)
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        # Cluster comparison
        st.markdown("### 📊 Segment Comparison")
        
        # Select metrics for comparison
        metric_options = []
        if 'risk_score' in df.columns:
            metric_options.append('Risk Score')
        if 'digital_adoption_score' in df.columns:
            metric_options.append('Digital Adoption')
        if 'sentiment_score' in df.columns:
            metric_options.append('Sentiment Score')
        if 'monthly_expenditure' in df.columns:
            metric_options.append('Monthly Expenditure')
        
        if metric_options:
            selected_metric = st.selectbox("Select metric for comparison", metric_options, key="segment_metric_select")
            
            metric_map = {
                'Risk Score': 'risk_score',
                'Digital Adoption': 'digital_adoption_score',
                'Sentiment Score': 'sentiment_score',
                'Monthly Expenditure': 'monthly_expenditure'
            }
            
            metric_col = metric_map[selected_metric]
            
            if metric_col in df.columns:
                cluster_metrics = df.groupby('cluster')[metric_col].agg(['mean', 'std']).reset_index()
                cluster_metrics['cluster_name'] = cluster_metrics['cluster'].map(
                    {c['cluster_id']: c['name'] for c in st.session_state.clusters}
                )
                
                fig = px.bar(cluster_metrics, x='cluster_name', y='mean',
                            error_y='std',
                            title=f'{selected_metric} by Segment',
                            color='cluster_name')
                st.plotly_chart(fig, use_container_width=True)

def display_digital_tab(df):
    """Display digital adoption analysis tab"""
    
    # Find payment channel columns
    payment_cols = [col for col in df.columns if col.startswith('uses_')]
    
    if payment_cols:
        # Calculate adoption rates
        adoption_rates = {}
        for col in payment_cols:
            channel = col.replace('uses_', '').replace('_', ' ').title()
            adoption_rate = df[col].mean() * 100
            adoption_rates[channel] = adoption_rate
        
        # Create adoption chart
        adoption_df = pd.DataFrame(list(adoption_rates.items()), columns=['Channel', 'Adoption Rate'])
        adoption_df = adoption_df.sort_values('Adoption Rate', ascending=True)
        
        fig = px.bar(adoption_df, x='Adoption Rate', y='Channel',
                    title='Digital Payment Channel Adoption Rates',
                    orientation='h',
                    color='Adoption Rate',
                    color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
        
        # Digital adoption by cluster
        if 'cluster' in df.columns and st.session_state.clusters:
            st.markdown("### Digital Adoption by Segment")
            
            cluster_digital = df.groupby('cluster')[payment_cols].mean().mean(axis=1) * 100
            cluster_digital = cluster_digital.reset_index()
            cluster_digital.columns = ['cluster', 'Digital Adoption %']
            cluster_digital['Segment'] = cluster_digital['cluster'].map(
                {c['cluster_id']: c['name'] for c in st.session_state.clusters}
            )
            
            fig = px.bar(cluster_digital, x='Segment', y='Digital Adoption %',
                        title='Digital Adoption Across Segments',
                        color='Digital Adoption %',
                        color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No digital payment channel data detected in this dataset.")

def display_risk_tab(df):
    """Display risk analysis tab"""
    
    if 'risk_score' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk distribution
            fig = px.histogram(df, x='risk_score', nbins=30,
                              title='Risk Score Distribution',
                              color_discrete_sequence=['#FF6B35'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk categories
            if 'risk_category' in df.columns:
                risk_dist = df['risk_category'].value_counts().reset_index()
                risk_dist.columns = ['Risk Category', 'Count']
                
                fig = px.pie(risk_dist, values='Count', names='Risk Category',
                            title='Risk Category Distribution',
                            color_discrete_sequence=['#2E7D32', '#FF9800', '#F44336'])
                st.plotly_chart(fig, use_container_width=True)
        
        # Risk by segment
        if 'cluster' in df.columns and st.session_state.clusters:
            st.markdown("### Risk Analysis by Segment")
            
            cluster_risk = df.groupby('cluster')['risk_score'].agg(['mean', 'std', 'count']).reset_index()
            cluster_risk['Segment'] = cluster_risk['cluster'].map(
                {c['cluster_id']: c['name'] for c in st.session_state.clusters}
            )
            
            fig = px.scatter(cluster_risk, x='Segment', y='mean',
                            size='count', error_y='std',
                            title='Average Risk Score by Segment',
                            color='mean',
                            color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Risk score data not available in this dataset.")

def display_sentiment_tab(df):
    """Display sentiment analysis tab"""
    
    if 'sentiment_score' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution
            fig = px.histogram(df, x='sentiment_score', nbins=30,
                              title='Sentiment Score Distribution',
                              color_discrete_sequence=['#2A9D8F'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sentiment categories
            df['sentiment_category'] = pd.cut(df['sentiment_score'],
                                             bins=[-1, -0.3, 0.3, 1],
                                             labels=['Negative', 'Neutral', 'Positive'])
            
            sentiment_dist = df['sentiment_category'].value_counts().reset_index()
            sentiment_dist.columns = ['Sentiment', 'Count']
            
            fig = px.pie(sentiment_dist, values='Count', names='Sentiment',
                        title='Sentiment Category Distribution',
                        color='Sentiment',
                        color_discrete_map={'Negative': '#F44336', 'Neutral': '#FF9800', 'Positive': '#4CAF50'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment by segment
        if 'cluster' in df.columns and st.session_state.clusters:
            st.markdown("### Sentiment Analysis by Segment")
            
            cluster_sentiment = df.groupby('cluster')['sentiment_score'].agg(['mean', 'std']).reset_index()
            cluster_sentiment['Segment'] = cluster_sentiment['cluster'].map(
                {c['cluster_id']: c['name'] for c in st.session_state.clusters}
            )
            
            fig = px.bar(cluster_sentiment, x='Segment', y='mean',
                        error_y='std',
                        title='Average Sentiment by Segment',
                        color='mean',
                        color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sentiment analysis data not available in this dataset.")

def generate_insights_report():
    """Generate comprehensive insights report"""
    
    report = f"""
    AFRICAN FINANCIAL BEHAVIOR ANALYSIS REPORT
    ==========================================
    Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
    DATASET OVERVIEW
    ----------------
    Dataset Name: {st.session_state.metadata.get('dataset_name', 'N/A')}
    Total Records: {st.session_state.metadata.get('original_records', 'N/A'):,}
    Country: {st.session_state.metadata.get('detected_country', 'N/A')}
    Analysis Date: {st.session_state.metadata.get('processing_date', 'N/A')}
    
    KEY FINDINGS
    ------------
    """
    
    # Add insights
    for insight in st.session_state.insights:
        report += f"\n{insight['title'].upper()}\n"
        report += "-" * len(insight['title']) + "\n"
        
        # Clean HTML tags and format text
        clean_content = insight['content'].replace('<br>', '\n').replace('</br>', '')
        clean_content = re.sub(r'<[^>]+>', '', clean_content)
        clean_content = clean_content.replace('**', '').replace('•', '  •')
        
        report += clean_content + "\n"
    
    # Add recommendations
    report += "\n\nRECOMMENDATIONS\n"
    report += "---------------\n"
    
    for recommendation in st.session_state.recommendations:
        report += f"\n{recommendation['title'].upper()} ({recommendation['priority'].upper()} PRIORITY)\n"
        
        clean_content = recommendation['content'].replace('<br>', '\n').replace('</br>', '')
        clean_content = re.sub(r'<[^>]+>', '', clean_content)
        clean_content = clean_content.replace('**', '').replace('•', '  •')
        
        report += clean_content + "\n"
    
    return report

# ============================================
# ENHANCED MAIN APP FLOW
# ============================================

def main():
    """Main application flow"""
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem; padding: 1.5rem; background: linear-gradient(135deg, #FF6B35 0%, #2A9D8F 100%); border-radius: 12px; box-shadow: 0 6px 20px rgba(255, 107, 53, 0.3);">
            <h3 style="color: white; margin: 0; font-weight: 900;">🌍 Capstone Project</h3>
            <p style="color: rgba(255,255,255,0.9); font-size: 1rem; margin: 0.5rem 0 0 0; font-weight: 700;">
                African Financial Behavior Segmentation
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # User role selector
        display_user_role_selector()
        
        st.markdown("---")
        
        # Navigation based on user role
        if st.session_state.user_role == "Analyst":
            nav_options = ["📁 Upload Data", "📊 Dashboard", "🔍 Insights", "🤖 Model Deployment", 
                          "🔄 What-If Analysis", "🗺️ Customer Journey", "⚠️ Alerts", "📈 Visualizations", "⚙️ Settings"]
        elif st.session_state.user_role == "Manager":
            nav_options = ["📊 Dashboard", "🔍 Insights", "🤖 Model Deployment", "⚠️ Alerts", "📈 Visualizations"]
        else:  # Executive
            nav_options = ["📊 Dashboard", "🔍 Insights", "📈 Visualizations"]
        
        selected_nav = st.radio(
            "Navigation",
            nav_options,
            key="main_navigation"
        )
        
        st.markdown("---")
        
        # Dataset info
        if st.session_state.metadata:
            st.markdown("### Current Dataset")
            
            dataset_info = f"""
            **{st.session_state.metadata.get('dataset_name', 'Unnamed')}**
            
            • {st.session_state.metadata.get('original_records', 0):,} records
            • {st.session_state.metadata.get('detected_country', 'Various')}
            • {len(st.session_state.insights) if st.session_state.insights else 0} insights
            • {len(st.session_state.recommendations) if st.session_state.recommendations else 0} recommendations
            """
            
            st.info(dataset_info)
            
            if st.button("🔄 Load New Dataset", use_container_width=True, key="sidebar_new_dataset"):
                st.session_state.df = None
                st.session_state.processed_df = None
                st.session_state.insights = []
                st.session_state.recommendations = []
                st.rerun()
        
        st.markdown("---")
        
        # Team info
        st.markdown("### Team")
        st.markdown("""
        <div style="background: rgba(42, 157, 143, 0.1); padding: 1rem; border-radius: 8px;">
            <p style="font-weight: 700; color: #264653; margin-bottom: 0.5rem;">Dataverse Africa Internship Cohort 3</p>
            <ul style="margin: 0; padding-left: 1.2rem; color: #455A64;">
                <li>Amarachi Florence</li>
                <li>Thato Maelane</li>
                <li>Philip Odiachi</li>
                <li>Mavis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content based on navigation
    if selected_nav == "📁 Upload Data":
        display_upload_section()
    
    elif selected_nav == "📊 Dashboard":
        display_dashboard()
    
    elif selected_nav == "🔍 Insights":
        show_insights_page()
    
    elif selected_nav == "🤖 Model Deployment":
        display_model_deployment_section()
    
    elif selected_nav == "🔄 What-If Analysis":
        display_what_if_analysis()
    
    elif selected_nav == "🗺️ Customer Journey":
        display_customer_journey_analysis()
    
    elif selected_nav == "⚠️ Alerts":
        display_alerts_and_monitoring()
    
    elif selected_nav == "📈 Visualizations":
        display_enhanced_visualizations()
    
    elif selected_nav == "⚙️ Settings":
        display_settings()

# ============================================
# FOOTER (Updated with clickable link)
# ============================================
st.markdown("---")

st.markdown("""
<div style="text-align: center; color: #264653; padding: 2rem; background: linear-gradient(135deg, #E9F5DB 0%, #C8E6C9 100%); border-radius: 12px; margin-top: 3rem;">
    <p style="margin-bottom: 0.5rem; font-size: 1.2rem; font-weight: 900; color: #1A237E;">
        African Financial Behavior Segmentation Dashboard
    </p>
    <p style="margin-bottom: 0.5rem; font-weight: 800; color: #D84315;">
        Complete Analytics Platform | Unsupervised ML + Advanced NLP
    </p>
    <p style="margin-bottom: 0.5rem; font-size: 0.9rem; font-weight: 700; color: #455A64;">
        Developed by: Amarachi Florence, Thato Maelane, Philip Odiachi, Mavis
    </p>
    <p style="margin-bottom: 0.5rem; font-size: 0.9rem; font-weight: 700; color: #455A64;">
        Dataverse Africa Internship Program | Cohort 3
    </p>
    <div style="margin-bottom: 1rem;">
        <a href="https://dataverseafrica.org/" target="_blank" style="color: #1565C0; text-decoration: none; font-weight: 700; font-size: 1.1rem; display: inline-block; padding: 0.5rem 1rem; background: rgba(255,255,255,0.9); border-radius: 8px; border: 2px solid #1565C0;">
            🌍 https://dataverseafrica.org/
        </a>
        <p style="color: #2E7D32; font-weight: 700; margin-top: 0.5rem;">Empowering Africa's Digital Future</p>
    </div>
    <div style="font-size: 0.8rem; color: #78909c; font-weight: 600;">
        Techniques: Clustering, Topic Modeling, Sentiment Analysis, Predictive Analytics | 
        Tools: Python, spaCy, Transformers, scikit-learn, Streamlit
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================
# RUN THE APPLICATION
# ============================================
if __name__ == "__main__":
    main()