"""
African Financial Behavior Segmentation Dashboard
Streamlit Web Application with Auto Insights
"""

# ============================================
# SET PAGE CONFIG
# ============================================
import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime 
import os
import warnings
import io
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # ADDED: VADER import

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="African Financial Segmentation Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS (MINIMAL)
# ============================================
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    p, div, span, label, .stMarkdown {
        color: #f0f2f6 !important;
    }
    
    .title-container {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    } 
    
    .title-container h1 {
        margin-bottom: 0.5rem;
        font-size: 2.5rem;
    }
    
    .title-container h3 {
        margin-top: 0;
        font-size: 1.3rem;
        opacity: 0.9;
    }
    
    .cluster-card {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white; 
        text-align: center;
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    } 
    
    .cluster-card h4 {
        font-size: 16px;
        margin-bottom: 10px;
        line-height: 1.3;
        font-weight: bold;
    } 
    
    .cluster-card p {
        font-size: 14px;
        margin: 5px 0;
        line-height: 1.4;
    }
    
    .info-box {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #3b82f6;
    }
    
    .info-box h4 {
        color: #60a5fa;
        margin-top: 0;
    }
    
    .feature-box {
        background: rgba(30, 64, 175, 0.2);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #3b82f6;
    }
    
    .feature-box h5 {
        color: #60a5fa;
        margin: 0 0 0.5rem 0;
    }
    
    .feature-box p {
        margin: 0;
        font-size: 14px;
    }
    
    .step-container {
        background: rgba(30, 58, 138, 0.3);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid rgba(59, 130, 246, 0.5);
    }
    
    .step-number {
        display: inline-block;
        background: #3b82f6;
        color: white;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        text-align: center;
        line-height: 30px;
        margin-right: 10px;
        font-weight: bold;
    }
    
    .step-title {
        display: inline-block;
        color: #60a5fa;
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    .footer-container {
        background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 3rem;
        text-align: center;
    }
    
    .footer-text {
        font-size: 14px;
        color: white !important;
        margin: 0.3rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# INITIALIZE VADER SENTIMENT ANALYZER
# ============================================
# Initialize VADER sentiment analyzer
@st.cache_resource
def load_sentiment_analyzer():
    """Load and cache the VADER sentiment analyzer."""
    try:
        analyzer = SentimentIntensityAnalyzer()
        return analyzer
    except Exception as e:
        st.error(f"Error loading VADER sentiment analyzer: {str(e)}")
        # Fallback to basic sentiment analysis
        return None

vader_analyzer = load_sentiment_analyzer()

# ============================================
# SESSION STATE INITIALIZATION
# ============================================
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'clusters' not in st.session_state:
    st.session_state.clusters = None
if 'insights' not in st.session_state:
    st.session_state.insights = []
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# ============================================
# HELPER FUNCTIONS
# ============================================

def load_dataset(uploaded_file):
    """Load dataset from uploaded file"""
    try:
        file_name = uploaded_file.name.lower()
        
        if file_name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            st.error("Unsupported file format. Please upload CSV or Excel file.")
            return None
        
        # Clean column names
        df.columns = [col.strip().replace(' ', '_').replace('-', '_').lower() for col in df.columns]
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def create_synthetic_dataset(n_records=5200):
    """Create realistic synthetic dataset for demonstration"""
    np.random.seed(42)
    
    data = {
        'customer_id': np.random.randint(10000, 50000, n_records),
        'monthly_expenditure': np.clip(np.random.lognormal(11.5, 0.8, n_records), 20000, 500000),
        'income_level': np.random.choice(['low', 'lower_middle', 'middle', 'upper_middle', 'high'], 
                                        n_records, p=[0.25, 0.30, 0.25, 0.15, 0.05]),
        'spending_category': np.random.choice(['groceries', 'rent', 'utilities', 'transport', 'health', 
                                             'education', 'entertainment', 'online_shopping', 'savings'],
                                            n_records, p=[0.20, 0.15, 0.12, 0.10, 0.08, 0.10, 0.10, 0.10, 0.05]),
        'saving_behavior': np.random.choice(['poor', 'average', 'good'], n_records, p=[0.35, 0.45, 0.20]),
        'credit_score': np.clip(np.random.normal(580, 100, n_records), 300, 850),
        'loan_status': np.random.choice(['no_loan', 'active_loan', 'default_risk'], 
                                       n_records, p=[0.60, 0.35, 0.05]),
        'customer_feedback': np.random.choice([
            'charges are confusing and unclear',
            'loan process takes too long',
            'payment failed multiple times',
            'the service is excellent',
            'i need better support',
            "i don't understand my loan deductions",
            'overall experience is great',
            'customer service is slow to respond',
            'app keeps crashing during payments',
            'the interface is confusing',
            'transaction was fast and smooth',
            'very satisfied with the platform'
        ], n_records),
        'transaction_channel': np.random.choice(['ussd', 'web', 'mobile_app', 'pos'], 
                                               n_records, p=[0.40, 0.20, 0.25, 0.15]),
        'location': np.random.choice(['lagos', 'abuja', 'kano', 'ibadan', 'port_harcourt', 
                                    'kaduna', 'enugu', 'accra', 'nairobi', 'kampala'],
                                   n_records, p=[0.25, 0.15, 0.10, 0.10, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05]),
    }
    
    df = pd.DataFrame(data)
    
    # Add realistic loan amounts
    mask_active = df['loan_status'] == 'active_loan'
    mask_default = df['loan_status'] == 'default_risk'
    
    df['loan_amount'] = 0
    df.loc[mask_active, 'loan_amount'] = np.clip(np.random.lognormal(13, 0.5, mask_active.sum()), 50000, 3000000)
    df.loc[mask_default, 'loan_amount'] = np.clip(np.random.lognormal(13.5, 0.7, mask_default.sum()), 100000, 5000000)
    
    return df

def analyze_sentiment(text):
    """Analyze sentiment using VADER (improved NLP)."""
    if pd.isna(text) or str(text).strip() == '':
        return 0.0
    
    # Use VADER if available, otherwise fallback to basic analysis
    if vader_analyzer is not None:
        try:
            # Get VADER sentiment scores
            scores = vader_analyzer.polarity_scores(str(text))
            # Return compound score (-1 to 1) where -1 is negative, 1 is positive
            return scores['compound']
        except Exception as e:
            st.warning(f"VADER analysis failed for text, using fallback: {str(e)[:50]}...")
    
    # Fallback to basic sentiment analysis (original function)
    text = str(text).lower()
    
    positive_words = ['excellent', 'great', 'good', 'fast', 'smooth', 'easy', 'helpful', 
                     'satisfied', 'happy', 'recommend', 'wonderful', 'perfect', 'love', 'amazing']
    negative_words = ['confusing', 'unclear', 'failed', 'crashing', 'slow', 'problem', 
                     'issue', 'error', 'bad', 'poor', 'terrible', 'awful', 'horrible', 'hate', 'difficult']
    
    positive_score = sum(1 for word in positive_words if word in text)
    negative_score = sum(1 for word in negative_words if word in text)
    
    if positive_score + negative_score > 0:
        sentiment = (positive_score - negative_score) / (positive_score + negative_score)
    else:
        sentiment = 0.0
    
    return sentiment

def detect_column_types(df):
    """Detect column types in the dataset"""
    column_types = {}
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Check for financial amount columns
        amount_keywords = ['amount', 'value', 'price', 'cost', 'expense', 'expenditure', 'spend', 'loan', 'salary']
        if any(keyword in col_lower for keyword in amount_keywords):
            if df[col].dtype in ['int64', 'float64']:
                column_types[col] = 'financial_amount'
            else:
                column_types[col] = 'categorical'
        
        # Check for score/rating columns
        elif any(keyword in col_lower for keyword in ['score', 'rating', 'credit']):
            if df[col].dtype in ['int64', 'float64']:
                column_types[col] = 'score'
            else:
                column_types[col] = 'categorical'
        
        # Check for ID columns
        elif any(keyword in col_lower for keyword in ['id', 'customer', 'client', 'user', 'number']):
            if df[col].nunique() > len(df) * 0.9:
                column_types[col] = 'identifier'
            else:
                column_types[col] = 'categorical'
        
        # Check for date columns
        elif any(keyword in col_lower for keyword in ['date', 'time', 'year', 'month', 'day']):
            column_types[col] = 'date'
        
        # Check for text/feedback columns
        elif any(keyword in col_lower for keyword in ['feedback', 'review', 'comment', 'note', 'description']):
            column_types[col] = 'text'
        
        # Check for categorical columns
        elif df[col].dtype == 'object' and df[col].nunique() < 50:
            column_types[col] = 'categorical'
        
        # Check for text columns
        elif df[col].dtype == 'object' and df[col].nunique() >= 50:
            column_types[col] = 'text'
        
        # Default to numeric
        elif df[col].dtype in ['int64', 'float64']:
            column_types[col] = 'numeric'
        
        else:
            column_types[col] = 'other'
    
    return column_types

def process_data(df):
    """Process data for analysis - handles any dataset"""
    df_processed = df.copy()
    
    # Detect column types
    column_types = detect_column_types(df_processed)
    
    # Handle missing values
    for col in df_processed.columns:
        if df_processed[col].isnull().sum() > 0:
            if column_types.get(col, '') in ['numeric', 'financial_amount', 'score']:
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            else:
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode().iloc[0] if len(df_processed[col].mode()) > 0 else 'unknown')
    
    # Create risk score if credit data exists
    credit_cols = [col for col in df_processed.columns if any(keyword in col.lower() 
                   for keyword in ['credit', 'score', 'risk'])]
    
    if credit_cols:
        credit_col = credit_cols[0]
        if df_processed[credit_col].dtype in ['int64', 'float64']:
            # Normalize credit score to 0-1 risk (lower credit = higher risk)
            min_val = df_processed[credit_col].min()
            max_val = df_processed[credit_col].max()
            if max_val > min_val:
                df_processed['risk_score'] = 1 - (df_processed[credit_col] - min_val) / (max_val - min_val)
            else:
                df_processed['risk_score'] = 0.5
    
    # Analyze sentiment if text columns exist - NOW USING VADER
    text_cols = [col for col, dtype in column_types.items() if dtype == 'text']
    
    if text_cols:
        text_col = text_cols[0]
        df_processed['sentiment_score'] = df_processed[text_col].apply(analyze_sentiment)
        
        # Update sentiment labeling logic for VADER scores
        # VADER compound scores: positive (>0.05), negative (<-0.05), neutral (between -0.05 and 0.05)
        df_processed['sentiment_label'] = df_processed['sentiment_score'].apply(
            lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral')
        )
    
    return df_processed, column_types

def perform_clustering(df_processed):
    """Perform KMeans clustering on numeric features"""
    # Select numeric features
    numeric_features = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove potential ID columns and already created scores
    id_keywords = ['id', 'index', 'count', 'sentiment', 'risk_score']
    numeric_features = [col for col in numeric_features if not any(keyword in col.lower() for keyword in id_keywords)]
    
    if len(numeric_features) < 3:
        st.warning("Not enough numeric features for clustering. Using default segmentation.")
        df_processed['cluster'] = 0
        cluster_profiles = [{
            'cluster_id': 0,
            'size': len(df_processed),
            'percentage': 100.0,
            'cluster_name': 'All Customers'
        }]
        return df_processed, cluster_profiles
    
    # Use up to 8 features for clustering
    numeric_features = numeric_features[:8]
    X = df_processed[numeric_features].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA if many features
    if X_scaled.shape[1] > 5:
        pca = PCA(n_components=min(5, X_scaled.shape[1]), random_state=42)
        X_reduced = pca.fit_transform(X_scaled)
    else:
        X_reduced = X_scaled
    
    # Determine optimal number of clusters (4-6 for interpretability)
    n_clusters = min(6, max(4, len(df_processed) // 1000))
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_reduced)
    
    # Assign clusters
    df_processed['cluster'] = clusters
    
    # Create cluster profiles
    cluster_profiles = []
    for cluster_id in sorted(df_processed['cluster'].unique()):
        cluster_data = df_processed[df_processed['cluster'] == cluster_id]
        
        profile = {
            'cluster_id': cluster_id,
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(df_processed) * 100,
        }
        
        # Add numeric statistics for key columns
        key_numeric_cols = ['credit_score', 'monthly_expenditure', 'loan_amount', 'risk_score', 'sentiment_score']
        for col in key_numeric_cols:
            if col in cluster_data.columns:
                profile[f'avg_{col}'] = cluster_data[col].mean()
        
        # Add other numeric columns
        other_numeric = [col for col in cluster_data.select_dtypes(include=[np.number]).columns 
                        if col not in key_numeric_cols and col != 'cluster']
        for col in other_numeric[:3]:  # Top 3 other numeric columns
            profile[f'avg_{col}'] = cluster_data[col].mean()
        
        cluster_profiles.append(profile)
    
    # Name clusters based on characteristics
    for profile in cluster_profiles:
        characteristics = []
        
        # Check key metrics against overall averages
        for col in ['loan_amount', 'monthly_expenditure', 'credit_score', 'risk_score', 'sentiment_score']:
            avg_key = f'avg_{col}'
            if avg_key in profile:
                if col in df_processed.columns:
                    overall_avg = df_processed[col].mean()
                    if col == 'sentiment_score':
                        # For sentiment: positive is good, negative is bad
                        if profile[avg_key] > overall_avg + 0.2:
                            characteristics.append(f"Very Positive")
                        elif profile[avg_key] < overall_avg - 0.2:
                            characteristics.append(f"Very Negative")
                    else:
                        if profile[avg_key] > overall_avg * 1.3:
                            characteristics.append(f"High {col.replace('_', ' ')}")
                        elif profile[avg_key] < overall_avg * 0.7:
                            characteristics.append(f"Low {col.replace('_', ' ')}")
        
        if characteristics:
            # Combine characteristics for naming (limit to 2 for readability)
            unique_chars = list(set(characteristics))[:2]
            if len(unique_chars) >= 2:
                profile['cluster_name'] = f"{unique_chars[0]}, {unique_chars[1]} Customers"
            else:
                profile['cluster_name'] = f"{unique_chars[0]} Customers"
        else:
            profile['cluster_name'] = f"Segment {profile['cluster_id'] + 1}"
    
    return df_processed, cluster_profiles

def generate_natural_language_insights(df_processed, cluster_profiles):
    """Generate natural language insights for non-technical audience"""
    insights = []
    
    # 1. Executive Summary - Enhanced with NLP info
    exec_summary = f"""
    Executive Summary
    
    Our analysis of {len(df_processed):,} customer records has revealed valuable insights into your financial customer base. We've identified {len(cluster_profiles)} distinct customer segments, each with unique characteristics and behaviors that can inform your business strategy.
    
    This analysis uses advanced NLP techniques (VADER sentiment analysis) to understand customer feedback, providing more accurate insights into customer satisfaction and concerns.
    
    The analysis shows a well-distributed customer base with clear patterns in financial behavior. Understanding these segments will help you tailor your products, services, and communication to better meet customer needs and drive business growth.
    """
    insights.append(exec_summary)
    
    # 2. Data Quality Insights
    missing_pct = df_processed.isnull().sum().sum() / (len(df_processed) * len(df_processed.columns)) * 100
    data_quality = f"""
    Data Quality Assessment
    
    Your dataset shows excellent data quality with only {missing_pct:.1f}% missing values. This high-quality data foundation ensures reliable insights and recommendations.
    
    Key Data Points:
    - Total records analyzed: {len(df_processed):,}
    - Number of features: {len(df_processed.columns)}
    - Customer segments identified: {len(cluster_profiles)}
    - Data completeness: {100 - missing_pct:.1f}%
    
    The clean dataset allows for accurate segmentation and reliable business insights.
    """
    insights.append(data_quality)
    
    # 3. Financial Behavior Insights
    financial_cols = [col for col in df_processed.select_dtypes(include=[np.number]).columns 
                     if any(keyword in col.lower() for keyword in ['amount', 'expenditure', 'spend', 'credit', 'loan'])]
    
    if financial_cols:
        financial_insights = "Financial Behavior Analysis\n\n"
        
        for col in financial_cols[:3]:
            col_name = col.replace('_', ' ').title()
            avg_val = df_processed[col].mean()
            median_val = df_processed[col].median()
            
            # Format based on column type
            if 'expenditure' in col.lower() or 'spend' in col.lower():
                financial_insights += f"""
                Monthly Spending Patterns:
                Customers show consistent spending behavior with an average monthly expenditure of ₦{avg_val:,.0f}. The median spending of ₦{median_val:,.0f} indicates a balanced distribution without extreme outliers affecting the average.
                
                What this means: Your customer base demonstrates stable financial behavior with predictable spending patterns that can inform product pricing and service offerings.
                """
            elif 'credit' in col.lower():
                financial_insights += f"""
                Credit Profile:
                The average credit score across your customer base is {avg_val:.0f}, with a median of {median_val:.0f}. This places most customers in the fair to good credit range.
                
                Business implication: This credit profile suggests opportunities for both secured and unsecured lending products, with appropriate risk management strategies for different customer segments.
                """
            elif 'loan' in col.lower() and 'amount' in col.lower():
                loan_customers = (df_processed[col] > 0).sum()
                loan_pct = loan_customers / len(df_processed) * 100
                financial_insights += f"""
                Loan Portfolio:
                {loan_customers:,} customers ({loan_pct:.1f}%) have active loans, with an average loan amount of ₦{avg_val:,.0f}.
                
                Strategic insight: This represents a significant lending opportunity, particularly for customers who currently don't have loans but have good credit profiles.
                """
        
        insights.append(financial_insights)
    
    # 4. Customer Segmentation Insights
    seg_insights = "Customer Segmentation Analysis\n\n"
    
    # Sort clusters by size
    sorted_clusters = sorted(cluster_profiles, key=lambda x: x['size'], reverse=True)
    
    seg_insights += f"""
    We've identified {len(cluster_profiles)} customer segments that represent distinct behavioral patterns in your customer base:
    
    Largest Segment: {sorted_clusters[0]['cluster_name']} ({sorted_clusters[0]['size']:,} customers, {sorted_clusters[0]['percentage']:.1f}%)
    This is your core customer group that should receive primary focus in marketing and product development.
    
    Key Segments Identified:
    """
    
    for profile in sorted_clusters[:3]:
        seg_insights += f"""
        {profile['cluster_name']} ({profile['percentage']:.1f}% of customers):
        - Segment size: {profile['size']:,} customers
        - Business opportunity: This segment represents a {'substantial' if profile['percentage'] > 15 else 'significant' if profile['percentage'] > 5 else 'niche'} portion of your customer base
        """
        
        # Add specific insights based on cluster characteristics
        if 'avg_credit_score' in profile:
            seg_insights += f"- Average credit score: {profile['avg_credit_score']:.0f}\n"
        
        if 'avg_monthly_expenditure' in profile:
            seg_insights += f"- Average monthly spending: ₦{profile['avg_monthly_expenditure']:,.0f}\n"
        
        if 'avg_sentiment_score' in profile:
            sentiment_val = profile['avg_sentiment_score']
            sentiment_desc = "positive" if sentiment_val > 0.1 else "neutral" if sentiment_val >= -0.1 else "negative"
            seg_insights += f"- Average sentiment: {sentiment_desc} ({sentiment_val:.2f})\n"
    
    seg_insights += """
    
    Why segmentation matters:
    Understanding these distinct customer groups allows for:
    1. Targeted marketing - Communicate differently with each segment
    2. Product personalization - Develop offerings that match segment needs
    3. Risk management - Apply appropriate controls for each segment
    4. Resource allocation - Focus efforts where they have most impact
    """
    
    insights.append(seg_insights)
    
    # 5. Sentiment Insights (if available) - Enhanced with VADER info
    if 'sentiment_label' in df_processed.columns:
        sentiment_dist = df_processed['sentiment_label'].value_counts(normalize=True) * 100
        positive_pct = sentiment_dist.get('Positive', 0)
        avg_sentiment = df_processed['sentiment_score'].mean()
        
        sentiment_insights = f"""
        Customer Sentiment Analysis (Using VADER NLP)
        
        Advanced NLP Analysis:
        Our system uses VADER (Valence Aware Dictionary and sEntiment Reasoner), a state-of-the-art sentiment analysis tool specifically designed for social media and informal text. This provides more accurate sentiment detection for customer feedback.
        
        Overall Satisfaction:
        Your customers show {positive_pct:.1f}% positive sentiment based on advanced NLP analysis. The average sentiment score is {avg_sentiment:.2f} on a scale from -1 (very negative) to +1 (very positive).
        
        Sentiment Distribution:
        - Positive feedback: {sentiment_dist.get('Positive', 0):.1f}%
        - Neutral feedback: {sentiment_dist.get('Neutral', 0):.1f}%
        - Negative feedback: {sentiment_dist.get('Negative', 0):.1f}%
        
        Customer experience opportunity: The {sentiment_dist.get('Negative', 0):.1f}% of negative feedback represents an opportunity to improve specific pain points and enhance overall customer satisfaction.
        
        VADER Sentiment Analysis Benefits:
        • Better understanding of informal language and slang
        • Accurate detection of sentiment intensity
        • Recognition of emoticons and informal expressions
        • Context-aware sentiment scoring
        """
        insights.append(sentiment_insights)
    
    return insights

def generate_natural_language_recommendations(df_processed, cluster_profiles):
    """Generate natural language recommendations for non-technical audience"""
    recommendations = []
    
    # Sort clusters by size
    sorted_clusters = sorted(cluster_profiles, key=lambda x: x['size'], reverse=True)
    
    # 1. Strategic Recommendations
    strategic_rec = f"""
     Strategic Business Recommendations
    
    Based on our analysis of your {len(df_processed):,} customers, we recommend the following strategic focus areas:
    
    Primary Market Focus:
    Your largest customer segment, {sorted_clusters[0]['cluster_name']}, represents {sorted_clusters[0]['percentage']:.1f}% of your customer base. We recommend:
    
    1. Develop core products specifically designed for this segment's needs
    2. Allocate approximately {sorted_clusters[0]['percentage']:.0f}% of marketing resources to retain and grow this segment
    3. Create loyalty programs that reward this segment's preferred behaviors
    
    Growth Opportunities:
    The {sorted_clusters[-1]['cluster_name']} segment, while smaller at {sorted_clusters[-1]['percentage']:.1f}%, represents a niche opportunity:
    
    1. Develop specialized offerings that address this segment's unique needs
    2. Test innovative products with this segment before broader rollout
    3. Allocate {max(5, sorted_clusters[-1]['percentage']):.0f}% of development budget to explore this niche market
    """
    recommendations.append(strategic_rec)
    
    # 2. Product Development Recommendations
    product_rec = """
    💼 Product & Service Development
    
    Based on Customer Needs Analysis:
    
    For Customers with Strong Credit Profiles:
    1. Premium credit products with higher limits and better terms
    2. Investment opportunities matching their financial capacity
    3. Priority banking services for enhanced customer experience
    
    For Customers Building Financial Stability:
    1. Financial education programs to improve money management skills
    2. Savings-focused products with automatic deposit features
    3. Credit-building tools to help improve credit scores over time
    
    Digital Service Enhancement:
    1. Mobile banking features that simplify common transactions
    2. Financial planning tools within your digital platforms
    3. Personalized notifications based on customer behavior patterns
    """
    recommendations.append(product_rec)
    
    # 3. Marketing & Communication Recommendations
    marketing_rec = """
     Marketing & Customer Communication
    
    Segment-Specific Communication Strategy:
    
    For Your Largest Segments:
    1. Mass communication channels like email and SMS for general updates
    2. Educational content about your core products and services
    3. Seasonal promotions aligned with common spending patterns
    
    For Niche Segments:
    1. Personalized communication addressing specific needs and interests
    2. Direct outreach from relationship managers or specialists
    3. Exclusive offers that recognize their unique value
    
    Communication Channels:
    1. Digital-first approach for tech-savvy segments
    2. Traditional channels (phone, branch) for less digital segments
    3. Omnichannel experience ensuring consistency across all touchpoints
    """
    recommendations.append(marketing_rec)
    
    # 4. Risk Management Recommendations
    risk_rec = """
     Risk Management & Compliance
    
    Proactive Risk Strategies:
    
    1. Tiered Risk Assessment:
       - Apply different risk rules for each customer segment
       - Monitor high-risk segments more closely
       - Offer risk-appropriate products to each segment
    
    2. Credit Risk Management:
       - Dynamic credit limits based on segment behavior
       - Early warning systems for potential defaults
       - Proactive outreach for customers showing risk patterns
    
    3. Operational Excellence:
       - Regular data quality checks to ensure accurate insights
       - Automated monitoring of key risk indicators
       - Continuous improvement of risk models based on new data
    """
    recommendations.append(risk_rec)
    
    # 5. Action Plan
    action_plan = """
     Immediate Action Plan
    
    First 2 Weeks - Foundation Building:
    1. Review findings with key stakeholders from marketing, product, and risk teams
    2. Identify quick wins - opportunities that can be implemented within 30 days
    3. Establish measurement - define how you'll track success of segmentation strategy
    
    Month 1 - Initial Implementation:
    1. Develop segment profiles for your customer service teams
    2. Create targeted marketing campaigns for your top 2 segments
    3. Update CRM systems with segmentation data for better customer insights
    
    Quarter 1 - Full Integration:
    1. Launch new products designed for specific customer segments
    2. Implement automated segmentation in your customer onboarding
    3. Measure impact and refine your approach based on results
    
    Ongoing - Continuous Improvement:
    1. Quarterly reviews of segment performance and characteristics
    2. Regular updates to your segmentation models as customer behavior evolves
    3. Continuous training for staff on segment-specific strategies
    """
    recommendations.append(action_plan)
    
    return recommendations

def create_interactive_charts(df_processed, cluster_profiles):
    """Create interactive Plotly charts with data labels"""
    
    charts = {}
    
    try:
        # 1. Cluster Distribution Pie Chart - Updated colors for better visibility
        cluster_sizes = [p['size'] for p in cluster_profiles]
        cluster_names = [p['cluster_name'] for p in cluster_profiles]
        percentages = [p['percentage'] for p in cluster_profiles]
        
        # Use darker, more visible colors
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        fig1 = go.Figure(data=[go.Pie(
            labels=cluster_names,
            values=cluster_sizes,
            hole=0.3,
            textinfo='label+percent+value',
            textposition='inside',
            marker=dict(colors=colors[:len(cluster_names)]),
            textfont=dict(size=14, color='black', family="Arial")  # Increased font size
        )])
        
        fig1.update_layout(
            title={
                'text': 'Customer Segment Distribution',
                'font': {'size': 22, 'color': 'white', 'family': 'Arial'}
            },
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            showlegend=True,
            legend=dict(
                font=dict(color='white', size=12),
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5
            ),
            height=550
        )
        
        charts['cluster_distribution'] = fig1
        
        # 2. Financial Metrics Comparison
        financial_cols = [col for col in df_processed.select_dtypes(include=[np.number]).columns 
                         if any(keyword in col.lower() for keyword in ['amount', 'expenditure', 'spend', 'credit', 'score'])]
        
        if financial_cols and len(cluster_profiles) > 1:
            # Use the first financial column for comparison
            compare_col = financial_cols[0]
            col_name = compare_col.replace('_', ' ').title()
            
            cluster_values = []
            for profile in cluster_profiles:
                cluster_data = df_processed[df_processed['cluster'] == profile['cluster_id']]
                cluster_values.append(cluster_data[compare_col].mean())
            
            fig2 = go.Figure(data=[go.Bar(
                x=[p['cluster_name'] for p in cluster_profiles],
                y=cluster_values,
                text=[f'₦{v:,.0f}' if 'amount' in compare_col.lower() or 'expenditure' in compare_col.lower() else f'{v:.0f}' 
                      for v in cluster_values],
                textposition='auto',
                textfont=dict(size=14, color='white', family="Arial"),  # Increased font size
                marker_color=px.colors.sequential.Viridis,
                marker=dict(line=dict(width=1, color='white'))
            )])
            
            fig2.update_layout(
                title={
                    'text': f'Average {col_name} by Customer Segment',
                    'font': {'size': 22, 'color': 'white', 'family': 'Arial'}
                },
                xaxis_title='Customer Segment',
                yaxis_title=col_name,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                xaxis=dict(
                    tickfont=dict(color='white', size=13, family="Arial"),
                    gridcolor='rgba(255,255,255,0.1)',
                    tickangle=45
                ),
                yaxis=dict(
                    tickfont=dict(color='white', size=13, family="Arial"),
                    gridcolor='rgba(255,255,255,0.1)'
                ),
                height=550
            )
            
            charts['financial_comparison'] = fig2
        
        # 3. Customer Metrics Dashboard
        if len(cluster_profiles) >= 3:
            # Prepare data for grouped bar chart
            metrics_to_show = []
            for col in ['credit_score', 'monthly_expenditure', 'loan_amount', 'sentiment_score']:
                if col in df_processed.columns:
                    metrics_to_show.append(col)
            
            if len(metrics_to_show) >= 2:
                fig3 = go.Figure()
                
                colors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63']
                
                for idx, metric in enumerate(metrics_to_show[:3]):
                    metric_values = []
                    for profile in cluster_profiles:
                        cluster_data = df_processed[df_processed['cluster'] == profile['cluster_id']]
                        metric_values.append(cluster_data[metric].mean())
                    
                    fig3.add_trace(go.Bar(
                        name=metric.replace('_', ' ').title(),
                        x=[p['cluster_name'] for p in cluster_profiles],
                        y=metric_values,
                        text=[f'₦{v:,.0f}' if 'amount' in metric or 'expenditure' in metric else f'{v:.2f}' 
                              for v in metric_values],
                        textposition='auto',
                        textfont=dict(size=12, color='white', family="Arial"),  # Increased font size
                        marker_color=colors[idx % len(colors)]
                    ))
                
                fig3.update_layout(
                    title={
                        'text': 'Key Metrics Across Customer Segments',
                        'font': {'size': 22, 'color': 'white', 'family': 'Arial'}
                    },
                    xaxis_title='Customer Segment',
                    yaxis_title='Average Value',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    xaxis=dict(
                        tickfont=dict(color='white', size=13, family="Arial"),
                        gridcolor='rgba(255,255,255,0.1)',
                        tickangle=45
                    ),
                    yaxis=dict(
                        tickfont=dict(color='white', size=13, family="Arial"),
                        gridcolor='rgba(255,255,255,0.1)'
                    ),
                    barmode='group',
                    legend=dict(
                        font=dict(color='white', size=12, family="Arial"),
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    height=550
                )
                
                charts['metrics_dashboard'] = fig3
        
        # 4. Sentiment Analysis (if available)
        if 'sentiment_label' in df_processed.columns:
            sentiment_counts = df_processed['sentiment_label'].value_counts()
            
            fig4 = go.Figure(data=[go.Bar(
                x=sentiment_counts.index,
                y=sentiment_counts.values,
                text=[f'{v:,}' for v in sentiment_counts.values],
                textposition='auto',
                textfont=dict(size=16, color='white', family="Arial"),  # Increased font size
                marker_color=['#4CAF50', '#FFC107', '#F44336'],
                marker=dict(line=dict(width=1, color='white'))
            )])
            
            fig4.update_layout(
                title={
                    'text': 'Customer Sentiment Distribution (VADER Analysis)',
                    'font': {'size': 22, 'color': 'white', 'family': 'Arial'}
                },
                xaxis_title='Sentiment',
                yaxis_title='Number of Customers',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                xaxis=dict(
                    tickfont=dict(color='white', size=14, family="Arial"),
                    gridcolor='rgba(255,255,255,0.1)'
                ),
                yaxis=dict(
                    tickfont=dict(color='white', size=14, family="Arial"),
                    gridcolor='rgba(255,255,255,0.1)'
                ),
                height=450
            )
            
            charts['sentiment_analysis'] = fig4
        
        # 5. Top Spending Categories (if available)
        if 'spending_category' in df_processed.columns:
            spending_counts = df_processed['spending_category'].value_counts().head(10)
            
            fig5 = go.Figure(data=[go.Bar(
                y=spending_counts.index.str.replace('_', ' ').str.title(),
                x=spending_counts.values,
                orientation='h',
                text=[f'{v:,}' for v in spending_counts.values],
                textposition='auto',
                textfont=dict(size=14, color='white', family="Arial"),  # Increased font size
                marker_color=px.colors.sequential.Plasma,
                marker=dict(line=dict(width=1, color='white'))
            )])
            
            fig5.update_layout(
                title={
                    'text': 'Top 10 Spending Categories',
                    'font': {'size': 22, 'color': 'white', 'family': 'Arial'}
                },
                xaxis_title='Number of Customers',
                yaxis_title='Spending Category',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                xaxis=dict(
                    tickfont=dict(color='white', size=13, family="Arial"),
                    gridcolor='rgba(255,255,255,0.1)'
                ),
                yaxis=dict(
                    tickfont=dict(color='white', size=13, family="Arial"),
                    gridcolor='rgba(255,255,255,0.1)'
                ),
                height=550
            )
            
            charts['spending_categories'] = fig5
            
    except Exception as e:
        st.error(f"Error creating charts: {str(e)}")
    
    return charts

# ============================================
# FOOTER FUNCTION
# ============================================

def display_footer():
    """Display footer on all pages"""
    st.markdown("---")
    
    # Footer content - plain text without HTML
    st.markdown('<div class="footer-container">', unsafe_allow_html=True)
    st.markdown('<p class="footer-text">African Financial Segmentation Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="footer-text">Complete Analytics Platform | Unsupervised ML + Advanced NLP (VADER)</p>', unsafe_allow_html=True)
    st.markdown('<p class="footer-text">Dataverse Africa Internship Program | Cohort 3</p>', unsafe_allow_html=True)
    
    # URL with clickable link using markdown
    st.markdown('[🌍 https://dataverseafrica.org/](https://dataverseafrica.org/)')
    
    st.markdown('<p class="footer-text" style="color: #4ade80; font-weight: bold;">Empowering Africa\'s Digital Future</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main Streamlit application"""
    
    # Title in blue box
    st.markdown("""
    <div class="title-container">
        <h1>🌍 African Financial Segmentation Dashboard</h1>
        <h3>Automated Customer Insights & Recommendations (VADER NLP Enhanced)</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Navigation")
        
        menu = st.radio(
            "Select Page",
            ["📁 Upload Data", "📈 Data Overview", "🔍 Analysis Results", "📊 Visualizations", "💡 Insights", "📄 Recommendations", "📚 User Guide"]
        )
        
        st.markdown("---")
        
        st.markdown("### 👥 Project Team")
        st.write("Dataverse Africa Cohort 3")
        st.write("Amarachi Florence")
        st.write("Thato Maelane")
        st.write("Philip Odiachi")
        st.write("Mavis")
        
        st.markdown("---")
        
        # NLP Info
        st.markdown("### 🔬 NLP Technology")
        st.info("This app uses **VADER Sentiment Analysis** for more accurate customer feedback analysis.")
        
        st.markdown("---")
        
        # Generate sample data button - FIXED WITH RERUN
        if st.button("Generate Sample Data", use_container_width=True, type="secondary"):
            with st.spinner("Creating sample dataset..."):
                st.session_state.df = create_synthetic_dataset(5200)
                st.success(f"Generated {len(st.session_state.df):,} sample records")
                st.rerun()  # CRITICAL FIX: Force app to refresh
        
        # Clear data button
        if st.button("Clear All Data", use_container_width=True, type="secondary"):
            for key in ['df', 'processed_df', 'clusters', 'insights', 'recommendations', 'analysis_complete']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # Main content based on menu selection
    if menu == "📁 Upload Data":
        show_upload_page()
    elif menu == "📈 Data Overview":
        show_data_overview()
    elif menu == "🔍 Analysis Results":
        show_analysis_results()
    elif menu == "📊 Visualizations":
        show_visualizations()
    elif menu == "💡 Insights":
        show_insights()
    elif menu == "📄 Recommendations":
        show_recommendations()
    elif menu == "📚 User Guide":
        show_user_guide()
    
    # Display footer on all pages
    display_footer()

def show_upload_page():
    """Show data upload page"""
    
    st.markdown("###  Upload Your Financial Dataset")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("#### Upload Your Financial Data Here")
        st.write("Upload your CSV or Excel file to begin the analysis. Our system will automatically detect column types, clean the data, and generate valuable insights about your customers.")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your financial customer data for analysis"
        )
        
        if uploaded_file is not None:
            with st.spinner("Loading dataset..."):
                df = load_dataset(uploaded_file)
                
                if df is not None:
                    st.session_state.df = df
                    st.success(f" Dataset loaded successfully!")
                    
                    # Show dataset info in metrics
                    col1a, col2a, col3a, col4a = st.columns(4)
                    
                    with col1a:
                        st.metric("Total Records", f"{len(df):,}")
                    
                    with col2a:
                        st.metric("Columns", len(df.columns))
                    
                    with col3a:
                        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
                        st.metric("Numeric Columns", numeric_cols)
                    
                    with col4a:
                        missing = df.isnull().sum().sum()
                        st.metric("Missing Values", f"{missing:,}")
                    
                    # Check for text columns for sentiment analysis
                    text_cols = [col for col in df.columns if any(keyword in col.lower() 
                               for keyword in ['feedback', 'review', 'comment', 'text'])]
                    if text_cols:
                        st.info(f" Text column detected for sentiment analysis: **{text_cols[0]}**")
                    
                    # Show preview
                    with st.expander(" Data Preview (First 10 rows)", expanded=True):
                        st.dataframe(df.head(10), use_container_width=True)
                    
                    # Process data button
                    if st.button(" Start Complete Analysis", type="primary", use_container_width=True):
                        with st.spinner("Processing data and performing analysis..."):
                            # Process data
                            st.session_state.processed_df, _ = process_data(df)
                            
                            # Perform clustering
                            st.session_state.processed_df, st.session_state.clusters = perform_clustering(st.session_state.processed_df)
                            
                            # Generate insights in natural language
                            st.session_state.insights = generate_natural_language_insights(st.session_state.processed_df, st.session_state.clusters)
                            
                            # Generate recommendations in natural language
                            st.session_state.recommendations = generate_natural_language_recommendations(st.session_state.processed_df, st.session_state.clusters)
                            
                            st.session_state.analysis_complete = True
                            st.success(" Analysis complete! Navigate to other tabs to view results.")
                            st.balloons()
                            st.rerun()
    
    with col2:
        st.write("####  Quick Tips")
        st.info("**Accepted File Formats:**")
        st.write("- CSV files (.csv)")
        st.write("- Excel files (.xlsx, .xls)")
        st.info("**NLP Technology:**")
        st.write("- Uses **VADER Sentiment Analysis**")
        st.write("- Better for informal text and feedback")
        st.info("**Need sample data?**")
        st.write("Click 'Generate Sample Data' in the sidebar to get started immediately!")

def show_data_overview():
    """Show data overview and statistics"""
    
    if st.session_state.df is None:
        st.info(" Please upload a dataset first or generate sample data.")
        return
    
    df = st.session_state.df
    
    st.markdown("###  Dataset Overview")
    
    # Key metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    
    with col2:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        st.metric("Numeric Columns", len(numeric_cols))
    
    with col3:
        categorical_cols = df.select_dtypes(include=['object']).columns
        st.metric("Categorical Columns", len(categorical_cols))
    
    with col4:
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
        st.metric("Missing Data", f"{missing_pct:.1f}%")
    
    # NLP capability check
    text_cols = [col for col in df.columns if any(keyword in col.lower() 
               for keyword in ['feedback', 'review', 'comment', 'text'])]
    if text_cols:
        st.info(f" **NLP Ready:** Found text column '{text_cols[0]}' for sentiment analysis using VADER.")
    
    # Data preview tabs
    st.markdown("####  Data Exploration")
    tab1, tab2, tab3, tab4 = st.tabs(["Data Preview", "Column Information", "Missing Values Analysis", "Statistical Summary"])
    
    with tab1:
        st.write("Below is a preview of your dataset showing the first 20 records. This helps you verify that the data has been loaded correctly.")
        st.dataframe(df.head(20), use_container_width=True)
    
    with tab2:
        st.write("This table shows detailed information about each column in your dataset, including data types and unique values.")
        col_info = pd.DataFrame({
            'Column Name': df.columns,
            'Data Type': df.dtypes.values,
            'Non-Null Count': df.notnull().sum().values,
            'Unique Values': df.nunique().values,
            'Sample Values': [', '.join(map(str, df[col].dropna().unique()[:3])) for col in df.columns]
        })
        st.dataframe(col_info, use_container_width=True)
    
    with tab3:
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        missing_df = pd.DataFrame({
            'Column Name': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing Percent': missing_percent.values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Percent', ascending=False)
        
        if len(missing_df) > 0:
            st.write("The following columns have missing values that may need attention:")
            st.dataframe(missing_df, use_container_width=True)
            st.warning(f"⚠️ Dataset has {missing_df['Missing Count'].sum():,} missing values")
        else:
            st.success("✅ No missing values found in your dataset")
    
    with tab4:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.write("Statistical summary of numeric columns in your dataset:")
            stats_df = df[numeric_cols].describe().T
            stats_df['IQR'] = stats_df['75%'] - stats_df['25%']
            stats_df['CV'] = (stats_df['std'] / stats_df['mean']) * 100
            st.dataframe(stats_df, use_container_width=True)
        else:
            st.info("No numeric columns found for statistical summary")

def show_analysis_results():
    """Show analysis results"""
    
    if not st.session_state.analysis_complete:
        st.info(" Please complete data analysis first.")
        if st.session_state.df is not None:
            if st.button("Run Complete Analysis", type="primary", use_container_width=True):
                with st.spinner("Processing data and generating insights..."):
                    # Process data
                    st.session_state.processed_df, _ = process_data(st.session_state.df)
                    
                    # Perform clustering
                    st.session_state.processed_df, st.session_state.clusters = perform_clustering(st.session_state.processed_df)
                    
                    # Generate insights
                    st.session_state.insights = generate_natural_language_insights(st.session_state.processed_df, st.session_state.clusters)
                    
                    # Generate recommendations
                    st.session_state.recommendations = generate_natural_language_recommendations(st.session_state.processed_df, st.session_state.clusters)
                    
                    st.session_state.analysis_complete = True
                    st.rerun()
        return
    
    df_processed = st.session_state.processed_df
    cluster_profiles = st.session_state.clusters
    
    st.markdown("###  Analysis Results")
    
    # Show NLP technology used
    if 'sentiment_score' in df_processed.columns:
        st.info(" **NLP Analysis:** Customer sentiment analyzed using VADER (Valence Aware Dictionary and sEntiment Reasoner) for accurate informal text understanding.")
    
    # Segment overview in a grid
    st.markdown("####  Customer Segments Identified")
    st.write("Our analysis has identified the following customer segments based on their financial behavior patterns:")
    
    # Create responsive grid for cluster cards
    num_clusters = len(cluster_profiles)
    if num_clusters <= 3:
        cols = st.columns(num_clusters)
        for col, profile in zip(cols, cluster_profiles):
            with col:
                st.markdown(f"""
                <div class="cluster-card">
                    <h4>{profile['cluster_name']}</h4>
                    <p><strong>{profile['size']:,}</strong> customers</p>
                    <p><strong>{profile['percentage']:.1f}%</strong> of total</p>
                </div>
                """, unsafe_allow_html=True)
    elif num_clusters == 4:
        col1, col2 = st.columns(2)
        with col1:
            for profile in cluster_profiles[:2]:
                st.markdown(f"""
                <div class="cluster-card">
                    <h4>{profile['cluster_name']}</h4>
                    <p><strong>{profile['size']:,}</strong> customers</p>
                    <p><strong>{profile['percentage']:.1f}%</strong> of total</p>
                </div>
                """, unsafe_allow_html=True)
        with col2:
            for profile in cluster_profiles[2:]:
                st.markdown(f"""
                <div class="cluster-card">
                    <h4>{profile['cluster_name']}</h4>
                    <p><strong>{profile['size']:,}</strong> customers</p>
                    <p><strong>{profile['percentage']:.1f}%</strong> of total</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        # For 5+ clusters, use 3 columns
        for i in range(0, num_clusters, 3):
            col1, col2, col3 = st.columns(3)
            with col1:
                if i < num_clusters:
                    profile = cluster_profiles[i]
                    st.markdown(f"""
                    <div class="cluster-card">
                        <h4>{profile['cluster_name']}</h4>
                        <p><strong>{profile['size']:,}</strong> customers</p>
                        <p><strong>{profile['percentage']:.1f}%</strong> of total</p>
                    </div>
                    """, unsafe_allow_html=True)
            with col2:
                if i+1 < num_clusters:
                    profile = cluster_profiles[i+1]
                    st.markdown(f"""
                    <div class="cluster-card">
                        <h4>{profile['cluster_name']}</h4>
                        <p><strong>{profile['size']:,}</strong> customers</p>
                        <p><strong>{profile['percentage']:.1f}%</strong> of total</p>
                    </div>
                    """, unsafe_allow_html=True)
            with col3:
                if i+2 < num_clusters:
                    profile = cluster_profiles[i+2]
                    st.markdown(f"""
                    <div class="cluster-card">
                        <h4>{profile['cluster_name']}</h4>
                        <p><strong>{profile['size']:,}</strong> customers</p>
                        <p><strong>{profile['percentage']:.1f}%</strong> of total</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Detailed segment analysis
    st.markdown("####  Detailed Segment Analysis")
    st.write("Here's a detailed breakdown of each customer segment with key metrics and characteristics:")
    
    # Create profiles DataFrame
    profiles_data = []
    for profile in cluster_profiles:
        row = {
            'Segment Name': profile['cluster_name'],
            'Number of Customers': f"{profile['size']:,}",
            'Percentage of Total': f"{profile['percentage']:.1f}%"
        }
        
        # Add key metrics
        key_metrics = ['credit_score', 'monthly_expenditure', 'loan_amount', 'risk_score', 'sentiment_score']
        for metric in key_metrics:
            key = f'avg_{metric}'
            if key in profile:
                if 'expenditure' in metric or 'amount' in metric:
                    row[metric.replace('_', ' ').title()] = f"₦{profile[key]:,.0f}"
                elif 'sentiment' in metric:
                    sentiment_val = profile[key]
                    sentiment_desc = "Positive" if sentiment_val > 0.1 else "Neutral" if sentiment_val >= -0.1 else "Negative"
                    row[metric.replace('_', ' ').title()] = f"{sentiment_desc} ({sentiment_val:.2f})"
                else:
                    row[metric.replace('_', ' ').title()] = f"{profile[key]:.1f}"
        
        profiles_data.append(row)
    
    if profiles_data:
        profiles_df = pd.DataFrame(profiles_data)
        st.dataframe(profiles_df, use_container_width=True)
    
    # Processed data preview
    st.markdown("####  Processed Data Sample")
    st.write("Below is a sample of your processed data with added features like risk scores and VADER sentiment analysis:")
    
    with st.expander("View Processed Data (First 10 rows)", expanded=False):
        st.dataframe(df_processed.head(10), use_container_width=True)
    
    # Download options
    st.markdown("####  Download Analysis Results")
    st.write("You can download the analysis results for further use or sharing with stakeholders:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download processed data
        csv = df_processed.to_csv(index=False)
        st.download_button(
            label=" Download Processed Data",
            data=csv,
            file_name="processed_financial_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Download cluster profiles
        profiles_df = pd.DataFrame(cluster_profiles)
        profiles_csv = profiles_df.to_csv(index=False)
        st.download_button(
            label=" Download Cluster Profiles",
            data=profiles_csv,
            file_name="cluster_profiles.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col3:
        # Download analysis summary
        summary = f"""
        Financial Segmentation Analysis Summary
        ======================================
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        NLP Technology: VADER Sentiment Analysis
        
        Dataset: {len(df_processed):,} records
        Segments Identified: {len(cluster_profiles)}
        
        Segment Summary:
        """
        
        for profile in cluster_profiles:
            summary += f"\n{profile['cluster_name']}: {profile['size']:,} customers ({profile['percentage']:.1f}%)"
        
        st.download_button(
            label=" Download Summary",
            data=summary,
            file_name="analysis_summary.txt",
            mime="text/plain",
            use_container_width=True
        )

def show_visualizations():
    """Show visualization charts"""
    
    if not st.session_state.analysis_complete:
        st.info(" Please complete data analysis first to view visualizations.")
        return
    
    df_processed = st.session_state.processed_df
    cluster_profiles = st.session_state.clusters
    
    st.markdown("###  Interactive Visualizations")
    st.write("Explore your data through these interactive visualizations. Hover over charts to see detailed information and click on legend items to filter data.")
    
    # Create charts
    charts = create_interactive_charts(df_processed, cluster_profiles)
    
    # Display charts in a logical order
    if 'cluster_distribution' in charts:
        st.markdown("#### Customer Segment Distribution")
        st.write("This pie chart shows how your customers are distributed across the different segments we identified.")
        st.plotly_chart(charts['cluster_distribution'], use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'financial_comparison' in charts:
            st.markdown("#### Financial Metrics by Segment")
            st.write("Compare key financial metrics across different customer segments.")
            st.plotly_chart(charts['financial_comparison'], use_container_width=True)
        
        if 'sentiment_analysis' in charts:
            st.markdown("#### Customer Sentiment Analysis")
            st.write("Understanding customer satisfaction through VADER sentiment analysis of feedback.")
            st.plotly_chart(charts['sentiment_analysis'], use_container_width=True)
    
    with col2:
        if 'metrics_dashboard' in charts:
            st.markdown("#### Multiple Metrics Comparison")
            st.write("Side-by-side comparison of different metrics across customer segments.")
            st.plotly_chart(charts['metrics_dashboard'], use_container_width=True)
        
        if 'spending_categories' in charts:
            st.markdown("#### Top Spending Categories")
            st.write("Most common spending categories among your customers.")
            st.plotly_chart(charts['spending_categories'], use_container_width=True)
    
    # Additional data tables
    st.markdown("####  Additional Statistics")
    tab1, tab2, tab3 = st.tabs(["Numeric Summary", "Categorical Summary", "Segment Comparison"])
    
    with tab1:
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()[:8]
        if numeric_cols:
            st.write("Statistical summary of key numeric columns in your processed dataset:")
            stats_df = df_processed[numeric_cols].describe().T
            stats_df['IQR'] = stats_df['75%'] - stats_df['25%']
            stats_df['CV'] = (stats_df['std'] / stats_df['mean']) * 100
            st.dataframe(stats_df, use_container_width=True)
    
    with tab2:
        categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()[:5]
        if categorical_cols:
            st.write("Distribution of categorical values in your dataset:")
            for col in categorical_cols:
                value_counts = df_processed[col].value_counts().head(10)
                value_df = pd.DataFrame({
                    'Value': value_counts.index,
                    'Count': value_counts.values,
                    'Percentage': (value_counts.values / len(df_processed)) * 100
                })
                st.markdown(f"**{col.replace('_', ' ').title()}**")
                st.dataframe(value_df, use_container_width=True)
    
    with tab3:
        if cluster_profiles:
            st.write("Detailed comparison of all customer segments with their characteristics:")
            # Create detailed profile table
            profile_details = []
            for profile in cluster_profiles:
                row = {'Segment Name': profile['cluster_name']}
                
                # Add all profile attributes
                for key, value in profile.items():
                    if key not in ['cluster_name', 'cluster_id']:
                        if isinstance(value, float):
                            if 'percentage' in key:
                                row[key.replace('_', ' ').title()] = f"{value:.1f}%"
                            elif any(metric in key for metric in ['expenditure', 'amount']):
                                row[key.replace('_', ' ').title()] = f"₦{value:,.0f}"
                            else:
                                row[key.replace('_', ' ').title()] = f"{value:.2f}"
                        else:
                            row[key.replace('_', ' ').title()] = value
                
                profile_details.append(row)
            
            if profile_details:
                st.dataframe(pd.DataFrame(profile_details), use_container_width=True)

def show_insights():
    """Show business insights"""
    
    if not st.session_state.analysis_complete:
        st.info(" Please complete data analysis first to view insights.")
        return
    
    insights = st.session_state.insights
    
    st.markdown("###  Insights")
    st.write("These insights were generated based on your dataset analysis. They provide a comprehensive understanding of your customer base and their financial behaviors.")
    
    # Display insights in natural paragraphs
    for insight in insights:
        st.write(insight)
        st.markdown("---")
    
    # Key findings summary
    st.markdown("###  Key Findings Summary")
    st.write("Based on our analysis of your customer data, here are the most important findings that should inform your business strategy:")
    
    # Extract and display key findings
    key_findings = [
        f"• Customer Segmentation: Identified {len(st.session_state.clusters)} distinct customer groups with unique behaviors",
        f"• Data Quality: Your dataset shows excellent completeness with minimal missing values",
        f"• Financial Patterns: Clear spending and credit patterns emerged across different customer segments",
        "• Business Opportunity: Each segment represents specific opportunities for targeted products and services",
        "• Strategic Foundation: These insights provide a data-driven foundation for customer-centric decision making"
    ]
    
    for finding in key_findings:
        st.write(finding)

def show_recommendations():
    """Show recommendations"""
    
    if not st.session_state.analysis_complete:
        st.info(" Please complete data analysis first to view recommendations.")
        return
    
    recommendations = st.session_state.recommendations
    
    st.markdown("###  Recommendations")
    st.write("Recommendations were generated based on your dataset analysis. They provide actionable guidance for improving your business strategies and customer engagement.")
    
    # Display recommendations in natural paragraphs
    for rec in recommendations:
        st.markdown(rec)
        st.markdown("---")
    
    # Action plan summary
    st.markdown("###  Recommended Action Plan")
    st.write("Based on the analysis, here's a practical action plan to implement these recommendations:")
    
    action_items = [
        "This Week - Foundation Building: Review the insights with your team and identify which recommendations align with your current business priorities.",
        "Next 2 Weeks - Planning Phase: Develop specific action plans for your top 2-3 customer segments, including targeted marketing and product adjustments.",
        "Month 1 - Initial Implementation: Begin implementing segment-specific strategies and set up measurement systems to track their effectiveness.",
        "Quarter 1 - Full Integration: Roll out comprehensive changes based on the segmentation insights and refine your approach based on early results.",
        "Ongoing - Continuous Improvement: Establish regular reviews of customer segments and update your strategies as customer behavior evolves."
    ]
    
    for item in action_items:
        st.write(item)
    
    # Download comprehensive report
    st.markdown("---")
    st.markdown("###  Generate Comprehensive Report")
    st.write("Download a complete report of all insights and recommendations for sharing with stakeholders or for your records:")
    
    if st.button(" Generate Full Analysis Report", use_container_width=True, type="primary"):
        # Create comprehensive report
        df_processed = st.session_state.processed_df
        cluster_profiles = st.session_state.clusters
        
        report_content = f"""
        COMPREHENSIVE FINANCIAL SEGMENTATION ANALYSIS REPORT
        ====================================================
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        EXECUTIVE SUMMARY
        -----------------
        This report presents a comprehensive analysis of your customer data, identifying {len(cluster_profiles)} distinct customer segments and providing actionable recommendations for business improvement.
        
        Total Customers Analyzed: {len(df_processed):,}
        Customer Segments Identified: {len(cluster_profiles)}
        Analysis Date: {datetime.now().strftime('%Y-%m-%d')}
        NLP Technology: VADER Sentiment Analysis
        
        KEY METRICS
        -----------
        Dataset Size: {len(df_processed):,} records
        Features Analyzed: {len(df_processed.columns)}
        Data Completeness: {100 - df_processed.isnull().sum().sum()/(len(df_processed)*len(df_processed.columns))*100:.1f}%
        
        CUSTOMER SEGMENTS
        -----------------
        """
        
        # Add cluster profiles
        for profile in cluster_profiles:
            report_content += f"""
            {profile['cluster_name']}
            • Size: {profile['size']:,} customers ({profile['percentage']:.1f}% of total)
            """
            
            # Add key metrics
            for key in ['avg_credit_score', 'avg_monthly_expenditure', 'avg_loan_amount', 'avg_risk_score', 'avg_sentiment_score']:
                if key in profile:
                    metric_name = key[4:].replace('_', ' ').title()
                    if 'expenditure' in key or 'amount' in key:
                        report_content += f"• {metric_name}: ₦{profile[key]:,.0f}\n"
                    elif 'sentiment' in key:
                        sentiment_val = profile[key]
                        sentiment_desc = "Positive" if sentiment_val > 0.1 else "Neutral" if sentiment_val >= -0.1 else "Negative"
                        report_content += f"• {metric_name}: {sentiment_desc} ({sentiment_val:.2f})\n"
                    else:
                        report_content += f"• {metric_name}: {profile[key]:.1f}\n"
        
        report_content += """
        
        KEY RECOMMENDATIONS
        -------------------
        1. Strategic Focus:
           • Prioritize your largest customer segment for core product development
           • Develop specialized offerings for niche segments
           • Allocate resources based on segment potential and current performance
        
        2. Product Development:
           • Create risk-appropriate financial products for different segments
           • Develop digital solutions that match segment technology adoption
           • Implement tiered service levels based on segment value
        
        3. Marketing Strategy:
           • Develop segment-specific communication approaches
           • Use different channels for different customer groups
           • Create personalized offers based on segment characteristics
        
        4. Risk Management:
           • Apply different risk rules for each customer segment
           • Implement early warning systems for high-risk patterns
           • Develop proactive outreach for customers showing concerning behaviors
        
        IMPLEMENTATION ROADMAP
        ----------------------
        Week 1-2: Review findings and develop implementation plans
        Month 1: Begin targeted marketing and product adjustments
        Quarter 1: Full implementation of segmentation strategies
        Ongoing: Regular review and refinement of approaches
        
        ====================================================
        This report was automatically generated by the
        African Financial Segmentation Dashboard
        
        Dataverse Africa Cohort 3 Project
        Empowering Africa's Digital Future
        ====================================================
        """
        
        # Download button
        st.download_button(
            label=" Download Full Report",
            data=report_content,
            file_name=f"financial_segmentation_report_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )

def show_user_guide():
    """Show user guide and instructions"""
    
    st.markdown("## 📚 User Guide & Instructions")
    st.write("Welcome to the African Financial Segmentation Dashboard! This guide will help you understand how to use the application effectively.")
    
    # Overview
    st.markdown("###  Overview")
    st.write("""
    This dashboard uses advanced machine learning algorithms to automatically segment your financial customers into distinct groups based on their behavior patterns. The insights generated help you understand your customers better and make data-driven business decisions.
    
    **Key Technology: VADER Sentiment Analysis**
    - This app uses VADER (Valence Aware Dictionary and sEntiment Reasoner) for customer feedback analysis
    - VADER is specifically designed for social media and informal text analysis
    - Provides more accurate sentiment detection for customer feedback than traditional methods
    """)
    
    # How it Works
    st.markdown("### ⚙️ How It Works")
    
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.markdown('<span class="step-number">1</span><span class="step-title">Data Upload</span>', unsafe_allow_html=True)
    st.write("Upload your financial dataset (CSV or Excel format) containing customer information, financial transactions, and other relevant data.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.markdown('<span class="step-number">2</span><span class="step-title">Automatic Processing</span>', unsafe_allow_html=True)
    st.write("The system automatically detects column types, handles missing values, and performs feature engineering to prepare your data for analysis.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.markdown('<span class="step-number">3</span><span class="step-title">NLP Sentiment Analysis</span>', unsafe_allow_html=True)
    st.write("Using VADER NLP technology, the system analyzes customer feedback to determine sentiment scores and categorize feedback as Positive, Neutral, or Negative.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.markdown('<span class="step-number">4</span><span class="step-title">Customer Segmentation</span>', unsafe_allow_html=True)
    st.write("Using K-Means clustering algorithm, the system groups customers into distinct segments based on their financial behavior patterns and sentiment scores.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.markdown('<span class="step-number">5</span><span class="step-title">Insights Generation</span>', unsafe_allow_html=True)
    st.write("Natural language insights are automatically generated, explaining what each customer segment represents and their business implications.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.markdown('<span class="step-number">6</span><span class="step-title">Actionable Recommendations</span>', unsafe_allow_html=True)
    st.write("Based on the analysis, the system provides specific recommendations for marketing, product development, and risk management.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Data Requirements
    st.markdown("###  Data Requirements")
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("#### Accepted Formats")
    st.write("- CSV files (.csv)")
    st.write("- Excel files (.xlsx, .xls)")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("#### Recommended Columns")
    st.write("For best results, include these columns in your dataset:")
    st.write("- **Customer identifiers**: Customer ID, account numbers")
    st.write("- **Financial amounts**: Transaction amounts, balances, expenditures")
    st.write("- **Credit scores**: Credit ratings, risk scores")
    st.write("- **Transaction data**: Dates, categories, frequencies")
    st.write("- **Customer feedback**: Reviews, comments, satisfaction ratings (for VADER sentiment analysis)")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Key Features
    st.markdown("###  Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.markdown("####  Automatic Column Detection")
        st.write("System automatically identifies and categorizes different types of data columns for optimal processing.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.markdown("####  Smart Missing Value Handling")
        st.write("Intelligent handling of missing data using appropriate imputation methods for each column type.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.markdown("####  Advanced NLP (VADER)")
        st.write("Uses VADER sentiment analysis for accurate understanding of informal customer feedback and slang.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.markdown("#### 🔄 Adaptive Clustering")
        st.write("Automatically determines optimal number of customer segments based on your data characteristics.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.markdown("#### 💬 Natural Language Insights")
        st.write("Generates easy-to-understand business insights in plain language, no technical expertise needed.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.markdown("####  Interactive Visualizations")
        st.write("Beautiful, interactive charts that help you explore and understand your customer segments.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Getting Started
    st.markdown("###  Getting Started")
    
    st.markdown("""
    1. **Navigate to the Upload Data page** from the sidebar menu
    2. **Upload your financial dataset** or click 'Generate Sample Data' to try with demo data
    3. **Click 'Start Complete Analysis'** to begin the automatic segmentation process
    4. **Explore the results** through the Analysis Results, Visualizations, Insights, and Recommendations pages
    5. **Download reports** for sharing with your team or stakeholders
    """)
    
    # Tips for Best Results
    st.markdown("###  Tips for Best Results")
    
    tips = [
        "**Clean your data** before uploading - remove any sensitive information",
        "**Include at least 1000 records** for meaningful segmentation results",
        "**Ensure consistent formatting** in your dataset columns",
        "**Use descriptive column names** for better automatic detection",
        "**Include customer feedback text** for VADER sentiment analysis",
        "**Regularly update your data** for current insights and recommendations"
    ]
    
    for tip in tips:
        st.write(f"- {tip}")

# ============================================
# RUN THE APPLICATION
# ============================================
if __name__ == "__main__":
    main()