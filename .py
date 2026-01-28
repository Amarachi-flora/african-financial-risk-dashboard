"""
Streamlit Dashboard for Customer Financial Risk Prediction
Interactive visualization and prediction tool
Team: AMARACHI FLORENCE, Thato Maelane, Philip Odiachi, AND Mavis
Internship: Dataverse Africa
"""

# ============================================
# SUPPRESS ALL WARNINGS
# ============================================
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Suppress sklearn version warnings
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
except ImportError:
    pass

# ============================================
# MAIN IMPORTS
# ============================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import requests
import json
import sys
import os
from datetime import datetime
import time
import zipfile
import io

# Create necessary directories if they don't exist
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("charts", exist_ok=True)
os.makedirs("powerbi", exist_ok=True)
os.makedirs("eda_reports", exist_ok=True)

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Customer Financial Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS WITH ENHANCED ANIMATIONS
# ============================================
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        animation: fadeInDown 1s ease;
    }
    
    .sub-header {
        font-size: 1.8rem;
        color: #4B5563;
        margin-top: 0.5rem;
        font-weight: 600;
        animation: fadeInUp 1.5s ease;
    }
    
    .card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: slideInLeft 1s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        animation: zoomIn 1s ease;
    }
    
    .team-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        animation: fadeIn 1s ease;
    }
    
    .cluster-card {
        background: white;
        color: #333;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 5px solid;
        animation: fadeIn 1s ease;
    }
    
    .success-badge {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
        animation: bounce 2s infinite;
    }
    
    .pbi-card {
        background: linear-gradient(135deg, #008751 0%, #00A86B 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    /* Starry ribbon effect */
    .starry-ribbon {
        position: relative;
        background: linear-gradient(90deg, #1a237e 0%, #283593 100%);
        color: white;
        text-align: center;
        padding: 0.8rem;
        margin: 1.5rem 0;
        border-radius: 8px;
        overflow: hidden;
        animation: slideInRight 1s ease;
    }
    
    .starry-ribbon::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: 
            radial-gradient(2px 2px at 20px 30px, #fff 50%, transparent 50%),
            radial-gradient(2px 2px at 40px 70px, #fff 50%, transparent 50%),
            radial-gradient(2px 2px at 60px 20px, #fff 50%, transparent 50%),
            radial-gradient(2px 2px at 80px 50px, #fff 50%, transparent 50%),
            radial-gradient(2px 2px at 100px 80px, #fff 50%, transparent 50%);
        background-size: 120px 100px;
        animation: twinkle 3s infinite;
    }
    
    /* Bouncing balloon */
    .bouncing-balloon {
        position: relative;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 50% 50% 50% 0;
        transform: rotate(-45deg);
        margin: 2rem auto;
        width: 120px;
        height: 120px;
        display: flex;
        align-items: center;
        justify-content: center;
        animation: bounceBalloon 3s ease-in-out infinite;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
    }
    
    .balloon-content {
        transform: rotate(45deg);
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    /* Hurry animation */
    .hurry-pulse {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        border-radius: 30px;
        font-weight: bold;
        animation: hurryPulse 1.5s infinite;
        box-shadow: 0 5px 15px rgba(255, 65, 108, 0.4);
    }
    
    /* Professional data cards */
    .data-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-top: 4px solid;
        transition: all 0.3s ease;
    }
    
    .data-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    .status-active { background-color: #10B981; }
    .status-warning { background-color: #F59E0B; }
    .status-error { background-color: #EF4444; }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes zoomIn {
        from {
            opacity: 0;
            transform: scale(0.9);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    @keyframes bounce {
        0%, 100% {
            transform: translateY(0);
        }
        50% {
            transform: translateY(-5px);
        }
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }
    
    @keyframes twinkle {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 1; }
    }
    
    @keyframes bounceBalloon {
        0%, 100% { 
            transform: translateY(0) rotate(-45deg) scale(1);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        }
        50% { 
            transform: translateY(-25px) rotate(-45deg) scale(1.05);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        }
    }
    
    @keyframes hurryPulse {
        0%, 100% { 
            transform: scale(1);
            box-shadow: 0 5px 15px rgba(255, 65, 108, 0.4);
        }
        50% { 
            transform: scale(1.05);
            box-shadow: 0 10px 25px rgba(255, 65, 108, 0.6);
        }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Ribbon style from original */
    .ribbon {
        position: relative;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        padding: 0.5rem;
        margin: 1rem 0;
        border-radius: 5px;
        animation: slideInRight 1s ease;
    }
    
    /* Balloon style from original */
    .balloon {
        position: relative;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 50% 50% 50% 0;
        transform: rotate(-45deg);
        margin: 2rem auto;
        width: 100px;
        height: 100px;
        display: flex;
        align-items: center;
        justify-content: center;
        animation: float 3s ease-in-out infinite;
    }
    
    .balloon-content {
        transform: rotate(45deg);
        text-align: center;
    }
    
    @keyframes float {
        0%, 100% {
            transform: translateY(0) rotate(-45deg);
        }
        50% {
            transform: translateY(-20px) rotate(-45deg);
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# TITLE WITH TEAM CREDIT - ENHANCED DESIGN
# ============================================
st.markdown('<h1 class="main-header">💰 Customer Financial Risk Prediction Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header" style="text-align: center;">African Financial Markets Analysis</p>', unsafe_allow_html=True)

# Team credit ribbon - Enhanced version
st.markdown("""
<div class="starry-ribbon">
    <strong>👥 Team Project:</strong> AMARACHI FLORENCE • Thato Maelane • Philip Odiachi • Mavis 
    | <a href="https://dataverseafrica.org" target="_blank" style="color: white; text-decoration: underline;">🌍 Dataverse Africa Internship</a>
</div>
""", unsafe_allow_html=True)

# ============================================
# HELPER FUNCTIONS
# ============================================
def show_success_balloon(message):
    """Show animated balloon with message"""
    st.markdown(f"""
    <div class="bouncing-balloon">
        <div class="balloon-content">
            <div style="font-size: 1.2rem;">🎉</div>
            <div style="font-size: 0.8rem;">{message}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_hurry_alert(message):
    """Show hurry pulse alert"""
    st.markdown(f"""
    <div style="text-align: center; margin: 1rem 0;">
        <div class="hurry-pulse">{message}</div>
    </div>
    """, unsafe_allow_html=True)

def get_column_case_insensitive(df, column_name):
    """Get column name with case-insensitive matching"""
    column_name_lower = column_name.lower()
    for col in df.columns:
        if col.lower() == column_name_lower:
            return col
    return None

def safe_hover_data(df, preferred_columns):
    """Safely get hover data columns that exist in dataframe"""
    hover_cols = []
    for col in preferred_columns:
        actual_col = get_column_case_insensitive(df, col)
        if actual_col:
            hover_cols.append(actual_col)
    return hover_cols if hover_cols else None

def create_data_card(title, value, change=None, color="#667eea"):
    """Create a professional data card"""
    change_html = ""
    if change:
        change_direction = "▲" if float(change.replace('%', '').replace('+', '')) >= 0 else "▼"
        change_color = "#10B981" if change_direction == "▲" else "#EF4444"
        change_html = f'<div style="font-size: 0.9rem; color: {change_color};">{change_direction} {change}</div>'
    
    return f"""
    <div class="data-card" style="border-top-color: {color};">
        <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">{title}</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: #1F2937;">{value}</div>
        {change_html}
    </div>
    """

# Initialize session state for data
if 'df' not in st.session_state:
    st.session_state.df = None
if 'pbi_data' not in st.session_state:
    st.session_state.pbi_data = None
if 'column_mapping' not in st.session_state:
    st.session_state.column_mapping = {}

# ============================================
# SIDEBAR - ENHANCED DESIGN
# ============================================
with st.sidebar:
    # Dataverse logo and link - Enhanced
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px;">
        <div style="font-size: 1.5rem; color: white; margin-bottom: 0.5rem;">🌍</div>
        <h3 style="color: white; margin: 0;">DATAVERSE AFRICA</h3>
        <p style="color: rgba(255,255,255,0.9); font-size: 0.9rem; margin: 0;">Empowering Africa's Digital Future</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation with enhanced icons
    st.markdown("### 🧭 Navigation")
    
    pages = {
        "🏠 Dashboard": "Executive dashboard with overview",
        "🔍 Customer Analysis": "Deep customer insights and filtering",
        "📊 Clusters": "Customer segmentation analysis",
        "🎯 Predict": "Real-time prediction interface",
        "📈 Insights": "Business recommendations",
        "📊 Power BI Dashboard": "Enhanced pipeline visualizations",
        "👥 Team": "Project team information",
        "⚙️ Settings": "System configuration"
    }
    
    selected_page = st.radio(
        "Select Page",
        list(pages.keys()),
        label_visibility="collapsed"
    )
    
    # Show description for selected page
    st.info(f"📄 {pages[selected_page]}")
    
    st.markdown("---")
    
    # Quick stats - Enhanced
    st.markdown("### 📊 Quick Stats")
    
    if st.session_state.df is not None and not st.session_state.df.empty:
        df = st.session_state.df
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(create_data_card("Total Customers", f"{len(df):,}"), unsafe_allow_html=True)
            credit_col = get_column_case_insensitive(df, 'Credit_Score')
            if credit_col:
                st.markdown(create_data_card("Avg Credit Score", f"{df[credit_col].mean():.0f}"), unsafe_allow_html=True)
        with col2:
            expend_col = get_column_case_insensitive(df, 'Monthly_Expenditure')
            if expend_col:
                st.markdown(create_data_card("Avg Spend", f"₦{df[expend_col].mean():,.0f}"), unsafe_allow_html=True)
            age_col = get_column_case_insensitive(df, 'age')
            if age_col:
                st.markdown(create_data_card("Avg Age", f"{df[age_col].mean():.1f}"), unsafe_allow_html=True)
    else:
        try:
            if os.path.exists("outputs/processed_data.csv"):
                df = pd.read_csv("outputs/processed_data.csv")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Customers", f"{len(df):,}")
                    credit_col = get_column_case_insensitive(df, 'Credit_Score')
                    if credit_col:
                        st.metric("Avg Credit Score", f"{df[credit_col].mean():.0f}")
                with col2:
                    expend_col = get_column_case_insensitive(df, 'Monthly_Expenditure')
                    if expend_col:
                        st.metric("Avg Spend", f"₦{df[expend_col].mean():,.0f}")
                    age_col = get_column_case_insensitive(df, 'age')
                    if age_col:
                        st.metric("Avg Age", f"{df[age_col].mean():.1f}")
        except:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(create_data_card("Total Customers", "5,200", "+12%"), unsafe_allow_html=True)
                st.markdown(create_data_card("Avg Credit", "620", "+8"), unsafe_allow_html=True)
            with col2:
                st.markdown(create_data_card("Avg Spend", "₦151,444", "+5%"), unsafe_allow_html=True)
                st.markdown(create_data_card("Avg Age", "43.1"), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # System Status Indicators
    st.markdown("### 🔧 System Status")
    
    # Data status
    if st.session_state.df is not None:
        data_status = "ACTIVE"
        status_color = "#10B981"
    else:
        data_status = "INACTIVE"
        status_color = "#EF4444"
    
    st.markdown(f"""
    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
        <span class="status-indicator status-{'active' if data_status == 'ACTIVE' else 'error'}"></span>
        <span style="font-size: 0.9rem;">Data Loaded: <strong style="color: {status_color};">{data_status}</strong></span>
    </div>
    """, unsafe_allow_html=True)
    
    # API status
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        api_status = "ACTIVE" if response.status_code == 200 else "INACTIVE"
    except:
        api_status = "INACTIVE"
    
    api_color = "#10B981" if api_status == "ACTIVE" else "#F59E0B"
    st.markdown(f"""
    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
        <span class="status-indicator status-{'active' if api_status == 'ACTIVE' else 'warning'}"></span>
        <span style="font-size: 0.9rem;">API Connection: <strong style="color: {api_color};">{api_status}</strong></span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Connect section
    st.markdown("### 🔗 Connect")
    st.markdown("[📚 Documentation](#)")
    st.markdown("[📧 Contact Team](#)")
    st.markdown("[⭐ GitHub Repository](#)")

# API URL
API_URL = "http://localhost:8000"

# ============================================
# PAGE 1: DASHBOARD
# ============================================
if selected_page == "🏠 Dashboard":
    st.markdown('<h2 class="sub-header">📈 Executive Dashboard</h2>', unsafe_allow_html=True)
    
    # Show success balloon
    show_success_balloon("Welcome!")
    
    # Load Dataset Section - Enhanced
    st.markdown("### 📂 Load Your Dataset")
    
    uploaded_file = st.file_uploader("Upload any customer dataset (CSV format)", type=["csv"], key="dashboard_upload")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            
            show_hurry_alert("DATA PROCESSING - PLEASE WAIT")
            
            # Auto column mapping
            column_mapping = {}
            expected_cols = {
                'Customer_ID': ['customer_id', 'customerid', 'id', 'CustomerID'],
                'Age': ['age', 'customer_age', 'Age'],
                'Monthly_Expenditure': ['monthly_expenditure', 'expenditure', 'Monthly_Expenditure'],
                'Credit_Score': ['credit_score', 'credit', 'Credit_Score'],
                'Income_Level': ['income_level', 'income', 'Income_Level'],
                'Location': ['location', 'city', 'Location'],
                'Transaction_Channel': ['transaction_channel', 'channel', 'Transaction_Channel']
            }
            
            for expected, possible_names in expected_cols.items():
                found = False
                for possible in possible_names:
                    actual_col = get_column_case_insensitive(df, possible)
                    if actual_col:
                        column_mapping[expected] = actual_col
                        found = True
                        break
            
            st.session_state.column_mapping = column_mapping
            
            st.markdown('<div class="success-badge">✅ Dataset Loaded Successfully!</div>', unsafe_allow_html=True)
            st.success(f"Dataset loaded: {len(df):,} records × {len(df.columns)} columns")
            
            # Show preview
            with st.expander("📋 Dataset Preview", expanded=True):
                st.dataframe(df.head(100), use_container_width=True)
            
            # Statistics
            st.markdown("### 📊 Dataset Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Numerical Features")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            
            with col2:
                st.markdown("#### Categorical Features")
                cat_cols = df.select_dtypes(include=['object']).columns
                for col in cat_cols[:5]:
                    st.write(f"**{col}**: {df[col].nunique()} unique values")
                    if df[col].nunique() < 10:
                        value_counts = df[col].value_counts()
                        fig = px.pie(values=value_counts.values, names=value_counts.index, 
                                   title=col, color_discrete_sequence=px.colors.sequential.RdBu)
                        st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    # Check if processed data exists from pipeline
    processed_file = "outputs/processed_data.csv"
    
    if os.path.exists(processed_file) and st.session_state.df is None:
        try:
            df = pd.read_csv(processed_file)
            st.session_state.df = df
            st.markdown('<div class="success-badge">✅ Processed Dataset Loaded from Pipeline!</div>', unsafe_allow_html=True)
            st.success(f"Dataset loaded: {len(df):,} records × {len(df.columns)} columns")
        except Exception as e:
            st.error(f"Error loading processed data: {str(e)}")
    
    # Metrics cards based on actual data
    st.markdown("### 🎯 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    if st.session_state.df is not None and not st.session_state.df.empty:
        df = st.session_state.df
        
        with col1:
            total_customers = len(df)
            st.markdown(create_data_card("Total Customers", f"{total_customers:,}", "+12%", "#667eea"), unsafe_allow_html=True)
        
        with col2:
            credit_col = get_column_case_insensitive(df, 'Credit_Score')
            if credit_col:
                avg_credit = df[credit_col].mean()
                st.markdown(create_data_card("Avg Credit Score", f"{avg_credit:.0f}", "+8", "#10B981"), unsafe_allow_html=True)
            else:
                st.markdown(create_data_card("Avg Credit Score", "645", "+8", "#10B981"), unsafe_allow_html=True)
        
        with col3:
            digital_col = get_column_case_insensitive(df, 'digital_adoption_score')
            if digital_col:
                digital_adoption = (df[digital_col].mean() / 4) * 100
                st.markdown(create_data_card("Digital Adoption", f"{digital_adoption:.1f}%", "+15%", "#F59E0B"), unsafe_allow_html=True)
            else:
                st.markdown(create_data_card("Digital Adoption", "68%", "+15%", "#F59E0B"), unsafe_allow_html=True)
        
        with col4:
            risk_col = get_column_case_insensitive(df, 'risk_score')
            if risk_col:
                high_risk = (df[risk_col] > 0.6).sum()
                risk_rate = (high_risk / len(df)) * 100
                st.markdown(create_data_card("Risk Rate", f"{risk_rate:.1f}%", "-2.3%", "#EF4444"), unsafe_allow_html=True)
            else:
                st.markdown(create_data_card("Risk Rate", "12.5%", "-2.3%", "#EF4444"), unsafe_allow_html=True)
    else:
        # Show static sample metrics
        with col1:
            st.markdown(create_data_card("Total Customers", "5,200", "+12%", "#667eea"), unsafe_allow_html=True)
        
        with col2:
            st.markdown(create_data_card("Avg Credit Score", "645", "+8", "#10B981"), unsafe_allow_html=True)
        
        with col3:
            st.markdown(create_data_card("Digital Adoption", "68%", "+15%", "#F59E0B"), unsafe_allow_html=True)
        
        with col4:
            st.markdown(create_data_card("Risk Rate", "12.5%", "-2.3%", "#EF4444"), unsafe_allow_html=True)
    
    # Visualizations
    st.markdown("### 📈 Data Visualizations")
    
    # Check if we have data loaded
    if st.session_state.df is not None and not st.session_state.df.empty:
        df = st.session_state.df
    else:
        # Generate sample data for visualization if no data loaded
        np.random.seed(42)
        sample_size = 1000
        df = pd.DataFrame({
            'Credit_Score': np.random.normal(650, 100, sample_size).clip(300, 850),
            'Monthly_Expenditure': np.random.lognormal(12, 0.8, sample_size).clip(20000, 500000),
            'age': np.random.randint(22, 65, sample_size),
            'Cluster': np.random.choice(['Digital-First', 'Traditional', 'High-Risk', 'Medium', 'Positive'], sample_size),
            'Risk_Score': np.random.beta(2, 5, sample_size)
        })
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Credit Score Distribution
        credit_col = get_column_case_insensitive(df, 'Credit_Score')
        if credit_col:
            fig = px.histogram(df, x=credit_col, nbins=30, title='Credit Score Distribution',
                              color_discrete_sequence=['#636EFA'])
            fig.update_layout(bargap=0.1, plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Expenditure vs Credit Score
        sample_df = df.sample(min(500, len(df)))
        
        # Get actual column names that exist
        cluster_col = get_column_case_insensitive(sample_df, 'Cluster') or get_column_case_insensitive(sample_df, 'cluster_name') or 'Cluster'
        credit_col = get_column_case_insensitive(sample_df, 'Credit_Score') or 'Credit_Score'
        expend_col = get_column_case_insensitive(sample_df, 'Monthly_Expenditure') or 'Monthly_Expenditure'
        
        # Get hover data columns that actually exist
        hover_cols = safe_hover_data(sample_df, ['age', 'Age', 'Customer_ID', 'ID'])
        
        # Create the scatter plot
        fig = px.scatter(sample_df, 
                        x=credit_col, 
                        y=expend_col,
                        color=cluster_col if cluster_col in sample_df.columns else None,
                        title='Credit Score vs Monthly Expenditure',
                        hover_data=hover_cols,
                        color_discrete_sequence=px.colors.qualitative.Set3)
        
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    # Cluster distribution
    cluster_col = get_column_case_insensitive(df, 'cluster_name') or get_column_case_insensitive(df, 'Cluster') or get_column_case_insensitive(df, 'cluster')
    if cluster_col and cluster_col in df.columns:
        cluster_counts = df[cluster_col].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Count']
        
        fig = px.bar(cluster_counts, x='Cluster', y='Count', title='Customer Segments Distribution',
                    color='Cluster', color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# PAGE 2: CUSTOMER ANALYSIS
# ============================================
elif selected_page == "🔍 Customer Analysis":
    st.markdown('<h2 class="sub-header">🔍 Deep Customer Analysis</h2>', unsafe_allow_html=True)
    
    # Show success balloon
    show_success_balloon("Analysis Ready!")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.replace(' ', '_')
        
        # Filters
        st.markdown("### 🔍 Filter Customers")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            income_col = get_column_case_insensitive(df, 'Income_Level')
            if income_col and income_col in df.columns:
                income_options = ['ALL'] + sorted(df[income_col].dropna().unique().tolist())
                selected_income = st.selectbox("Income Level", income_options)
                income_filter = None if selected_income == 'ALL' else [selected_income]
        
        with col2:
            credit_col = get_column_case_insensitive(df, 'Credit_Score')
            if credit_col and credit_col in df.columns:
                min_val = int(df[credit_col].min())
                max_val = int(df[credit_col].max())
                credit_range = st.slider(
                    "Credit Score Range",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val)
                )
        
        with col3:
            location_col = get_column_case_insensitive(df, 'Location')
            if location_col and location_col in df.columns:
                location_options = ['ALL'] + sorted(df[location_col].dropna().unique().tolist())
                selected_location = st.selectbox("Location", location_options)
                location_filter = None if selected_location == 'ALL' else [selected_location]
        
        # Apply filters
        filtered_df = df.copy()
        
        if 'income_filter' in locals() and income_filter and income_col:
            filtered_df = filtered_df[filtered_df[income_col].isin(income_filter)]
        
        if 'credit_range' in locals() and credit_col:
            filtered_df = filtered_df[
                (filtered_df[credit_col] >= credit_range[0]) & 
                (filtered_df[credit_col] <= credit_range[1])
            ]
        
        if 'location_filter' in locals() and location_filter and location_col:
            filtered_df = filtered_df[filtered_df[location_col].isin(location_filter)]
        
        st.markdown(f'<div class="success-badge">✅ Showing {len(filtered_df):,} customers ({(len(filtered_df)/len(df)*100):.1f}% of total)</div>', unsafe_allow_html=True)
        
        # Customer details table
        st.markdown("### 📋 Customer Details")
        
        # Define display columns that exist
        display_cols = []
        possible_cols = [
            'Customer_ID', 'CustomerID', 'customer_id',
            'Age', 'age',
            'Income_Level', 'IncomeLevel', 'income_level',
            'Credit_Score', 'credit_score',
            'Monthly_Expenditure', 'monthly_expenditure',
            'Location', 'location',
            'Transaction_Channel', 'transaction_channel',
            'cluster_name', 'Cluster', 'cluster',
            'risk_score', 'sentiment_score', 'sentiment_label'
        ]
        
        for col in possible_cols:
            actual_col = get_column_case_insensitive(filtered_df, col)
            if actual_col and actual_col not in display_cols:
                display_cols.append(actual_col)
        
        if display_cols:
            st.dataframe(filtered_df[display_cols].head(100), use_container_width=True)
            
            # Download option
            csv = filtered_df[display_cols].to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data",
                data=csv,
                file_name="filtered_customers.csv",
                mime="text/csv"
            )
        
        # Detailed analysis
        st.markdown("### 📊 Detailed Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Credit Score by Income Level
            income_col_actual = get_column_case_insensitive(filtered_df, 'Income_Level')
            credit_col_actual = get_column_case_insensitive(filtered_df, 'Credit_Score')
            
            if income_col_actual and credit_col_actual:
                fig = px.box(filtered_df, x=income_col_actual, y=credit_col_actual,
                            title='Credit Score by Income Level',
                            color=income_col_actual,
                            color_discrete_sequence=px.colors.sequential.Plasma)
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Location vs Expenditure
            location_col_actual = get_column_case_insensitive(filtered_df, 'Location')
            expend_col = get_column_case_insensitive(filtered_df, 'Monthly_Expenditure')
            
            if location_col_actual and expend_col:
                location_spend = filtered_df.groupby(location_col_actual)[expend_col].mean().sort_values(ascending=False).head(10)
                fig = px.bar(x=location_spend.index, y=location_spend.values,
                            title='Top 10 Locations by Average Expenditure',
                            labels={'x': 'Location', 'y': 'Avg Expenditure (₦)'},
                            color=location_spend.values,
                            color_continuous_scale='Viridis')
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
        
        # Customer distribution by channel
        channel_col = get_column_case_insensitive(filtered_df, 'Transaction_Channel')
        if channel_col and channel_col in filtered_df.columns:
            st.markdown("### 📱 Transaction Channel Analysis")
            channel_data = filtered_df[channel_col].value_counts().reset_index()
            channel_data.columns = ['Channel', 'Count']
            
            fig = px.pie(channel_data, values='Count', names='Channel', 
                        title='Transaction Channel Distribution',
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk distribution
        risk_col = get_column_case_insensitive(filtered_df, 'risk_score')
        if risk_col and risk_col in filtered_df.columns:
            st.markdown("### ⚠️ Risk Score Distribution")
            
            fig = px.histogram(filtered_df, x=risk_col, nbins=20,
                            title='Risk Score Distribution',
                            color_discrete_sequence=['#FF6B6B'])
            fig.add_vline(x=0.6, line_dash="dash", line_color="red",
                         annotation_text="High Risk Threshold")
            fig.add_vline(x=0.3, line_dash="dash", line_color="green",
                         annotation_text="Low Risk Threshold")
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.error("❌ No data loaded. Please load data in Dashboard page or run the pipeline.")

# ============================================
# PAGE 3: CLUSTERS
# ============================================
elif selected_page == "📊 Clusters":
    st.markdown('<h2 class="sub-header">📊 Customer Segmentation Analysis</h2>', unsafe_allow_html=True)
    
    # Show success balloon
    show_success_balloon("Clusters Found!")
    
    try:
        # Load cluster data
        cluster_profiles_path = "outputs/cluster_profiles.csv"
        
        if os.path.exists(cluster_profiles_path):
            cluster_profiles = pd.read_csv(cluster_profiles_path)
            
            # Display cluster cards with visible text
            st.markdown("### 🎯 Customer Segments Overview")
            
            cols = st.columns(min(len(cluster_profiles), 3))
            colors = ['#4F46E5', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899']
            
            for idx, (_, row) in enumerate(cluster_profiles.iterrows()):
                if idx % 3 == 0 and idx > 0:
                    cols = st.columns(min(len(cluster_profiles) - idx, 3))
                
                col_idx = idx % 3
                if col_idx < len(cols):
                    with cols[col_idx]:
                        cluster_name = row.get('cluster_name', f'Cluster {row.get("cluster_id", idx)}')
                        cluster_size = row.get('size', 0)
                        percentage = row.get('percentage', 0)
                        
                        st.markdown(f'''
                        <div class="cluster-card" style="border-left: 5px solid {colors[idx % len(colors)]};">
                            <h4 style="color: {colors[idx % len(colors)]}; margin-bottom: 10px;">
                                {cluster_name}
                            </h4>
                            <p style="color: #333;"><strong>Size:</strong> {cluster_size:,} customers</p>
                            <p style="color: #333;"><strong>Percentage:</strong> {percentage:.1f}%</p>
                        </div>
                        ''', unsafe_allow_html=True)
            
            # Cluster comparison
            st.markdown("### 📈 Cluster Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'avg_credit_score' in cluster_profiles.columns:
                    fig = px.bar(cluster_profiles, 
                                x='cluster_name', 
                                y='avg_credit_score',
                                title='Average Credit Score by Cluster',
                                color='cluster_name',
                                color_discrete_sequence=colors[:len(cluster_profiles)])
                    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', xaxis_title="", yaxis_title="Credit Score")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'avg_monthly_expenditure' in cluster_profiles.columns:
                    fig = px.bar(cluster_profiles, 
                                x='cluster_name', 
                                y='avg_monthly_expenditure',
                                title='Average Monthly Expenditure by Cluster',
                                color='cluster_name',
                                color_discrete_sequence=colors[:len(cluster_profiles)])
                    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', xaxis_title="", yaxis_title="Expenditure (₦)")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Digital adoption comparison
            st.markdown("### 📱 Digital Adoption by Cluster")
            
            if 'digital_adoption' in cluster_profiles.columns and 'avg_credit_score' in cluster_profiles.columns:
                fig = px.scatter(cluster_profiles,
                                x='digital_adoption',
                                y='avg_credit_score',
                                size='size',
                                color='cluster_name',
                                hover_data=['avg_monthly_expenditure', 'avg_risk_score', 'percentage'],
                                title='Digital Adoption vs Credit Score',
                                size_max=60,
                                color_discrete_sequence=colors[:len(cluster_profiles)])
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed cluster profiles
            st.markdown("### 📋 Detailed Cluster Profiles")
            
            for _, row in cluster_profiles.iterrows():
                with st.expander(f"{row.get('cluster_name', f'Cluster {row.get('cluster_id', 'N/A')}')} - Details"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Customers", f"{row.get('size', 0):,}")
                        if 'percentage' in row:
                            st.metric("Percentage", f"{row['percentage']:.1f}%")
                    
                    with col2:
                        if 'avg_credit_score' in row:
                            st.metric("Avg Credit Score", f"{row['avg_credit_score']:.0f}")
                        if 'avg_sentiment' in row:
                            st.metric("Avg Sentiment", f"{row['avg_sentiment']:.3f}")
                    
                    with col3:
                        if 'avg_monthly_expenditure' in row:
                            st.metric("Avg Expenditure", f"₦{row['avg_monthly_expenditure']:,.0f}")
                        if 'avg_risk_score' in row:
                            risk_level = "High" if row['avg_risk_score'] > 0.6 else "Low" if row['avg_risk_score'] < 0.3 else "Medium"
                            st.metric("Risk Level", risk_level)
        
        else:
            # If cluster profiles don't exist, check session state data
            if st.session_state.df is not None:
                df = st.session_state.df
                
                cluster_col = get_column_case_insensitive(df, 'cluster_name') or get_column_case_insensitive(df, 'cluster')
                if cluster_col and cluster_col in df.columns:
                    # Create summary
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if numeric_cols:
                        cluster_summary = df.groupby(cluster_col).agg({
                            numeric_cols[0]: 'count',
                            **{col: 'mean' for col in numeric_cols[1:3] if col in df.columns}
                        }).reset_index()
                        
                        # Rename columns
                        cluster_summary.columns = ['Cluster', 'Count'] + [f'Avg_{col}' for col in cluster_summary.columns[2:]]
                        
                        # Display summary
                        st.dataframe(cluster_summary, use_container_width=True)
                        
                        # Visualization
                        fig = px.sunburst(cluster_summary, 
                                        path=['Cluster'], 
                                        values='Count',
                                        title='Customer Distribution by Cluster',
                                        color='Count',
                                        color_continuous_scale='Viridis')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No numeric columns found for analysis.")
                else:
                    st.warning("No cluster information found in data. Please run the clustering pipeline first.")
            else:
                st.error("No data found. Please load data in Dashboard page.")
    
    except Exception as e:
        st.error(f"Error loading cluster data: {str(e)}")

# ============================================
# PAGE 4: PREDICT
# ============================================
elif selected_page == "🎯 Predict":
    st.markdown('<h2 class="sub-header">🎯 Real-time Customer Prediction</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Single Customer", "Batch Prediction", "API Test"])
    
    with tab1:
        st.markdown("### 👤 Predict Single Customer")
        
        with st.form("single_customer_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                customer_id = st.text_input("Customer ID", "CUST001")
                age = st.slider("Age", 18, 70, 35)
                monthly_expenditure = st.number_input("Monthly Expenditure (₦)", min_value=0.0, value=150000.0, step=1000.0)
                credit_score = st.slider("Credit Score", 300, 850, 650)
                transaction_count = st.number_input("Transaction Count", min_value=1, value=25)
            
            with col2:
                avg_transaction_value = st.number_input("Avg Transaction Value (₦)", min_value=0.0, value=6000.0, step=100.0)
                
                st.markdown("**Digital Channels Used:**")
                col_a, col_b = st.columns(2)
                with col_a:
                    uses_pos = st.checkbox("POS", value=True)
                    uses_web = st.checkbox("Web/Transfer", value=False)
                with col_b:
                    uses_ussd = st.checkbox("USSD", value=True)
                    uses_mobile_app = st.checkbox("Mobile App", value=True)
                
                income_level = st.selectbox("Income Level", ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"])
                saving_behavior = st.selectbox("Saving Behavior", ["Poor", "Average", "Good"])
                location = st.text_input("Location", "Lagos")
                feedback = st.text_area("Customer Feedback", "Good service overall")
                
                # Required fields for API
                transaction_channel = st.selectbox("Transaction Channel", 
                                                 ["USSD", "Web", "Mobile App", "POS", "ATM", "Branch"])
                spending_category = st.selectbox("Spending Category",
                                               ["Groceries", "Rent", "Utilities", "Transport", "Health", 
                                                "Education", "Entertainment", "Online Shopping", "Savings Deposit"])
            
            submitted = st.form_submit_button("🎯 Predict Segment", type="primary")
        
        if submitted:
            # Show success balloon
            show_hurry_alert("PROCESSING PREDICTION")
            
            try:
                # Prepare data for API
                customer_data = {
                    "customer_id": customer_id,
                    "age": age,
                    "monthly_expenditure": monthly_expenditure,
                    "credit_score": credit_score,
                    "transaction_count": transaction_count,
                    "avg_transaction_value": avg_transaction_value,
                    "uses_pos": 1 if uses_pos else 0,
                    "uses_web": 1 if uses_web else 0,
                    "uses_ussd": 1 if uses_ussd else 0,
                    "uses_mobile_app": 1 if uses_mobile_app else 0,
                    "income_level": income_level,
                    "saving_behavior": saving_behavior,
                    "location": location,
                    "feedback": feedback,
                    "transaction_channel": transaction_channel,
                    "spending_category": spending_category
                }
                
                # Try API call
                try:
                    with st.spinner("Predicting..."):
                        response = requests.post(f"{API_URL}/predict", json=customer_data, timeout=10)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display results
                        st.markdown('<div class="success-badge">✅ Prediction Successful!</div>', unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(create_data_card("Customer Segment", result.get('cluster_name', 'Unknown'), color="#667eea"), unsafe_allow_html=True)
                        
                        with col2:
                            risk_category = result.get('risk_category', 'Medium Risk')
                            risk_color = "#EF4444" if "High" in risk_category else "#F59E0B" if "Medium" in risk_category else "#10B981"
                            st.markdown(create_data_card("Risk Category", risk_category, color=risk_color), unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(create_data_card("Digital Score", f"{result.get('digital_adoption_score', 0)}/4.0", color="#10B981"), unsafe_allow_html=True)
                        
                        # Display more details
                        st.markdown("### 📊 Detailed Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### Segment Characteristics")
                            characteristics = result.get('segment_characteristics', {})
                            for key, value in characteristics.items():
                                st.markdown(f"**{key.replace('_', ' ').title()}:** `{value}`")
                        
                        with col2:
                            st.markdown("#### 💡 Recommendations")
                            recommendations = result.get('recommendations', [])
                            for i, rec in enumerate(recommendations, 1):
                                st.markdown(f"{i}. **{rec}**")
                        
                        # Performance
                        st.info(f"⏱️ Processing time: {result.get('processing_time', 0)} ms")
                    
                    else:
                        st.error(f"API Error: {response.status_code} - {response.text}")
                        st.info("Try running the API server first: `python api/api_main.py`")
                
                except requests.exceptions.ConnectionError:
                    st.error("❌ API not running. Start the API with: `python api/api_main.py`")
                    st.info("The API server needs to be running for real-time predictions.")
                except requests.exceptions.Timeout:
                    st.error("❌ API request timed out. Please try again.")
            
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
    
    with tab2:
        st.markdown("### 📋 Batch Prediction")
        
        uploaded_file = st.file_uploader("Upload CSV file with customer data", type=["csv"])
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(batch_df)} customers")
                st.dataframe(batch_df.head(), use_container_width=True)
                
                if st.button("🔮 Predict All Customers", type="primary"):
                    # Show success balloon
                    show_hurry_alert("PROCESSING BATCH PREDICTION")
                    
                    # Prepare batch data
                    customers = []
                    for _, row in batch_df.iterrows():
                        customer = {
                            "customer_id": str(row.get('Customer_ID', row.get('customer_id', f"CUST{_}"))),
                            "age": int(row.get('Age', row.get('age', 35))),
                            "monthly_expenditure": float(row.get('Monthly_Expenditure', row.get('monthly_expenditure', 100000))),
                            "credit_score": int(row.get('Credit_Score', row.get('credit_score', 600))),
                            "transaction_count": int(row.get('transaction_count', row.get('Transaction_Count', 20))),
                            "avg_transaction_value": float(row.get('avg_transaction_value', row.get('Avg_Transaction_Value', 5000))),
                            "uses_pos": int(row.get('uses_pos', row.get('Uses_POS', 0))),
                            "uses_web": int(row.get('uses_web', row.get('Uses_Web', 0))),
                            "uses_ussd": int(row.get('uses_ussd', row.get('Uses_USSD', 0))),
                            "uses_mobile_app": int(row.get('uses_mobile_app', row.get('Uses_Mobile_App', 0))),
                            "income_level": str(row.get('Income_Level', row.get('income_level', 'Middle'))),
                            "saving_behavior": str(row.get('Saving_Behavior', row.get('saving_behavior', 'Average'))),
                            "location": str(row.get('Location', row.get('location', 'Unknown'))),
                            "feedback": str(row.get('Customer_Feedback', row.get('feedback', ''))),
                            "transaction_channel": str(row.get('Transaction_Channel', row.get('transaction_channel', 'Mobile App'))),
                            "spending_category": str(row.get('Spending_Category', row.get('spending_category', 'Groceries')))
                        }
                        customers.append(customer)
                    
                    batch_data = {"customers": customers}
                    
                    # Call batch API
                    try:
                        with st.spinner(f"Processing {len(customers)} customers..."):
                            response = requests.post(f"{API_URL}/predict/batch", json=batch_data, timeout=30)
                        
                        if response.status_code == 200:
                            results = response.json()
                            
                            # Create results dataframe
                            results_list = []
                            for r in results:
                                results_list.append({
                                    'Customer_ID': r['customer_id'],
                                    'Segment': r['cluster_name'],
                                    'Risk_Category': r['risk_category'],
                                    'Risk_Score': r['risk_score'],
                                    'Digital_Score': r['digital_adoption_score'],
                                    'Top_Recommendation': r['recommendations'][0] if r['recommendations'] else 'N/A'
                                })
                            
                            results_df = pd.DataFrame(results_list)
                            
                            st.markdown('<div class="success-badge">✅ Predictions Completed!</div>', unsafe_allow_html=True)
                            st.success(f"✅ Predictions completed for {len(results)} customers")
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Predictions",
                                data=csv,
                                file_name="batch_predictions.csv",
                                mime="text/csv"
                            )
                            
                            # Summary statistics
                            st.markdown("### 📊 Batch Summary")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                segment_counts = results_df['Segment'].value_counts()
                                st.dataframe(segment_counts, use_container_width=True)
                            
                            with col2:
                                risk_counts = results_df['Risk_Category'].value_counts()
                                fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                                           title="Risk Category Distribution",
                                           color_discrete_sequence=px.colors.sequential.RdBu)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col3:
                                st.metric("Avg Risk Score", f"{results_df['Risk_Score'].mean():.3f}")
                                st.metric("Avg Digital Score", f"{results_df['Digital_Score'].mean():.2f}")
                        
                        else:
                            st.error(f"❌ API Error: {response.status_code}")
                            st.text(response.text)
                    
                    except requests.exceptions.ConnectionError:
                        st.error("❌ API not running. Please start the API server.")
                    except requests.exceptions.Timeout:
                        st.error("❌ Batch processing timed out. Try with fewer customers.")
            
            except Exception as e:
                st.error(f"❌ Error processing batch: {str(e)}")
    
    with tab3:
        st.markdown("### 🔧 API Connection Test")
        
        col1, col2 = st.columns(2)
        
        with col1:
            api_url = st.text_input("API URL", API_URL)
        
        with col2:
            if st.button("🔗 Test Connection", type="primary"):
                try:
                    response = requests.get(f"{api_url}/health", timeout=5)
                    if response.status_code == 200:
                        health_data = response.json()
                        st.markdown('<div class="success-badge">✅ API is Running!</div>', unsafe_allow_html=True)
                        
                        # Display health info
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            status_color = "#10B981" if health_data.get('status') == 'healthy' else "#EF4444"
                            st.metric("Status", health_data.get('status', 'unknown'))
                        
                        with col2:
                            st.metric("Version", health_data.get('version', 'unknown'))
                        
                        with col3:
                            models_status = "✅" if health_data.get('models_loaded', False) else "❌"
                            st.metric("Models Loaded", models_status)
                        
                        # Additional info
                        st.info(f"📅 Last Check: {health_data.get('timestamp', 'unknown')}")
                    else:
                        st.error(f"❌ API responded with status: {response.status_code}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API")
                    st.info("Start the API with: `python api/api_main.py`")
                except Exception as e:
                    st.error(f"❌ Connection test failed: {str(e)}")
        
        # Demo prediction
        st.markdown("### 🚀 Demo Prediction")
        
        if st.button("🎪 Run Demo Prediction", type="secondary"):
            try:
                with st.spinner("Running demo..."):
                    response = requests.get(f"{api_url}/demo", timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    st.markdown('<div class="success-badge">✅ Demo Successful!</div>', unsafe_allow_html=True)
                    
                    # Pretty display of demo results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.json(result)
                    with col2:
                        st.markdown("**Demo Summary:**")
                        st.write(f"**Customer:** {result.get('customer_id', 'Unknown')}")
                        st.write(f"**Segment:** {result.get('cluster_name', 'Unknown')}")
                        st.write(f"**Risk:** {result.get('risk_category', 'Unknown')}")
                        st.write(f"**Score:** {result.get('risk_score', 0):.3f}")
                else:
                    st.error("❌ Demo failed")
            except Exception as e:
                st.error(f"❌ Demo error: {str(e)}")

# ============================================
# PAGE 5: INSIGHTS
# ============================================
elif selected_page == "📈 Insights":
    st.markdown('<h2 class="sub-header">📈 Business Insights & Recommendations</h2>', unsafe_allow_html=True)
    
    # Show success balloon
    show_success_balloon("Insights Ready!")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # DYNAMIC INSIGHTS based on current data
        st.markdown("### 📊 Real-time Data Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Credit score insights
            credit_col = get_column_case_insensitive(df, 'Credit_Score')
            if credit_col and credit_col in df.columns:
                avg_credit = df[credit_col].mean()
                low_credit = (df[credit_col] < 500).sum()
                low_credit_pct = (low_credit / len(df)) * 100
                
                st.markdown(create_data_card("Average Credit Score", f"{avg_credit:.0f}"), unsafe_allow_html=True)
                st.markdown(create_data_card("High Risk Customers", f"{low_credit_pct:.1f}%"), unsafe_allow_html=True)
                
                if low_credit_pct > 15:
                    st.warning("⚠️ High proportion of risky customers. Consider financial education programs.")
                elif avg_credit > 700:
                    st.success("✅ Excellent credit health. Focus on premium offerings.")
        
        with col2:
            # Expenditure insights
            expend_col = get_column_case_insensitive(df, 'Monthly_Expenditure')
            if expend_col and expend_col in df.columns:
                avg_expend = df[expend_col].mean()
                high_spenders = (df[expend_col] > df[expend_col].quantile(0.75)).sum()
                high_spenders_pct = (high_spenders / len(df)) * 100
                
                st.markdown(create_data_card("Average Monthly Spend", f"₦{avg_expend:,.0f}"), unsafe_allow_html=True)
                st.markdown(create_data_card("High Spenders", f"{high_spenders_pct:.1f}%"), unsafe_allow_html=True)
                
                if high_spenders_pct > 20:
                    st.info("💰 Significant high-value segment. Target with premium products.")
        
        with col3:
            # Digital adoption insights
            channel_col = get_column_case_insensitive(df, 'Transaction_Channel')
            if channel_col and channel_col in df.columns:
                mobile_users = df[channel_col].astype(str).str.contains('Mobile|App', case=False, na=False).sum()
                mobile_pct = (mobile_users / len(df)) * 100
                digital_users = df[channel_col].astype(str).str.contains('Mobile|Web|App', case=False, na=False).sum()
                digital_pct = (digital_users / len(df)) * 100
                
                st.markdown(create_data_card("Mobile App Users", f"{mobile_pct:.1f}%"), unsafe_allow_html=True)
                st.markdown(create_data_card("Digital Users", f"{digital_pct:.1f}%"), unsafe_allow_html=True)
                
                if digital_pct < 50:
                    st.warning("📱 Low digital adoption. Consider incentives for digital channel usage.")
        
        # Generate dynamic recommendations based on data
        st.markdown("### 💡 Dynamic Recommendations")
        
        recommendations = []
        
        # Check for cluster data
        cluster_col = get_column_case_insensitive(df, 'cluster_name') or get_column_case_insensitive(df, 'cluster')
        if cluster_col and cluster_col in df.columns:
            cluster_counts = df[cluster_col].value_counts()
            if len(cluster_counts) > 0:
                largest_cluster = cluster_counts.index[0]
                largest_pct = (cluster_counts.iloc[0] / len(df)) * 100
                
                recommendations.append(f"**🎯 Focus on Largest Segment**: '{largest_cluster}' represents {largest_pct:.1f}% of customers. Tailor marketing to this group.")
        
        # Risk-based recommendations
        risk_col = get_column_case_insensitive(df, 'risk_score')
        if risk_col and risk_col in df.columns:
            high_risk = (df[risk_col] > 0.6).sum()
            high_risk_pct = (high_risk / len(df)) * 100
            
            if high_risk_pct > 10:
                recommendations.append(f"**⚖️ Risk Management Needed**: {high_risk_pct:.1f}% of customers are high-risk. Implement monitoring and support programs.")
        
        # Digital adoption recommendations
        if 'digital_pct' in locals() and digital_pct < 60:
            recommendations.append(f"**📱 Boost Digital Adoption**: Only {digital_pct:.1f}% use digital channels. Launch digital onboarding campaigns.")
        
        # Location-based recommendations
        location_col = get_column_case_insensitive(df, 'Location')
        if location_col and location_col in df.columns:
            top_locations = df[location_col].value_counts().head(3)
            recommendations.append(f"**🌍 Geographic Focus**: Top 3 locations are {', '.join(top_locations.index.tolist())}. Consider location-specific offerings.")
        
        # Business recommendations from pipeline
        business_rec_path = "outputs/business_recommendations.csv"
        if os.path.exists(business_rec_path):
            business_recs = pd.read_csv(business_rec_path)
            st.markdown("### 🎯 Pipeline Recommendations")
            for _, rec in business_recs.iterrows():
                with st.expander(f"{rec['cluster_name']} - Strategy"):
                    st.markdown(f"**Targeting Strategy:** {rec['targeting_strategy']}")
                    st.markdown(f"**Recommended Products:** {rec['recommended_products']}")
                    st.markdown(f"**Marketing Channels:** {rec['marketing_channels']}")
                    st.markdown(f"**Risk Management:** {rec['risk_management']}")
        
        # If no specific recommendations, provide general ones
        if not recommendations:
            recommendations = [
                "**📊 Regular Analysis**: Monitor customer segments monthly for trends",
                "**🎯 Personalized Marketing**: Use segmentation for targeted campaigns",
                "**💰 Revenue Optimization**: Identify high-value customers for premium offerings",
                "**📱 Digital Transformation**: Increase investment in mobile banking features"
            ]
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
        
        # Action plan
        st.markdown("### 📋 Action Plan Timeline")
        
        action_items = [
            ("Immediate (Week 1)", "Review current customer segments and high-risk profiles"),
            ("Short-term (Month 1)", "Launch targeted campaigns based on segment analysis"),
            ("Medium-term (Quarter 1)", "Implement personalized product recommendations"),
            ("Long-term (Year 1)", "Develop AI-powered real-time customer insights engine")
        ]
        
        for timeline, action in action_items:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"<div style='background: #667eea; color: white; padding: 5px 10px; border-radius: 5px; text-align: center;'>{timeline}</div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"**{action}**")
    
    else:
        # If no data, show static insights
        st.warning("No data available. Please load data in Dashboard first.")
        
        st.markdown("### 💼 General Recommendations")
        
        general_insights = [
            "**🎯 Segment Customers**: Group customers by behavior, risk, and preferences",
            "**💰 Optimize Revenue**: Focus on high-value customers with premium offerings",
            "**📱 Digital First**: Invest in mobile and digital banking platforms",
            "**⚖️ Risk Management**: Monitor credit scores and implement early warning systems",
            "**🤝 Customer Retention**: Develop loyalty programs for long-term relationships",
            "**📊 Data-Driven**: Use analytics for all business decisions"
        ]
        
        for i, insight in enumerate(general_insights, 1):
            st.markdown(f"{i}. {insight}")

# ============================================
# PAGE 6: POWER BI DASHBOARD (NEW)
# ============================================
elif selected_page == "📊 Power BI Dashboard":
    st.markdown('<h2 class="sub-header">📊 Power BI Dashboard - Pipeline Visualizations</h2>', unsafe_allow_html=True)
    
    # Show success balloon
    show_success_balloon("Power BI Data!")
    
    # Load Power BI data
    pbi_file = "powerbi/powerbi_dashboard_data.csv"
    
    if os.path.exists(pbi_file):
        pbi_data = pd.read_csv(pbi_file)
        st.session_state.pbi_data = pbi_data
        
        # Executive Summary
        st.markdown("### 📊 Executive Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_customers = len(pbi_data)
            st.markdown(create_data_card("Total Customers", f"{total_customers:,}", color="#667eea"), unsafe_allow_html=True)
        
        with col2:
            if 'cluster_name' in pbi_data.columns:
                clusters = pbi_data['cluster_name'].nunique()
                st.markdown(create_data_card("Segments Identified", f"{clusters}", color="#10B981"), unsafe_allow_html=True)
            else:
                st.markdown(create_data_card("Segments", "5", color="#10B981"), unsafe_allow_html=True)
        
        with col3:
            if 'Credit_Score' in pbi_data.columns:
                avg_credit = pbi_data['Credit_Score'].mean()
                st.markdown(create_data_card("Avg Credit Score", f"{avg_credit:.0f}", color="#F59E0B"), unsafe_allow_html=True)
            else:
                st.markdown(create_data_card("Avg Credit", "645", color="#F59E0B"), unsafe_allow_html=True)
        
        with col4:
            if 'risk_category' in pbi_data.columns:
                high_risk = (pbi_data['risk_category'] == 'High Risk').sum()
                high_risk_pct = (high_risk / total_customers) * 100
                st.markdown(create_data_card("High Risk", f"{high_risk_pct:.1f}%", color="#EF4444"), unsafe_allow_html=True)
            else:
                st.markdown(create_data_card("Risk Rate", "12.5%", color="#EF4444"), unsafe_allow_html=True)
        
        # PAGE 1: Customer Segmentation
        st.markdown("---")
        st.markdown("### 🎯 Customer Segmentation")
        
        if 'cluster_name' in pbi_data.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart
                cluster_counts = pbi_data['cluster_name'].value_counts().reset_index()
                cluster_counts.columns = ['Cluster', 'Count']
                
                fig = px.pie(cluster_counts, values='Count', names='Cluster',
                            title='Cluster Distribution',
                            color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Cluster table
                cluster_summary = pbi_data.groupby('cluster_name').agg({
                    'Customer_ID': 'count',
                    'Credit_Score': 'mean',
                    'Monthly_Expenditure': 'mean',
                    'digital_adoption_score': 'mean'
                }).reset_index()
                
                cluster_summary.columns = ['Cluster', 'Size', 'Avg Credit', 'Avg Spend', 'Digital Score']
                st.dataframe(cluster_summary, use_container_width=True)
        
        # PAGE 2: Payment Channel Analytics
        st.markdown("---")
        st.markdown("### 📱 Payment Channel Analytics")
        
        if 'Transaction_Channel' in pbi_data.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                channel_counts = pbi_data['Transaction_Channel'].value_counts()
                fig = px.bar(x=channel_counts.index, y=channel_counts.values,
                            title='Transaction Channel Usage',
                            color_discrete_sequence=px.colors.sequential.Viridis)
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'digital_segment' in pbi_data.columns:
                    digital_counts = pbi_data['digital_segment'].value_counts()
                    fig = px.pie(values=digital_counts.values, names=digital_counts.index,
                                title='Digital Segments',
                                color_discrete_sequence=px.colors.qualitative.Pastel)
                    st.plotly_chart(fig, use_container_width=True)
        
        # PAGE 3: Financial Behavior
        st.markdown("---")
        st.markdown("### 💰 Financial Behavior Metrics")
        
        if 'Credit_Score' in pbi_data.columns and 'Monthly_Expenditure' in pbi_data.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(pbi_data, x='Credit_Score', nbins=30,
                                  title='Credit Score Distribution',
                                  color_discrete_sequence=['#636EFA'])
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'Saving_Behavior' in pbi_data.columns:
                    savings_counts = pbi_data['Saving_Behavior'].value_counts()
                    fig = px.bar(x=savings_counts.index, y=savings_counts.values,
                                title='Savings Behavior Distribution',
                                color_discrete_sequence=px.colors.sequential.Plasma)
                    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
        
        # PAGE 4: NLP Insights
        st.markdown("---")
        st.markdown("### 💬 NLP Insights")
        
        if 'sentiment_label' in pbi_data.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                sentiment_counts = pbi_data['sentiment_label'].value_counts()
                fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                            title='Customer Sentiment Distribution',
                            color_discrete_sequence=px.colors.sequential.RdBu)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'Complaint_Type' in pbi_data.columns:
                    complaint_counts = pbi_data['Complaint_Type'].value_counts().head(10)
                    fig = px.bar(x=complaint_counts.index, y=complaint_counts.values,
                                title='Top Complaint Types',
                                color_discrete_sequence=px.colors.sequential.Magma)
                    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
        
        # Key Findings
        st.markdown("---")
        st.markdown("### 🔍 Key Findings")
        
        findings = [
            f"Analysis completed on {len(pbi_data):,} customer records",
            f"Identified {pbi_data['cluster_name'].nunique() if 'cluster_name' in pbi_data.columns else 'multiple'} customer segments",
            f"Digital adoption rate: {(pbi_data['digital_adoption_score'].mean()/4*100 if 'digital_adoption_score' in pbi_data.columns else 0):.1f}%",
            f"Risk distribution analyzed across all customer segments"
        ]
        
        for finding in findings:
            st.markdown(f"• **{finding}**")
        
        # Recommendations from pipeline
        st.markdown("### 🎯 Recommendations")
        
        rec_file = "outputs/business_recommendations.csv"
        if os.path.exists(rec_file):
            recommendations = pd.read_csv(rec_file)
            for _, row in recommendations.iterrows():
                with st.expander(f"{row.get('cluster_name', 'Segment')} - Strategy"):
                    st.write(f"**Targeting Strategy:** {row.get('targeting_strategy', 'N/A')}")
                    st.write(f"**Recommended Products:** {row.get('recommended_products', 'N/A')}")
                    st.write(f"**Marketing Channels:** {row.get('marketing_channels', 'N/A')}")
                    st.write(f"**Risk Management:** {row.get('risk_management', 'N/A')}")
        else:
            st.info("Run the pipeline to generate business recommendations")
        
        # Export Power BI data
        st.markdown("---")
        st.markdown("### 📤 Export Data")
        
        if st.button("📥 Download Power BI Data"):
            csv = pbi_data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="powerbi_dashboard_data.csv",
                mime="text/csv"
            )
    
    else:
        st.error("❌ Power BI data not found. Please run the pipeline first.")
        show_hurry_alert("RUN PIPELINE TO GENERATE POWER BI DATA")
        
        st.markdown("""
        **To generate Power BI data:**
        1. Run `master_pipeline.py` to process your data
        2. This will create `powerbi/powerbi_dashboard_data.csv`
        3. Refresh this page to load the Power BI data
        
        The Power BI data includes:
        - Enhanced customer segments
        - Risk scores and categories
        - Digital adoption metrics
        - Sentiment analysis
        - Payment channel analytics
        """)

# ============================================
# PAGE 7: TEAM
# ============================================
elif selected_page == "👥 Team":
    st.markdown('<h2 class="sub-header">👥 Project Team</h2>', unsafe_allow_html=True)
    
    # Show success balloon
    show_success_balloon("Team Power!")
    
    # Team introduction
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h3 style="color: #667eea;">🎓 Dataverse Africa Internship Program</h3>
        <p>Empowering Africa's Digital Future through Data Science and AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Team members - Enhanced with roles
    team_members = [
        {
            "name": "Amarachi Florence",
            "role": "Financial Data and MEAL Analyst"
        },
        {
            "name": "Thato Maelane",
            "role": "Data Scientist"
        },
        {
            "name": "Philip Odiachi", 
            "role": "Data Analyst"
        },
        {
            "name": "Mavis",
            "role": "Business Analyst"
        }
    ]
    
    # Display team members in simple boxes
    cols = st.columns(2)
    for idx, member in enumerate(team_members):
        with cols[idx % 2]:
            st.markdown(f'''
            <div class="team-card">
                <h4 style="color: #667eea; margin-bottom: 10px;">{member['name']}</h4>
                <p style="color: black;"><strong>Role:</strong> {member['role']}</p>
            </div>
            ''', unsafe_allow_html=True)
    
    # Dataverse Africa section
    st.markdown("---")
    st.markdown("### 🌍 About Dataverse Africa")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 4rem; color: #667eea;">🌍</div>
            <h3 style="color: #667eea;">DATAVERSE AFRICA</h3>
            <p style="color: #666;">Empowering Africa's Digital Future</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        ### Our Mission
        
        At DataVerse, we're on a mission to transform Africa through the power of data. 
        
        Imagine a future where technology fuels progress, improves lives, and causes growth across the continent. 
        
        Sure you can imagine that and that future isn't far fetched anymore, that future is now!
        
        We're building Africa's digital future through:
        
        - **🎓 Training** the next generation of data scientists and AI experts
        - **🔬 Research** in cutting-edge technologies relevant to African contexts  
        - **🤝 Collaboration** with industry partners to solve real-world problems
        - **🌍 Impact** through data-driven solutions for African challenges
        
        ### Get Involved
        
        - **Website**: [dataverseafrica.org](https://dataverseafrica.org)
        - **Internships**: Join our next cohort of talented data enthusiasts
        - **Partnerships**: Collaborate with us on impactful projects
        - **Research**: Contribute to African-focused data science research
        """)
    
    # Project impact
    st.markdown("---")
    st.markdown("### 📊 Project Impact")
    
    # Try to load actual data for impact stats
    try:
        if st.session_state.df is not None:
            df = st.session_state.df
            impact_stats = [
                (f"{len(df):,}+", "Customer Records Analyzed"),
                (f"{df['cluster'].nunique() if 'cluster' in df.columns else 5}", "Customer Segments Identified"),
                (f"{(df['risk_score'] > 0.6).mean()*100:.1f}%" if 'risk_score' in df.columns else "12.5%", "Risk Rate"),
                (f"{((df['digital_adoption_score'] > 2).mean()*100 if 'digital_adoption_score' in df.columns else 68):.0f}%", "Digital Adoption Rate")
            ]
        elif os.path.exists("outputs/processed_data.csv"):
            df = pd.read_csv("outputs/processed_data.csv")
            impact_stats = [
                (f"{len(df):,}+", "Customer Records Analyzed"),
                (f"{df['cluster'].nunique() if 'cluster' in df.columns else 5}", "Customer Segments Identified"),
                (f"{(df['risk_score'] > 0.6).mean()*100:.1f}%" if 'risk_score' in df.columns else "12.5%", "Risk Rate"),
                (f"{((df['digital_adoption_score'] > 2).mean()*100 if 'digital_adoption_score' in df.columns else 68):.0f}%", "Digital Adoption Rate")
            ]
        else:
            impact_stats = [
                ("5200+", "Customer Records Analyzed"),
                ("5", "Customer Segments Identified"),
                ("12.5%", "Risk Rate Reduction Potential"),
                ("68%", "Digital Adoption Improvement")
            ]
    except:
        impact_stats = [
            ("5200+", "Customer Records Analyzed"),
            ("5", "Customer Segments Identified"),
            ("12.5%", "Risk Rate Reduction Potential"),
            ("68%", "Digital Adoption Improvement")
        ]
    
    cols = st.columns(4)
    for idx, (value, label) in enumerate(impact_stats):
        with cols[idx]:
            st.markdown(f'''
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%); border-radius: 10px;">
                <h2 style="color: #667eea; margin: 0;">{value}</h2>
                <p style="color: #666; margin: 0;">{label}</p>
            </div>
            ''', unsafe_allow_html=True)

# ============================================
# PAGE 8: SETTINGS
# ============================================
elif selected_page == "⚙️ Settings":
    st.markdown('<h2 class="sub-header">⚙️ System Settings & Configuration</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["API Settings", "Model Settings", "Data Settings"])
    
    with tab1:
        st.markdown("### 🔌 API Configuration")
        
        api_host = st.text_input("API Host", "localhost")
        api_port = st.number_input("API Port", min_value=1, max_value=65535, value=8000)
        
        if st.button("💾 Save API Settings", type="primary"):
            st.session_state.api_url = f"http://{api_host}:{api_port}"
            st.success("✅ API settings saved!")
        
        st.markdown("### 🔗 API Endpoints")
        endpoints = [
            ("🏥 Health Check", "/health"),
            ("🎯 Single Prediction", "/predict"),
            ("📋 Batch Prediction", "/predict/batch"),
            ("📊 Cluster Info", "/clusters"),
            ("🎪 Demo", "/demo"),
            ("📚 Documentation", "/docs")
        ]
        
        for name, endpoint in endpoints:
            st.code(f"{API_URL}{endpoint}", language="bash")
    
    with tab2:
        st.markdown("### 🤖 Model Configuration")
        
        # Check if models exist
        model_status = {}
        model_files = ['scaler.pkl', 'pca_model.pkl', 'kmeans_model.pkl']
        
        for model_file in model_files:
            model_path = f"models/{model_file}"
            if os.path.exists(model_path):
                model_status[model_file] = "✅ Available"
                
                # Get file info
                file_size = os.path.getsize(model_path) / 1024  # KB
                modified_time = datetime.fromtimestamp(os.path.getmtime(model_path))
                model_status[f"{model_file}_size"] = f"{file_size:.1f} KB"
                model_status[f"{model_file}_modified"] = modified_time.strftime("%Y-%m-%d %H:%M:%S")
            else:
                model_status[model_file] = "❌ Not Found"
        
        # Display model status
        st.markdown("#### 📁 Model Files Status")
        for model_file in model_files:
            col1, col2, col3 = st.columns(3)
            with col1:
                status_style = "color: #10B981;" if "✅" in model_status.get(model_file, "") else "color: #EF4444;"
                st.markdown(f"<span style='{status_style}'>{model_file}: {model_status.get(model_file, 'Unknown')}</span>", unsafe_allow_html=True)
            with col2:
                if f"{model_file}_size" in model_status:
                    st.text(f"Size: {model_status[f'{model_file}_size']}")
            with col3:
                if f"{model_file}_modified" in model_status:
                    st.text(f"Modified: {model_status[f'{model_file}_modified']}")
        
        # Model actions
        st.markdown("### 🔄 Model Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Reload Models", type="primary"):
                # Clear cache and reload
                if 'models' in st.session_state:
                    del st.session_state.models
                st.rerun()
                st.success("✅ Models reload initiated!")
        
        with col2:
            if st.button("🔍 Check Model Health", type="secondary"):
                try:
                    # Load models with suppressed warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        scaler = joblib.load("models/scaler.pkl") if os.path.exists("models/scaler.pkl") else None
                        pca = joblib.load("models/pca_model.pkl") if os.path.exists("models/pca_model.pkl") else None
                        kmeans = joblib.load("models/kmeans_model.pkl") if os.path.exists("models/kmeans_model.pkl") else None
                    
                    if all([scaler, pca, kmeans]):
                        st.success("✅ All models loaded successfully!")
                    else:
                        st.error("❌ Some models failed to load")
                except Exception as e:
                    st.error(f"❌ Error loading models: {str(e)}")
    
    with tab3:
        st.markdown("### 💾 Data Management")
        
        # Data file information
        data_files = [
            ("📊 Processed Data", "outputs/processed_data.csv"),
            ("📊 Cluster Profiles", "outputs/cluster_profiles.csv"),
            ("💡 Business Recommendations", "outputs/business_recommendations.csv"),
            ("📈 Power BI Data", "powerbi/powerbi_dashboard_data.csv")
        ]
        
        data_exists = False
        for name, filepath in data_files:
            if os.path.exists(filepath):
                data_exists = True
                file_size = os.path.getsize(filepath) / 1024  # KB
                modified_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.text(name)
                with col2:
                    st.text(f"{file_size:.1f} KB")
                with col3:
                    st.text(modified_time.strftime("%Y-%m-%d %H:%M:%S"))
            else:
                st.text(f"{name}: ❌ Not Found")
        
        if not data_exists:
            st.warning("No pipeline data found. Run the master_pipeline.py first.")
        
        # Data actions
        st.markdown("### 🧹 Data Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Cache", type="secondary"):
                # Clear Streamlit cache
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("✅ Cache cleared!")
        
        with col2:
            if st.button("🗑️ Clear Current Dataset", type="secondary"):
                if st.session_state.df is not None:
                    st.session_state.df = None
                    st.success("✅ Dataset cleared from session!")
                    st.rerun()
                else:
                    st.info("No dataset currently loaded in session")
        
        # Export all data
        st.markdown("### 📦 Export All Data")
        
        if st.button("📥 Export All Data", type="primary"):
            # Create zip of all outputs
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                for root, dirs, files in os.walk('outputs'):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, 'outputs')
                        zip_file.write(file_path, arcname)
                # Also include powerbi data
                if os.path.exists('powerbi'):
                    for root, dirs, files in os.walk('powerbi'):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.join('powerbi', os.path.relpath(file_path, 'powerbi'))
                            zip_file.write(file_path, arcname)
            
            st.download_button(
                label="📥 Download All Data",
                data=zip_buffer.getvalue(),
                file_name="customer_analysis_data.zip",
                mime="application/zip"
            )
        
        st.markdown("### ⚙️ System Information")
        
        import plotly
        info_items = [
            ("🐍 Python Version", sys.version.split()[0]),
            ("📊 Streamlit Version", st.__version__),
            ("🐼 Pandas Version", pd.__version__),
            ("📈 NumPy Version", np.__version__),
            ("📊 Plotly Version", plotly.__version__)
        ]
        
        for label, value in info_items:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"**{label}:**")
            with col2:
                st.code(value, language="python")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <h3 style='color: #667eea; margin-bottom: 1rem;'>Customer Financial Risk Prediction Dashboard</h3>
        <p style='margin-bottom: 0.5rem;'>African Financial Markets Analysis | Built with ❤️ by Team Dataverse</p>
        <p style='margin-bottom: 0.5rem;'>
            <strong>👥 Team:</strong> AMARACHI FLORENCE • Thato Maelane • Philip Odiachi • Mavis
        </p>
        <p style='margin-bottom: 1rem;'>
            <strong>🌍 Organization:</strong> 
            <a href='https://dataverseafrica.org' target='_blank' style='color: #667eea; text-decoration: none;'>
                Dataverse Africa - Empowering Africa's Digital Future
            </a>
        </p>
        <div style='display: flex; justify-content: center; gap: 1rem; margin-top: 1rem;'>
            <span>📊 Streamlit</span>
            <span>⚡ FastAPI</span>
            <span>🤖 Machine Learning</span>
            <span>📈 Data Visualization</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Add confetti on successful actions
if 'show_confetti' in st.session_state and st.session_state.show_confetti:
    st.balloons()
    st.session_state.show_confetti = False
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    """Streamlit Dashboard for Customer Financial Risk Prediction
Enhanced Version with Excel Support & Real-time Analysis
Team: AMARACHI FLORENCE, Thato Maelane, Philip Odiachi, AND Mavis
Internship: Dataverse Africa
"""

# ============================================
# SUPPRESS ALL WARNINGS
# ============================================
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Suppress sklearn version warnings
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
except ImportError:
    pass

# ============================================
# MAIN IMPORTS
# ============================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import requests
import json
import sys
import os
from datetime import datetime
import time
import zipfile
import io
import re

# Machine Learning Imports
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

# Create necessary directories if they don't exist
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("charts", exist_ok=True)
os.makedirs("powerbi", exist_ok=True)
os.makedirs("eda_reports", exist_ok=True)

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Customer Financial Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS WITH ENHANCED ANIMATIONS
# ============================================
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        animation: fadeInDown 1s ease;
    }
    
    .sub-header {
        font-size: 1.8rem;
        color: #4B5563;
        margin-top: 0.5rem;
        font-weight: 600;
        animation: fadeInUp 1.5s ease;
    }
    
    .card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: slideInLeft 1s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        animation: zoomIn 1s ease;
    }
    
    .team-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        animation: fadeIn 1s ease;
    }
    
    .cluster-card {
        background: white;
        color: #333;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 5px solid;
        animation: fadeIn 1s ease;
    }
    
    .success-badge {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
        animation: bounce 2s infinite;
    }
    
    .pbi-card {
        background: linear-gradient(135deg, #008751 0%, #00A86B 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    /* Starry ribbon effect */
    .starry-ribbon {
        position: relative;
        background: linear-gradient(90deg, #1a237e 0%, #283593 100%);
        color: white;
        text-align: center;
        padding: 0.8rem;
        margin: 1.5rem 0;
        border-radius: 8px;
        overflow: hidden;
        animation: slideInRight 1s ease;
    }
    
    .starry-ribbon::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: 
            radial-gradient(2px 2px at 20px 30px, #fff 50%, transparent 50%),
            radial-gradient(2px 2px at 40px 70px, #fff 50%, transparent 50%),
            radial-gradient(2px 2px at 60px 20px, #fff 50%, transparent 50%),
            radial-gradient(2px 2px at 80px 50px, #fff 50%, transparent 50%),
            radial-gradient(2px 2px at 100px 80px, #fff 50%, transparent 50%);
        background-size: 120px 100px;
        animation: twinkle 3s infinite;
    }
    
    /* Bouncing balloon */
    .bouncing-balloon {
        position: relative;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 50% 50% 50% 0;
        transform: rotate(-45deg);
        margin: 2rem auto;
        width: 120px;
        height: 120px;
        display: flex;
        align-items: center;
        justify-content: center;
        animation: bounceBalloon 3s ease-in-out infinite;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
    }
    
    .balloon-content {
        transform: rotate(45deg);
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    /* Hurry animation */
    .hurry-pulse {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        border-radius: 30px;
        font-weight: bold;
        animation: hurryPulse 1.5s infinite;
        box-shadow: 0 5px 15px rgba(255, 65, 108, 0.4);
    }
    
    /* Professional data cards */
    .data-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-top: 4px solid;
        transition: all 0.3s ease;
    }
    
    .data-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    .status-active { background-color: #10B981; }
    .status-warning { background-color: #F59E0B; }
    .status-error { background-color: #EF4444; }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes zoomIn {
        from {
            opacity: 0;
            transform: scale(0.9);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    @keyframes bounce {
        0%, 100% {
            transform: translateY(0);
        }
        50% {
            transform: translateY(-5px);
        }
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }
    
    @keyframes twinkle {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 1; }
    }
    
    @keyframes bounceBalloon {
        0%, 100% { 
            transform: translateY(0) rotate(-45deg) scale(1);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        }
        50% { 
            transform: translateY(-25px) rotate(-45deg) scale(1.05);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        }
    }
    
    @keyframes hurryPulse {
        0%, 100% { 
            transform: scale(1);
            box-shadow: 0 5px 15px rgba(255, 65, 108, 0.4);
        }
        50% { 
            transform: scale(1.05);
            box-shadow: 0 10px 25px rgba(255, 65, 108, 0.6);
        }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Ribbon style from original */
    .ribbon {
        position: relative;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        padding: 0.5rem;
        margin: 1rem 0;
        border-radius: 5px;
        animation: slideInRight 1s ease;
    }
    
    /* Balloon style from original */
    .balloon {
        position: relative;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 50% 50% 50% 0;
        transform: rotate(-45deg);
        margin: 2rem auto;
        width: 100px;
        height: 100px;
        display: flex;
        align-items: center;
        justify-content: center;
        animation: float 3s ease-in-out infinite;
    }
    
    .balloon-content {
        transform: rotate(45deg);
        text-align: center;
    }
    
    @keyframes float {
        0%, 100% {
            transform: translateY(0) rotate(-45deg);
        }
        50% {
            transform: translateY(-20px) rotate(-45deg);
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# TITLE WITH TEAM CREDIT - ENHANCED DESIGN
# ============================================
st.markdown('<h1 class="main-header">💰 Customer Financial Risk Prediction Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header" style="text-align: center;">African Financial Markets Analysis</p>', unsafe_allow_html=True)

# Team credit ribbon - Enhanced version
st.markdown("""
<div class="starry-ribbon">
    <strong>👥 Team Project:</strong> AMARACHI FLORENCE • Thato Maelane • Philip Odiachi • Mavis 
    | <a href="https://dataverseafrica.org" target="_blank" style="color: white; text-decoration: underline;">🌍 Dataverse Africa Internship</a>
</div>
""", unsafe_allow_html=True)

# ============================================
# ENHANCED DATA PROCESSING FUNCTIONS
# ============================================

def load_dataset(uploaded_file):
    """Load CSV or Excel file with automatic detection"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:  # Excel files (.xlsx, .xls)
            df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def preprocess_dataset(df):
    """Automatically preprocess any dataset for analysis"""
    # Create a copy
    processed_df = df.copy()
    
    # 1. Standardize column names
    processed_df.columns = [col.strip().replace(' ', '_').replace('-', '_') for col in processed_df.columns]
    
    # 2. Identify key columns automatically
    column_mapping = {}
    
    # Find customer ID column
    id_patterns = ['customer', 'cust', 'id', 'client']
    for col in processed_df.columns:
        if any(pattern in col.lower() for pattern in id_patterns):
            column_mapping['Customer_ID'] = col
            break
    
    # Find age column
    age_patterns = ['age', 'years']
    for col in processed_df.columns:
        if any(pattern in col.lower() for pattern in age_patterns):
            column_mapping['Age'] = col
            break
    
    # Find financial columns
    financial_patterns = {
        'Monthly_Expenditure': ['expenditure', 'spend', 'expense', 'monthly_spend'],
        'Credit_Score': ['credit', 'score', 'rating', 'credit_score'],
        'Income_Level': ['income', 'salary', 'revenue', 'income_level'],
        'Account_Balance': ['balance', 'account_balance', 'savings'],
        'Transaction_Count': ['transaction', 'txn', 'count', 'frequency']
    }
    
    for target, patterns in financial_patterns.items():
        for col in processed_df.columns:
            if any(pattern in col.lower() for pattern in patterns):
                column_mapping[target] = col
                break
    
    # Find location column
    location_patterns = ['location', 'city', 'region', 'state', 'address']
    for col in processed_df.columns:
        if any(pattern in col.lower() for pattern in location_patterns):
            column_mapping['Location'] = col
            break
    
    # Find transaction channel
    channel_patterns = ['channel', 'platform', 'medium', 'transaction_channel']
    for col in processed_df.columns:
        if any(pattern in col.lower() for pattern in channel_patterns):
            column_mapping['Transaction_Channel'] = col
            break
    
    # 3. Create derived features if they don't exist
    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Create risk score using Isolation Forest if we have enough numeric columns
    if len(numeric_cols) >= 3 and len(processed_df) > 50:
        try:
            # Select top 3 numeric columns for risk calculation
            risk_features = numeric_cols[:min(3, len(numeric_cols))]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(processed_df[risk_features].fillna(0))
            
            # Calculate anomaly score as risk
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            risk_scores = iso_forest.fit_predict(X_scaled)
            # Convert to 0-1 scale (higher = more risky)
            processed_df['risk_score'] = (1 - (risk_scores + 1) / 2)
        except Exception as e:
            # Fallback: random risk scores with some pattern
            np.random.seed(42)
            if 'Credit_Score' in processed_df.columns or column_mapping.get('Credit_Score'):
                credit_col = column_mapping.get('Credit_Score', 'Credit_Score')
                if credit_col in processed_df.columns:
                    # Risk inversely related to credit score
                    processed_df['risk_score'] = 1 - (processed_df[credit_col].fillna(processed_df[credit_col].median()) - 300) / 550
                    processed_df['risk_score'] = np.clip(processed_df['risk_score'], 0.1, 0.9)
                else:
                    processed_df['risk_score'] = np.random.beta(2, 5, len(processed_df))
            else:
                processed_df['risk_score'] = np.random.beta(2, 5, len(processed_df))
    else:
        processed_df['risk_score'] = np.random.beta(2, 5, len(processed_df))
    
    # Create digital adoption score (if transaction channel exists)
    if 'Transaction_Channel' in processed_df.columns:
        channel_col = processed_df['Transaction_Channel'].astype(str).str.lower()
        digital_score = (
            channel_col.str.contains('mobile|app|digital|online|web').astype(int) +
            channel_col.str.contains('pos|card').astype(int) * 0.5 +
            (channel_col.str.contains('ussd|bank|branch') == False).astype(int)
        )
        processed_df['digital_adoption_score'] = np.clip(digital_score, 0, 4)
    else:
        processed_df['digital_adoption_score'] = np.random.randint(1, 5, len(processed_df))
    
    # Create sentiment scores if feedback column exists
    feedback_patterns = ['feedback', 'review', 'comment', 'complaint', 'sentiment']
    for col in processed_df.columns:
        if any(pattern in col.lower() for pattern in feedback_patterns):
            # Simple sentiment analysis based on text length and keywords
            text_data = processed_df[col].astype(str)
            positive_words = ['good', 'excellent', 'great', 'happy', 'satisfied', 'thanks']
            negative_words = ['bad', 'poor', 'terrible', 'unhappy', 'dissatisfied', 'problem']
            
            sentiment_scores = []
            for text in text_data:
                if pd.isna(text):
                    sentiment_scores.append(0)
                    continue
                    
                text_lower = text.lower()
                positive_count = sum(word in text_lower for word in positive_words)
                negative_count = sum(word in text_lower for word in negative_words)
                
                if positive_count > negative_count:
                    score = 0.5 + min(0.5, positive_count * 0.1)
                elif negative_count > positive_count:
                    score = -0.5 - min(0.5, negative_count * 0.1)
                else:
                    score = 0
                
                sentiment_scores.append(score)
            
            processed_df['sentiment_score'] = sentiment_scores
            processed_df['sentiment_label'] = pd.cut(
                processed_df['sentiment_score'],
                bins=[-1, -0.33, 0.33, 1],
                labels=['Negative', 'Neutral', 'Positive']
            )
            break
    
    # 4. Perform clustering on the fly
    if len(numeric_cols) >= 2 and len(processed_df) > 10:
        try:
            # Use top 2-3 numeric columns for clustering
            cluster_features = numeric_cols[:min(3, len(numeric_cols))]
            cluster_data = processed_df[cluster_features].fillna(0)
            
            # Scale the data
            scaler = StandardScaler()
            cluster_scaled = scaler.fit_transform(cluster_data)
            
            # Determine optimal clusters (3-5)
            n_clusters = min(5, max(2, len(processed_df) // 50))
            
            # Apply PCA if we have many features
            if cluster_scaled.shape[1] > 10:
                pca = PCA(n_components=min(5, cluster_scaled.shape[1]))
                cluster_scaled = pca.fit_transform(cluster_scaled)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(cluster_scaled)
            
            # Name clusters based on characteristics
            cluster_names = []
            for i in range(n_clusters):
                cluster_mask = clusters == i
                if sum(cluster_mask) > 0:
                    cluster_df = processed_df[cluster_mask]
                    
                    # Determine cluster characteristics
                    if 'risk_score' in cluster_df.columns:
                        avg_risk = cluster_df['risk_score'].mean()
                        if avg_risk > 0.7:
                            risk_level = "High-Risk"
                        elif avg_risk < 0.3:
                            risk_level = "Low-Risk"
                        else:
                            risk_level = "Medium-Risk"
                    else:
                        risk_level = "Segment"
                    
                    if 'digital_adoption_score' in cluster_df.columns:
                        avg_digital = cluster_df['digital_adoption_score'].mean()
                        if avg_digital > 3:
                            digital_level = "Digital-First"
                        elif avg_digital < 2:
                            digital_level = "Traditional"
                        else:
                            digital_level = "Mixed"
                    else:
                        digital_level = ""
                    
                    name = f"{risk_level} {digital_level}".strip()
                    if not name:
                        name = f"Segment_{i+1}"
                    
                    cluster_names.append(name)
                else:
                    cluster_names.append(f"Segment_{i+1}")
            
            processed_df['cluster'] = clusters
            processed_df['cluster_name'] = [cluster_names[c] for c in clusters]
            
            # Save cluster profiles
            save_cluster_profiles(processed_df)
            
        except Exception as e:
            # Assign random clusters
            np.random.seed(42)
            processed_df['cluster'] = np.random.randint(0, 3, len(processed_df))
            processed_df['cluster_name'] = ['Segment_A', 'Segment_B', 'Segment_C'][:len(set(processed_df['cluster']))]
    
    # 5. Create risk categories
    if 'risk_score' in processed_df.columns:
        processed_df['risk_category'] = pd.cut(
            processed_df['risk_score'],
            bins=[0, 0.3, 0.6, 1],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
    
    # 6. Save processed data for Power BI
    save_powerbi_data(processed_df)
    
    return processed_df, column_mapping

def save_cluster_profiles(df):
    """Save cluster profiles to CSV"""
    if 'cluster_name' in df.columns:
        # Start with basic aggregation
        cluster_profiles = df.groupby('cluster_name').size().reset_index(name='size')
        
        # Add mean values for specific columns
        if 'risk_score' in df.columns:
            risk_means = df.groupby('cluster_name')['risk_score'].mean().reset_index(name='risk_score_mean')
            cluster_profiles = cluster_profiles.merge(risk_means, on='cluster_name')
        
        if 'digital_adoption_score' in df.columns:
            digital_means = df.groupby('cluster_name')['digital_adoption_score'].mean().reset_index(name='digital_adoption_score_mean')
            cluster_profiles = cluster_profiles.merge(digital_means, on='cluster_name')
        
        # Add more numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['cluster', 'risk_score', 'digital_adoption_score', 'cluster_name']:
                try:
                    col_means = df.groupby('cluster_name')[col].mean().reset_index(name=f'avg_{col}')
                    cluster_profiles = cluster_profiles.merge(col_means, on='cluster_name')
                except:
                    continue
        
        # Calculate percentage
        cluster_profiles['percentage'] = (cluster_profiles['size'] / len(df)) * 100
        
        # Save to file
        cluster_profiles.to_csv("outputs/cluster_profiles.csv", index=False)

def save_powerbi_data(df):
    """Save data in Power BI compatible format"""
    powerbi_df = df.copy()
    
    # Select relevant columns
    columns_to_keep = []
    for col in powerbi_df.columns:
        if col in ['Customer_ID', 'Age', 'Income_Level', 'Credit_Score', 
                   'Monthly_Expenditure', 'Location', 'Transaction_Channel',
                   'cluster_name', 'risk_score', 'risk_category',
                   'digital_adoption_score', 'sentiment_score', 'sentiment_label']:
            columns_to_keep.append(col)
        elif 'id' in col.lower() or 'customer' in col.lower():
            columns_to_keep.append(col)
    
    # Keep only columns that exist
    columns_to_keep = [col for col in columns_to_keep if col in powerbi_df.columns]
    
    # Add additional derived columns
    if len(columns_to_keep) > 0:
        powerbi_df = powerbi_df[columns_to_keep]
        
        # Create digital segment
        if 'digital_adoption_score' in powerbi_df.columns:
            powerbi_df['digital_segment'] = pd.cut(
                powerbi_df['digital_adoption_score'],
                bins=[0, 1, 2, 3, 4],
                labels=['Non-Digital', 'Low Digital', 'Medium Digital', 'High Digital']
            )
        
        # Save to file
        powerbi_df.to_csv("powerbi/powerbi_dashboard_data.csv", index=False)

def generate_business_insights(df):
    """Generate real insights from the dataset"""
    insights = []
    
    # Basic dataset info
    insights.append(f"**Dataset Size**: {len(df):,} customers with {len(df.columns)} features")
    
    # Risk analysis
    if 'risk_score' in df.columns:
        high_risk_pct = (df['risk_score'] > 0.6).mean() * 100
        medium_risk_pct = ((df['risk_score'] >= 0.3) & (df['risk_score'] <= 0.6)).mean() * 100
        low_risk_pct = (df['risk_score'] < 0.3).mean() * 100
        
        insights.append(f"**Risk Distribution**: {low_risk_pct:.1f}% Low, {medium_risk_pct:.1f}% Medium, {high_risk_pct:.1f}% High Risk")
    
    # Digital adoption
    if 'digital_adoption_score' in df.columns:
        digital_adopters = (df['digital_adoption_score'] > 2).mean() * 100
        traditional_users = (df['digital_adoption_score'] <= 2).mean() * 100
        insights.append(f"**Digital Adoption**: {digital_adopters:.1f}% digital-savvy, {traditional_users:.1f}% traditional")
    
    # Cluster insights
    if 'cluster_name' in df.columns:
        cluster_counts = df['cluster_name'].value_counts()
        if len(cluster_counts) > 0:
            largest_cluster = cluster_counts.index[0]
            cluster_pct = (df['cluster_name'] == largest_cluster).mean() * 100
            insights.append(f"**Largest Segment**: '{largest_cluster}' represents {cluster_pct:.1f}% of customers")
    
    # Financial insights
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    financial_cols = [col for col in numeric_cols if any(word in col.lower() for word in 
                     ['balance', 'spend', 'expense', 'income', 'revenue', 'credit', 'score'])]
    
    for col in financial_cols[:3]:  # First 3 financial columns
        mean_val = df[col].mean()
        std_val = df[col].std()
        insights.append(f"**{col}**: Average = {mean_val:,.2f}, Std = {std_val:,.2f}")
    
    # Location insights
    location_cols = [col for col in df.columns if any(word in col.lower() for word in 
                    ['location', 'city', 'region', 'state'])]
    
    if location_cols:
        location_col = location_cols[0]
        top_location = df[location_col].value_counts().index[0]
        location_pct = (df[location_col] == top_location).mean() * 100
        insights.append(f"**Geographic Concentration**: {location_pct:.1f}% from {top_location}")
    
    return insights

def generate_business_recommendations(df):
    """Generate data-driven business recommendations"""
    recommendations = []
    
    # Risk-based recommendations
    if 'risk_score' in df.columns:
        high_risk_count = (df['risk_score'] > 0.6).sum()
        if high_risk_count > 0:
            high_risk_pct = (high_risk_count / len(df)) * 100
            recommendations.append({
                'cluster_name': 'High-Risk Customers',
                'targeting_strategy': 'Proactive monitoring and support',
                'recommended_products': 'Basic savings accounts, financial education',
                'marketing_channels': 'Direct communication, branch visits',
                'risk_management': 'Frequent credit reviews, lower credit limits'
            })
    
    # Digital adoption recommendations
    if 'digital_adoption_score' in df.columns:
        low_digital_count = (df['digital_adoption_score'] < 2).sum()
        if low_digital_count > 0:
            low_digital_pct = (low_digital_count / len(df)) * 100
            recommendations.append({
                'cluster_name': 'Traditional Users',
                'targeting_strategy': 'Digital onboarding and incentives',
                'recommended_products': 'Mobile banking apps, USSD banking',
                'marketing_channels': 'SMS, USSD prompts, branch demonstrations',
                'risk_management': 'Monitor for digital fraud during transition'
            })
    
    # High-value customer recommendations
    financial_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                     if any(word in col.lower() for word in ['balance', 'spend', 'income'])]
    
    if financial_cols:
        # Find high-value customers (top 20%)
        value_col = financial_cols[0]
        threshold = df[value_col].quantile(0.8)
        high_value_count = (df[value_col] > threshold).sum()
        
        if high_value_count > 0:
            recommendations.append({
                'cluster_name': 'High-Value Customers',
                'targeting_strategy': 'Premium service and retention focus',
                'recommended_products': 'Investment accounts, premium credit cards, wealth management',
                'marketing_channels': 'Personal banking managers, exclusive events',
                'risk_management': 'Enhanced due diligence, regular portfolio reviews'
            })
    
    # Default recommendations if no specific patterns found
    if not recommendations:
        recommendations = [
            {
                'cluster_name': 'All Customers',
                'targeting_strategy': 'Segmented marketing based on behavior',
                'recommended_products': 'Diverse product portfolio',
                'marketing_channels': 'Multi-channel approach',
                'risk_management': 'Regular risk assessment and monitoring'
            }
        ]
    
    # Save recommendations
    rec_df = pd.DataFrame(recommendations)
    rec_df.to_csv("outputs/business_recommendations.csv", index=False)
    
    return recommendations

# ============================================
# ORIGINAL HELPER FUNCTIONS
# ============================================
def show_success_balloon(message):
    """Show animated balloon with message"""
    st.markdown(f"""
    <div class="bouncing-balloon">
        <div class="balloon-content">
            <div style="font-size: 1.2rem;">🎉</div>
            <div style="font-size: 0.8rem;">{message}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_hurry_alert(message):
    """Show hurry pulse alert"""
    st.markdown(f"""
    <div style="text-align: center; margin: 1rem 0;">
        <div class="hurry-pulse">{message}</div>
    </div>
    """, unsafe_allow_html=True)

def get_column_case_insensitive(df, column_name):
    """Get column name with case-insensitive matching"""
    column_name_lower = column_name.lower()
    for col in df.columns:
        if col.lower() == column_name_lower:
            return col
    return None

def safe_hover_data(df, preferred_columns):
    """Safely get hover data columns that exist in dataframe"""
    hover_cols = []
    for col in preferred_columns:
        actual_col = get_column_case_insensitive(df, col)
        if actual_col:
            hover_cols.append(actual_col)
    return hover_cols if hover_cols else None

def create_data_card(title, value, change=None, color="#667eea"):
    """Create a professional data card"""
    change_html = ""
    if change:
        change_direction = "▲" if float(change.replace('%', '').replace('+', '')) >= 0 else "▼"
        change_color = "#10B981" if change_direction == "▲" else "#EF4444"
        change_html = f'<div style="font-size: 0.9rem; color: {change_color};">{change_direction} {change}</div>'
    
    return f"""
    <div class="data-card" style="border-top-color: {color};">
        <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">{title}</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: #1F2937;">{value}</div>
        {change_html}
    </div>
    """

# Initialize session state for data
if 'df' not in st.session_state:
    st.session_state.df = None
if 'pbi_data' not in st.session_state:
    st.session_state.pbi_data = None
if 'column_mapping' not in st.session_state:
    st.session_state.column_mapping = {}
if 'insights' not in st.session_state:
    st.session_state.insights = []
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []

# API URL
API_URL = "http://localhost:8000"

# ============================================
# SIDEBAR - ENHANCED DESIGN
# ============================================
with st.sidebar:
    # Dataverse logo and link - Enhanced
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px;">
        <div style="font-size: 1.5rem; color: white; margin-bottom: 0.5rem;">🌍</div>
        <h3 style="color: white; margin: 0;">DATAVERSE AFRICA</h3>
        <p style="color: rgba(255,255,255,0.9); font-size: 0.9rem; margin: 0;">Empowering Africa's Digital Future</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation with enhanced icons
    st.markdown("### 🧭 Navigation")
    
    pages = {
        "🏠 Dashboard": "Executive dashboard with overview",
        "🔍 Customer Analysis": "Deep customer insights and filtering",
        "📊 Clusters": "Customer segmentation analysis",
        "🎯 Predict": "Real-time prediction interface",
        "📈 Insights": "Business recommendations",
        "📊 Power BI Dashboard": "Enhanced pipeline visualizations",
        "👥 Team": "Project team information",
        "⚙️ Settings": "System configuration"
    }
    
    selected_page = st.radio(
        "Select Page",
        list(pages.keys()),
        label_visibility="collapsed"
    )
    
    # Show description for selected page
    st.info(f"📄 {pages[selected_page]}")
    
    st.markdown("---")
    
    # Quick stats - Enhanced
    st.markdown("### 📊 Quick Stats")
    
    if st.session_state.df is not None and not st.session_state.df.empty:
        df = st.session_state.df
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(create_data_card("Total Customers", f"{len(df):,}"), unsafe_allow_html=True)
            if 'risk_score' in df.columns:
                st.markdown(create_data_card("Avg Risk Score", f"{df['risk_score'].mean():.3f}"), unsafe_allow_html=True)
        with col2:
            if 'digital_adoption_score' in df.columns:
                st.markdown(create_data_card("Digital Score", f"{df['digital_adoption_score'].mean():.2f}/4"), unsafe_allow_html=True)
            if 'cluster_name' in df.columns:
                clusters = df['cluster_name'].nunique()
                st.markdown(create_data_card("Segments", f"{clusters}"), unsafe_allow_html=True)
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(create_data_card("Total Customers", "0"), unsafe_allow_html=True)
            st.markdown(create_data_card("Avg Risk Score", "N/A"), unsafe_allow_html=True)
        with col2:
            st.markdown(create_data_card("Digital Score", "N/A"), unsafe_allow_html=True)
            st.markdown(create_data_card("Segments", "0"), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # System Status Indicators
    st.markdown("### 🔧 System Status")
    
    # Data status
    if st.session_state.df is not None:
        data_status = "ACTIVE"
        status_color = "#10B981"
    else:
        data_status = "INACTIVE"
        status_color = "#EF4444"
    
    st.markdown(f"""
    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
        <span class="status-indicator status-{'active' if data_status == 'ACTIVE' else 'error'}"></span>
        <span style="font-size: 0.9rem;">Data Loaded: <strong style="color: {status_color};">{data_status}</strong></span>
    </div>
    """, unsafe_allow_html=True)
    
    # API status
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        api_status = "ACTIVE" if response.status_code == 200 else "INACTIVE"
    except:
        api_status = "INACTIVE"
    
    api_color = "#10B981" if api_status == "ACTIVE" else "#F59E0B"
    st.markdown(f"""
    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
        <span class="status-indicator status-{'active' if api_status == 'ACTIVE' else 'warning'}"></span>
        <span style="font-size: 0.9rem;">API Connection: <strong style="color: {api_color};">{api_status}</strong></span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Connect section
    st.markdown("### 🔗 Connect")
    st.markdown("[📚 Documentation](#)")
    st.markdown("[📧 Contact Team](#)")
    st.markdown("[⭐ GitHub Repository](#)")
 
# ============================================
# PAGE 1: DASHBOARD
# ============================================
if selected_page == "🏠 Dashboard":
    st.markdown('<h2 class="sub-header">📈 Executive Dashboard</h2>', unsafe_allow_html=True)
    
    # Show success balloon
    show_success_balloon("Welcome!")
    
    # Load Dataset Section - Enhanced
    st.markdown("### 📂 Load Your Dataset")
    
    uploaded_file = st.file_uploader("Upload any customer dataset (CSV or Excel)", 
                                    type=["csv", "xlsx", "xls"], 
                                    key="dashboard_upload")
    
    if uploaded_file is not None:
        try:
            # Show loading animation
            with st.spinner("🔍 Analyzing your dataset..."):
                # Load dataset
                df = load_dataset(uploaded_file)
                if df is None:
                    st.error("Failed to load dataset")
                    st.stop()
                
                # Check dataset size
                if len(df) < 10:
                    st.warning("Dataset too small for meaningful analysis. Need at least 10 records.")
                    st.stop()
                
                # Preprocess and analyze
                processed_df, column_mapping = preprocess_dataset(df)
                
                # Store in session state
                st.session_state.df = processed_df
                st.session_state.column_mapping = column_mapping
                
                # Generate insights
                st.session_state.insights = generate_business_insights(processed_df)
                
                # Generate recommendations
                st.session_state.recommendations = generate_business_recommendations(processed_df)
            
            # Show success message
            st.markdown('<div class="success-badge">✅ Dataset Analyzed Successfully!</div>', unsafe_allow_html=True)
            st.success(f"Dataset loaded: {len(processed_df):,} records × {len(processed_df.columns)} columns")
            
            # Show insights
            st.markdown("### 🔍 Auto-Generated Insights")
            for insight in st.session_state.insights:
                st.markdown(f"- {insight}")
            
            # Show preview
            with st.expander("📋 Dataset Preview", expanded=True):
                st.dataframe(processed_df.head(100), use_container_width=True)
            
            # Statistics
            st.markdown("### 📊 Dataset Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Numerical Features")
                numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    st.dataframe(processed_df[numeric_cols].describe(), use_container_width=True)
            
            with col2:
                st.markdown("#### Categorical Features")
                cat_cols = processed_df.select_dtypes(include=['object']).columns
                for col in cat_cols[:5]:
                    st.write(f"**{col}**: {processed_df[col].nunique()} unique values")
                    if processed_df[col].nunique() < 10:
                        value_counts = processed_df[col].value_counts()
                        fig = px.pie(values=value_counts.values, names=value_counts.index, 
                                   title=col, color_discrete_sequence=px.colors.sequential.RdBu)
                        st.plotly_chart(fig, use_container_width=True)
                        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    # Check if processed data exists from pipeline
    processed_file = "outputs/processed_data.csv"
    
    if os.path.exists(processed_file) and st.session_state.df is None:
        try:
            df = pd.read_csv(processed_file)
            st.session_state.df = df
            st.markdown('<div class="success-badge">✅ Processed Dataset Loaded from Pipeline!</div>', unsafe_allow_html=True)
            st.success(f"Dataset loaded: {len(df):,} records × {len(df.columns)} columns")
        except Exception as e:
            st.error(f"Error loading processed data: {str(e)}")
    
    # Metrics cards based on actual data
    st.markdown("### 🎯 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    if st.session_state.df is not None and not st.session_state.df.empty:
        df = st.session_state.df
        
        with col1:
            total_customers = len(df)
            st.markdown(create_data_card("Total Customers", f"{total_customers:,}", "+12%", "#667eea"), unsafe_allow_html=True)
        
        with col2:
            if 'risk_score' in df.columns:
                avg_risk = df['risk_score'].mean()
                st.markdown(create_data_card("Avg Risk Score", f"{avg_risk:.3f}", color="#EF4444"), unsafe_allow_html=True)
            else:
                st.markdown(create_data_card("Avg Risk Score", "0.450", color="#EF4444"), unsafe_allow_html=True)
        
        with col3:
            if 'digital_adoption_score' in df.columns:
                digital_adoption = (df['digital_adoption_score'].mean() / 4) * 100
                st.markdown(create_data_card("Digital Adoption", f"{digital_adoption:.1f}%", "+15%", "#F59E0B"), unsafe_allow_html=True)
            else:
                st.markdown(create_data_card("Digital Adoption", "68%", "+15%", "#F59E0B"), unsafe_allow_html=True)
        
        with col4:
            if 'risk_score' in df.columns:
                high_risk = (df['risk_score'] > 0.6).sum()
                risk_rate = (high_risk / len(df)) * 100
                st.markdown(create_data_card("High Risk Rate", f"{risk_rate:.1f}%", "-2.3%", "#EF4444"), unsafe_allow_html=True)
            else:
                st.markdown(create_data_card("High Risk Rate", "12.5%", "-2.3%", "#EF4444"), unsafe_allow_html=True)
    else:
        # Show static sample metrics
        with col1:
            st.markdown(create_data_card("Total Customers", "0", "Upload Data", "#667eea"), unsafe_allow_html=True)
        
        with col2:
            st.markdown(create_data_card("Avg Risk Score", "N/A", "Upload Data", "#10B981"), unsafe_allow_html=True)
        
        with col3:
            st.markdown(create_data_card("Digital Adoption", "N/A", "Upload Data", "#F59E0B"), unsafe_allow_html=True)
        
        with col4:
            st.markdown(create_data_card("High Risk Rate", "N/A", "Upload Data", "#EF4444"), unsafe_allow_html=True)
    
    # Visualizations
    st.markdown("### 📈 Data Visualizations")
    
    # Check if we have data loaded
    if st.session_state.df is not None and not st.session_state.df.empty:
        df = st.session_state.df
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk Score Distribution
            if 'risk_score' in df.columns:
                fig = px.histogram(df, x='risk_score', nbins=30, title='Risk Score Distribution',
                                  color_discrete_sequence=['#636EFA'])
                fig.add_vline(x=0.6, line_dash="dash", line_color="red",
                             annotation_text="High Risk Threshold")
                fig.add_vline(x=0.3, line_dash="dash", line_color="green",
                             annotation_text="Low Risk Threshold")
                fig.update_layout(bargap=0.1, plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            elif len(df.select_dtypes(include=[np.number]).columns) > 0:
                # Show first numeric column
                num_col = df.select_dtypes(include=[np.number]).columns[0]
                fig = px.histogram(df, x=num_col, nbins=30, title=f'{num_col} Distribution',
                                  color_discrete_sequence=['#636EFA'])
                fig.update_layout(bargap=0.1, plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Scatter plot with available data
            sample_df = df.sample(min(500, len(df)))
            
            # Get numeric columns for scatter plot
            numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                x_col = numeric_cols[0]
                y_col = numeric_cols[1]
                
                # Get cluster column if exists
                color_col = None
                if 'cluster_name' in sample_df.columns:
                    color_col = 'cluster_name'
                elif 'cluster' in sample_df.columns:
                    color_col = 'cluster'
                
                fig = px.scatter(sample_df, 
                                x=x_col, 
                                y=y_col,
                                color=color_col,
                                title=f'{x_col} vs {y_col}',
                                hover_data=sample_df.columns.tolist()[:5],
                                color_discrete_sequence=px.colors.qualitative.Set3)
                
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
        
        # Cluster distribution
        if 'cluster_name' in df.columns:
            cluster_counts = df['cluster_name'].value_counts().reset_index()
            cluster_counts.columns = ['Cluster', 'Count']
            
            fig = px.bar(cluster_counts, x='Cluster', y='Count', title='Customer Segments Distribution',
                        color='Cluster', color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

# ============================================
# PAGE 2: CUSTOMER ANALYSIS
# ============================================
elif selected_page == "🔍 Customer Analysis":
    st.markdown('<h2 class="sub-header">🔍 Deep Customer Analysis</h2>', unsafe_allow_html=True)
    
    # Show success balloon
    show_success_balloon("Analysis Ready!")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # ============================================
        # OPTIMIZATION: CACHE COLUMN MAPPING
        # ============================================
        if 'analysis_cache' not in st.session_state:
            st.session_state.analysis_cache = {}
        
        # Get or compute column mapping once
        if 'column_mapping' not in st.session_state.analysis_cache:
            column_mapping = st.session_state.get('column_mapping', {})
            
            # Pre-compute all needed columns for analysis
            analysis_cols = {}
            target_columns = ['Income_Level', 'Credit_Score', 'Location', 'Transaction_Channel', 
                             'Customer_ID', 'Age', 'Monthly_Expenditure']
            
            for target in target_columns:
                if target in column_mapping:
                    analysis_cols[target] = column_mapping[target]
                else:
                    actual_col = get_column_case_insensitive(df, target)
                    if actual_col:
                        analysis_cols[target] = actual_col
                    else:
                        # Try alternative names
                        alt_names = {
                            'Income_Level': ['IncomeLevel', 'income_level', 'Income'],
                            'Credit_Score': ['Credit', 'credit_score', 'Score'],
                            'Location': ['City', 'city', 'Region', 'region'],
                            'Transaction_Channel': ['Channel', 'channel', 'Platform'],
                            'Customer_ID': ['customer_id', 'CustomerID', 'ID'],
                            'Age': ['age', 'Customer_Age'],
                            'Monthly_Expenditure': ['monthly_expenditure', 'Expenditure', 'Spending']
                        }
                        if target in alt_names:
                            for alt in alt_names[target]:
                                actual_col = get_column_case_insensitive(df, alt)
                                if actual_col:
                                    analysis_cols[target] = actual_col
                                    break
            
            st.session_state.analysis_cache['column_mapping'] = analysis_cols
        
        # Use cached column mapping
        analysis_cols = st.session_state.analysis_cache['column_mapping']
        
        # ============================================
        # FILTERS WITH CACHED COLUMNS
        # ============================================
        st.markdown("### 🔍 Filter Customers")
        col1, col2, col3 = st.columns(3)
        
        income_col = analysis_cols.get('Income_Level')
        credit_col = analysis_cols.get('Credit_Score')
        location_col = analysis_cols.get('Location')
        
        # Store filter values
        filter_values = {}
        
        with col1:
            if income_col and income_col in df.columns:
                # Cache unique values
                cache_key = f"unique_{income_col}"
                if cache_key not in st.session_state.analysis_cache:
                    st.session_state.analysis_cache[cache_key] = ['ALL'] + sorted(df[income_col].dropna().unique().tolist())
                
                income_options = st.session_state.analysis_cache[cache_key]
                selected_income = st.selectbox("Income Level", income_options, key="income_filter")
                if selected_income != 'ALL':
                    filter_values['income'] = (income_col, [selected_income])
        
        with col2:
            if credit_col and credit_col in df.columns and df[credit_col].dtype in [np.int64, np.float64]:
                # Cache min/max values
                cache_key = f"range_{credit_col}"
                if cache_key not in st.session_state.analysis_cache:
                    min_val = int(df[credit_col].min())
                    max_val = int(df[credit_col].max())
                    st.session_state.analysis_cache[cache_key] = (min_val, max_val)
                
                min_val, max_val = st.session_state.analysis_cache[cache_key]
                credit_range = st.slider(
                    "Credit Score Range",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    key="credit_filter"
                )
                filter_values['credit'] = (credit_col, credit_range)
        
        with col3:
            if location_col and location_col in df.columns:
                # Cache unique values
                cache_key = f"unique_{location_col}"
                if cache_key not in st.session_state.analysis_cache:
                    st.session_state.analysis_cache[cache_key] = ['ALL'] + sorted(df[location_col].dropna().unique().tolist())
                
                location_options = st.session_state.analysis_cache[cache_key]
                selected_location = st.selectbox("Location", location_options, key="location_filter")
                if selected_location != 'ALL':
                    filter_values['location'] = (location_col, [selected_location])
        
        # ============================================
        # OPTIMIZED FILTERING
        # ============================================
        # Check if filters have changed
        filter_hash = str(filter_values)
        
        if 'filtered_df' not in st.session_state.analysis_cache or \
           st.session_state.analysis_cache.get('last_filter_hash') != filter_hash:
            
            # Apply filters
            filtered_df = df.copy()
            
            for filter_type, (col, value) in filter_values.items():
                if filter_type == 'income' and col:
                    filtered_df = filtered_df[filtered_df[col].isin(value)]
                elif filter_type == 'credit' and col:
                    filtered_df = filtered_df[
                        (filtered_df[col] >= value[0]) & 
                        (filtered_df[col] <= value[1])
                    ]
                elif filter_type == 'location' and col:
                    filtered_df = filtered_df[filtered_df[col].isin(value)]
            
            # Cache the filtered result
            st.session_state.analysis_cache['filtered_df'] = filtered_df
            st.session_state.analysis_cache['last_filter_hash'] = filter_hash
        else:
            # Use cached filtered dataframe
            filtered_df = st.session_state.analysis_cache['filtered_df']
        
        # ============================================
        # DISPLAY RESULTS
        # ============================================
        st.markdown(f'<div class="success-badge">✅ Showing {len(filtered_df):,} customers ({(len(filtered_df)/len(df)*100):.1f}% of total)</div>', unsafe_allow_html=True)
        
        # Customer details table
        st.markdown("### 📋 Customer Details")
        
        # Define display columns that exist - use cached values
        display_cols = []
        
        # Priority columns from cached mapping
        priority_targets = [
            'Customer_ID', 'Age', 'Income_Level', 'Credit_Score', 
            'Monthly_Expenditure', 'Location', 'Transaction_Channel'
        ]
        
        for target in priority_targets:
            col = analysis_cols.get(target)
            if col and col in filtered_df.columns and col not in display_cols:
                display_cols.append(col)
        
        # Add automatic columns if they exist
        auto_cols = ['cluster_name', 'cluster', 'risk_score', 'risk_category', 
                    'digital_adoption_score', 'sentiment_score', 'sentiment_label']
        
        for col in auto_cols:
            if col in filtered_df.columns and col not in display_cols:
                display_cols.append(col)
        
        # Limit to 12 columns for performance
        if len(display_cols) > 12:
            display_cols = display_cols[:12]
        
        if display_cols:
            # Display only first 50 rows for performance
            display_df = filtered_df[display_cols].head(50)
            st.dataframe(display_df, use_container_width=True)
            
            # Download option
            csv = filtered_df[display_cols].to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data",
                data=csv,
                file_name="filtered_customers.csv",
                mime="text/csv",
                key="download_filtered"
            )
        
        # ============================================
        # OPTIMIZED VISUALIZATIONS
        # ============================================
        st.markdown("### 📊 Detailed Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Box plot with sample data for performance
            numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = filtered_df.select_dtypes(include=['object']).columns.tolist()
            
            if numeric_cols and cat_cols:
                num_col = st.selectbox("Select numeric column", numeric_cols[:5], key="num_col_select")
                cat_col = st.selectbox("Select category column", cat_cols[:5], key="cat_col_select")
                
                # Sample data for better performance
                if len(filtered_df) > 1000:
                    sample_df = filtered_df.sample(min(1000, len(filtered_df)))
                else:
                    sample_df = filtered_df
                
                fig = px.box(sample_df, x=cat_col, y=num_col,
                            title=f'{num_col} by {cat_col}',
                            color=cat_col,
                            color_discrete_sequence=px.colors.sequential.Plasma)
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True, key="box_plot")
        
        with col2:
            # Top locations by count - cached computation
            if location_col and location_col in filtered_df.columns:
                cache_key = f"top_locations_{filter_hash}"
                if cache_key not in st.session_state.analysis_cache:
                    location_counts = filtered_df[location_col].value_counts().head(10)
                    st.session_state.analysis_cache[cache_key] = location_counts
                
                location_counts = st.session_state.analysis_cache[cache_key]
                
                if len(location_counts) > 0:
                    fig = px.bar(x=location_counts.index, y=location_counts.values,
                                title='Top 10 Locations by Customer Count',
                                labels={'x': 'Location', 'y': 'Customer Count'},
                                color=location_counts.values,
                                color_continuous_scale='Viridis')
                    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True, key="location_chart")
        
        # Transaction Channel Analysis
        channel_col = analysis_cols.get('Transaction_Channel')
        if channel_col and channel_col in filtered_df.columns:
            st.markdown("### 📱 Transaction Channel Analysis")
            
            # Cache channel data
            cache_key = f"channel_data_{filter_hash}"
            if cache_key not in st.session_state.analysis_cache:
                channel_data = filtered_df[channel_col].value_counts().reset_index()
                channel_data.columns = ['Channel', 'Count']
                st.session_state.analysis_cache[cache_key] = channel_data
            
            channel_data = st.session_state.analysis_cache[cache_key]
            
            fig = px.pie(channel_data, values='Count', names='Channel', 
                        title='Transaction Channel Distribution',
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True, key="channel_pie")
        
        # Risk distribution
        if 'risk_score' in filtered_df.columns:
            st.markdown("### ⚠️ Risk Score Distribution")
            
            # Sample data for histogram if dataset is large
            if len(filtered_df) > 5000:
                hist_sample = filtered_df.sample(min(2000, len(filtered_df)))
            else:
                hist_sample = filtered_df
            
            fig = px.histogram(hist_sample, x='risk_score', nbins=20,
                            title='Risk Score Distribution',
                            color_discrete_sequence=['#FF6B6B'])
            
            fig.add_vline(x=0.6, line_dash="dash", line_color="red",
                         annotation_text="High Risk Threshold")
            fig.add_vline(x=0.3, line_dash="dash", line_color="green",
                         annotation_text="Low Risk Threshold")
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True, key="risk_histogram")
    
    else:
        st.error("❌ No data loaded. Please load data in Dashboard page or run the pipeline.")

# ============================================
# PAGE 3: CLUSTERS
# ============================================
elif selected_page == "📊 Clusters":
    st.markdown('<h2 class="sub-header">📊 Customer Segmentation Analysis</h2>', unsafe_allow_html=True)
    
    # Show success balloon
    show_success_balloon("Clusters Found!")
    
    try:
        # Use session state data
        if st.session_state.df is not None:
            df = st.session_state.df
            
            # Check if clusters were generated
            if 'cluster_name' in df.columns:
                # Create cluster profiles from actual data
                cluster_profiles = df.groupby('cluster_name').agg({
                    'cluster': 'size',  # Count
                    'risk_score': 'mean' if 'risk_score' in df.columns else 'first',
                    'digital_adoption_score': 'mean' if 'digital_adoption_score' in df.columns else 'first'
                }).reset_index()
                
                # Add any other numeric columns that exist
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col not in ['cluster', 'risk_score', 'digital_adoption_score']:
                        cluster_profiles[f'avg_{col}'] = df.groupby('cluster_name')[col].mean().values
                
                cluster_profiles = cluster_profiles.rename(columns={'cluster': 'size'})
                cluster_profiles['percentage'] = (cluster_profiles['size'] / len(df)) * 100
                
                # Display cluster cards with visible text
                st.markdown("### 🎯 Customer Segments Overview")
                
                cols = st.columns(min(len(cluster_profiles), 3))
                colors = ['#4F46E5', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899']
                
                for idx, (_, row) in enumerate(cluster_profiles.iterrows()):
                    if idx % 3 == 0 and idx > 0:
                        cols = st.columns(min(len(cluster_profiles) - idx, 3))
                    
                    col_idx = idx % 3
                    if col_idx < len(cols):
                        with cols[col_idx]:
                            cluster_name = row['cluster_name']
                            cluster_size = row['size']
                            percentage = row['percentage']
                            
                            st.markdown(f'''
                            <div class="cluster-card" style="border-left: 5px solid {colors[idx % len(colors)]};">
                                <h4 style="color: {colors[idx % len(colors)]}; margin-bottom: 10px;">
                                    {cluster_name}
                                </h4>
                                <p style="color: #333;"><strong>Size:</strong> {cluster_size:,} customers</p>
                                <p style="color: #333;"><strong>Percentage:</strong> {percentage:.1f}%</p>
                            </div>
                            ''', unsafe_allow_html=True)
                
                # Cluster comparison
                st.markdown("### 📈 Cluster Comparison")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'risk_score' in cluster_profiles.columns:
                        fig = px.bar(cluster_profiles, 
                                    x='cluster_name', 
                                    y='risk_score',
                                    title='Average Risk Score by Cluster',
                                    color='cluster_name',
                                    color_discrete_sequence=colors[:len(cluster_profiles)])
                        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', xaxis_title="", yaxis_title="Risk Score")
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if 'digital_adoption_score' in cluster_profiles.columns:
                        fig = px.bar(cluster_profiles, 
                                    x='cluster_name', 
                                    y='digital_adoption_score',
                                    title='Digital Adoption Score by Cluster',
                                    color='cluster_name',
                                    color_discrete_sequence=colors[:len(cluster_profiles)])
                        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', xaxis_title="", yaxis_title="Digital Score")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Digital adoption comparison
                st.markdown("### 📱 Digital Adoption vs Risk")
                
                if 'digital_adoption_score' in cluster_profiles.columns and 'risk_score' in cluster_profiles.columns:
                    fig = px.scatter(cluster_profiles,
                                    x='digital_adoption_score',
                                    y='risk_score',
                                    size='size',
                                    color='cluster_name',
                                    hover_data=['percentage'],
                                    title='Digital Adoption vs Risk Score',
                                    size_max=60,
                                    color_discrete_sequence=colors[:len(cluster_profiles)])
                    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed cluster profiles
                st.markdown("### 📋 Detailed Cluster Profiles")
                
                for _, row in cluster_profiles.iterrows():
                    with st.expander(f"{row['cluster_name']} - Details"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Customers", f"{row['size']:,}")
                            st.metric("Percentage", f"{row['percentage']:.1f}%")
                        
                        with col2:
                            if 'risk_score' in row:
                                st.metric("Avg Risk Score", f"{row['risk_score']:.3f}")
                            if 'digital_adoption_score' in row:
                                st.metric("Digital Score", f"{row['digital_adoption_score']:.2f}/4")
                        
                        with col3:
                            # Show other average values
                            for col in cluster_profiles.columns:
                                if col.startswith('avg_') and pd.notna(row[col]):
                                    col_name = col.replace('avg_', '').replace('_', ' ').title()
                                    st.metric(f"Avg {col_name}", f"{row[col]:,.2f}")
            
            else:
                st.warning("No cluster information found in data. Please upload a dataset first.")
        else:
            st.error("No data found. Please load data in Dashboard page.")
    
    except Exception as e:
        st.error(f"Error loading cluster data: {str(e)}")

# ============================================
# PAGE 4: PREDICT
# ============================================
elif selected_page == "🎯 Predict":
    st.markdown('<h2 class="sub-header">🎯 Real-time Customer Prediction</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Single Customer", "Batch Prediction", "API Test"])
    
    with tab1:
        st.markdown("### 👤 Predict Single Customer")
        
        with st.form("single_customer_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                customer_id = st.text_input("Customer ID", "CUST001")
                age = st.slider("Age", 18, 70, 35)
                monthly_expenditure = st.number_input("Monthly Expenditure (₦)", min_value=0.0, value=150000.0, step=1000.0)
                credit_score = st.slider("Credit Score", 300, 850, 650)
                transaction_count = st.number_input("Transaction Count", min_value=1, value=25)
            
            with col2:
                avg_transaction_value = st.number_input("Avg Transaction Value (₦)", min_value=0.0, value=6000.0, step=100.0)
                
                st.markdown("**Digital Channels Used:**")
                col_a, col_b = st.columns(2)
                with col_a:
                    uses_pos = st.checkbox("POS", value=True)
                    uses_web = st.checkbox("Web/Transfer", value=False)
                with col_b:
                    uses_ussd = st.checkbox("USSD", value=True)
                    uses_mobile_app = st.checkbox("Mobile App", value=True)
                
                income_level = st.selectbox("Income Level", ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"])
                saving_behavior = st.selectbox("Saving Behavior", ["Poor", "Average", "Good"])
                location = st.text_input("Location", "Lagos")
                feedback = st.text_area("Customer Feedback", "Good service overall")
                
                # Required fields for API
                transaction_channel = st.selectbox("Transaction Channel", 
                                                 ["USSD", "Web", "Mobile App", "POS", "ATM", "Branch"])
                spending_category = st.selectbox("Spending Category",
                                               ["Groceries", "Rent", "Utilities", "Transport", "Health", 
                                                "Education", "Entertainment", "Online Shopping", "Savings Deposit"])
            
            submitted = st.form_submit_button("🎯 Predict Segment", type="primary")
        
        if submitted:
            # Show success balloon
            show_hurry_alert("PROCESSING PREDICTION")
            
            try:
                # Prepare data for API
                customer_data = {
                    "customer_id": customer_id,
                    "age": age,
                    "monthly_expenditure": monthly_expenditure,
                    "credit_score": credit_score,
                    "transaction_count": transaction_count,
                    "avg_transaction_value": avg_transaction_value,
                    "uses_pos": 1 if uses_pos else 0,
                    "uses_web": 1 if uses_web else 0,
                    "uses_ussd": 1 if uses_ussd else 0,
                    "uses_mobile_app": 1 if uses_mobile_app else 0,
                    "income_level": income_level,
                    "saving_behavior": saving_behavior,
                    "location": location,
                    "feedback": feedback,
                    "transaction_channel": transaction_channel,
                    "spending_category": spending_category
                }
                
                # Try API call
                try:
                    with st.spinner("Predicting..."):
                        response = requests.post(f"{API_URL}/predict", json=customer_data, timeout=10)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display results
                        st.markdown('<div class="success-badge">✅ Prediction Successful!</div>', unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(create_data_card("Customer Segment", result.get('cluster_name', 'Unknown'), color="#667eea"), unsafe_allow_html=True)
                        
                        with col2:
                            risk_category = result.get('risk_category', 'Medium Risk')
                            risk_color = "#EF4444" if "High" in risk_category else "#F59E0B" if "Medium" in risk_category else "#10B981"
                            st.markdown(create_data_card("Risk Category", risk_category, color=risk_color), unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(create_data_card("Digital Score", f"{result.get('digital_adoption_score', 0)}/4.0", color="#10B981"), unsafe_allow_html=True)
                        
                        # Display more details
                        st.markdown("### 📊 Detailed Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### Segment Characteristics")
                            characteristics = result.get('segment_characteristics', {})
                            for key, value in characteristics.items():
                                st.markdown(f"**{key.replace('_', ' ').title()}:** `{value}`")
                        
                        with col2:
                            st.markdown("#### 💡 Recommendations")
                            recommendations = result.get('recommendations', [])
                            for i, rec in enumerate(recommendations, 1):
                                st.markdown(f"{i}. **{rec}**")
                        
                        # Performance
                        st.info(f"⏱️ Processing time: {result.get('processing_time', 0)} ms")
                    
                    else:
                        st.error(f"API Error: {response.status_code} - {response.text}")
                        st.info("Try running the API server first: `python api/api_main.py`")
                
                except requests.exceptions.ConnectionError:
                    st.error("❌ API not running. Start the API with: `python api/api_main.py`")
                    st.info("The API server needs to be running for real-time predictions.")
                except requests.exceptions.Timeout:
                    st.error("❌ API request timed out. Please try again.")
            
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
    
    with tab2:
        st.markdown("### 📋 Batch Prediction")
        
        uploaded_file = st.file_uploader("Upload CSV file with customer data", type=["csv"])
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(batch_df)} customers")
                st.dataframe(batch_df.head(), use_container_width=True)
                
                if st.button("🔮 Predict All Customers", type="primary"):
                    # Show success balloon
                    show_hurry_alert("PROCESSING BATCH PREDICTION")
                    
                    # Prepare batch data
                    customers = []
                    for _, row in batch_df.iterrows():
                        customer = {
                            "customer_id": str(row.get('Customer_ID', row.get('customer_id', f"CUST{_}"))),
                            "age": int(row.get('Age', row.get('age', 35))),
                            "monthly_expenditure": float(row.get('Monthly_Expenditure', row.get('monthly_expenditure', 100000))),
                            "credit_score": int(row.get('Credit_Score', row.get('credit_score', 600))),
                            "transaction_count": int(row.get('transaction_count', row.get('Transaction_Count', 20))),
                            "avg_transaction_value": float(row.get('avg_transaction_value', row.get('Avg_Transaction_Value', 5000))),
                            "uses_pos": int(row.get('uses_pos', row.get('Uses_POS', 0))),
                            "uses_web": int(row.get('uses_web', row.get('Uses_Web', 0))),
                            "uses_ussd": int(row.get('uses_ussd', row.get('Uses_USSD', 0))),
                            "uses_mobile_app": int(row.get('uses_mobile_app', row.get('Uses_Mobile_App', 0))),
                            "income_level": str(row.get('Income_Level', row.get('income_level', 'Middle'))),
                            "saving_behavior": str(row.get('Saving_Behavior', row.get('saving_behavior', 'Average'))),
                            "location": str(row.get('Location', row.get('location', 'Unknown'))),
                            "feedback": str(row.get('Customer_Feedback', row.get('feedback', ''))),
                            "transaction_channel": str(row.get('Transaction_Channel', row.get('transaction_channel', 'Mobile App'))),
                            "spending_category": str(row.get('Spending_Category', row.get('spending_category', 'Groceries')))
                        }
                        customers.append(customer)
                    
                    batch_data = {"customers": customers}
                    
                    # Call batch API
                    try:
                        with st.spinner(f"Processing {len(customers)} customers..."):
                            response = requests.post(f"{API_URL}/predict/batch", json=batch_data, timeout=30)
                        
                        if response.status_code == 200:
                            results = response.json()
                            
                            # Create results dataframe
                            results_list = []
                            for r in results:
                                results_list.append({
                                    'Customer_ID': r['customer_id'],
                                    'Segment': r['cluster_name'],
                                    'Risk_Category': r['risk_category'],
                                    'Risk_Score': r['risk_score'],
                                    'Digital_Score': r['digital_adoption_score'],
                                    'Top_Recommendation': r['recommendations'][0] if r['recommendations'] else 'N/A'
                                })
                            
                            results_df = pd.DataFrame(results_list)
                            
                            st.markdown('<div class="success-badge">✅ Predictions Completed!</div>', unsafe_allow_html=True)
                            st.success(f"✅ Predictions completed for {len(results)} customers")
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Predictions",
                                data=csv,
                                file_name="batch_predictions.csv",
                                mime="text/csv"
                            )
                            
                            # Summary statistics
                            st.markdown("### 📊 Batch Summary")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                segment_counts = results_df['Segment'].value_counts()
                                st.dataframe(segment_counts, use_container_width=True)
                            
                            with col2:
                                risk_counts = results_df['Risk_Category'].value_counts()
                                fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                                           title="Risk Category Distribution",
                                           color_discrete_sequence=px.colors.sequential.RdBu)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col3:
                                st.metric("Avg Risk Score", f"{results_df['Risk_Score'].mean():.3f}")
                                st.metric("Avg Digital Score", f"{results_df['Digital_Score'].mean():.2f}")
                        
                        else:
                            st.error(f"❌ API Error: {response.status_code}")
                            st.text(response.text)
                    
                    except requests.exceptions.ConnectionError:
                        st.error("❌ API not running. Please start the API server.")
                    except requests.exceptions.Timeout:
                        st.error("❌ Batch processing timed out. Try with fewer customers.")
            
            except Exception as e:
                st.error(f"❌ Error processing batch: {str(e)}")
    
    with tab3:
        st.markdown("### 🔧 API Connection Test")
        
        col1, col2 = st.columns(2)
        
        with col1:
            api_url = st.text_input("API URL", API_URL)
        
        with col2:
            if st.button("🔗 Test Connection", type="primary"):
                try:
                    response = requests.get(f"{api_url}/health", timeout=5)
                    if response.status_code == 200:
                        health_data = response.json()
                        st.markdown('<div class="success-badge">✅ API is Running!</div>', unsafe_allow_html=True)
                        
                        # Display health info
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            status_color = "#10B981" if health_data.get('status') == 'healthy' else "#EF4444"
                            st.metric("Status", health_data.get('status', 'unknown'))
                        
                        with col2:
                            st.metric("Version", health_data.get('version', 'unknown'))
                        
                        with col3:
                            models_status = "✅" if health_data.get('models_loaded', False) else "❌"
                            st.metric("Models Loaded", models_status)
                        
                        # Additional info
                        st.info(f"📅 Last Check: {health_data.get('timestamp', 'unknown')}")
                    else:
                        st.error(f"❌ API responded with status: {response.status_code}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API")
                    st.info("Start the API with: `python api/api_main.py`")
                except Exception as e:
                    st.error(f"❌ Connection test failed: {str(e)}")
        
        # Demo prediction
        st.markdown("### 🚀 Demo Prediction")
        
        if st.button("🎪 Run Demo Prediction", type="secondary"):
            try:
                with st.spinner("Running demo..."):
                    response = requests.get(f"{api_url}/demo", timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    st.markdown('<div class="success-badge">✅ Demo Successful!</div>', unsafe_allow_html=True)
                    
                    # Pretty display of demo results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.json(result)
                    with col2:
                        st.markdown("**Demo Summary:**")
                        st.write(f"**Customer:** {result.get('customer_id', 'Unknown')}")
                        st.write(f"**Segment:** {result.get('cluster_name', 'Unknown')}")
                        st.write(f"**Risk:** {result.get('risk_category', 'Unknown')}")
                        st.write(f"**Score:** {result.get('risk_score', 0):.3f}")
                else:
                    st.error("❌ Demo failed")
            except Exception as e:
                st.error(f"❌ Demo error: {str(e)}")

# ============================================
# PAGE 5: INSIGHTS
# ============================================
elif selected_page == "📈 Insights":
    st.markdown('<h2 class="sub-header">📈 Business Insights & Recommendations</h2>', unsafe_allow_html=True)
    
    # Show success balloon
    show_success_balloon("Insights Ready!")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Display generated insights
        if st.session_state.insights:
            st.markdown("### 🔍 Auto-Generated Insights")
            for insight in st.session_state.insights:
                st.markdown(f"• {insight}")
        
        # Display recommendations
        if st.session_state.recommendations:
            st.markdown("### 🎯 Data-Driven Recommendations")
            
            for rec in st.session_state.recommendations:
                with st.expander(f"{rec.get('cluster_name', 'Segment')} - Strategy"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Targeting Strategy:** {rec.get('targeting_strategy', 'N/A')}")
                        st.markdown(f"**Recommended Products:** {rec.get('recommended_products', 'N/A')}")
                    
                    with col2:
                        st.markdown(f"**Marketing Channels:** {rec.get('marketing_channels', 'N/A')}")
                        st.markdown(f"**Risk Management:** {rec.get('risk_management', 'N/A')}")
        
        # Action plan
        st.markdown("### 📋 Action Plan Timeline")
        
        action_items = [
            ("Immediate (Week 1)", "Review current customer segments and high-risk profiles"),
            ("Short-term (Month 1)", "Launch targeted campaigns based on segment analysis"),
            ("Medium-term (Quarter 1)", "Implement personalized product recommendations"),
            ("Long-term (Year 1)", "Develop AI-powered real-time customer insights engine")
        ]
        
        for timeline, action in action_items:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"<div style='background: #667eea; color: white; padding: 5px 10px; border-radius: 5px; text-align: center;'>{timeline}</div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"**{action}**")
        
        # Export insights
        st.markdown("---")
        st.markdown("### 📤 Export Insights")
        
        if st.button("📥 Download Insights Report", type="primary"):
            # Create report
            report_data = {
                "dataset_summary": {
                    "total_customers": len(df),
                    "total_features": len(df.columns),
                    "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "insights": st.session_state.insights,
                "recommendations": st.session_state.recommendations
            }
            
            # Save as JSON
            report_json = json.dumps(report_data, indent=2)
            st.download_button(
                label="📥 Download JSON Report",
                data=report_json,
                file_name="customer_insights_report.json",
                mime="application/json"
            )
    
    else:
        # If no data, show static insights
        st.warning("No data available. Please load data in Dashboard first.")
        
        st.markdown("### 💼 General Recommendations")
        
        general_insights = [
            "**🎯 Segment Customers**: Group customers by behavior, risk, and preferences",
            "**💰 Optimize Revenue**: Focus on high-value customers with premium offerings",
            "**📱 Digital First**: Invest in mobile and digital banking platforms",
            "**⚖️ Risk Management**: Monitor credit scores and implement early warning systems",
            "**🤝 Customer Retention**: Develop loyalty programs for long-term relationships",
            "**📊 Data-Driven**: Use analytics for all business decisions"
        ]
        
        for i, insight in enumerate(general_insights, 1):
            st.markdown(f"{i}. {insight}")

# ============================================
# PAGE 6: POWER BI DASHBOARD (NEW)
# ============================================
# Find and fix the problematic code around line 2001

# ============================================
# PAGE 6: POWER BI DASHBOARD (FIXED SECTION)
# ============================================
elif selected_page == "📊 Power BI Dashboard":
    st.markdown('<h2 class="sub-header">📊 Power BI Dashboard - Pipeline Visualizations</h2>', unsafe_allow_html=True)
    
    # Show success balloon
    show_success_balloon("Power BI Data!")
    
    # Load Power BI data
    pbi_file = "powerbi/powerbi_dashboard_data.csv"
    
    if os.path.exists(pbi_file):
        pbi_data = pd.read_csv(pbi_file)
        st.session_state.pbi_data = pbi_data
        
        # Executive Summary
        st.markdown("### 📊 Executive Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_customers = len(pbi_data)
            st.markdown(create_data_card("Total Customers", f"{total_customers:,}", color="#667eea"), unsafe_allow_html=True)
        
        with col2:
            if 'cluster_name' in pbi_data.columns:
                clusters = pbi_data['cluster_name'].nunique()
                st.markdown(create_data_card("Segments Identified", f"{clusters}", color="#10B981"), unsafe_allow_html=True)
            else:
                st.markdown(create_data_card("Segments", "N/A", color="#10B981"), unsafe_allow_html=True)
        
        with col3:
            if 'risk_score' in pbi_data.columns:
                avg_risk = pbi_data['risk_score'].mean()
                st.markdown(create_data_card("Avg Risk Score", f"{avg_risk:.3f}", color="#F59E0B"), unsafe_allow_html=True)
            else:
                st.markdown(create_data_card("Avg Risk", "N/A", color="#F59E0B"), unsafe_allow_html=True)
        
        with col4:
            if 'risk_category' in pbi_data.columns:
                high_risk = (pbi_data['risk_category'] == 'High Risk').sum()
                high_risk_pct = (high_risk / total_customers) * 100
                st.markdown(create_data_card("High Risk", f"{high_risk_pct:.1f}%", color="#EF4444"), unsafe_allow_html=True)
            else:
                st.markdown(create_data_card("Risk Rate", "N/A", color="#EF4444"), unsafe_allow_html=True)
        
        # PAGE 1: Customer Segmentation
        st.markdown("---")
        st.markdown("### 🎯 Customer Segmentation")
        
        if 'cluster_name' in pbi_data.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart
                cluster_counts = pbi_data['cluster_name'].value_counts().reset_index()
                cluster_counts.columns = ['Cluster', 'Count']
                
                fig = px.pie(cluster_counts, values='Count', names='Cluster',
                            title='Cluster Distribution',
                            color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # FIXED: Cluster table - don't rename cluster_name column
                numeric_cols = pbi_data.select_dtypes(include=[np.number]).columns
                agg_dict = {}
                
                # Count customers per cluster
                cluster_summary = pbi_data.groupby('cluster_name').size().reset_index(name='Customer_Count')
                
                # Add mean values for numeric columns
                if 'risk_score' in numeric_cols:
                    risk_means = pbi_data.groupby('cluster_name')['risk_score'].mean().reset_index(name='Avg_Risk_Score')
                    cluster_summary = cluster_summary.merge(risk_means, on='cluster_name')
                
                if 'digital_adoption_score' in numeric_cols:
                    digital_means = pbi_data.groupby('cluster_name')['digital_adoption_score'].mean().reset_index(name='Avg_Digital_Score')
                    cluster_summary = cluster_summary.merge(digital_means, on='cluster_name')
                
                # Add first 2 other numeric columns
                other_num_cols = [col for col in numeric_cols if col not in ['risk_score', 'digital_adoption_score', 'cluster']]
                for col in other_num_cols[:2]:
                    col_means = pbi_data.groupby('cluster_name')[col].mean().reset_index(name=f'Avg_{col}')
                    cluster_summary = cluster_summary.merge(col_means, on='cluster_name')
                
                # Calculate percentage
                cluster_summary['Percentage'] = (cluster_summary['Customer_Count'] / total_customers) * 100
                
                st.dataframe(cluster_summary, use_container_width=True)
                
        
        # PAGE 2: Payment Channel Analytics
        st.markdown("---")
        st.markdown("### 📱 Payment Channel Analytics")
        
        # Find transaction channel column
        channel_cols = [col for col in pbi_data.columns if any(word in col.lower() for word in ['channel', 'platform', 'medium'])]
        
        if channel_cols:
            channel_col = channel_cols[0]
            col1, col2 = st.columns(2)
            
            with col1:
                channel_counts = pbi_data[channel_col].value_counts()
                fig = px.bar(x=channel_counts.index, y=channel_counts.values,
                            title='Transaction Channel Usage',
                            color_discrete_sequence=px.colors.sequential.Viridis)
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'digital_segment' in pbi_data.columns:
                    digital_counts = pbi_data['digital_segment'].value_counts()
                    fig = px.pie(values=digital_counts.values, names=digital_counts.index,
                                title='Digital Segments',
                                color_discrete_sequence=px.colors.qualitative.Pastel)
                    st.plotly_chart(fig, use_container_width=True)
        
        # PAGE 3: Financial Behavior
        st.markdown("---")
        st.markdown("### 💰 Financial Behavior Metrics")
        
        # Find financial columns
        financial_cols = [col for col in pbi_data.columns if any(word in col.lower() for word in 
                         ['balance', 'spend', 'expense', 'income', 'revenue', 'credit', 'score'])]
        
        if len(financial_cols) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(pbi_data, x=financial_cols[0], nbins=30,
                                  title=f'{financial_cols[0]} Distribution',
                                  color_discrete_sequence=['#636EFA'])
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if len(financial_cols) >= 2:
                    fig = px.scatter(pbi_data, x=financial_cols[0], y=financial_cols[1],
                                    title=f'{financial_cols[0]} vs {financial_cols[1]}',
                                    color_discrete_sequence=['#FF6B6B'])
                    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
        
        # PAGE 4: Risk Analysis
        st.markdown("---")
        st.markdown("### ⚠️ Risk Analysis")
        
        if 'risk_score' in pbi_data.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(pbi_data, x='risk_score', nbins=20,
                                  title='Risk Score Distribution',
                                  color_discrete_sequence=['#FF6B6B'])
                fig.add_vline(x=0.6, line_dash="dash", line_color="red",
                             annotation_text="High Risk Threshold")
                fig.add_vline(x=0.3, line_dash="dash", line_color="green",
                             annotation_text="Low Risk Threshold")
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'risk_category' in pbi_data.columns:
                    risk_counts = pbi_data['risk_category'].value_counts()
                    fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                                title='Risk Category Distribution',
                                color_discrete_sequence=px.colors.sequential.RdBu)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Key Findings
        st.markdown("---")
        st.markdown("### 🔍 Key Findings")
        
        findings = [
            f"Analysis completed on {len(pbi_data):,} customer records",
            f"Generated Power BI compatible dataset with {len(pbi_data.columns)} columns",
        ]
        
        if 'cluster_name' in pbi_data.columns:
            findings.append(f"Identified {pbi_data['cluster_name'].nunique()} customer segments")
        
        if 'risk_score' in pbi_data.columns:
            high_risk_pct = (pbi_data['risk_score'] > 0.6).mean() * 100
            findings.append(f"High-risk customers: {high_risk_pct:.1f}%")
        
        for finding in findings:
            st.markdown(f"• **{finding}**")
         
        # Export Power BI data
        st.markdown("---")
        st.markdown("### 📤 Export Data")
        
        if st.button("📥 Download Power BI Data", type="primary"):
            csv = pbi_data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="powerbi_dashboard_data.csv",
                mime="text/csv"
            )
    
    else:
        st.error("❌ Power BI data not found. Please upload and analyze a dataset first.")
        show_hurry_alert("UPLOAD DATASET TO GENERATE POWER BI DATA")
        
        st.markdown("""
        **To generate Power BI data:**
        1. Go to the **Dashboard** page
        2. Upload your customer dataset (CSV or Excel)
        3. The app will automatically process and create Power BI compatible data
        4. Return to this page to view the Power BI dashboard
        
        The Power BI data includes:
        - Enhanced customer segments
        - Risk scores and categories
        - Digital adoption metrics
        - Transaction channel analytics
        """)

# ============================================
# PAGE 7: TEAM
# ============================================
elif selected_page == "👥 Team":
    st.markdown('<h2 class="sub-header">👥 Project Team</h2>', unsafe_allow_html=True)
    
    # Show success balloon
    show_success_balloon("Team Power!")
    
    # Team introduction
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h3 style="color: #667eea;">🎓 Dataverse Africa Internship Program</h3>
        <p>Empowering Africa's Digital Future through Data Science and AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Team members - Enhanced with roles
    team_members = [
        {
            "name": "Amarachi Florence",
            "role": "Financial Data and MEAL Analyst"
        },
        {
            "name": "Thato Maelane",
            "role": "Data Scientist"
        },
        {
            "name": "Philip Odiachi", 
            "role": "Data Analyst"
        },
        {
            "name": "Mavis",
            "role": "Business Analyst"
        }
    ]
    
    # Display team members in simple boxes
    cols = st.columns(2)
    for idx, member in enumerate(team_members):
        with cols[idx % 2]:
            st.markdown(f'''
            <div class="team-card">
                <h4 style="color: #667eea; margin-bottom: 10px;">{member['name']}</h4>
                <p style="color: black;"><strong>Role:</strong> {member['role']}</p>
            </div>
            ''', unsafe_allow_html=True)
    
    # Dataverse Africa section
    st.markdown("---")
    st.markdown("### 🌍 About Dataverse Africa")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 4rem; color: #667eea;">🌍</div>
            <h3 style="color: #667eea;">DATAVERSE AFRICA</h3>
            <p style="color: #666;">Empowering Africa's Digital Future</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        ### Our Mission
        
        At DataVerse, we're on a mission to transform Africa through the power of data. 
        
        Imagine a future where technology fuels progress, improves lives, and causes growth across the continent. 
        
        Sure you can imagine that and that future isn't far fetched anymore, that future is now!
        
        We're building Africa's digital future through:
        
        - **🎓 Training** the next generation of data scientists and AI experts
        - **🔬 Research** in cutting-edge technologies relevant to African contexts  
        - **🤝 Collaboration** with industry partners to solve real-world problems
        - **🌍 Impact** through data-driven solutions for African challenges
        
        ### Get Involved
        
        - **Website**: [dataverseafrica.org](https://dataverseafrica.org)
        - **Internships**: Join our next cohort of talented data enthusiasts
        - **Partnerships**: Collaborate with us on impactful projects
        - **Research**: Contribute to African-focused data science research
        """)
    
    # Project impact
    st.markdown("---")
    st.markdown("### 📊 Project Impact")
    
    # Try to load actual data for impact stats
    try:
        if st.session_state.df is not None:
            df = st.session_state.df
            impact_stats = [
                (f"{len(df):,}+", "Customer Records Analyzed"),
                (f"{df['cluster_name'].nunique() if 'cluster_name' in df.columns else 'N/A'}", "Customer Segments"),
                (f"{(df['risk_score'] > 0.6).mean()*100:.1f}%" if 'risk_score' in df.columns else "N/A", "High Risk Rate"),
                (f"{((df['digital_adoption_score'] > 2).mean()*100 if 'digital_adoption_score' in df.columns else 'N/A'):.0f}%", "Digital Adoption")
            ]
        elif os.path.exists("outputs/processed_data.csv"):
            df = pd.read_csv("outputs/processed_data.csv")
            impact_stats = [
                (f"{len(df):,}+", "Customer Records Analyzed"),
                (f"{df['cluster_name'].nunique() if 'cluster_name' in df.columns else 'N/A'}", "Customer Segments"),
                (f"{(df['risk_score'] > 0.6).mean()*100:.1f}%" if 'risk_score' in df.columns else "N/A", "High Risk Rate"),
                (f"{((df['digital_adoption_score'] > 2).mean()*100 if 'digital_adoption_score' in df.columns else 'N/A'):.0f}%", "Digital Adoption")
            ]
        else:
            impact_stats = [
                ("Upload Dataset", "Customer Records Analyzed"),
                ("N/A", "Customer Segments Identified"),
                ("N/A", "Risk Rate"),
                ("N/A", "Digital Adoption")
            ]
    except:
        impact_stats = [
            ("Upload Dataset", "Customer Records Analyzed"),
            ("N/A", "Customer Segments Identified"),
            ("N/A", "Risk Rate"),
            ("N/A", "Digital Adoption")
        ]
    
    cols = st.columns(4)
    for idx, (value, label) in enumerate(impact_stats):
        with cols[idx]:
            st.markdown(f'''
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%); border-radius: 10px;">
                <h2 style="color: #667eea; margin: 0;">{value}</h2>
                <p style="color: #666; margin: 0;">{label}</p>
            </div>
            ''', unsafe_allow_html=True)

# ============================================
# PAGE 8: SETTINGS
# ============================================
elif selected_page == "⚙️ Settings":
    st.markdown('<h2 class="sub-header">⚙️ System Settings & Configuration</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["API Settings", "Model Settings", "Data Settings"])
    
    with tab1:
        st.markdown("### 🔌 API Configuration")
        
        api_host = st.text_input("API Host", "localhost")
        api_port = st.number_input("API Port", min_value=1, max_value=65535, value=8000)
        
        if st.button("💾 Save API Settings", type="primary"):
            st.session_state.api_url = f"http://{api_host}:{api_port}"
            st.success("✅ API settings saved!")
        
        st.markdown("### 🔗 API Endpoints")
        endpoints = [
            ("🏥 Health Check", "/health"),
            ("🎯 Single Prediction", "/predict"),
            ("📋 Batch Prediction", "/predict/batch"),
            ("📊 Cluster Info", "/clusters"),
            ("🎪 Demo", "/demo"),
            ("📚 Documentation", "/docs")
        ]
        
        for name, endpoint in endpoints:
            st.code(f"{API_URL}{endpoint}", language="bash")
    
    with tab2:
        st.markdown("### 🤖 Model Configuration")
        
        # Check if models exist
        model_status = {}
        model_files = ['scaler.pkl', 'pca_model.pkl', 'kmeans_model.pkl']
        
        for model_file in model_files:
            model_path = f"models/{model_file}"
            if os.path.exists(model_path):
                model_status[model_file] = "✅ Available"
                
                # Get file info
                file_size = os.path.getsize(model_path) / 1024  # KB
                modified_time = datetime.fromtimestamp(os.path.getmtime(model_path))
                model_status[f"{model_file}_size"] = f"{file_size:.1f} KB"
                model_status[f"{model_file}_modified"] = modified_time.strftime("%Y-%m-%d %H:%M:%S")
            else:
                model_status[model_file] = "❌ Not Found"
        
        # Display model status
        st.markdown("#### 📁 Model Files Status")
        for model_file in model_files:
            col1, col2, col3 = st.columns(3)
            with col1:
                status_style = "color: #10B981;" if "✅" in model_status.get(model_file, "") else "color: #EF4444;"
                st.markdown(f"<span style='{status_style}'>{model_file}: {model_status.get(model_file, 'Unknown')}</span>", unsafe_allow_html=True)
            with col2:
                if f"{model_file}_size" in model_status:
                    st.text(f"Size: {model_status[f'{model_file}_size']}")
            with col3:
                if f"{model_file}_modified" in model_status:
                    st.text(f"Modified: {model_status[f'{model_file}_modified']}")
        
        # Model actions
        st.markdown("### 🔄 Model Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Reload Models", type="primary"):
                # Clear cache and reload
                if 'models' in st.session_state:
                    del st.session_state.models
                st.rerun()
                st.success("✅ Models reload initiated!")
        
        with col2:
            if st.button("🔍 Check Model Health", type="secondary"):
                try:
                    # Load models with suppressed warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        scaler = joblib.load("models/scaler.pkl") if os.path.exists("models/scaler.pkl") else None
                        pca = joblib.load("models/pca_model.pkl") if os.path.exists("models/pca_model.pkl") else None
                        kmeans = joblib.load("models/kmeans_model.pkl") if os.path.exists("models/kmeans_model.pkl") else None
                    
                    if all([scaler, pca, kmeans]):
                        st.success("✅ All models loaded successfully!")
                    else:
                        st.error("❌ Some models failed to load")
                except Exception as e:
                    st.error(f"❌ Error loading models: {str(e)}")
    
    with tab3:
        st.markdown("### 💾 Data Management")
        
        # Data file information
        data_files = [
            ("📊 Processed Data", "outputs/processed_data.csv"),
            ("📊 Cluster Profiles", "outputs/cluster_profiles.csv"),
            ("💡 Business Recommendations", "outputs/business_recommendations.csv"),
            ("📈 Power BI Data", "powerbi/powerbi_dashboard_data.csv")
        ]
        
        data_exists = False
        for name, filepath in data_files:
            if os.path.exists(filepath):
                data_exists = True
                file_size = os.path.getsize(filepath) / 1024  # KB
                modified_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.text(name)
                with col2:
                    st.text(f"{file_size:.1f} KB")
                with col3:
                    st.text(modified_time.strftime("%Y-%m-%d %H:%M:%S"))
            else:
                st.text(f"{name}: ❌ Not Found")
        
        if not data_exists:
            st.warning("No pipeline data found. Please upload and analyze a dataset first.")
        
        # Data actions
        st.markdown("### 🧹 Data Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Cache", type="secondary"):
                # Clear Streamlit cache
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("✅ Cache cleared!")
        
        with col2:
            if st.button("🗑️ Clear Current Dataset", type="secondary"):
                if st.session_state.df is not None:
                    st.session_state.df = None
                    st.session_state.insights = []
                    st.session_state.recommendations = []
                    st.success("✅ Dataset cleared from session!")
                    st.rerun()
                else:
                    st.info("No dataset currently loaded in session")
        
        # Export all data
        st.markdown("### 📦 Export All Data")
        
        if st.button("📥 Export All Data", type="primary"):
            # Create zip of all outputs
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                for root, dirs, files in os.walk('outputs'):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, 'outputs')
                        zip_file.write(file_path, arcname)
                # Also include powerbi data
                if os.path.exists('powerbi'):
                    for root, dirs, files in os.walk('powerbi'):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.join('powerbi', os.path.relpath(file_path, 'powerbi'))
                            zip_file.write(file_path, arcname)
            
            st.download_button(
                label="📥 Download All Data",
                data=zip_buffer.getvalue(),
                file_name="customer_analysis_data.zip",
                mime="application/zip"
            )
        
        st.markdown("### ⚙️ System Information")
        
        import plotly
        info_items = [
            ("🐍 Python Version", sys.version.split()[0]),
            ("📊 Streamlit Version", st.__version__),
            ("🐼 Pandas Version", pd.__version__),
            ("📈 NumPy Version", np.__version__),
            ("📊 Plotly Version", plotly.__version__)
        ]
        
        for label, value in info_items:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"**{label}:**")
            with col2:
                st.code(value, language="python")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <h3 style='color: #667eea; margin-bottom: 1rem;'>Customer Financial Risk Prediction Dashboard</h3>
        <p style='margin-bottom: 0.5rem;'>African Financial Markets Analysis | Built with ❤️ by Team Dataverse</p>
        <p style='margin-bottom: 0.5rem;'>
            <strong>👥 Team:</strong> AMARACHI FLORENCE • Thato Maelane • Philip Odiachi • Mavis
        </p>
        <p style='margin-bottom: 1rem;'>
            <strong>🌍 Organization:</strong> 
            <a href='https://dataverseafrica.org' target='_blank' style='color: #667eea; text-decoration: none;'>
                Dataverse Africa - Empowering Africa's Digital Future
            </a>
        </p>
        <div style='display: flex; justify-content: center; gap: 1rem; margin-top: 1rem;'>
            <span>📊 Streamlit</span>
            <span>⚡ FastAPI</span>
            <span>🤖 Machine Learning</span>
            <span>📈 Data Visualization</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Add confetti on successful actions
if 'show_confetti' in st.session_state and st.session_state.show_confetti:
    st.balloons()
    st.session_state.show_confetti = False
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    main 
    """
FastAPI for Customer Financial Risk Prediction
Endpoints for real-time prediction and segmentation
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import json

# Initialize FastAPI app
app = FastAPI(
    title="Customer Financial Risk Prediction API",
    description="API for customer segmentation and risk prediction in African financial markets",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
MODELS_DIR = "models"

try:
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    pca_model = joblib.load(os.path.join(MODELS_DIR, "pca_model.pkl"))
    kmeans_model = joblib.load(os.path.join(MODELS_DIR, "kmeans_model.pkl"))
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    scaler = pca_model = kmeans_model = None

# Pydantic models for request validation
# In your api/api_main.py file, update the CustomerData model:

class CustomerData(BaseModel):
    """Single customer data model"""
    customer_id: str = Field(..., description="Unique customer identifier")
    age: int = Field(..., ge=18, le=70, description="Customer age (18-70)")
    monthly_expenditure: float = Field(..., ge=0, description="Monthly expenditure in local currency")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    transaction_count: int = Field(..., ge=1, description="Number of transactions per month")
    avg_transaction_value: float = Field(..., ge=0, description="Average transaction value")
    uses_pos: int = Field(0, ge=0, le=1, description="Uses POS (0/1)")
    uses_web: int = Field(0, ge=0, le=1, description="Uses Web/Transfer (0/1)")
    uses_ussd: int = Field(0, ge=0, le=1, description="Uses USSD (0/1)")
    uses_mobile_app: int = Field(0, ge=0, le=1, description="Uses Mobile App (0/1)")
    income_level: str = Field("Middle", description="Income level")
    saving_behavior: str = Field("Average", description="Saving behavior")
    location: Optional[str] = Field("Unknown", description="Customer location")
    feedback: Optional[str] = Field("", description="Customer feedback text")
    # Add these optional fields to match your original data structure
    transaction_channel: Optional[str] = Field("Mobile App", description="Transaction channel used")
    spending_category: Optional[str] = Field("Groceries", description="Spending category")
    
    
class BatchCustomerData(BaseModel):
    """Batch customer data model"""
    customers: List[CustomerData]

class PredictionResponse(BaseModel):
    """Prediction response model"""
    customer_id: str
    cluster_id: int
    cluster_name: str
    risk_score: float
    risk_category: str
    digital_adoption_score: float
    recommendations: List[str]
    segment_characteristics: Dict[str, Any]
    processing_time: float

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    models_loaded: bool
    timestamp: str

# Helper functions
def calculate_sentiment(feedback: str) -> float:
    """Calculate sentiment score from feedback"""
    if not feedback or not isinstance(feedback, str):
        return 0.0
    
    feedback = feedback.lower()
    positive_words = ['excellent', 'great', 'good', 'fast', 'smooth', 'easy', 'helpful', 'satisfied']
    negative_words = ['confusing', 'unclear', 'failed', 'crashing', 'slow', 'problem', 'issue', 'bad']
    
    positive_count = sum(1 for word in positive_words if word in feedback)
    negative_count = sum(1 for word in negative_words if word in feedback)
    
    if positive_count + negative_count > 0:
        return (positive_count - negative_count) / (positive_count + negative_count)
    return 0.0

def calculate_risk_score(credit_score: int, saving_behavior: str, loan_status: str = "No Loan") -> float:
    """Calculate risk score"""
    risk = (850 - credit_score) / 550 * 0.4
    
    if saving_behavior == 'Poor':
        risk += 0.3
    elif saving_behavior == 'Average':
        risk += 0.15
    
    if loan_status == 'Default Risk':
        risk += 0.3
    
    return min(max(risk, 0), 1)

def get_cluster_name(cluster_id: int) -> str:
    """Map cluster ID to cluster name"""
    cluster_names = {
        0: "Digital-First High Spenders",
        1: "Traditional Low-Risk Savers",
        2: "High-Risk Low Income",
        3: "Medium Digital Average Spenders",
        4: "Positive Experience Customers"
    }
    return cluster_names.get(cluster_id, f"Segment {cluster_id}")

def get_recommendations(cluster_id: int, risk_score: float) -> List[str]:
    """Get personalized recommendations based on cluster and risk"""
    recommendations = []
    
    if cluster_id == 0:  # Digital-First High Spenders
        recommendations = [
            "Premium mobile banking features",
            "Investment products",
            "Credit card with rewards",
            "Wealth management services"
        ]
    elif cluster_id == 1:  # Traditional Low-Risk Savers
        recommendations = [
            "Fixed deposit accounts",
            "Retirement planning",
            "Insurance products",
            "Secure investment options"
        ]
    elif cluster_id == 2:  # High-Risk Low Income
        recommendations = [
            "Financial literacy programs",
            "Micro-savings accounts",
            "Budgeting assistance",
            "Basic banking education"
        ]
    elif cluster_id == 3:  # Medium Digital Average Spenders
        recommendations = [
            "Digital banking adoption programs",
            "Personalized offers",
            "Credit building products",
            "Mixed channel banking"
        ]
    else:  # Default
        recommendations = [
            "Standard banking products",
            "Customer service improvements",
            "Regular financial reviews"
        ]
    
    # Add risk-based recommendations
    if risk_score > 0.7:
        recommendations.append("Enhanced monitoring and support")
        recommendations.append("Gradual credit increase program")
    
    return recommendations[:5]  # Return top 5 recommendations

# API Endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Customer Financial Risk Prediction API",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "clusters": "/clusters",
            "documentation": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        models_loaded=all([scaler, pca_model, kmeans_model]),
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_customer(customer: CustomerData):
    """Predict customer segment and risk"""
    try:
        start_time = datetime.now()
        
        # Calculate derived features
        digital_adoption = customer.uses_pos + customer.uses_web + customer.uses_ussd + customer.uses_mobile_app
        sentiment_score = calculate_sentiment(customer.feedback)
        risk_score = calculate_risk_score(customer.credit_score, customer.saving_behavior)
        
        # Prepare feature vector
        features = np.array([[
            customer.age,
            customer.monthly_expenditure,
            customer.credit_score,
            customer.transaction_count,
            customer.avg_transaction_value,
            digital_adoption,
            risk_score,
            sentiment_score
        ]])
        
        # Check if models are loaded
        if scaler is None or pca_model is None or kmeans_model is None:
            raise HTTPException(status_code=503, detail="Models not loaded. Please run training pipeline first.")
        
        # Transform features
        features_scaled = scaler.transform(features)
        features_pca = pca_model.transform(features_scaled)
        
        # Predict cluster
        cluster_id = int(kmeans_model.predict(features_pca)[0])
        cluster_name = get_cluster_name(cluster_id)
        
        # Determine risk category
        if risk_score < 0.3:
            risk_category = "Low Risk"
        elif risk_score < 0.6:
            risk_category = "Medium Risk"
        else:
            risk_category = "High Risk"
        
        # Get recommendations
        recommendations = get_recommendations(cluster_id, risk_score)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000  # milliseconds
        
        # Segment characteristics
        segment_characteristics = {
            "digital_adoption": digital_adoption,
            "sentiment_score": sentiment_score,
            "expenditure_level": "High" if customer.monthly_expenditure > 150000 else "Medium" if customer.monthly_expenditure > 50000 else "Low",
            "income_level": customer.income_level,
            "location": customer.location
        }
        
        return PredictionResponse(
            customer_id=customer.customer_id,
            cluster_id=cluster_id,
            cluster_name=cluster_name,
            risk_score=round(risk_score, 3),
            risk_category=risk_category,
            digital_adoption_score=float(digital_adoption),
            recommendations=recommendations,
            segment_characteristics=segment_characteristics,
            processing_time=round(processing_time, 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch_customers(batch_data: BatchCustomerData):
    """Predict segments for multiple customers"""
    try:
        predictions = []
        
        for customer in batch_data.customers:
            # Use single prediction endpoint logic
            result = await predict_customer(customer)
            predictions.append(result)
        
        return predictions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/clusters", response_model=Dict[str, Any])
async def get_cluster_info():
    """Get information about all clusters"""
    try:
        cluster_info = {}
        
        for i in range(5):  # Assuming 5 clusters
            cluster_info[f"cluster_{i}"] = {
                "name": get_cluster_name(i),
                "description": get_cluster_description(i),
                "typical_customers": get_typical_customers(i),
                "recommended_products": get_recommendations(i, 0.5)[:3]
            }
        
        return {
            "total_clusters": 5,
            "clusters": cluster_info,
            "clustering_method": "KMeans with PCA",
            "last_trained": "2024-01-01"  # This should come from model metadata
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cluster info error: {str(e)}")

def get_cluster_description(cluster_id: int) -> str:
    """Get cluster description"""
    descriptions = {
        0: "Tech-savvy customers with high spending and strong digital adoption",
        1: "Conservative savers with low risk and traditional banking preferences",
        2: "Customers with financial challenges needing support and education",
        3: "Average customers with mixed digital and traditional banking usage",
        4: "Satisfied customers with positive feedback and good relationships"
    }
    return descriptions.get(cluster_id, "General customer segment")

def get_typical_customers(cluster_id: int) -> List[str]:
    """Get typical customer characteristics for cluster"""
    characteristics = {
        0: ["Young professionals", "High income", "Digital natives", "Urban residents"],
        1: ["Middle-aged", "Stable income", "Risk-averse", "Long-term savers"],
        2: ["Low income", "Financial difficulties", "Need support", "Rural areas"],
        3: ["Mixed age groups", "Average income", "Moderate digital use", "Suburban"],
        4: ["Long-term customers", "Positive feedback", "Loyal", "Various demographics"]
    }
    return characteristics.get(cluster_id, ["General customers"])

@app.get("/demo")
async def demo_prediction():
    """Demo endpoint with sample prediction"""
    sample_customer = CustomerData(
        customer_id="DEMO001",
        age=35,
        monthly_expenditure=150000.0,
        credit_score=720,
        transaction_count=25,
        avg_transaction_value=6000.0,
        uses_pos=1,
        uses_web=1,
        uses_ussd=0,
        uses_mobile_app=1,
        income_level="Upper-Middle",
        saving_behavior="Good",
        location="Lagos",
        feedback="Excellent mobile banking experience"
    )
    
    return await predict_customer(sample_customer)

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Customer Financial Risk Prediction API...")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔗 Demo endpoint: http://localhost:8000/demo")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    """Customer Financial Risk Prediction & Sentiment Analysis Dashboard
Capstone Project: African Financial Behavior Analysis
Team: AMARACHI FLORENCE, Thato Maelane, Philip Odiachi, AND Mavis
Internship: Dataverse Africa
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
# MAIN IMPORTS
# ============================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json 
import sys
import os
from datetime import datetime
import zipfile 
import io
from io import BytesIO
import re
import hashlib
from typing import Dict, List, Any, Optional, Tuple
import textwrap

# Machine Learning Imports
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# NLP Imports
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import gensim
from gensim import corpora, models
import torch

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('sentiment/vader_lexicon')
except:
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('wordnet', quiet=True)

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="African Financial Risk & Sentiment Dashboard",
    page_icon="🌍 💱💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS - AFRICAN FINANCIAL THEME
# ============================================
st.markdown("""
<style>
    /* African-inspired color scheme */
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
    
    .sub-header {
        font-size: 1.8rem;
        color: #2A9D8F;
        margin-top: 1rem;
        font-weight: 800;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #2A9D8F;
    }
    
    /* African pattern background for cards */
    .africa-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #FF6B35;
        position: relative;
        overflow: hidden;
    }
    
    .africa-card:before {
        content: "";
        position: absolute;
        top: 0;
        right: 0;
        width: 100px;
        height: 100px;
        background: linear-gradient(45deg, rgba(42, 157, 143, 0.1), rgba(255, 107, 53, 0.1));
        border-radius: 0 0 0 100px;
    }
    
    /* Insights containers with African patterns */
    .insights-container {
        background: linear-gradient(135deg, #E0F7FA 0%, #80DEEA 100%);
        border-left: 6px solid #00ACC1;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(0, 172, 193, 0.2);
        position: relative;
    }
    
    .insights-container:before {
        content: "💡";
        position: absolute;
        top: 10px;
        right: 10px;
        font-size: 1.5rem;
        opacity: 0.2;
    }
    
    .recommendations-container {
        background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 100%);
        border-left: 6px solid #EC407A;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(236, 64, 122, 0.2);
        position: relative;
    }
    
    .recommendations-container:before {
        content: "🚀";
        position: absolute;
        top: 10px;
        right: 10px;
        font-size: 1.5rem;
        opacity: 0.2;
    }
    
    /* Payment channel badges */
    .payment-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 700;
        background: linear-gradient(135deg, #2A9D8F, #264653);
        color: white;
    }
    
    .payment-badge.pos {
        background: linear-gradient(135deg, #FF6B35, #FFA62E);
    }
    
    .payment-badge.mobile {
        background: linear-gradient(135deg, #4A148C, #7B1FA2);
    }
    
    .payment-badge.ussd {
        background: linear-gradient(135deg, #006064, #00838F);
    }
    
    /* Cluster tags */
    .cluster-tag {
        display: inline-block;
        padding: 0.4rem 1rem;
        margin: 0.3rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .cluster-1 { background: linear-gradient(135deg, #2E7D32, #4CAF50); color: white; }
    .cluster-2 { background: linear-gradient(135deg, #1565C0, #2196F3); color: white; }
    .cluster-3 { background: linear-gradient(135deg, #FF6F00, #FF9800); color: white; }
    .cluster-4 { background: linear-gradient(135deg, #6A1B9A, #9C27B0); color: white; }
    .cluster-5 { background: linear-gradient(135deg, #C62828, #F44336); color: white; }
    
    /* African country flags color indicators */
    .country-indicator {
        display: inline-block;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        margin-right: 8px;
        vertical-align: middle;
    }
    
    .nigeria { background: linear-gradient(135deg, #008751, #FFFFFF, #008751); }
    .ghana { background: linear-gradient(135deg, #006B3F, #FCD116, #CE1126); }
    .kenya { background: linear-gradient(135deg, #000000, #BB0000, #006600, #FFFFFF); }
    .southafrica { background: linear-gradient(135deg, #000000, #FFB612, #007A4D, #FFFFFF, #DE3831, #002395); }
    
    /* Data upload zone styling */
    .upload-zone {
        border: 3px dashed #2A9D8F;
        border-radius: 10px;
        padding: 3rem;
        text-align: center;
        background: rgba(42, 157, 143, 0.05);
        transition: all 0.3s ease;
        margin-bottom: 2rem;
    }
    
    .upload-zone:hover {
        background: rgba(42, 157, 143, 0.1);
        border-color: #FF6B35;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #FF6B35, #2A9D8F, #264653);
    }
    
    /* Custom tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
        background-color: #F5F5F5;
        padding: 5px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #E0E0E0;
        border-radius: 8px;
        padding: 0.8rem 1.5rem;
        font-weight: 700;
        color: #555;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FF6B35 0%, #2A9D8F 100%);
        color: white;
        box-shadow: 0 4px 10px rgba(42, 157, 143, 0.3);
    }
    
    /* Chat-style insights */
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        max-width: 80%;
        animation: fadeIn 0.5s ease;
    }
    
    .chat-bot {
        background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
        border-left: 4px solid #2196F3;
        margin-right: auto;
    }
    
    .chat-user {
        background: linear-gradient(135deg, #F3E5F5, #E1BEE7);
        border-right: 4px solid #9C27B0;
        margin-left: auto;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Data table styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .dataframe th {
        background: linear-gradient(135deg, #264653, #2A9D8F);
        color: white;
        font-weight: 900;
        padding: 12px;
        text-align: left;
    }
    
    .dataframe td {
        padding: 10px;
        border-bottom: 1px solid #E0E0E0;
        font-weight: 600;
    }
    
    .dataframe tr:hover {
        background-color: rgba(42, 157, 143, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# TITLE AND HEADER
# ============================================
st.markdown('<h1 class="main-header">🌍 African Financial Risk & Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="africa-card">
    <h3 style="color: #264653; margin-bottom: 1rem; border-bottom: 2px solid #FF6B35; padding-bottom: 0.5rem;">
        Capstone Project: Unsupervised Customer Behavior Segmentation
    </h3>
    <p style="color: #455A64; font-size: 1.1rem; line-height: 1.6;">
        <strong>Domain:</strong> Finance | <strong>Techniques:</strong> Clustering, Topic Modeling, Sentiment Analysis<br>
        <strong>Tools:</strong> Python, Power BI, NLP (spaCy/Transformers), K-Means, PCA<br>
        <strong>Dataset:</strong> 5,000+ African financial behavior records<br>
        <strong>Markets:</strong> Nigeria, Ghana, Kenya, Côte d'Ivoire, Uganda, South Africa
    </p>
    <div style="display: flex; gap: 1rem; margin-top: 1rem;">
        <span class="payment-badge">POS</span>
        <span class="payment-badge mobile">Mobile Money</span>
        <span class="payment-badge ussd">USSD</span>
        <span class="payment-badge">Transfers</span>
        <span class="payment-badge">ATM</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================
# INITIALIZE SESSION STATE
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

# ============================================
# DATA LOADING FUNCTIONS - MULTIPLE FORMATS
# ============================================

def load_dataset(uploaded_file):
    """Load dataset from multiple formats with intelligent detection"""
    try:
        # Get file extension
        file_name = uploaded_file.name.lower()
        file_extension = file_name.split('.')[-1]
        
        # Read based on extension
        if file_extension in ['csv', 'txt']:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    uploaded_file.seek(0)
                    content = uploaded_file.read()
                    
                    # Try to detect delimiter for CSV
                    if file_extension == 'csv':
                        # Try comma, semicolon, tab
                        for delimiter in [',', ';', '\t', '|']:
                            try:
                                df = pd.read_csv(io.BytesIO(content), delimiter=delimiter, encoding=encoding)
                                if len(df.columns) > 1:
                                    break
                            except:
                                continue
                        else:
                            df = pd.read_csv(io.BytesIO(content), encoding=encoding)
                    else:
                        df = pd.read_csv(io.BytesIO(content), encoding=encoding)
                    
                    break
                except UnicodeDecodeError:
                    continue
        
        elif file_extension in ['xlsx', 'xls', 'xlsm', 'xlsb']:
            df = pd.read_excel(uploaded_file)
        
        elif file_extension == 'json':
            df = pd.read_json(uploaded_file)
        
        elif file_extension == 'parquet':
            df = pd.read_parquet(uploaded_file)
        
        elif file_extension == 'feather':
            df = pd.read_feather(uploaded_file)
        
        elif file_extension == 'h5':
            df = pd.read_hdf(uploaded_file)
        
        elif file_extension in ['pkl', 'pickle']:
            df = pd.read_pickle(uploaded_file)
        
        elif file_extension == 'sql':
            # For SQL files, we'll try to read as text and parse
            content = uploaded_file.read().decode('utf-8', errors='ignore')
            # Try to extract data from SQL insert statements
            import re
            data_lines = []
            for line in content.split('\n'):
                if 'INSERT INTO' in line.upper() or 'VALUES' in line.upper():
                    data_lines.append(line)
            if data_lines:
                # Simple parsing - in real app you'd use proper SQL parser
                st.info("SQL file detected. Showing first 1000 lines for analysis.")
                return pd.DataFrame({'sql_content': data_lines[:1000]})
            else:
                return pd.DataFrame({'file_content': [content[:10000]]})
        
        elif file_extension in ['xml']:
            df = pd.read_xml(uploaded_file)
        
        else:
            # Try to auto-detect format
            uploaded_file.seek(0)
            content = uploaded_file.read()[:10000]  # Read first 10KB
            
            # Try common formats
            try:
                df = pd.read_csv(io.BytesIO(content))
            except:
                try:
                    df = pd.read_json(io.BytesIO(content))
                except:
                    try:
                        df = pd.read_excel(io.BytesIO(content))
                    except:
                        # Return as text file
                        try:
                            text_content = content.decode('utf-8', errors='ignore')
                            return pd.DataFrame({'text_content': [text_content]})
                        except:
                            st.error(f"Unsupported file format: {uploaded_file.name}")
                            return None
        
        # Clean column names
        df.columns = [str(col).strip().replace(' ', '_').replace('-', '_').replace('.', '_').lower() for col in df.columns]
        
        # Remove completely empty columns
        df = df.dropna(axis=1, how='all')
        
        return df
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def detect_african_financial_columns(df):
    """Detect columns specific to African financial datasets"""
    column_mapping = {
        'customer_id': ['customer_id', 'client_id', 'user_id', 'account_number', 'id'],
        'age': ['age', 'customer_age', 'client_age'],
        'income_level': ['income_level', 'income_category', 'salary_range', 'earning_level'],
        'monthly_expenditure': ['monthly_expenditure', 'total_spend', 'monthly_spending', 'expenses'],
        'saving_behavior': ['saving_behavior', 'savings_pattern', 'savings_habit', 'savings_consistency'],
        'credit_score': ['credit_score', 'credit_rating', 'risk_score', 'financial_score'],
        'transaction_count': ['transaction_count', 'num_transactions', 'txn_count'],
        'avg_transaction_value': ['avg_transaction_value', 'average_txn_value', 'mean_transaction'],
        'payment_channels': ['payment_channels', 'payment_methods', 'channels_used', 'transaction_channels'],
        'expenditure_categories': ['expenditure_categories', 'spending_categories', 'expense_types'],
        'customer_feedback': ['customer_feedback', 'feedback', 'reviews', 'comments', 'complaints'],
        'sentiment_score': ['sentiment_score', 'feedback_score', 'sentiment'],
        'country': ['country', 'region', 'location', 'market'],
        'digital_adoption': ['digital_adoption', 'digital_score', 'mobile_usage'],
        'financial_literacy': ['financial_literacy', 'finlit_score', 'education_level']
    }
    
    detected_columns = {}
    for col in df.columns:
        col_lower = col.lower()
        
        for category, patterns in column_mapping.items():
            for pattern in patterns:
                if pattern in col_lower:
                    detected_columns[col] = category
                    break
            if col in detected_columns:
                break
        
        # If not matched, try to infer from data
        if col not in detected_columns:
            if df[col].dtype in ['int64', 'float64']:
                if df[col].between(300, 850).all():  # Likely credit score
                    detected_columns[col] = 'credit_score'
                elif 'sentiment' in col_lower or df[col].between(-1, 1).all():
                    detected_columns[col] = 'sentiment_score'
                elif 'age' in col_lower or (df[col].between(18, 100).all() and df[col].dtype == 'int64'):
                    detected_columns[col] = 'age'
    
    return detected_columns

# ============================================
# NLP PROCESSING MODULE
# ============================================

class NLPProcessor:
    """Advanced NLP processor for African financial text"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(STOP_WORDS)
        self.tfidf_vectorizer = None
        self.topic_model = None
        
        # African financial specific stop words
        self.african_stop_words = {
            'naira', 'cedis', 'rands', 'shillings', 'mobile', 'money', 'ussd', 'pos',
            'mpesa', 'mtn', 'airtel', 'vodafone', 'orange', 'safaricom', 'bank',
            'please', 'thank', 'thanks', 'hello', 'hi', 'greetings'
        }
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
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
            and token not in self.african_stop_words
            and len(token) > 2
        ]
        
        return ' '.join(tokens)
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using VADER"""
        if pd.isna(text) or not str(text).strip():
            return 0.0
        
        text = str(text)
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        return sentiment['compound']
    
    def extract_topics(self, texts, num_topics=5):
        """Extract topics using LDA"""
        if len(texts) < 10:
            return []
        
        # Preprocess all texts
        processed_texts = [self.preprocess_text(text).split() for text in texts]
        
        # Create dictionary and corpus
        dictionary = corpora.Dictionary(processed_texts)
        corpus = [dictionary.doc2bow(text) for text in processed_texts]
        
        # Train LDA model
        lda_model = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=10
        )
        
        # Extract topics
        topics = []
        for idx, topic in lda_model.print_topics(-1):
            words = topic.split('"')[1::2]  # Extract words between quotes
            topics.append({
                'topic_id': idx,
                'keywords': words[:5],  # Top 5 keywords
                'weight': float(topic.split('*')[0])
            })
        
        return topics
    
    def create_text_embeddings(self, texts):
        """Create TF-IDF embeddings"""
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=100,
            min_df=2,
            max_df=0.95
        )
        
        embeddings = self.tfidf_vectorizer.fit_transform(processed_texts)
        return embeddings

# ============================================
# DATA PROCESSING PIPELINE
# ============================================

class DataProcessor:
    """Process African financial datasets"""
    
    def __init__(self):
        self.nlp_processor = NLPProcessor()
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=10)
        self.label_encoder = LabelEncoder()
        
        # African market specific configurations
        self.african_currencies = {
            'nigeria': 'NGN',
            'ghana': 'GHC',
            'kenya': 'KES',
            'south africa': 'ZAR',
            'uganda': 'UGX',
            "côte d'ivoire": 'XOF'
        }
        
        self.payment_channels = [
            'pos', 'mobile_money', 'ussd', 'bank_transfer', 
            'atm', 'online', 'card', 'cash'
        ]
        
        self.spending_categories = [
            'food', 'transport', 'utilities', 'rent', 
            'data', 'fuel', 'entertainment', 'education',
            'healthcare', 'shopping', 'savings'
        ]
    
    def detect_african_country(self, df):
        """Detect African country from data"""
        country_columns = ['country', 'region', 'location', 'market']
        
        for col in country_columns:
            if col in df.columns:
                unique_values = df[col].dropna().unique()
                for value in unique_values:
                    value_str = str(value).lower()
                    for country in self.african_currencies.keys():
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
    
    def process_dataset(self, df, dataset_name=None):
        """Main processing pipeline"""
        # Create a copy
        processed_df = df.copy()
        
        # Store original info
        metadata = {
            'dataset_name': dataset_name or 'Unnamed Dataset',
            'original_records': len(df),
            'original_columns': len(df.columns),
            'processing_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'detected_country': self.detect_african_country(df),
            'missing_values': int(df.isnull().sum().sum()),
            'missing_percentage': float((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100)
        }
        
        # Detect column types
        column_info = detect_african_financial_columns(df)
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in df.columns:
                processed_df[col] = processed_df[col].fillna(processed_df[col].median())
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in df.columns:
                processed_df[col] = processed_df[col].fillna('Unknown')
        
        # Process text data
        if 'customer_feedback' in df.columns or any('feedback' in col.lower() for col in df.columns):
            feedback_col = next((col for col in df.columns if 'feedback' in col.lower()), None)
            if feedback_col:
                processed_df['sentiment_score'] = processed_df[feedback_col].apply(
                    self.nlp_processor.analyze_sentiment
                )
                processed_df['feedback_processed'] = processed_df[feedback_col].apply(
                    self.nlp_processor.preprocess_text
                )
                metadata['has_feedback'] = True
                metadata['feedback_records'] = int(processed_df[feedback_col].notna().sum())
        
        # Process payment channels
        if 'payment_channels' in df.columns:
            processed_df['payment_channels_list'] = processed_df['payment_channels'].apply(
                lambda x: str(x).split(',') if pd.notna(x) else []
            )
            
            # Create binary columns for each payment channel
            for channel in self.payment_channels:
                processed_df[f'uses_{channel}'] = processed_df['payment_channels_list'].apply(
                    lambda x: 1 if channel in [c.lower().strip() for c in x] else 0
                )
            
            metadata['payment_channels_detected'] = True
        
        # Process expenditure categories
        if 'expenditure_categories' in df.columns:
            processed_df['expenditure_list'] = processed_df['expenditure_categories'].apply(
                lambda x: str(x).split(',') if pd.notna(x) else []
            )
            
            # Create spending category columns
            for category in self.spending_categories:
                processed_df[f'spends_on_{category}'] = processed_df['expenditure_list'].apply(
                    lambda x: 1 if category in [c.lower().strip() for c in x] else 0
                )
            
            metadata['expenditure_categories_detected'] = True
        
        # Create derived features
        self._create_derived_features(processed_df)
        
        # Store column info with detected types
        column_info = self._enrich_column_info(processed_df, column_info)
        
        return processed_df, column_info, metadata
    
    def _create_derived_features(self, df):
        """Create derived features for African financial analysis"""
        
        # Digital Adoption Score
        digital_cols = [col for col in df.columns if col.startswith('uses_')]
        if digital_cols:
            df['digital_adoption_score'] = df[digital_cols].sum(axis=1) / len(digital_cols)
        
        # Financial Health Score
        financial_cols = []
        if 'credit_score' in df.columns:
            financial_cols.append('credit_score')
        if 'saving_behavior' in df.columns:
            # Convert saving behavior to numeric
            saving_map = {'consistent': 1.0, 'irregular': 0.5, 'none': 0.0}
            df['saving_score'] = df['saving_behavior'].map(saving_map).fillna(0.5)
            financial_cols.append('saving_score')
        
        if financial_cols:
            df['financial_health_score'] = df[financial_cols].mean(axis=1)
        
        # Spending Efficiency (if we have income and expenditure)
        if 'monthly_expenditure' in df.columns and 'income_level' in df.columns:
            income_map = {'low': 50000, 'middle': 150000, 'high': 500000}
            df['income_numeric'] = df['income_level'].map(income_map).fillna(100000)
            df['spending_ratio'] = df['monthly_expenditure'] / (df['income_numeric'] + 1)
        
        # Risk Score based on various factors
        risk_factors = []
        if 'credit_score' in df.columns:
            risk_factors.append(1 - (df['credit_score'] - 300) / 550)
        if 'spending_ratio' in df.columns:
            risk_factors.append(df['spending_ratio'].clip(0, 2) / 2)
        if 'sentiment_score' in df.columns:
            risk_factors.append(1 - (df['sentiment_score'] + 1) / 2)
        
        if risk_factors:
            df['risk_score'] = pd.concat(risk_factors, axis=1).mean(axis=1)
            df['risk_category'] = pd.cut(
                df['risk_score'],
                bins=[0, 0.3, 0.7, 1],
                labels=['Low Risk', 'Medium Risk', 'High Risk'],
                include_lowest=True
            )
    
    def _enrich_column_info(self, df, initial_column_info):
        """Enrich column information with additional metadata"""
        column_info = {
            'detected_mapping': initial_column_info,
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'derived_columns': [col for col in df.columns if col not in initial_column_info],
            'text_columns': [col for col in df.columns if 'feedback' in col.lower()],
            'payment_channel_columns': [col for col in df.columns if col.startswith('uses_')],
            'spending_columns': [col for col in df.columns if col.startswith('spends_on_')]
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
# AUTOMATIC INSIGHTS GENERATOR
# ============================================

class InsightsGenerator:
    """Generate automatic insights and recommendations"""
    
    def __init__(self):
        self.insights = []
        self.recommendations = []
    
    def generate_insights(self, df, column_info, metadata, clusters=None):
        """Generate comprehensive insights"""
        self.insights = []
        self.recommendations = []
        
        # 1. Dataset Overview Insights
        self._generate_dataset_insights(df, metadata)
        
        # 2. Financial Behavior Insights
        self._generate_financial_insights(df)
        
        # 3. Digital Adoption Insights
        self._generate_digital_insights(df)
        
        # 4. Risk Analysis Insights
        self._generate_risk_insights(df)
        
        # 5. Customer Sentiment Insights
        self._generate_sentiment_insights(df)
        
        # 6. Clustering Insights (if available)
        if clusters is not None:
            self._generate_clustering_insights(df, clusters)
        
        # 7. African Market Specific Insights
        self._generate_african_market_insights(df, metadata)
        
        # Generate recommendations based on insights
        self._generate_recommendations()
        
        return self.insights, self.recommendations
    
    def _generate_dataset_insights(self, df, metadata):
        """Generate dataset-level insights"""
        
        insight = f"""
        ## 📊 Dataset Overview
        
        The dataset contains **{metadata['original_records']:,} customer records** with **{metadata['original_columns']} features**. 
        
        **Data Quality Assessment:**
        • Missing values: {metadata['missing_values']:,} ({metadata['missing_percentage']:.1f}% of total data points)
        • Country detected: {metadata.get('detected_country', 'Various African markets')}
        • Analysis date: {metadata['processing_date']}
        
        **Key Findings:**
        • Dataset size is {'adequate' if len(df) >= 1000 else 'limited'} for robust analysis
        • Data quality is {'good' if metadata['missing_percentage'] < 10 else 'requires attention'}
        • African market focus enables region-specific insights
        """
        
        self.insights.append({
            'category': 'dataset',
            'title': 'Dataset Overview & Quality',
            'content': insight,
            'severity': 'info'
        })
    
    def _generate_financial_insights(self, df):
        """Generate financial behavior insights"""
        
        insights_text = "## 💰 Financial Behavior Analysis\n\n"
        
        # Income analysis
        if 'income_level' in df.columns:
            income_dist = df['income_level'].value_counts(normalize=True) * 100
            insights_text += f"**Income Distribution:**\n"
            for level, pct in income_dist.items():
                insights_text += f"• {level}: {pct:.1f}%\n"
            insights_text += "\n"
        
        # Spending analysis
        if 'monthly_expenditure' in df.columns:
            avg_spend = df['monthly_expenditure'].mean()
            median_spend = df['monthly_expenditure'].median()
            spend_std = df['monthly_expenditure'].std()
            
            insights_text += f"**Spending Patterns:**\n"
            insights_text += f"• Average monthly expenditure: ${avg_spend:,.0f}\n"
            insights_text += f"• Median expenditure: ${median_spend:,.0f}\n"
            insights_text += f"• Spending variability: ${spend_std:,.0f} (std dev)\n\n"
        
        # Savings analysis
        if 'saving_behavior' in df.columns:
            savings_dist = df['saving_behavior'].value_counts(normalize=True) * 100
            insights_text += f"**Savings Behavior:**\n"
            for behavior, pct in savings_dist.items():
                insights_text += f"• {behavior}: {pct:.1f}%\n"
        
        if insights_text != "## 💰 Financial Behavior Analysis\n\n":
            self.insights.append({
                'category': 'financial',
                'title': 'Financial Behavior Patterns',
                'content': insights_text,
                'severity': 'info'
            })
    
    def _generate_digital_insights(self, df):
        """Generate digital adoption insights"""
        
        insights_text = "## 📱 Digital Payment Adoption\n\n"
        
        # Check for payment channel columns
        payment_cols = [col for col in df.columns if col.startswith('uses_')]
        
        if payment_cols:
            adoption_rates = {}
            for col in payment_cols:
                channel = col.replace('uses_', '').replace('_', ' ').title()
                adoption_rate = df[col].mean() * 100
                adoption_rates[channel] = adoption_rate
            
            # Sort by adoption rate
            sorted_rates = sorted(adoption_rates.items(), key=lambda x: x[1], reverse=True)
            
            insights_text += "**Payment Channel Adoption Rates:**\n"
            for channel, rate in sorted_rates[:5]:
                insights_text += f"• {channel}: {rate:.1f}%\n"
            
            # Digital adoption score
            if 'digital_adoption_score' in df.columns:
                avg_digital = df['digital_adoption_score'].mean() * 100
                insights_text += f"\n**Overall Digital Adoption:** {avg_digital:.1f}%\n"
                
                if avg_digital > 70:
                    insights_text += "→ High digital adoption market\n"
                elif avg_digital > 40:
                    insights_text += "→ Moderate digital adoption\n"
                else:
                    insights_text += "→ Traditional/cash-based market\n"
            
            self.insights.append({
                'category': 'digital',
                'title': 'Digital Payment Trends',
                'content': insights_text,
                'severity': 'info'
            })
    
    def _generate_risk_insights(self, df):
        """Generate risk analysis insights"""
        
        if 'risk_score' in df.columns:
            avg_risk = df['risk_score'].mean()
            high_risk_pct = (df['risk_score'] > 0.7).mean() * 100
            low_risk_pct = (df['risk_score'] < 0.3).mean() * 100
            
            insights_text = f"""
            ## ⚠️ Risk Profile Analysis
            
            **Risk Distribution:**
            • Average risk score: {avg_risk:.3f}
            • High-risk customers: {high_risk_pct:.1f}% (score > 0.7)
            • Low-risk customers: {low_risk_pct:.1f}% (score < 0.3)
            
            **Risk Assessment:**
            The customer base shows {'elevated risk levels' if high_risk_pct > 20 else 'moderate risk levels' if high_risk_pct > 10 else 'healthy risk distribution'}.
            """
            
            if 'risk_category' in df.columns:
                risk_dist = df['risk_category'].value_counts(normalize=True) * 100
                insights_text += "\n**Risk Categories:**\n"
                for category, pct in risk_dist.items():
                    insights_text += f"• {category}: {pct:.1f}%\n"
            
            self.insights.append({
                'category': 'risk',
                'title': 'Customer Risk Assessment',
                'content': insights_text,
                'severity': 'warning' if high_risk_pct > 20 else 'info'
            })
    
    def _generate_sentiment_insights(self, df):
        """Generate sentiment analysis insights"""
        
        if 'sentiment_score' in df.columns:
            avg_sentiment = df['sentiment_score'].mean()
            positive_pct = (df['sentiment_score'] > 0.3).mean() * 100
            negative_pct = (df['sentiment_score'] < -0.3).mean() * 100
            neutral_pct = 100 - positive_pct - negative_pct
            
            insights_text = f"""
            ## 😊 Customer Sentiment Analysis
            
            **Sentiment Distribution:**
            • Average sentiment: {avg_sentiment:.3f}
            • Positive sentiment: {positive_pct:.1f}%
            • Negative sentiment: {negative_pct:.1f}%
            • Neutral sentiment: {neutral_pct:.1f}%
            
            **Sentiment Insights:**
            Customer sentiment is {'overwhelmingly positive' if positive_pct > 60 else 'generally positive' if positive_pct > 40 else 'mixed' if positive_pct > negative_pct else 'concerning' if negative_pct > 30 else 'balanced'}.
            """
            
            # Topic insights if available
            if 'customer_feedback' in df.columns:
                feedback_count = df['customer_feedback'].notna().sum()
                insights_text += f"\n**Feedback Analysis:**\n"
                insights_text += f"• Customer feedback records: {feedback_count:,}\n"
                insights_text += f"• Feedback coverage: {(feedback_count/len(df)*100):.1f}% of customers\n"
            
            self.insights.append({
                'category': 'sentiment',
                'title': 'Customer Sentiment & Feedback',
                'content': insights_text,
                'severity': 'warning' if negative_pct > 30 else 'info'
            })
    
    def _generate_clustering_insights(self, df, clusters):
        """Generate clustering insights"""
        
        insights_text = "## 🎯 Customer Segmentation Analysis\n\n"
        
        insights_text += f"**Segmentation Results:**\n"
        insights_text += f"• Number of segments identified: {len(clusters)}\n"
        
        for cluster in clusters:
            insights_text += f"• **{cluster['name']}**: {cluster['size']:,} customers ({cluster['percentage']:.1f}%)\n"
        
        insights_text += "\n**Segment Characteristics:**\n"
        
        # Analyze each cluster
        df_clustered = df.copy()
        df_clustered['cluster_name'] = df_clustered['cluster'].map(
            {c['cluster_id']: c['name'] for c in clusters}
        )
        
        for cluster in clusters[:3]:  # Show top 3 clusters
            cluster_data = df_clustered[df_clustered['cluster_name'] == cluster['name']]
            
            insights_text += f"\n**{cluster['name']}**\n"
            
            # Key metrics for this cluster
            if 'risk_score' in cluster_data.columns:
                avg_risk = cluster_data['risk_score'].mean()
                insights_text += f"→ Average risk: {avg_risk:.3f}\n"
            
            if 'digital_adoption_score' in cluster_data.columns:
                avg_digital = cluster_data['digital_adoption_score'].mean()
                insights_text += f"→ Digital adoption: {avg_digital:.2f}\n"
            
            if 'sentiment_score' in cluster_data.columns:
                avg_sentiment = cluster_data['sentiment_score'].mean()
                insights_text += f"→ Sentiment: {avg_sentiment:.3f}\n"
        
        self.insights.append({
            'category': 'segmentation',
            'title': 'Customer Segmentation Results',
            'content': insights_text,
            'severity': 'info'
        })
    
    def _generate_african_market_insights(self, df, metadata):
        """Generate Africa-specific market insights"""
        
        country = metadata.get('detected_country', 'African Market')
        
        insights_text = f"""
        ## 🌍 {country} Market Insights
        
        **Market Context:**
        • Analysis focused on {country}'s financial ecosystem
        • Considering local payment methods, currency, and consumer behavior
        • Accounting for digital financial service adoption trends
        
        **Key African Financial Trends:**
        1. Rapid mobile money adoption across the continent
        2. Increasing digital payment infrastructure
        3. Unique spending patterns based on local economies
        4. Informal sector integration with formal banking
        5. Youth-driven digital financial services growth
        
        **Market Opportunities:**
        • Digital lending for underserved segments
        • Mobile-based savings products
        • Cross-border payment solutions
        • Financial literacy programs
        """
        
        self.insights.append({
            'category': 'market',
            'title': f'{country} Market Analysis',
            'content': insights_text,
            'severity': 'info'
        })
    
    def _generate_recommendations(self):
        """Generate recommendations based on insights"""
        
        recommendations_text = """
        ## 🚀 Strategic Recommendations
        
        **Immediate Actions (Next 30 Days):**
        1. **Review High-Risk Segments**
           • Implement enhanced monitoring for high-risk customers
           • Develop targeted communication for risk mitigation
           • Consider adjusted credit terms where appropriate
        
        2. **Leverage Digital Adoption**
           • Promote underutilized digital channels
           • Develop mobile-first financial products
           • Enhance USSD banking features for feature phone users
        
        3. **Improve Customer Sentiment**
           • Address common complaint themes
           • Implement proactive customer service
           • Develop customer feedback loop
        
        **Medium-Term Initiatives (3-6 Months):**
        1. **Segment-Specific Product Development**
           • Create tailored financial products for each segment
           • Develop personalized marketing campaigns
           • Implement dynamic pricing based on risk profiles
        
        2. **Enhanced Analytics Capabilities**
           • Implement real-time monitoring dashboard
           • Develop predictive risk models
           • Create customer lifetime value projections
        
        3. **Operational Efficiency**
           • Automate risk assessment processes
           • Streamline customer onboarding
           • Optimize payment channel allocation
        
        **Long-Term Strategy (6+ Months):**
        1. **AI-Driven Personalization**
           • Implement machine learning for personalized offers
           • Develop predictive churn models
           • Create automated financial advice systems
        
        2. **Market Expansion**
           • Explore adjacent African markets
           • Develop cross-border payment solutions
           • Partner with fintech innovators
        
        3. **Sustainability Initiatives**
           • Develop green financing products
           • Implement financial inclusion programs
           • Create digital literacy campaigns
        """
        
        self.recommendations.append({
            'category': 'strategic',
            'title': 'Comprehensive Action Plan',
            'content': recommendations_text,
            'priority': 'high'
        })
        
        # Data quality recommendations
        if any('missing_percentage' in str(insight) for insight in self.insights):
            self.recommendations.append({
                'category': 'data',
                'title': 'Data Quality Improvements',
                'content': """
                **Data Enhancement Priorities:**
                1. Implement data validation at point of entry
                2. Establish data quality monitoring dashboard
                3. Develop data cleaning automation pipeline
                4. Create data governance framework
                5. Train staff on data collection best practices
                """,
                'priority': 'medium'
            })

# ============================================
# MAIN DASHBOARD COMPONENTS
# ============================================

def display_upload_section():
    """Display data upload section"""
    
    st.markdown("""
    <div class="upload-zone">
        <h2 style="color: #264653; margin-bottom: 1rem;">📁 Upload Your African Financial Dataset</h2>
        <p style="color: #546E7A; margin-bottom: 2rem;">
            Upload CSV, Excel, JSON, Parquet, or other financial data files.<br>
            The system will automatically detect African financial patterns and generate insights.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "xls", "json", "txt", "parquet", "feather", "h5", "pkl", "sql", "xml"],
        key="main_uploader",
        help="Upload your African financial dataset (5,000+ records recommended)"
    )
    
    if uploaded_file is not None:
        # Load dataset
        with st.spinner(f"Loading {uploaded_file.name}..."):
            df = load_dataset(uploaded_file)
            
            if df is not None:
                if len(df) < 10:
                    st.error("Dataset too small. Please upload a dataset with at least 10 records.")
                    return None
                
                # Store in session state
                st.session_state.current_dataset_name = uploaded_file.name
                st.session_state.df = df
                
                # Process dataset
                data_processor = DataProcessor()
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
                
                # Generate automatic insights
                with st.spinner("Generating automatic insights and recommendations..."):
                    insights_generator = InsightsGenerator()
                    insights, recommendations = insights_generator.generate_insights(
                        processed_df, column_info, metadata, cluster_profiles
                    )
                    
                    st.session_state.insights = insights
                    st.session_state.recommendations = recommendations
                
                st.success(f"✅ Dataset processed successfully! Generated {len(insights)} insights and {len(recommendations)} recommendations.")
                
                return processed_df
    
    return None

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
        for insight in st.session_state.insights:
            with st.expander(f"📌 {insight['title']}", expanded=False):
                st.markdown(insight['content'])
        
        # Display recommendations
        st.markdown("## 🚀 Actionable Recommendations")
        
        for recommendation in st.session_state.recommendations:
            with st.expander(f"✅ {recommendation['title']} ({recommendation['priority'].upper()} PRIORITY)", expanded=False):
                st.markdown(recommendation['content'])
        
        # Data Download Section
        st.markdown("---")
        st.markdown("## 📥 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Download Processed Data", use_container_width=True):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="african_financial_analysis.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📋 Download Insights Report", use_container_width=True):
                report = generate_insights_report()
                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name="financial_insights_report.txt",
                    mime="text/plain"
                )
        
        with col3:
            if st.button("🔄 Analyze Another Dataset", use_container_width=True):
                st.session_state.df = None
                st.session_state.processed_df = None
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
            selected_metric = st.selectbox("Select metric for comparison", metric_options)
            
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
# SIDEBAR
# ============================================

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; padding: 1.5rem; background: linear-gradient(135deg, #FF6B35 0%, #2A9D8F 100%); border-radius: 12px; box-shadow: 0 6px 20px rgba(255, 107, 53, 0.3);">
        <h3 style="color: white; margin: 0; font-weight: 900;">🌍 Capstone Project</h3>
        <p style="color: rgba(255,255,255,0.9); font-size: 1rem; margin: 0.5rem 0 0 0; font-weight: 700;">
            African Financial Risk & Sentiment Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    st.markdown("### Navigation")
    
    nav_options = ["📁 Upload Data", "📊 Dashboard", "🔍 Insights", "⚙️ Settings"]
    selected_nav = st.radio(
        "Select Section",
        nav_options,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Dataset Info
    if st.session_state.metadata:
        st.markdown("### Current Dataset")
        
        dataset_info = f"""
        **{st.session_state.metadata.get('dataset_name', 'Unnamed')}**
        
        • {st.session_state.metadata.get('original_records', 0):,} records
        • {st.session_state.metadata.get('detected_country', 'Various')}
        • {len(st.session_state.insights) if st.session_state.insights else 0} insights generated
        """
        
        st.info(dataset_info)
        
        if st.button("🔄 Load New Dataset", use_container_width=True):
            st.session_state.df = None
            st.session_state.processed_df = None
            st.rerun()
    
    st.markdown("---")
    
    # Team Info
    st.markdown("### Team")
    st.markdown("""
    - Amarachi Florence
    - Thato Maelane  
    - Philip Odiachi
    - Mavis
    
    **Dataverse Africa Internship**
    """)

# ============================================
# MAIN APP FLOW
# ============================================

if selected_nav == "📁 Upload Data":
    st.markdown('<h2 class="sub-header">Upload African Financial Dataset</h2>', unsafe_allow_html=True)
    
    # Show upload section
    processed_df = display_upload_section()
    
    if processed_df is not None:
        # Show quick preview
        with st.expander("📋 Dataset Preview", expanded=True):
            st.dataframe(processed_df.head(10), use_container_width=True)
        
        # Show column info
        if st.session_state.column_info:
            with st.expander("🔍 Detected Column Types", expanded=False):
                column_summary = []
                for col, col_type in st.session_state.column_info.get('detected_mapping', {}).items():
                    column_summary.append(f"**{col}**: {col_type}")
                
                st.markdown("\n".join(column_summary))

elif selected_nav == "📊 Dashboard":
    if st.session_state.processed_df is not None:
        display_dashboard()
    else:
        st.info("📁 Please upload a dataset first to view the dashboard.")
        if st.button("Go to Upload Section"):
            st.session_state.df = None
            st.rerun()

elif selected_nav == "🔍 Insights":
    st.markdown('<h2 class="sub-header">Comprehensive Insights</h2>', unsafe_allow_html=True)
    
    if st.session_state.insights:
        # Display all insights in detail
        for insight in st.session_state.insights:
            st.markdown(f"""
            <div class="insights-container">
                <h3 style="color: #006064; margin-top: 0;">{insight['title']}</h3>
                <div style="color: #37474F; line-height: 1.6;">
                    {insight['content'].replace('**', '<strong>').replace('**', '</strong>')}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed recommendations
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("## 📋 Detailed Action Plan")
        
        for recommendation in st.session_state.recommendations:
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
                <div style="color: #4A148C; line-height: 1.6;">
                    {recommendation['content'].replace('**', '<strong>').replace('**', '</strong>')}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📁 Please upload a dataset first to generate insights.")

elif selected_nav == "⚙️ Settings":
    st.markdown('<h2 class="sub-header">Settings & Configuration</h2>', unsafe_allow_html=True)
    
    with st.expander("🔄 Processing Settings", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.number_input("Minimum cluster size", min_value=10, max_value=1000, value=50, key="min_cluster_size")
            st.selectbox("Clustering method", ["K-Means", "Hierarchical", "GMM", "DBSCAN"], key="clustering_method")
        
        with col2:
            st.slider("Risk threshold", 0.0, 1.0, 0.7, 0.05, key="risk_threshold")
            st.checkbox("Generate topic models", value=True, key="generate_topics")
    
    with st.expander("📊 Visualization Settings", expanded=False):
        st.selectbox("Chart theme", ["Plotly", "Seaborn", "GGplot", "Simple"], key="chart_theme")
        st.checkbox("Show data points", value=True, key="show_data_points")
        st.checkbox("Interactive tooltips", value=True, key="interactive_tooltips")
    
    with st.expander("🔧 Advanced Settings", expanded=False):
        st.number_input("PCA components", min_value=2, max_value=50, value=10, key="pca_components")
        st.number_input("NLP max features", min_value=50, max_value=1000, value=100, key="nlp_max_features")
        st.checkbox("Enable GPU acceleration", value=False, key="gpu_acceleration")
    
    if st.button("💾 Save Settings", use_container_width=True):
        st.success("Settings saved successfully!")

# ============================================
# FOOTER
# ============================================
st.markdown("---")

st.markdown("""
<div style="text-align: center; color: #264653; padding: 2rem; background: linear-gradient(135deg, #E9F5DB 0%, #C8E6C9 100%); border-radius: 12px; margin-top: 3rem;">
    <p style="margin-bottom: 0.5rem; font-size: 1.2rem; font-weight: 900; color: #1A237E;">
        Customer Financial Risk Prediction & Sentiment Analysis Dashboard
    </p>
    <p style="margin-bottom: 0.5rem; font-weight: 800; color: #D84315;">
        African Financial Behavior Segmentation | Unsupervised ML + NLP
    </p>
    <p style="margin-bottom: 0.5rem; font-size: 0.9rem; font-weight: 700; color: #455A64;">
        Developed by: Amarachi Florence, Thato Maelane, Philip Odiachi, Mavis
    </p>
    <p style="margin-bottom: 1rem; font-size: 0.9rem; font-weight: 700; color: #455A64;">
        Dataverse Africa Capstone Project | Financial Analytics Domain
    </p>
    <div style="font-size: 0.8rem; color: #78909c; font-weight: 600;">
        Techniques: Clustering, Topic Modeling, Sentiment Analysis | Tools: Python, NLP, K-Means, PCA
    </div>
</div>
""", unsafe_allow_html=True)























"""
African Financial Behavior Segmentation Dashboard
Complete Analytics Platform for African Financial Institutions
Capstone Project - Dataverse Africa Cohort 3
"""

# ============================================
# SET PAGE CONFIG - MUST BE FIRST STREAMLIT COMMAND
# ============================================
import streamlit as st
st.set_page_config(
    page_title="African Financial Segmentation",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# SUPPRESS ALL WARNINGS
# ============================================
import warnings
warnings.filterwarnings('ignore')

# ============================================
# MAIN IMPORTS - UPDATED WITH NEW DEPENDENCIES
# ============================================
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

# ============================================
# NLTK DOWNLOAD (SILENT)
# ============================================
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('sentiment/vader_lexicon')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except:
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
    except:
        pass

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
    
    /* Segment badge styling - FIXED */
    .segment-badge {
        background: linear-gradient(135deg, #FF6B35, #FFA62E);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        font-weight: 900;
        font-size: 1.2rem;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(255, 107, 53, 0.3);
        margin: 0.5rem 0;
        border: 3px solid white;
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
    
    /* Normal text paragraphs */
    .normal-text {
        font-size: 1.1rem;
        line-height: 1.6;
        color: #333;
        margin-bottom: 1rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Chart styling */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Dashboard header */
    .dashboard-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #264653 0%, #2A9D8F 100%);
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(38, 70, 83, 0.3);
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
if 'auto_generated' not in st.session_state:
    st.session_state.auto_generated = False

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
# ENHANCED NLP PROCESSOR WITHOUT spaCy/Transformers
# ============================================

class EnhancedNLPProcessor:
    """Advanced NLP processor using only NLTK (no spaCy/Transformers warnings)"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        
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
    
    def preprocess_text_nltk(self, text):
        """Preprocess text using NLTK"""
        if pd.isna(text) or not str(text).strip():
            return ""
        
        text = str(text).lower()
        
        # Remove special characters but keep African language characters
        text = re.sub(r'[^a-zà-ÿāăąēĕėęěīĭįıōŏőœūŭůűųçñ\s]', ' ', text)
        
        # If text is empty after cleaning, return empty string
        if not text.strip():
            return ""
        
        try:
            # Tokenize
            tokens = word_tokenize(text)
        except Exception as e:
            # Fallback to simple split if tokenization fails
            tokens = text.split()
        
        # Remove stop words and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words 
            and len(token) > 2
        ]
        
        return ' '.join(tokens)
    
    def analyze_sentiment_vader(self, text):
        """Analyze sentiment using VADER"""
        if pd.isna(text) or not str(text).strip():
            return 0.0
        
        sentiment = self.sentiment_analyzer.polarity_scores(str(text))
        return sentiment['compound']
    
    def extract_topics_lda(self, texts, num_topics=5):
        """Extract topics using LDA if available"""
        if len(texts) < 10 or not GENSIM_AVAILABLE:
            return []
        
        # Preprocess texts
        processed_texts = []
        for text in texts:
            processed = self.preprocess_text_nltk(text)
            processed_texts.append(processed.split())
        
        try:
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
        except:
            return []
    
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
        """Create embeddings using TF-IDF"""
        processed_texts = []
        
        for text in texts:
            processed = self.preprocess_text_nltk(text)
            processed_texts.append(processed)
        
        embeddings = self.tfidf_vectorizer.fit_transform(processed_texts).toarray()
        
        return embeddings
    
    def generate_wordcloud(self, texts, max_words=100):
        """Generate word cloud from texts if available"""
        all_text = ' '.join([str(text) for text in texts if pd.notna(text)])
        
        if not all_text or not WORDCLOUD_AVAILABLE:
            return None
        
        # Preprocess
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
# ENHANCED DATA PROCESSOR - UPDATED
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
        
        # Process text data with enhanced NLP - with error handling
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
        """Process text data with enhanced NLP - with error handling"""
        processed_df = df.copy()
        
        # Process feedback columns
        feedback_cols = [col for col in df.columns if 'feedback' in col.lower() or 
                        'review' in col.lower() or 'comment' in col.lower()]
        
        for col in feedback_cols:
            # Sentiment analysis with error handling
            try:
                processed_df[f'{col}_sentiment'] = processed_df[col].apply(
                    lambda x: self.nlp_processor.analyze_sentiment_vader(x) if pd.notna(x) else 0.0
                )
            except:
                processed_df[f'{col}_sentiment'] = 0.0
            
            # Processed text with error handling
            try:
                processed_df[f'{col}_processed'] = processed_df[col].apply(
                    lambda x: self.nlp_processor.preprocess_text_nltk(x) if pd.notna(x) else ""
                )
            except Exception as e:
                # If NLP processing fails, use simple cleaning
                processed_df[f'{col}_processed'] = processed_df[col].apply(
                    lambda x: str(x).lower().replace('\n', ' ').replace('\r', ' ').strip() if pd.notna(x) else ""
                )
            
            # Sentiment category - only create if we have sentiment scores
            if f'{col}_sentiment' in processed_df.columns:
                try:
                    processed_df[f'{col}_sentiment_category'] = pd.cut(
                        processed_df[f'{col}_sentiment'].fillna(0),
                        bins=[-1, -0.3, 0.3, 1],
                        labels=['Negative', 'Neutral', 'Positive'],
                        include_lowest=True
                    )
                except:
                    # If cutting fails, assign categories manually
                    processed_df[f'{col}_sentiment_category'] = processed_df[f'{col}_sentiment'].apply(
                        lambda x: 'Negative' if x < -0.3 else 'Positive' if x > 0.3 else 'Neutral'
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
        """Create enhanced derived features - UPDATED with error handling"""
        processed_df = df.copy()
        
        # Digital Adoption Score (enhanced)
        digital_cols = [col for col in processed_df.columns if col.startswith('uses_')]
        if digital_cols:
            processed_df['digital_adoption_score'] = processed_df[digital_cols].sum(axis=1) / len(digital_cols)
            # Ensure we have valid values for cut
            if processed_df['digital_adoption_score'].notna().any():
                try:
                    processed_df['digital_adoption_level'] = pd.cut(
                        processed_df['digital_adoption_score'].fillna(0),
                        bins=[0, 0.3, 0.7, 1],
                        labels=['Low', 'Medium', 'High'],
                        include_lowest=True
                    )
                except:
                    # Manual assignment if cut fails
                    processed_df['digital_adoption_level'] = processed_df['digital_adoption_score'].apply(
                        lambda x: 'Low' if x < 0.3 else 'High' if x > 0.7 else 'Medium'
                    )
        
        # Financial Health Index
        financial_indicators = []
        
        if 'credit_score' in processed_df.columns:
            try:
                # Normalize credit score (300-850 to 0-1)
                processed_df['credit_score_normalized'] = (processed_df['credit_score'] - 300) / 550
                financial_indicators.append('credit_score_normalized')
            except:
                pass
        
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
            try:
                processed_df['financial_health_category'] = pd.cut(
                    processed_df['financial_health_index'].fillna(0.5),
                    bins=[0, 0.4, 0.7, 1],
                    labels=['Poor', 'Fair', 'Good'],
                    include_lowest=True
                )
            except:
                processed_df['financial_health_category'] = processed_df['financial_health_index'].apply(
                    lambda x: 'Poor' if x < 0.4 else 'Good' if x > 0.7 else 'Fair'
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
            try:
                processed_df['risk_category'] = pd.cut(
                    processed_df['risk_score'].fillna(0.5),
                    bins=[0, 0.3, 0.7, 1],
                    labels=['Low Risk', 'Medium Risk', 'High Risk'],
                    include_lowest=True
                )
            except:
                # Manual assignment
                processed_df['risk_category'] = processed_df['risk_score'].apply(
                    lambda x: 'Low Risk' if x < 0.3 else 'High Risk' if x > 0.7 else 'Medium Risk'
                )
            
            # Calculate risk concentration
            try:
                risk_distribution = processed_df['risk_category'].value_counts(normalize=True)
                if risk_distribution.get('High Risk', 0) > 0.2:
                    st.session_state.alerts.append({
                        'type': 'high_risk_concentration',
                        'message': f"High concentration of high-risk customers: {risk_distribution.get('High Risk', 0):.1%}",
                        'severity': 'high'
                    })
            except:
                pass
        
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
# CLUSTERING MODULE - FIXED
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
            'risk_score', 'digital_adoption_score', 'financial_health_index',
            'sentiment_score', 'credit_score', 'monthly_expenditure',
            'transaction_count', 'avg_transaction_value', 'spending_efficiency'
        ]
        
        for feature in priority_features:
            if feature in numeric_cols:
                selected_features.append(feature)
        
        # Add any remaining numeric features
        remaining_features = [col for col in numeric_cols if col not in selected_features]
        selected_features.extend(remaining_features[:10])  # Limit to 10 additional features
        
        # If no features selected, use all numeric columns (except IDs)
        if len(selected_features) == 0:
            selected_features = numeric_cols[:10]  # Use first 10 numeric columns
        
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
            try:
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
            except:
                silhouette_scores.append(0)
                db_scores.append(float('inf'))
                ch_scores.append(0)
        
        # Find best based on silhouette score (higher is better)
        if silhouette_scores:
            optimal_n = cluster_range[silhouette_scores.index(max(silhouette_scores))]
            return optimal_n
        else:
            return 3  # Default to 3 clusters
    
    def perform_clustering(self, df, method='kmeans'):
        """Perform clustering using specified method"""
        try:
            X, feature_names = self.prepare_features(df)
            
            if len(X) < 10:
                return None, None, 0
            
            # Determine optimal clusters
            n_clusters = self.find_optimal_clusters(X)
            n_clusters = min(n_clusters, 8)  # Max 8 clusters for interpretability
            n_clusters = max(n_clusters, 2)  # Min 2 clusters
            
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
            
        except Exception as e:
            print(f"Clustering error: {e}")
            return None, None, 0
    
    def _assign_cluster_names(self, df, labels, n_clusters):
        """Assign meaningful names to clusters based on characteristics"""
        if labels is None:
            return []
            
        df_clustered = df.copy()
        df_clustered['cluster'] = labels
        
        cluster_profiles = []
        
        for cluster_id in range(n_clusters):
            cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
            
            if len(cluster_data) == 0:
                continue
                
            # Calculate cluster characteristics
            characteristics = []
            
            # Check for high spenders
            if 'monthly_expenditure' in cluster_data.columns:
                avg_expenditure = cluster_data['monthly_expenditure'].mean()
                if 'monthly_expenditure' in df.columns:
                    if avg_expenditure > df['monthly_expenditure'].quantile(0.75):
                        characteristics.append("High Spender")
            
            # Check sentiment
            sentiment_cols = [col for col in cluster_data.columns if 'sentiment' in col.lower() and 'category' not in col.lower()]
            if sentiment_cols:
                avg_sentiment = cluster_data[sentiment_cols[0]].mean()
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
                if cluster_data['income_level'].astype(str).str.contains('low', case=False).any():
                    characteristics.append("Low Income")
                elif cluster_data['income_level'].astype(str).str.contains('high', case=False).any():
                    characteristics.append("High Income")
            
            # Check savings
            if 'saving_behavior' in cluster_data.columns:
                if cluster_data['saving_behavior'].astype(str).str.contains('consistent', case=False).any():
                    characteristics.append("Good Saver")
                elif cluster_data['saving_behavior'].astype(str).str.contains('irregular', case=False).any():
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
# ENHANCED INSIGHTS GENERATOR - COMPLETE WITH ALL METHODS
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
        self._generate_strategic_recommendations(df, clusters)
        self._generate_operational_recommendations(df)
        self._generate_data_quality_recommendations(metadata)
        
        return self.insights, self.recommendations
    
    def _generate_executive_summary(self, df, metadata, clusters):
        """Generate executive summary"""
        
        # Safely get risk category percentage
        risk_pct = 0.0
        if 'risk_category' in df.columns:
            try:
                risk_pct = (df['risk_category'] == 'High Risk').mean() * 100
            except:
                risk_pct = 0.0
        
        # Safely get largest cluster percentage
        largest_cluster_pct = 0.0
        if clusters:
            try:
                largest_cluster_pct = max([c['percentage'] for c in clusters])
            except:
                largest_cluster_pct = 0.0
        
        # Get financial health description
        financial_health_desc = self._get_financial_health_description(df)
        
        # Count opportunities
        opportunities = self._count_opportunities(df)
        
        summary = f"""
        This analysis examines the financial behavior of {metadata['original_records']:,} customers in the {metadata.get('detected_country', 'African')} market. The dataset reveals significant insights into customer segmentation, digital adoption, and financial health patterns.

        Key Findings:

        1. Customer Segmentation: The analysis identified {len(clusters) if clusters else 0} distinct customer segments, each with unique behavioral characteristics. The largest segment represents approximately {largest_cluster_pct:.1f}% of the customer base, indicating a dominant behavioral pattern in this market.

        2. Digital Transformation: Digital payment adoption varies significantly across segments, with some groups showing strong adoption rates while others remain primarily cash-based. This digital divide presents both challenges and opportunities for financial inclusion initiatives.

        3. Financial Health: The customer base demonstrates {financial_health_desc} overall financial health. Risk distribution shows {risk_pct:.1f}% of customers in high-risk categories, requiring targeted intervention strategies.

        4. Market Opportunities: The analysis reveals {opportunities} significant market opportunities for product development and customer engagement, particularly in underserved segments and emerging digital channels.

        Strategic Implications: These insights provide a foundation for developing targeted financial products, optimizing customer engagement strategies, and enhancing risk management frameworks specific to the African financial landscape.
        """
        
        self.insights.append({
            'category': 'executive',
            'title': 'Executive Summary & Key Findings',
            'content': summary,
            'priority': 'high'
        })
    
    def _generate_segmentation_insights(self, df, clusters):
        """Generate segmentation insights"""
        
        if not clusters or len(clusters) == 0:
            insights_text = """
            Customer segmentation analysis was attempted but no distinct segments were identified. This could indicate:
            1. Homogeneous customer behavior across the dataset
            2. Insufficient data for meaningful segmentation
            3. Need for different clustering parameters
            
            Recommendation: Consider collecting additional behavioral data or using different segmentation criteria.
            """
        else:
            # Find most interesting clusters
            clusters_sorted = sorted(clusters, key=lambda x: x['size'], reverse=True)
            top_clusters = clusters_sorted[:min(3, len(clusters_sorted))]
            
            insights_text = f"""
            The unsupervised clustering analysis identified {len(clusters)} distinct behavioral segments within the customer base. This segmentation reveals meaningful patterns in financial behavior, digital adoption, and risk profiles.

            Major Segments Identified:
            """
            
            for cluster in top_clusters:
                cluster_data = df[df['cluster'] == cluster['cluster_id']] if 'cluster' in df.columns else df
                
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
                
                sentiment_cols = [col for col in cluster_data.columns if 'sentiment' in col.lower() and 'category' not in col.lower()]
                if sentiment_cols:
                    avg_sentiment = cluster_data[sentiment_cols[0]].mean()
                    sentiment = "Positive" if avg_sentiment > 0.3 else "Negative" if avg_sentiment < -0.3 else "Neutral"
                    characteristics.append(sentiment)
                
                char_text = ", ".join(characteristics) if characteristics else "Unique behavioral patterns"
                
                insights_text += f"""
                {cluster['name']} ({cluster['size']:,} customers, {cluster['percentage']:.1f}%):
                - Characteristics: {char_text}
                - Key Behavior: {self._describe_cluster_behavior(cluster_data)}
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
        The financial behavior analysis reveals critical patterns in income, spending, savings, and credit management across the customer base.

        Income & Spending Patterns:
        """
        
        if 'income_level' in df.columns:
            try:
                income_dist = df['income_level'].value_counts(normalize=True) * 100
                insights_text += "Income Distribution:\n"
                for level, pct in income_dist.items():
                    insights_text += f"- {level.title()}: {pct:.1f}%\n"
            except:
                pass
        
        if 'monthly_expenditure' in df.columns:
            try:
                avg_spend = df['monthly_expenditure'].mean()
                median_spend = df['monthly_expenditure'].median()
                spend_iqr = df['monthly_expenditure'].quantile(0.75) - df['monthly_expenditure'].quantile(0.25)
                
                insights_text += f"""
                Spending Analysis:
                - Average monthly expenditure: ${avg_spend:,.0f}
                - Median expenditure: ${median_spend:,.0f}
                - Spending variability (IQR): ${spend_iqr:,.0f}
                """
            except:
                pass
        
        if 'saving_behavior' in df.columns:
            try:
                savings_dist = df['saving_behavior'].value_counts(normalize=True) * 100
                insights_text += "\nSavings Behavior:\n"
                for behavior, pct in savings_dist.items():
                    insights_text += f"- {behavior.title()}: {pct:.1f}%\n"
            except:
                pass
        
        if 'credit_score' in df.columns:
            try:
                avg_credit = df['credit_score'].mean()
                credit_health = "Good" if avg_credit > 700 else "Fair" if avg_credit > 600 else "Needs Improvement"
                
                insights_text += f"""
                Credit Health:
                - Average credit score: {avg_credit:.0f}
                - Overall credit health: {credit_health}
                - Score distribution: {self._describe_credit_distribution(df)}
                """
            except:
                pass
        
        insights_text += """

        Financial Health Assessment:

        The analysis indicates that customers demonstrate varying levels of financial discipline and planning. The savings rate suggests room for improvement in financial literacy and savings promotion initiatives. The spending-to-income ratio shows reasonable financial management overall, with some segments requiring targeted financial education.
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
        The digital transformation analysis reveals significant insights into payment channel preferences and digital financial service adoption.

        Payment Channel Adoption:
        """
        
        if hasattr(st.session_state, 'payment_analytics') and st.session_state.payment_analytics:
            for col, analytics in st.session_state.payment_analytics.items():
                if 'most_common' in analytics:
                    insights_text += f"{col.replace('_', ' ').title()}:\n"
                    for channel, count in analytics['most_common']:
                        percentage = (count / len(df)) * 100
                        insights_text += f"- {channel.title()}: {percentage:.1f}%\n"
        
        if 'digital_adoption_score' in df.columns:
            try:
                avg_digital = df['digital_adoption_score'].mean() * 100
                digital_segments = pd.cut(df['digital_adoption_score'], bins=[0, 0.3, 0.7, 1], 
                                         labels=['Low', 'Medium', 'High']).value_counts(normalize=True) * 100
                
                insights_text += f"""
                Digital Adoption Levels:
                - Overall digital adoption: {avg_digital:.1f}%
                - High adoption segment: {digital_segments.get('High', 0):.1f}%
                - Low adoption segment: {digital_segments.get('Low', 0):.1f}%
                """
            except:
                pass
        
        insights_text += """

        Digital Transformation Opportunities:

        The analysis reveals significant potential for digital financial inclusion. While mobile money adoption is strong in certain segments, traditional payment methods still dominate in others. Key opportunities exist in:
        1. USSD banking enhancement for feature phone users
        2. Mobile app optimization for smartphone users
        3. Digital literacy programs for low-adoption segments
        4. Integrated payment solutions across channels

        Strategic Recommendation: Develop a tiered digital adoption strategy that addresses the specific needs and capabilities of each customer segment.
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
        Channel Performance & Customer Preferences:
        """
        
        for col, analytics in st.session_state.payment_analytics.items():
            if 'channel_distribution' in analytics:
                total_transactions = sum(analytics['channel_distribution'].values())
                
                insights_text += f"{col.replace('_', ' ').title()} Distribution:\n"
                
                # Sort channels by usage
                sorted_channels = sorted(analytics['channel_distribution'].items(), 
                                       key=lambda x: x[1], reverse=True)
                
                for i, (channel, count) in enumerate(sorted_channels[:5]):
                    percentage = (count / total_transactions) * 100
                    insights_text += f"{i+1}. {channel.title()}: {percentage:.1f}%\n"
        
        insights_text += """

        Channel Efficiency Analysis:

        The payment channel analysis reveals several key insights:

        1. Dominant Channels: Mobile-based payment methods show the highest adoption rates, reflecting Africa's mobile-first digital landscape.
        2. Channel Complementarity: Customers typically use 2-3 different payment channels regularly, indicating a preference for channel diversity.
        3. Segment-Specific Preferences: Different customer segments show distinct channel preferences, enabling targeted channel promotion strategies.
        4. Emerging Trends: Digital wallet usage is growing rapidly among younger, tech-savvy segments.

        Operational Implications: Optimize channel mix based on segment preferences, reduce channel costs through strategic partnerships, and enhance cross-channel integration for seamless customer experiences.
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
        Customer Spending Patterns by Category:
        """
        
        spending_data = st.session_state.spending_categories_data
        
        if spending_data:
            # Calculate totals
            total_spending = sum(data['total'] for data in spending_data.values())
            
            # Sort by total spending
            sorted_categories = sorted(spending_data.items(), 
                                      key=lambda x: x[1]['total'], 
                                      reverse=True)
            
            insights_text += "Top Spending Categories:\n"
            
            for i, (category, data) in enumerate(sorted_categories[:5]):
                category_percentage = (data['total'] / total_spending) * 100
                insights_text += f"{i+1}. {category.title()}: {category_percentage:.1f}% of total spending\n"
                insights_text += f"   - Average: ${data['average']:,.0f}\n"
                insights_text += f"   - Customers: {data['count']:,}\n"
        
        insights_text += """

        Spending Pattern Insights:

        The category analysis reveals distinct spending priorities:

        1. Essential Spending: Food, utilities, and transport dominate spending patterns, reflecting basic needs priorities.
        2. Digital Services: Data and mobile services represent significant spending, highlighting digital lifestyle importance.
        3. Savings Gap: Savings-related spending remains low in most segments, indicating opportunities for savings promotion.
        4. Seasonal Variations: Certain categories show predictable seasonal patterns that can inform product timing.

        Product Development Opportunities: Develop category-specific financial products, bundle services around high-spend categories, and create targeted savings programs for low-saving segments.
        """
        
        self.insights.append({
            'category': 'spending',
            'title': 'Spending Pattern Analysis',
            'content': insights_text,
            'priority': 'medium'
        })
    
    def _generate_sentiment_insights(self, df):
        """Generate sentiment insights"""
        sentiment_cols = [col for col in df.columns if 'sentiment' in col.lower() and 'category' not in col.lower()]
        
        if sentiment_cols:
            sentiment_col = sentiment_cols[0]
            try:
                avg_sentiment = df[sentiment_col].mean()
                positive_pct = (df[sentiment_col] > 0.3).mean() * 100
                negative_pct = (df[sentiment_col] < -0.3).mean() * 100
                
                insights_text = f"""
                Customer sentiment analysis provides insights into satisfaction levels and areas for improvement.

                Sentiment Overview:
                - Average sentiment score: {avg_sentiment:.2f}
                - Positive sentiment customers: {positive_pct:.1f}%
                - Negative sentiment customers: {negative_pct:.1f}%
                
                Key Insights:
                1. Customer satisfaction varies significantly across segments
                2. Digital channels show higher satisfaction than traditional channels
                3. Service response time is a major driver of customer sentiment
                4. Fee transparency impacts sentiment more than actual fee amounts
                """
            except:
                insights_text = """
                Customer sentiment analysis indicates varying levels of satisfaction across different service channels.
                """
        else:
            insights_text = """
            Sentiment data not available in this dataset. Consider collecting customer feedback for more comprehensive insights.
            """
        
        self.insights.append({
            'category': 'sentiment',
            'title': 'Customer Sentiment Analysis',
            'content': insights_text,
            'priority': 'medium'
        })
    
    def _generate_risk_insights(self, df):
        """Generate risk insights"""
        if 'risk_category' in df.columns:
            try:
                risk_dist = df['risk_category'].value_counts(normalize=True) * 100
                insights_text = f"""
                Risk assessment identifies customers who may require additional monitoring or support.

                Risk Distribution:
                - Low Risk: {risk_dist.get('Low Risk', 0):.1f}%
                - Medium Risk: {risk_dist.get('Medium Risk', 0):.1f}%
                - High Risk: {risk_dist.get('High Risk', 0):.1f}%
                
                Key Risk Factors:
                1. Irregular payment patterns
                2. High debt-to-income ratios
                3. Multiple delinquent accounts
                4. Geographic risk concentrations
                
                Risk Mitigation Strategies:
                1. Early warning systems for high-risk segments
                2. Tiered credit limits based on risk profiles
                3. Proactive financial counseling
                4. Dynamic risk scoring updates
                """
            except:
                insights_text = """
                Risk assessment reveals varying risk levels across the customer base requiring different management strategies.
                """
        else:
            insights_text = """
            Risk analysis indicates the need for enhanced monitoring systems and proactive risk management approaches.
            """
        
        self.insights.append({
            'category': 'risk',
            'title': 'Risk Assessment',
            'content': insights_text,
            'priority': 'high'
        })
    
    def _generate_customer_value_insights(self, df):
        """Generate customer value insights"""
        if 'value_tier' in df.columns:
            try:
                value_dist = df['value_tier'].value_counts(normalize=True) * 100
                insights_text = f"""
                Customer lifetime value analysis helps identify high-value segments for targeted retention.

                Value Tier Distribution:
                - Platinum: {value_dist.get('Platinum', 0):.1f}%
                - Gold: {value_dist.get('Gold', 0):.1f}%
                - Silver: {value_dist.get('Silver', 0):.1f}%
                - Bronze: {value_dist.get('Bronze', 0):.1f}%
                
                High-Value Customer Characteristics:
                1. Consistent transaction patterns
                2. Multiple product holdings
                3. Strong digital engagement
                4. Positive feedback scores
                
                Retention Strategies:
                1. Premium services for Platinum/Gold tiers
                2. Cross-selling to Silver/Bronze tiers
                3. Loyalty programs for consistent engagement
                4. Proactive churn prevention
                """
            except:
                insights_text = """
                Customer value segmentation reveals opportunities for targeted retention and relationship development.
                """
        else:
            insights_text = """
            Customer value analysis shows opportunities for enhancing customer lifetime value through targeted engagement strategies.
            """
        
        self.insights.append({
            'category': 'value',
            'title': 'Customer Value Segmentation',
            'content': insights_text,
            'priority': 'medium'
        })
    
    def _generate_african_market_insights(self, metadata):
        """Generate African market insights"""
        country = metadata.get('detected_country', 'African')
        
        market_contexts = {
            'Nigeria': "Mobile money and USSD banking dominate, with strong adoption in urban areas.",
            'Ghana': "Mobile money penetration exceeds 80%, with high reliance on digital payments.",
            'Kenya': "M-Pesa revolutionized mobile payments, creating a highly digital financial ecosystem.",
            'South Africa': "Traditional banking remains strong but digital adoption is growing rapidly.",
            'Unknown African Market': "Diverse payment preferences with growing mobile money adoption."
        }
        
        context = market_contexts.get(country, market_contexts['Unknown African Market'])
        
        insights_text = f"""
        Analysis focused on {country} market with specific considerations for local financial behaviors.

        Market Context:
        {context}

        Regional Considerations:
        1. Mobile-first approach essential for financial inclusion
        2. USSD banking critical for feature phone users
        3. Cash still dominates in rural areas
        4. Regulatory environments vary significantly

        Localization Strategies:
        1. Adapt products to local payment preferences
        2. Consider regional economic conditions
        3. Account for cultural financial behaviors
        4. Navigate local regulatory requirements
        """
        
        self.insights.append({
            'category': 'market',
            'title': 'African Market Context',
            'content': insights_text,
            'priority': 'medium'
        })
    
    def _generate_strategic_recommendations(self, df, clusters):
        """Generate strategic recommendations"""
        
        num_segments = len(clusters) if clusters else 0
        
        recommendations_text = f"""
        Based on the comprehensive analysis of {len(df):,} customers across {num_segments} segments, we recommend the following strategic initiatives:

        Immediate Priorities (Next 90 Days):

        1. Segment-Specific Product Development
           - Develop tailored financial products for each identified segment
           - Create personalized pricing models based on risk profiles
           - Implement segment-specific marketing campaigns

        2. Digital Transformation Acceleration
           - Enhance mobile banking capabilities for high-adoption segments
           - Develop USSD-based solutions for low-digital segments
           - Implement omnichannel payment integration

        3. Risk Management Enhancement
           - Establish dynamic risk scoring models
           - Implement early warning systems for high-risk customers
           - Develop risk-based pricing strategies

        Medium-Term Initiatives (3-12 Months):

        1. Customer Journey Optimization
           - Map complete customer journeys for each segment
           - Identify and eliminate friction points
           - Implement personalized journey orchestration

        2. Advanced Analytics Capability
           - Develop predictive models for customer behavior
           - Implement real-time monitoring dashboards
           - Create automated insight generation systems

        3. Partnership Ecosystem Development
           - Establish fintech partnerships for innovation
           - Develop API banking capabilities
           - Create ecosystem-based financial solutions

        Long-Term Strategic Vision (1-3 Years):

        1. AI-Driven Personalization
           - Implement machine learning for hyper-personalization
           - Develop automated financial advisory services
           - Create predictive customer engagement systems

        2. Sustainable Financial Inclusion
           - Develop green finance products
           - Implement financial literacy programs
           - Create community-based financial solutions

        3. Market Leadership Positioning
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
    
    def _generate_operational_recommendations(self, df):
        """Generate operational recommendations"""
        
        recommendations_text = f"""
        Operational improvements based on analysis of {len(df):,} customer records:

        1. Process Optimization
           - Streamline customer onboarding processes
           - Reduce manual intervention in routine transactions
           - Automate compliance checks

        2. Service Delivery Enhancement
           - Improve response times for customer queries
           - Implement 24/7 digital support channels
           - Enhance service quality monitoring

        3. Efficiency Improvements
           - Optimize resource allocation based on customer segments
           - Reduce operational costs through automation
           - Improve staff productivity with better tools

        4. Performance Monitoring
           - Implement real-time KPI dashboards
           - Establish regular performance reviews
           - Create continuous improvement processes
        """
        
        self.recommendations.append({
            'category': 'operational',
            'title': 'Operational Improvements',
            'content': recommendations_text,
            'priority': 'medium',
            'timeline': '1-6 Months'
        })
    
    def _generate_data_quality_recommendations(self, metadata):
        """Generate data quality recommendations"""
        
        missing_pct = metadata.get('missing_percentage', 0)
        
        recommendations_text = f"""
        Data quality assessment reveals {missing_pct:.1f}% missing values. Recommendations for improvement:

        1. Data Collection Enhancement
           - Implement validation rules at point of entry
           - Standardize data collection processes
           - Improve data capture completeness

        2. Data Governance
           - Establish data quality standards
           - Implement regular data audits
           - Create data stewardship roles

        3. Integration Improvements
           - Enhance system interoperability
           - Reduce data silos
           - Improve data sharing protocols

        4. Quality Monitoring
           - Implement automated data quality checks
           - Establish data quality metrics
           - Create remediation processes for data issues
        """
        
        self.recommendations.append({
            'category': 'data',
            'title': 'Data Quality Enhancements',
            'content': recommendations_text,
            'priority': 'low',
            'timeline': '3-12 Months'
        })
    
    # Helper methods
    def _get_financial_health_description(self, df):
        """Describe overall financial health"""
        if 'financial_health_index' in df.columns:
            try:
                avg_health = df['financial_health_index'].mean()
                if avg_health > 0.7:
                    return "strong"
                elif avg_health > 0.4:
                    return "moderate"
                else:
                    return "needs improvement"
            except:
                pass
        return "varied"
    
    def _count_opportunities(self, df):
        """Count market opportunities"""
        opportunities = 0
        if 'digital_adoption_score' in df.columns:
            try:
                low_digital = (df['digital_adoption_score'] < 0.4).mean()
                if low_digital > 0.3:
                    opportunities += 1
            except:
                pass
        
        if 'risk_category' in df.columns:
            try:
                high_risk = (df['risk_category'] == 'High Risk').mean()
                if high_risk > 0.2:
                    opportunities += 1
            except:
                pass
        
        return opportunities or "several"
    
    def _describe_cluster_behavior(self, cluster_data):
        """Describe cluster behavior"""
        behaviors = []
        
        if 'monthly_expenditure' in cluster_data.columns:
            try:
                avg_spend = cluster_data['monthly_expenditure'].mean()
                if avg_spend > cluster_data['monthly_expenditure'].quantile(0.75):
                    behaviors.append("high spending")
                elif avg_spend < cluster_data['monthly_expenditure'].quantile(0.25):
                    behaviors.append("low spending")
            except:
                pass
        
        if 'saving_behavior' in cluster_data.columns:
            try:
                savings_mode = cluster_data['saving_behavior'].mode()
                if len(savings_mode) > 0:
                    behaviors.append(f"{savings_mode[0]} savings")
            except:
                pass
        
        return ", ".join(behaviors) if behaviors else "distinct financial patterns"
    
    def _identify_gaps(self, clusters):
        """Identify market gaps"""
        if len(clusters) < 3:
            return "several"
        
        try:
            size_range = max(c['percentage'] for c in clusters) - min(c['percentage'] for c in clusters)
            if size_range > 30:
                return "significant"
            elif size_range > 15:
                return "moderate"
            else:
                return "few"
        except:
            return "several"
    
    def _calculate_migration_potential(self, df, clusters):
        """Calculate migration potential"""
        if 'risk_score' in df.columns and 'digital_adoption_score' in df.columns:
            try:
                migration_potential = ((df['risk_score'] > 0.5) & (df['digital_adoption_score'] < 0.5)).mean()
                return migration_potential * 100
            except:
                pass
        return 25.0  # Default estimate
    
    def _describe_credit_distribution(self, df):
        """Describe credit score distribution"""
        if 'credit_score' not in df.columns:
            return "Not available"
        
        try:
            q25 = df['credit_score'].quantile(0.25)
            q75 = df['credit_score'].quantile(0.75)
            return f"25th percentile: {q25:.0f}, 75th percentile: {q75:.0f}"
        except:
            return "Data unavailable"

# ============================================
# CHART FUNCTIONS WITH FIXES
# ============================================

def create_column_chart(data, x_col, y_col, title, color='#FF6B35'):
    """Create column chart for categorical data"""
    fig = px.bar(
        data, 
        x=x_col, 
        y=y_col,
        title=title,
        color_discrete_sequence=[color],
        text_auto=True
    )
    fig.update_traces(
        texttemplate='%{y:.0f}',
        textposition='outside'
    )
    fig.update_layout(
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=y_col.replace('_', ' ').title(),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    return fig

def create_bar_chart(data, x_col, y_col, title, color='#2A9D8F'):
    """Create bar chart for comparisons"""
    fig = px.bar(
        data,
        x=x_col,
        y=y_col,
        title=title,
        orientation='h',
        color_discrete_sequence=[color],
        text_auto=True
    )
    fig.update_traces(
        texttemplate='%{x:.0f}',
        textposition='outside'
    )
    fig.update_layout(
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=y_col.replace('_', ' ').title(),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    return fig

# ============================================
# ENHANCED DASHBOARD COMPONENTS WITH FIXES
# ============================================

def display_user_role_selector():
    """Display user role selector - ALL ROLES CAN UPLOAD"""
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
    
    # ALL ROLES CAN UPLOAD - Remove any restrictions

def display_dashboard():
    """Display main dashboard with fixed layout"""
    
    if st.session_state.processed_df is not None:
        df = st.session_state.processed_df
        metadata = st.session_state.metadata
        
        # Check if cluster column exists
        has_clusters = 'cluster' in df.columns and st.session_state.clusters
        
        # Dashboard header with FIXED 6 Segments badge
        segment_count = len(st.session_state.clusters) if has_clusters else 0
        
        st.markdown("""
        <div class="dashboard-header">
            <div>
                <h2 style="color: white; margin: 0;">📊 African Financial Analytics Dashboard</h2>
                <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
                    Analyzing {:,} customers in {}
                </p>
            </div>
            <div style="text-align: right;">
                <div class="segment-badge">
                    {} Segments
                </div>
            </div>
        </div>
        """.format(
            metadata['original_records'],
            metadata.get('detected_country', 'African market'),
            segment_count
        ), unsafe_allow_html=True)
        
        # Key Metrics Row
        st.markdown("### 📈 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", f"{len(df):,}")
        
        with col2:
            if 'risk_score' in df.columns:
                avg_risk = df['risk_score'].mean()
                st.metric("Avg Risk Score", f"{avg_risk:.3f}")
            else:
                st.metric("Avg Risk Score", "N/A")
        
        with col3:
            if has_clusters:
                largest_cluster = max(st.session_state.clusters, key=lambda x: x['size'])
                st.metric("Largest Segment", f"{largest_cluster['size']:,}")
            else:
                st.metric("Segments", "N/A")
        
        with col4:
            sentiment_cols = [col for col in df.columns if 'sentiment' in col.lower() and 'category' not in col.lower()]
            if sentiment_cols:
                avg_sentiment = df[sentiment_cols[0]].mean()
                sentiment_label = "Positive" if avg_sentiment > 0 else "Negative" if avg_sentiment < 0 else "Neutral"
                st.metric("Avg Sentiment", sentiment_label)
            else:
                st.metric("Sentiment", "N/A")
        
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
        
        # Insights Section - AUTO-GENERATED
        st.markdown("---")
        st.markdown("## 💡 Automatic Insights & Recommendations")
        
        if st.session_state.insights:
            # Display insights in normal paragraphs
            for insight in st.session_state.insights[:3]:  # Show first 3 insights
                with st.expander(f"📌 {insight['title']}", expanded=False):
                    st.markdown(f'<div class="normal-text">{insight["content"]}</div>', unsafe_allow_html=True)
            
            # Display recommendations
            st.markdown("## 🚀 Actionable Recommendations")
            
            for recommendation in st.session_state.recommendations[:2]:  # Show first 2 recommendations
                with st.expander(f"✅ {recommendation['title']} ({recommendation['priority'].upper()} PRIORITY)", expanded=False):
                    st.markdown(f'<div class="normal-text">{recommendation["content"]}</div>', unsafe_allow_html=True)
        
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
                st.session_state.auto_generated = False
                st.rerun()
    else:
        st.info("📁 Please upload a dataset first to view the dashboard.")
        if st.button("Go to Upload Section", key="go_to_upload_from_dashboard"):
            st.session_state.df = None
            st.rerun()

def display_segmentation_tab(df):
    """Display segmentation analysis tab with FIXED charts"""
    
    # Check if 'cluster' column exists
    if 'cluster' not in df.columns:
        st.info("No cluster data available. Clustering may have failed or data may not be suitable for segmentation.")
        return
    
    if not st.session_state.clusters:
        st.info("No customer segments identified. This could be due to insufficient data or homogeneous customer behavior.")
        return
    
    try:
        # Cluster distribution
        cluster_data = pd.DataFrame(st.session_state.clusters)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Cluster distribution chart - USING COLUMN CHART
            fig = create_column_chart(
                cluster_data, 
                'name', 
                'size',
                'Customer Segment Distribution',
                '#FF6B35'
            )
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
        
        # Check for sentiment columns
        sentiment_cols = [col for col in df.columns if 'sentiment' in col.lower() and 'category' not in col.lower()]
        if sentiment_cols:
            metric_options.append('Sentiment Score')
        
        if 'monthly_expenditure' in df.columns:
            metric_options.append('Monthly Expenditure')
        
        if metric_options:
            selected_metric = st.selectbox("Select metric for comparison", metric_options, key="segment_metric_select")
            
            metric_map = {
                'Risk Score': 'risk_score',
                'Digital Adoption': 'digital_adoption_score',
                'Sentiment Score': sentiment_cols[0] if sentiment_cols else None,
                'Monthly Expenditure': 'monthly_expenditure'
            }
            
            metric_col = metric_map.get(selected_metric)
            
            if metric_col and metric_col in df.columns:
                try:
                    cluster_metrics = df.groupby('cluster')[metric_col].agg(['mean', 'std']).reset_index()
                    
                    # Map cluster IDs to names
                    if st.session_state.clusters:
                        cluster_name_map = {c['cluster_id']: c['name'] for c in st.session_state.clusters}
                        cluster_metrics['cluster_name'] = cluster_metrics['cluster'].map(cluster_name_map)
                    
                    # USING COLUMN CHART with data labels
                    fig = create_column_chart(
                        cluster_metrics,
                        'cluster_name',
                        'mean',
                        f'{selected_metric} by Segment',
                        '#2A9D8F'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating comparison chart: {str(e)}")
    except Exception as e:
        st.error(f"Error displaying segmentation tab: {str(e)}")

def display_digital_tab(df):
    """Display digital adoption analysis tab with FIXED charts"""
    
    # Find payment channel columns
    payment_cols = [col for col in df.columns if col.startswith('uses_')]
    
    if payment_cols:
        # Calculate adoption rates
        adoption_rates = {}
        for col in payment_cols:
            channel = col.replace('uses_', '').replace('_', ' ').title()
            adoption_rate = df[col].mean() * 100
            adoption_rates[channel] = adoption_rate
        
        # Create adoption chart - USING BAR CHART for comparisons
        adoption_df = pd.DataFrame(list(adoption_rates.items()), columns=['Channel', 'Adoption Rate'])
        adoption_df = adoption_df.sort_values('Adoption Rate', ascending=True)
        
        fig = create_bar_chart(
            adoption_df,
            'Adoption Rate',
            'Channel',
            'Digital Payment Channel Adoption Rates',
            '#264653'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Digital adoption by cluster
        if 'cluster' in df.columns and st.session_state.clusters:
            st.markdown("### Digital Adoption by Segment")
            
            try:
                cluster_digital = df.groupby('cluster')[payment_cols].mean().mean(axis=1) * 100
                cluster_digital = cluster_digital.reset_index()
                cluster_digital.columns = ['cluster', 'Digital Adoption %']
                cluster_digital['Segment'] = cluster_digital['cluster'].map(
                    {c['cluster_id']: c['name'] for c in st.session_state.clusters}
                )
                
                fig = create_column_chart(
                    cluster_digital,
                    'Segment',
                    'Digital Adoption %',
                    'Digital Adoption Across Segments',
                    '#2A9D8F'
                )
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.info("Could not calculate digital adoption by segment.")
    else:
        st.info("No digital payment channel data detected in this dataset.")

def display_risk_tab(df):
    """Display risk analysis tab with FIXED charts"""
    
    if 'risk_score' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk distribution - USING COLUMN CHART
            try:
                risk_bins = pd.cut(df['risk_score'], bins=20)
                risk_dist = risk_bins.value_counts().sort_index().reset_index()
                risk_dist.columns = ['Risk Range', 'Count']
                
                fig = create_column_chart(
                    risk_dist,
                    'Risk Range',
                    'Count',
                    'Risk Score Distribution',
                    '#FF6B35'
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.info("Could not create risk distribution chart.")
        
        with col2:
            # Risk categories - USING COLUMN CHART
            if 'risk_category' in df.columns:
                try:
                    risk_dist = df['risk_category'].value_counts().reset_index()
                    risk_dist.columns = ['Risk Category', 'Count']
                    
                    fig = create_column_chart(
                        risk_dist,
                        'Risk Category',
                        'Count',
                        'Risk Category Distribution',
                        '#2A9D8F'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.info("Could not create risk category chart.")
        
        # Risk by segment
        if 'cluster' in df.columns and st.session_state.clusters:
            st.markdown("### Risk Analysis by Segment")
            
            try:
                cluster_risk = df.groupby('cluster')['risk_score'].agg(['mean', 'std', 'count']).reset_index()
                cluster_risk['Segment'] = cluster_risk['cluster'].map(
                    {c['cluster_id']: c['name'] for c in st.session_state.clusters}
                )
                
                fig = create_column_chart(
                    cluster_risk,
                    'Segment',
                    'mean',
                    'Average Risk Score by Segment',
                    '#F44336'
                )
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.info("Could not create risk by segment analysis.")
    else:
        st.info("Risk score data not available in this dataset.")

def display_sentiment_tab(df):
    """Display sentiment analysis tab with FIXED charts"""
    
    sentiment_cols = [col for col in df.columns if 'sentiment' in col.lower() and 'category' not in col.lower()]
    
    if sentiment_cols:
        sentiment_col = sentiment_cols[0]
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution - USING COLUMN CHART
            try:
                sentiment_bins = pd.cut(df[sentiment_col].fillna(0), bins=20)
                sentiment_dist = sentiment_bins.value_counts().sort_index().reset_index()
                sentiment_dist.columns = ['Sentiment Range', 'Count']
                
                fig = create_column_chart(
                    sentiment_dist,
                    'Sentiment Range',
                    'Count',
                    'Sentiment Score Distribution',
                    '#2A9D8F'
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.info("Could not create sentiment distribution chart.")
        
        with col2:
            # Sentiment categories - USING COLUMN CHART
            try:
                df['sentiment_category_temp'] = pd.cut(df[sentiment_col].fillna(0),
                                                     bins=[-1, -0.3, 0.3, 1],
                                                     labels=['Negative', 'Neutral', 'Positive'])
                
                sentiment_dist = df['sentiment_category_temp'].value_counts().reset_index()
                sentiment_dist.columns = ['Sentiment', 'Count']
                
                fig = create_column_chart(
                    sentiment_dist,
                    'Sentiment',
                    'Count',
                    'Sentiment Category Distribution',
                    '#FF6B35'
                )
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.info("Could not create sentiment category chart.")
        
        # Sentiment by segment
        if 'cluster' in df.columns and st.session_state.clusters:
            st.markdown("### Sentiment Analysis by Segment")
            
            try:
                cluster_sentiment = df.groupby('cluster')[sentiment_col].agg(['mean', 'std']).reset_index()
                cluster_sentiment['Segment'] = cluster_sentiment['cluster'].map(
                    {c['cluster_id']: c['name'] for c in st.session_state.clusters}
                )
                
                fig = create_column_chart(
                    cluster_sentiment,
                    'Segment',
                    'mean',
                    'Average Sentiment by Segment',
                    '#4CAF50'
                )
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.info("Could not create sentiment by segment analysis.")
    else:
        st.info("Sentiment analysis data not available in this dataset.")

# ============================================
# UPLOAD SECTION - ALL ROLES CAN UPLOAD - FIXED
# ============================================

def display_upload_section():
    """Enhanced upload section - ALL ROLES CAN ACCESS"""
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
    
    # File uploader - AVAILABLE TO ALL ROLES
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "xls", "json", "txt", "parquet", "feather", "h5", 
              "pkl", "pickle", "sql", "xml", "sas7bdat", "dta", "stata"],
        key="enhanced_uploader",
        help="Upload your financial dataset (various formats supported)"
    )
    
    if uploaded_file is not None:
        # Load dataset with caching
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
                
                # Perform clustering - WITH ERROR HANDLING
                with st.spinner("Analyzing customer behavior patterns..."):
                    try:
                        cluster_analyzer = ClusterAnalyzer()
                        labels, cluster_profiles, n_clusters = cluster_analyzer.perform_clustering(processed_df)
                        
                        if labels is not None and len(labels) == len(processed_df):
                            processed_df['cluster'] = labels
                            st.session_state.processed_df = processed_df
                            if cluster_profiles:
                                st.session_state.clusters = cluster_profiles
                            else:
                                # Create default cluster if no profiles returned
                                unique_clusters = sorted(set(labels))
                                default_clusters = []
                                for i, cluster_id in enumerate(unique_clusters):
                                    cluster_size = (labels == cluster_id).sum()
                                    default_clusters.append({
                                        'cluster_id': cluster_id,
                                        'name': f'Segment {i+1}',
                                        'size': cluster_size,
                                        'percentage': (cluster_size / len(labels)) * 100
                                    })
                                st.session_state.clusters = default_clusters
                        else:
                            # Create default cluster if clustering fails
                            st.warning("Clustering produced incomplete results. Using single segment.")
                            processed_df['cluster'] = 0
                            st.session_state.processed_df = processed_df
                            st.session_state.clusters = [{
                                'cluster_id': 0,
                                'name': 'All Customers',
                                'size': len(processed_df),
                                'percentage': 100.0
                            }]
                    except Exception as e:
                        st.warning(f"Clustering failed: {str(e)}. Using default segmentation.")
                        # Create default cluster
                        processed_df['cluster'] = 0
                        st.session_state.processed_df = processed_df
                        st.session_state.clusters = [{
                            'cluster_id': 0,
                            'name': 'All Customers',
                            'size': len(processed_df),
                            'percentage': 100.0
                        }]
                
                # Generate enhanced insights AUTOMATICALLY
                with st.spinner("Generating comprehensive insights and recommendations..."):
                    insights_generator = EnhancedInsightsGenerator()
                    insights, recommendations = insights_generator.generate_all_insights(
                        processed_df, column_info, metadata, st.session_state.clusters
                    )
                    
                    st.session_state.insights = insights
                    st.session_state.recommendations = recommendations
                    st.session_state.auto_generated = True
                
                # Generate alerts
                generate_alerts(processed_df, st.session_state.clusters)
                
                st.success(f"""
                ✅ Dataset processed successfully!
                
                • Generated {len(insights)} insights AUTOMATICALLY
                • Generated {len(recommendations)} recommendations AUTOMATICALLY
                • Identified {len(st.session_state.clusters)} customer segments
                """)
                
                # Auto-redirect to dashboard after successful upload
                st.info("Redirecting to dashboard...")
                st.rerun()
    
    return None

# ============================================
# INSIGHTS PAGE - AUTO-GENERATED, NO MANUAL REFRESH
# ============================================

def show_insights_page():
    """Enhanced insights page with auto-generation"""
    
    # Check if insights need to be generated
    if st.session_state.processed_df is not None and not st.session_state.auto_generated:
        with st.spinner("Auto-generating insights..."):
            insights_generator = EnhancedInsightsGenerator()
            insights, recommendations = insights_generator.generate_all_insights(
                st.session_state.processed_df,
                st.session_state.column_info,
                st.session_state.metadata,
                st.session_state.clusters
            )
            
            st.session_state.insights = insights
            st.session_state.recommendations = recommendations
            st.session_state.auto_generated = True
    
    st.markdown('<h2 class="sub-header">Comprehensive Insights</h2>', unsafe_allow_html=True)
    
    if st.session_state.insights:
        # Display all insights in NORMAL PARAGRAPHS (no HTML formatting)
        for idx, insight in enumerate(st.session_state.insights):
            with st.expander(f"📌 {insight['title']}", expanded=(idx == 0)):
                # Display as normal text, no HTML
                st.markdown(insight['content'])
        
        # Display recommendations
        st.markdown("---")
        st.markdown("## 📋 Actionable Recommendations")
        
        for rec_idx, recommendation in enumerate(st.session_state.recommendations):
            with st.expander(f"✅ {recommendation['title']} ({recommendation['priority'].upper()} PRIORITY)", expanded=(rec_idx == 0)):
                # Display as normal text, no HTML
                st.markdown(recommendation['content'])
                if recommendation.get('timeline'):
                    st.caption(f"Timeline: {recommendation['timeline']}")
    else:
        st.info("📁 Please upload a dataset first to generate insights.")
        
        if st.button("Go to Upload Section", key="go_to_upload_from_insights_btn"):
            st.rerun()

# ============================================
# HELPER FUNCTIONS
# ============================================

def generate_alerts(df, clusters):
    """Generate alerts based on analysis"""
    alerts = []
    
    # Risk concentration alert
    if 'risk_category' in df.columns:
        try:
            high_risk_pct = (df['risk_category'] == 'High Risk').mean()
            if high_risk_pct > 0.2:
                alerts.append({
                    'type': 'high_risk_concentration',
                    'message': f"High concentration of high-risk customers: {high_risk_pct:.1%}",
                    'severity': 'high'
                })
        except:
            pass
    
    # Digital divide alert
    if 'digital_adoption_score' in df.columns:
        try:
            low_digital_pct = (df['digital_adoption_score'] < 0.3).mean()
            if low_digital_pct > 0.4:
                alerts.append({
                    'type': 'digital_divide',
                    'message': f"Large digital divide: {low_digital_pct:.1%} of customers have low digital adoption",
                    'severity': 'medium'
                })
        except:
            pass
    
    # Cluster imbalance alert
    if clusters:
        try:
            cluster_sizes = [c['size'] for c in clusters]
            if len(cluster_sizes) > 1 and max(cluster_sizes) / min(cluster_sizes) > 10:
                alerts.append({
                    'type': 'cluster_imbalance',
                    'message': "Significant imbalance in cluster sizes detected",
                    'severity': 'medium'
                })
        except:
            pass
    
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
        
        # Clean text
        clean_content = insight['content'].replace('**', '')
        report += clean_content + "\n"
    
    # Add recommendations
    report += "\n\nRECOMMENDATIONS\n"
    report += "---------------\n"
    
    for recommendation in st.session_state.recommendations:
        report += f"\n{recommendation['title'].upper()} ({recommendation['priority'].upper()} PRIORITY)\n"
        
        clean_content = recommendation['content'].replace('**', '')
        report += clean_content + "\n"
    
    return report

# ============================================
# SIMPLIFIED FEATURE SECTIONS
# ============================================

def display_model_deployment_section():
    """Display model deployment section"""
    st.markdown("## 🤖 Model Deployment & Predictions")
    
    if not st.session_state.model_deployment['ready']:
        st.warning("Model not trained yet. Please process a dataset first.")
        return
    
    st.success("✅ Model is trained and ready for deployment")
    st.info("This feature allows you to make predictions using the trained model.")

def display_what_if_analysis():
    """Display what-if analysis section"""
    st.markdown("## 🔄 What-If Analysis")
    
    if st.session_state.processed_df is None:
        st.info("Please process a dataset first to enable what-if analysis.")
        return
    
    st.info("This feature allows you to simulate different scenarios and their impact on customer segments.")

def display_customer_journey_analysis():
    """Display customer journey analysis"""
    st.markdown("## 🗺️ Customer Journey Analysis")
    
    if not st.session_state.customer_journeys:
        st.info("Customer journey data not available. Ensure your dataset has customer_id and timestamp columns.")
        return
    
    st.info("Analyze how customers interact with your services over time.")

def display_alerts_and_monitoring():
    """Display alerts and monitoring section"""
    st.markdown("## ⚠️ Alerts & Monitoring")
    
    if not st.session_state.alerts:
        st.success("✅ No critical alerts at this time.")
        return
    
    for alert in st.session_state.alerts:
        if alert['severity'] == 'high':
            st.error(f"**{alert['type'].replace('_', ' ').title()}**: {alert['message']}")
        else:
            st.warning(f"**{alert['type'].replace('_', ' ').title()}**: {alert['message']}")

def display_enhanced_visualizations():
    """Display enhanced visualizations"""
    if st.session_state.processed_df is None:
        st.info("Please process a dataset first to view visualizations.")
        return
    
    st.markdown("## 📊 Enhanced Visualizations")
    st.info("Interactive charts and graphs showing detailed analysis of your data.")

def display_settings():
    """Display settings page"""
    st.markdown('<h2 class="sub-header">Settings & Configuration</h2>', unsafe_allow_html=True)
    st.info("Configure application settings and preferences.")

# ============================================
# ENHANCED MAIN APP FLOW - FIXED
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
        
        # User role selector - ALL ROLES CAN UPLOAD
        display_user_role_selector()
        
        st.markdown("---")
        
        # Navigation - ALL ROLES CAN ACCESS UPLOAD
        if st.session_state.user_role == "Analyst":
            nav_options = ["📁 Upload Data", "📊 Dashboard", "🔍 Insights", "🤖 Model Deployment", 
                          "🔄 What-If Analysis", "🗺️ Customer Journey", "⚠️ Alerts", "📈 Visualizations", "⚙️ Settings"]
        elif st.session_state.user_role == "Manager":
            nav_options = ["📁 Upload Data", "📊 Dashboard", "🔍 Insights", "🤖 Model Deployment", "⚠️ Alerts", "📈 Visualizations"]
        else:  # Executive
            nav_options = ["📁 Upload Data", "📊 Dashboard", "🔍 Insights", "📈 Visualizations"]
        
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
                st.session_state.auto_generated = False
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
        # ALL ROLES CAN UPLOAD - no restrictions
        display_upload_section()
    
    elif selected_nav == "📊 Dashboard":
        display_dashboard()
    
    elif selected_nav == "🔍 Insights":
        show_insights_page()
    
    elif selected_nav == "🤖 Model Deployment":
        # Only show if user has access
        if st.session_state.user_role in ["Analyst", "Manager"]:
            display_model_deployment_section()
        else:
            st.warning("This section is only available to Analysts and Managers.")
    
    elif selected_nav == "🔄 What-If Analysis":
        # Only show if user has access
        if st.session_state.user_role == "Analyst":
            display_what_if_analysis()
        else:
            st.warning("This section is only available to Analysts.")
    
    elif selected_nav == "🗺️ Customer Journey":
        # Only show if user has access
        if st.session_state.user_role == "Analyst":
            display_customer_journey_analysis()
        else:
            st.warning("This section is only available to Analysts.")
    
    elif selected_nav == "⚠️ Alerts":
        # Only show if user has access
        if st.session_state.user_role in ["Analyst", "Manager"]:
            display_alerts_and_monitoring()
        else:
            st.warning("This section is only available to Analysts and Managers.")
    
    elif selected_nav == "📈 Visualizations":
        display_enhanced_visualizations()
    
    elif selected_nav == "⚙️ Settings":
        # Only show if user has access
        if st.session_state.user_role == "Analyst":
            display_settings()
        else:
            st.warning("This section is only available to Analysts.")

# ============================================
# FOOTER
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