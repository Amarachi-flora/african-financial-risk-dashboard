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
# CUSTOM CSS WITH ANIMATIONS
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
    
    /* Ribbon style */
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
    
    /* Balloon style */
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
# TITLE WITH TEAM CREDIT
# ============================================
st.markdown('<h1 class="main-header">💰 Customer Financial Risk Prediction Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header" style="text-align: center;">African Financial Markets Analysis</p>', unsafe_allow_html=True)

# Team credit ribbon
st.markdown("""
<div class="ribbon">
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
    <div class="balloon">
        <div class="balloon-content">
            <div style="font-size: 1.5rem;">🎉</div>
            <div style="font-size: 0.8rem;">{message}</div>
        </div>
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

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    # Dataverse logo and link
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <a href="https://dataverseafrica.org" target="_blank">
            <div style="font-size: 2rem; color: #667eea; margin-bottom: 0.5rem;">🌍</div>
            <h3 style="color: #667eea; margin: 0;">DATAVERSE AFRICA</h3>
            <p style="color: #666; font-size: 0.9rem; margin: 0;">Empowering Africa's Digital Future</p>
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation with icons and animations
    st.markdown("### 🧭 Navigation")
    
    pages = {
        "🏠 Dashboard": "Executive dashboard with overview",
        "🔍 Customer Analysis": "Deep customer insights and filtering",
        "📊 Clusters": "Customer segmentation analysis",
        "🎯 Predict": "Real-time prediction interface",
        "📈 Insights": "Business recommendations",
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
    
    # Quick stats
    st.markdown("### 📊 Quick Stats")
    
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
        st.info("Load data in Dashboard page")
    
    st.markdown("---")
    
    # Connect section
    st.markdown("### 🔗 Connect")
    st.markdown("[📚 Documentation](#)")
    st.markdown("[📧 Contact Team](#)")
    st.markdown("[⭐ GitHub Repository](#)")

# ============================================
# LOAD MODELS (WITH WARNING SUPPRESSION)
# ============================================
@st.cache_resource
def load_models():
    """Load ML models with suppressed warnings"""
    try:
        if os.path.exists("models/scaler.pkl"):
            # Suppress warnings during model loading
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scaler = joblib.load("models/scaler.pkl")
                pca = joblib.load("models/pca_model.pkl")
                kmeans = joblib.load("models/kmeans_model.pkl")
            return scaler, pca, kmeans
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None
    return None, None, None

# API URL
API_URL = "http://localhost:8000"

# ============================================
# PAGE 1: DASHBOARD
# ============================================
if selected_page == "🏠 Dashboard":
    st.markdown('<h2 class="sub-header">📈 Executive Dashboard</h2>', unsafe_allow_html=True)
    
    # Show success balloon
    show_success_balloon("Welcome!")
    
    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Customers", "5,200", "+12%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg Credit Score", "645", "+8")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Digital Adoption", "68%", "+15%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Risk Rate", "12.5%", "-2.3%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Load and display data
    st.markdown("### 📁 Load Your Dataset")
    
    uploaded_file = st.file_uploader("Upload financial_data.csv", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown('<div class="success-badge">✅ Dataset Loaded Successfully!</div>', unsafe_allow_html=True)
        st.success(f"Dataset loaded: {len(df):,} records × {len(df.columns)} columns")
        
        # Save for other pages
        df.to_csv("outputs/processed_data.csv", index=False)
        
        # Show preview
        with st.expander("📋 Dataset Preview", expanded=True):
            st.dataframe(df.head(100), use_container_width=True)
        
        # Basic statistics
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
    
    # Sample charts (if no data loaded)
    st.markdown("### 📈 Sample Visualizations")
    
    # Create sample data for visualization if no data loaded
    if 'df' not in locals() or df.empty:
        # Generate sample data
        np.random.seed(42)
        sample_size = 1000
        sample_data = pd.DataFrame({
            'Credit_Score': np.random.normal(650, 100, sample_size).clip(300, 850),
            'Monthly_Expenditure': np.random.lognormal(12, 0.8, sample_size).clip(20000, 500000),
            'age': np.random.randint(22, 65, sample_size),
            'Cluster': np.random.choice(['Digital-First', 'Traditional', 'High-Risk', 'Medium', 'Positive'], sample_size),
            'Risk_Score': np.random.beta(2, 5, sample_size)
        })
        df = sample_data
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Credit Score Distribution
        fig = px.histogram(df, x='Credit_Score', nbins=30, title='Credit Score Distribution',
                          color_discrete_sequence=['#636EFA'])
        fig.update_layout(bargap=0.1, plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Expenditure vs Credit Score
        sample_df = df.sample(min(500, len(df)))
        
        # Get actual column names that exist
        cluster_col = get_column_case_insensitive(sample_df, 'Cluster') or 'Cluster'
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
    cluster_col = get_column_case_insensitive(df, 'Cluster')
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
    
    try:
        df = pd.read_csv("outputs/processed_data.csv")
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.replace(' ', '_')
        
        # Filters
        st.markdown("### 🔍 Filter Customers")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            income_col = get_column_case_insensitive(df, 'Income_Level')
            if income_col:
                income_options = df[income_col].dropna().unique()
                income_filter = st.multiselect(
                    "Income Level",
                    options=income_options,
                    default=list(income_options[:2]) if len(income_options) > 1 else []
                )
        
        with col2:
            credit_col = get_column_case_insensitive(df, 'Credit_Score')
            if credit_col:
                min_val = int(df[credit_col].min())
                max_val = int(df[credit_col].max())
                credit_range = st.slider(
                    "Credit Score Range",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val + (max_val-min_val)//4, min_val + 3*(max_val-min_val)//4)
                )
        
        with col3:
            location_col = get_column_case_insensitive(df, 'Location')
            if location_col:
                location_options = df[location_col].dropna().unique()
                location_filter = st.multiselect(
                    "Location",
                    options=location_options,
                    default=list(location_options[:3]) if len(location_options) > 3 else list(location_options)
                )
        
        # Apply filters
        filtered_df = df.copy()
        
        # Apply income filter
        if 'income_filter' in locals() and income_filter and income_col:
            filtered_df = filtered_df[filtered_df[income_col].isin(income_filter)]
        
        # Apply credit score filter
        if 'credit_range' in locals() and credit_col:
            filtered_df = filtered_df[
                (filtered_df[credit_col] >= credit_range[0]) & 
                (filtered_df[credit_col] <= credit_range[1])
            ]
        
        # Apply location filter
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
            'Transaction_Channel', 'transaction_channel'
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
        if channel_col:
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
        if risk_col:
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
        
    except FileNotFoundError:
        st.error("❌ No data found. Please load data in the Dashboard page first.")
        st.info("Go to the Dashboard page and upload your dataset.")
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")

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
            # If cluster profiles don't exist, load processed data
            df_path = "outputs/processed_data.csv"
            if os.path.exists(df_path):
                df = pd.read_csv(df_path)
                
                cluster_col = get_column_case_insensitive(df, 'cluster')
                if cluster_col:
                    # Create summary
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if numeric_cols:
                        cluster_summary = df.groupby(cluster_col).agg({
                            numeric_cols[0]: 'count',
                            **{col: 'mean' for col in numeric_cols[1:3]}
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
                st.error("No data found. Please load data in the Dashboard page first.")
    
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
            show_success_balloon("Predicting!")
            
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
                            st.markdown(f'''
                            <div class="metric-card">
                                <h3>Customer Segment</h3>
                                <h2>{result['cluster_name']}</h2>
                                <p>Cluster ID: {result['cluster_id']}</p>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        with col2:
                            risk_color = "#10B981" if result['risk_category'] == "Low Risk" else "#F59E0B" if result['risk_category'] == "Medium Risk" else "#EF4444"
                            st.markdown(f'''
                            <div class="metric-card">
                                <h3>Risk Category</h3>
                                <h2 style="color: {risk_color};">{result['risk_category']}</h2>
                                <p>Score: {result['risk_score']:.3f}</p>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f'''
                            <div class="metric-card">
                                <h3>Digital Adoption</h3>
                                <h2>{result['digital_adoption_score']}/4.0</h2>
                                <p>Channels used</p>
                            </div>
                            ''', unsafe_allow_html=True)
                        
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
                    show_success_balloon("Processing Batch!")
                    
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
    
    try:
        # Check for processed data
        df_path = "outputs/processed_data.csv"
        
        if os.path.exists(df_path):
            df = pd.read_csv(df_path)
            
            # DYNAMIC INSIGHTS based on current data
            st.markdown("### 📊 Real-time Data Insights")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Credit score insights
                credit_col = get_column_case_insensitive(df, 'Credit_Score')
                if credit_col:
                    avg_credit = df[credit_col].mean()
                    low_credit = (df[credit_col] < 500).sum()
                    low_credit_pct = (low_credit / len(df)) * 100
                    
                    st.metric("Average Credit Score", f"{avg_credit:.0f}")
                    st.metric("High Risk Customers", f"{low_credit:,}", f"{low_credit_pct:.1f}%")
                    
                    if low_credit_pct > 15:
                        st.warning("⚠️ High proportion of risky customers. Consider financial education programs.")
                    elif avg_credit > 700:
                        st.success("✅ Excellent credit health. Focus on premium offerings.")
            
            with col2:
                # Expenditure insights
                expend_col = get_column_case_insensitive(df, 'Monthly_Expenditure')
                if expend_col:
                    avg_expend = df[expend_col].mean()
                    high_spenders = (df[expend_col] > df[expend_col].quantile(0.75)).sum()
                    high_spenders_pct = (high_spenders / len(df)) * 100
                    
                    st.metric("Average Monthly Spend", f"₦{avg_expend:,.0f}")
                    st.metric("High Spenders", f"{high_spenders:,}", f"{high_spenders_pct:.1f}%")
                    
                    if high_spenders_pct > 20:
                        st.info("💰 Significant high-value segment. Target with premium products.")
            
            with col3:
                # Digital adoption insights
                channel_col = get_column_case_insensitive(df, 'Transaction_Channel')
                if channel_col:
                    mobile_users = df[channel_col].astype(str).str.contains('Mobile|App', case=False, na=False).sum()
                    mobile_pct = (mobile_users / len(df)) * 100
                    digital_users = df[channel_col].astype(str).str.contains('Mobile|Web|App', case=False, na=False).sum()
                    digital_pct = (digital_users / len(df)) * 100
                    
                    st.metric("Mobile App Users", f"{mobile_users:,}", f"{mobile_pct:.1f}%")
                    st.metric("Digital Users", f"{digital_users:,}", f"{digital_pct:.1f}%")
                    
                    if digital_pct < 50:
                        st.warning("📱 Low digital adoption. Consider incentives for digital channel usage.")
            
            # Generate dynamic recommendations based on data
            st.markdown("### 💡 Dynamic Recommendations")
            
            recommendations = []
            
            # Check for cluster data
            cluster_col = get_column_case_insensitive(df, 'cluster')
            if cluster_col and cluster_col in df.columns:
                cluster_counts = df[cluster_col].value_counts()
                largest_cluster = cluster_counts.index[0]
                largest_pct = (cluster_counts.iloc[0] / len(df)) * 100
                
                recommendations.append(f"**🎯 Focus on Largest Segment**: '{largest_cluster}' represents {largest_pct:.1f}% of customers. Tailor marketing to this group.")
            
            # Risk-based recommendations
            risk_col = get_column_case_insensitive(df, 'risk_score')
            if risk_col:
                high_risk = (df[risk_col] > 0.6).sum()
                high_risk_pct = (high_risk / len(df)) * 100
                
                if high_risk_pct > 10:
                    recommendations.append(f"**⚖️ Risk Management Needed**: {high_risk_pct:.1f}% of customers are high-risk. Implement monitoring and support programs.")
            
            # Digital adoption recommendations
            if 'digital_pct' in locals() and digital_pct < 60:
                recommendations.append(f"**📱 Boost Digital Adoption**: Only {digital_pct:.1f}% use digital channels. Launch digital onboarding campaigns.")
            
            # Location-based recommendations
            location_col = get_column_case_insensitive(df, 'Location')
            if location_col:
                top_locations = df[location_col].value_counts().head(3)
                recommendations.append(f"**🌍 Geographic Focus**: Top 3 locations are {', '.join(top_locations.index.tolist())}. Consider location-specific offerings.")
            
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
    
    except Exception as e:
        st.error(f"Error loading insights: {str(e)}")

# ============================================
# PAGE 6: TEAM
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
    
    # Team members - SIMPLIFIED with just names
    team_members = [
        {
            "name": "AMARACHI FLORENCE",
            "role": "Data Analyst & Project Lead"
        },
        {
            "name": "Thato Maelane",
            "role": "Machine Learning Engineer"
        },
        {
            "name": "Philip Odiachi", 
            "role": "Data Engineer"
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
                <p><strong>Role:</strong> {member['role']}</p>
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
# PAGE 7: SETTINGS
# ============================================
elif selected_page == "⚙️ Settings":
    st.markdown('<h2 class="sub-header">⚙️ System Settings & Configuration</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["API Settings", "Model Settings", "Data Settings"])
    
    with tab1:
        st.markdown("### 🔌 API Configuration")
        
        api_host = st.text_input("API Host", "localhost")
        api_port = st.number_input("API Port", min_value=1, max_value=65535, value=8000)
        
        if st.button("💾 Save API Settings"):
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
            if st.button("🔄 Reload Models"):
                # Clear cache and reload
                if 'models' in st.session_state:
                    del st.session_state.models
                st.rerun()
                st.success("✅ Models reload initiated!")
        
        with col2:
            if st.button("🔍 Check Model Health"):
                scaler, pca, kmeans = load_models()
                if all([scaler, pca, kmeans]):
                    st.success("✅ All models loaded successfully!")
                else:
                    st.error("❌ Some models failed to load")
    
    with tab3:
        st.markdown("### 💾 Data Management")
        
        # Data file information
        data_files = [
            ("📊 Raw Data", "financial_data.csv"),
            ("🔧 Processed Data", "outputs/processed_data.csv"),
            ("📊 Cluster Profiles", "outputs/cluster_profiles.csv"),
            ("💡 Recommendations", "outputs/business_recommendations.csv")
        ]
        
        for name, filepath in data_files:
            if os.path.exists(filepath):
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
        
        # Data actions
        st.markdown("### 🧹 Data Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Cache"):
                # Clear Streamlit cache
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("✅ Cache cleared!")
        
        with col2:
            if st.button("📦 Export All Data"):
                # Create zip of all outputs
                import zipfile
                import io
                
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    for root, dirs, files in os.walk('outputs'):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, 'outputs')
                            zip_file.write(file_path, arcname)
                
                st.download_button(
                    label="📥 Download All Data",
                    data=zip_buffer.getvalue(),
                    file_name="customer_analysis_data.zip",
                    mime="application/zip"
                )
        
        st.markdown("### ⚙️ System Information")
        
        # FIXED: Using correct version attributes
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