"""
AI Data Analyst - A Streamlit Web App for Automated Data Analysis
=================================================================

This application allows users to upload CSV datasets and automatically
performs comprehensive data analysis including cleaning, visualization,
and insight generation.

Author: AI Developer
Version: 1.0.0

Installation Instructions:
-------------------------
1. Install the required libraries:
   pip install streamlit pandas numpy matplotlib seaborn plotly scikit-learn

2. Run the Streamlit app:
   streamlit run ai_data_analyst.py

Requirements:
- Python 3.8 or higher
- Streamlit
- Pandas, NumPy
- Matplotlib, Seaborn, Plotly
- Scikit-learn (for predictions)
"""

# ============================================================================
# IMPORT LIBRARIES
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main App Styling */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        min-height: 100vh;
    }
    
    /* Title Styling */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
    }
    
    h1 {
        background: linear-gradient(90deg, #00d9ff, #a855f7, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700 !important;
    }
    
    h2 {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #6366f1;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    h3 {
        color: #94a3b8 !important;
        font-weight: 500 !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f1f5f9 !important;
    }
    
    /* DataFrame Styling */
    .dataframe {
        font-family: 'Inter', sans-serif;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #00d9ff !important;
        font-weight: 600 !important;
        font-size: 1.5rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }
    
    /* Cards */
    .card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    }
    
    /* Info/Success/Warning Boxes */
    .stAlert {
        border-radius: 12px !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #6366f1, #8b5cf6);
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_data(file):
    """Load data from uploaded CSV file."""
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


def get_column_types(df):
    """Categorize columns into numerical and categorical."""
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
    
    return {
        'numerical': numerical_cols,
        'categorical': categorical_cols,
        'datetime': datetime_cols
    }


def detect_missing_values(df):
    """Detect and analyze missing values."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
    return missing_df


def detect_duplicates(df):
    """Detect duplicate rows."""
    duplicates = df.duplicated().sum()
    return duplicates


def suggest_data_cleaning(df, missing_df, duplicates):
    """Generate data cleaning suggestions."""
    suggestions = []
    
    # Missing values suggestions
    if len(missing_df) > 0:
        for col in missing_df.index:
            pct = missing_df.loc[col, 'Missing %']
            if df[col].dtype in ['int64', 'float64']:
                if pct < 10:
                    suggestions.append(f"• **{col}**: Fill missing values with mean/median (only {pct:.1f}% missing)")
                else:
                    suggestions.append(f"• **{col}**: Consider dropping this column ({pct:.1f}% missing)")
            else:
                if pct < 10:
                    suggestions.append(f"• **{col}**: Fill missing values with mode or 'Unknown' ({pct:.1f}% missing)")
                else:
                    suggestions.append(f"• **{col}**: Consider dropping rows with missing values ({pct:.1f}% missing)")
    else:
        suggestions.append("• No missing values detected! Your data is clean.")
    
    # Duplicates suggestions
    if duplicates > 0:
        suggestions.append(f"• **{duplicates} duplicate rows** detected. Consider removing them for accurate analysis.")
    else:
        suggestions.append("• No duplicate rows detected.")
    
    return suggestions


def calculate_statistics(df, numerical_cols):
    """Calculate basic statistics for numerical columns."""
    if numerical_cols:
        return df[numerical_cols].describe()
    return pd.DataFrame()


def generate_correlation_heatmap(df, numerical_cols):
    """Generate correlation heatmap for numerical columns."""
    if len(numerical_cols) > 1:
        corr = df[numerical_cols].corr()
        return corr
    return None


def detect_outliers(df, numerical_cols):
    """Detect outliers using IQR method."""
    outliers = {}
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if outlier_count > 0:
            outliers[col] = {
                'count': outlier_count,
                'percentage': (outlier_count / len(df)) * 100,
                'bounds': (lower_bound, upper_bound)
            }
    return outliers


def generate_insights(df, column_types):
    """Generate plain-English insights from the dataset."""
    insights = []
    
    # Basic dataset info
    insights.append(f"📊 **Dataset Overview**: This dataset contains **{len(df)} rows** and **{len(df.columns)} columns**.")
    
    # Column type distribution
    num_count = len(column_types['numerical'])
    cat_count = len(column_types['categorical'])
    insights.append(f"📈 **Column Types**: The dataset has **{num_count} numerical** and **{cat_count} categorical** columns.")
    
    # Missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        missing_pct = (missing / (len(df) * len(df.columns))) * 100
        insights.append(f"⚠️ **Missing Values**: There are **{missing} missing values** ({missing_pct:.1f}% of total data).")
    
    # Duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        insights.append(f"🔄 **Duplicates**: Found **{duplicates} duplicate rows** ({duplicates/len(df)*100:.1f}% of data).")
    
    # Outliers
    if column_types['numerical']:
        outliers = detect_outliers(df, column_types['numerical'])
        if outliers:
            insights.append(f"📉 **Outliers Detected**: {len(outliers)} numerical columns have outliers.")
            for col, info in list(outliers.items())[:3]:
                insights.append(f"   - {col}: {info['count']} outliers ({info['percentage']:.1f}%)")
    
    # Correlations
    if len(column_types['numerical']) > 1:
        corr = df[column_types['numerical']].corr()
        high_corr = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                if abs(corr.iloc[i, j]) > 0.7:
                    high_corr.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
        
        if high_corr:
            insights.append(f"🔗 **Strong Correlations**: Found {len(high_corr)} pairs with strong correlation:")
            for col1, col2, corr_val in high_corr[:3]:
                direction = "positive" if corr_val > 0 else "negative"
                insights.append(f"   - {col1} ↔ {col2}: {corr_val:.2f} ({direction})")
    
    # Categorical insights
    if column_types['categorical']:
        for col in column_types['categorical'][:3]:
            top_value = df[col].mode()[0] if len(df[col].mode()) > 0 else "N/A"
            unique_count = df[col].nunique()
            insights.append(f"📋 **{col}**: Has {unique_count} unique values, most common is '{top_value}'.")
    
    return insights


def create_visualization_plots(df, column_types):
    """Create all visualization plots."""
    plots = {}
    
    # Histograms for numerical columns
    if column_types['numerical']:
        plots['histograms'] = {}
        for col in column_types['numerical']:
            fig = px.histogram(df, x=col, nbins=30, 
                            title=f"Distribution of {col}",
                            color_discrete_sequence=['#6366f1'])
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e2e8f0',
                title_font_size=16
            )
            plots['histograms'][col] = fig
        
        # Boxplots for numerical columns
        plots['boxplots'] = {}
        for col in column_types['numerical']:
            fig = px.box(df, y=col, title=f"Boxplot of {col}",
                        color_discrete_sequence=['#8b5cf6'])
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e2e8f0',
                title_font_size=16
            )
            plots['boxplots'][col] = fig
        
        # Correlation heatmap
        if len(column_types['numerical']) > 1:
            corr = df[column_types['numerical']].corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto",
                        color_continuous_scale='RdBu_r',
                        title="Correlation Heatmap")
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e2e8f0',
                title_font_size=16
            )
            plots['correlation'] = fig
    
    # Bar charts for categorical columns
    if column_types['categorical']:
        plots['barcharts'] = {}
        for col in column_types['categorical'][:5]:  # Limit to first 5
            value_counts = df[col].value_counts().head(10)
            fig = px.bar(x=value_counts.index, y=value_counts.values,
                        title=f"Top 10 values in {col}",
                        color_discrete_sequence=['#00d9ff'])
            fig.update_layout(
                xaxis_title=col,
                yaxis_title="Count",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e2e8f0',
                title_font_size=16,
                xaxis={'tickangle': -45}
            )
            plots['barcharts'][col] = fig
    
    return plots


def build_prediction_model(df, column_types, target_column):
    """Build a simple prediction model."""
    if not target_column or target_column not in df.columns:
        return None
    
    # Prepare features
    feature_cols = [col for col in column_types['numerical'] if col != target_column]
    
    if len(feature_cols) == 0:
        return None
    
    # Prepare data
    df_model = df[feature_cols + [target_column]].dropna()
    
    if len(df_model) < 10:
        return None
    
    X = df_model[feature_cols]
    y = df_model[target_column]
    
    # Determine if regression or classification
    if df[target_column].dtype in ['int64', 'float64']:
        # Regression
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return {
            'type': 'regression',
            'model': model,
            'score': model.score(X_test, y_test),
            'mse': mse,
            'feature_importance': dict(zip(feature_cols, model.coef_))
        }
    else:
        # Classification
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return {
            'type': 'classification',
            'model': model,
            'score': accuracy,
            'feature_importance': dict(zip(feature_cols, model.feature_importances_))
        }


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application function."""
    
    # Sidebar
    with st.sidebar:
        st.title("📊 AI Data Analyst")
        st.markdown("---")
        
        st.header("📁 Upload Data")
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        st.markdown("---")
        
        # Demo data option
        if st.button("Load Demo Data"):
            # Create sample data
            np.random.seed(42)
            n = 500
            data = {
                'Age': np.random.randint(18, 70, n),
                'Income': np.random.normal(50000, 15000, n),
                'Credit_Score': np.random.randint(300, 850, n),
                'Loan_Amount': np.random.exponential(10000, n),
                'Employment_Type': np.random.choice(['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], n),
                'City': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n),
                'Customer_Satisfaction': np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.1, 0.3, 0.35, 0.2])
            }
            # Add some missing values
            df_demo = pd.DataFrame(data)
            df_demo.loc[::20, 'Income'] = np.nan
            df_demo.loc[::30, 'Employment_Type'] = np.nan
            # Add duplicates
            df_demo = pd.concat([df_demo, df_demo.head(10)], ignore_index=True)
            
            df = df_demo
            st.session_state['df'] = df
            st.success("Demo data loaded!")
        
        st.markdown("---")
        
        st.header("⚙️ Settings")
        st.info("Upload a CSV file to get started with automatic data analysis!")
    
    # Main content
    st.title("🤖 AI Data Analyst")
    st.markdown("### Your Intelligent Data Analysis Assistant")
    
    # Check if data is loaded
    if 'df' in st.session_state and st.session_state['df'] is not None:
        df = st.session_state['df']
    elif uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.session_state['df'] = df
    else:
        # Welcome screen
        st.markdown("""
        <div class="card">
            <h2>👋 Welcome to AI Data Analyst!</h2>
            <p style="color: #cbd5e1; font-size: 1.1rem;">
                Upload a CSV file to get comprehensive data analysis including:
            </p>
            <ul style="color: #94a3b8; font-size: 1rem;">
                <li>📊 Data Overview & Statistics</li>
                <li>🧹 Data Cleaning Suggestions</li>
                <li>📈 Interactive Visualizations</li>
                <li>💡 AI-Powered Insights</li>
                <li>🔮 Predictive Modeling (Optional)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        return
    
    if df is None:
        return
    
    # Get column types
    column_types = get_column_types(df)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Data Overview",
        "🧹 Data Cleaning",
        "📈 Visualizations",
        "💡 Insights",
        "🔮 Predictions",
        "🔍 Data Filter"
    ])
    
    # ============ TAB 1: DATA OVERVIEW ============
    with tab1:
        st.header("Data Overview")
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Duplicates", df.duplicated().sum())
        
        st.markdown("---")
        
        # First 10 rows
        st.subheader("First 10 Rows")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Column information
        st.markdown("---")
        st.subheader("Column Information")
        
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.values,
            'Non-Null Count': df.count().values,
            'Null Count': df.isnull().sum().values,
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(col_info, use_container_width=True)
        
        # Statistics
        if column_types['numerical']:
            st.markdown("---")
            st.subheader("Statistical Summary")
            stats = calculate_statistics(df, column_types['numerical'])
            st.dataframe(stats, use_container_width=True)
    
    # ============ TAB 2: DATA CLEANING ============
    with tab2:
        st.header("Data Cleaning Suggestions")
        
        # Missing values analysis
        st.subheader("Missing Values Analysis")
        missing_df = detect_missing_values(df)
        
        if len(missing_df) > 0:
            st.warning(f"Found missing values in {len(missing_df)} columns!")
            
            # Visualize missing values
            fig = px.bar(missing_df, x=missing_df.index, y='Missing %',
                        title="Missing Values Percentage by Column",
                        color='Missing %',
                        color_continuous_scale='Reds')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e2e8f0'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("No missing values found in your dataset!")
        
        # Duplicates analysis
        st.markdown("---")
        st.subheader("Duplicate Analysis")
        duplicates = detect_duplicates(df)
        
        if duplicates > 0:
            st.warning(f"Found {duplicates} duplicate rows!")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Show Duplicates"):
                    st.dataframe(df[df.duplicated()].head(10), use_container_width=True)
            with col2:
                if st.button("Remove Duplicates"):
                    df = df.drop_duplicates()
                    st.session_state['df'] = df
                    st.success(f"Removed {duplicates} duplicate rows!")
        else:
            st.success("No duplicate rows found!")
        
        # Cleaning suggestions
        st.markdown("---")
        st.subheader("Suggested Actions")
        suggestions = suggest_data_cleaning(df, missing_df, duplicates)
        
        for suggestion in suggestions:
            st.markdown(suggestion)
        
        # Apply cleaning actions
        st.markdown("---")
        st.subheader("Apply Cleaning")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fill_method = st.selectbox("Fill Missing Values", 
                                    ["None", "Mean (Numerical)", "Median (Numerical)", 
                                    "Mode (Categorical)", "Forward Fill", "Backward Fill"])
            if st.button("Apply Fill") and fill_method != "None":
                df_clean = df.copy()
                for col in df_clean.columns:
                    if df_clean[col].dtype in ['int64', 'float64']:
                        if fill_method == "Mean (Numerical)":
                            df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                        elif fill_method == "Median (Numerical)":
                            df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    else:
                        if fill_method == "Mode (Categorical)":
                            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
                
                st.session_state['df'] = df_clean
                st.success("Missing values filled!")
        
        with col2:
            dropna_threshold = st.slider("Drop columns with > X% missing", 0, 100, 50)
            if st.button("Drop High-Missing Columns"):
                df_clean = df.dropna(axis=1, thresh=int(len(df) * (100 - dropna_threshold) / 100))
                st.session_state['df'] = df_clean
                st.success("High-missing columns dropped!")
        
        with col3:
            if st.button("Remove All Duplicates"):
                df_clean = df.drop_duplicates()
                st.session_state['df'] = df_clean
                st.success("Duplicates removed!")
    
    # ============ TAB 3: VISUALIZATIONS ============
    with tab3:
        st.header("Data Visualizations")
        
        # Create visualizations
        plots = create_visualization_plots(df, column_types)
        
        # Numerical columns visualizations
        if column_types['numerical']:
            st.subheader("📊 Numerical Columns")
            
            selected_num_col = st.selectbox("Select Numerical Column", 
                                        column_types['numerical'],
                                        key="viz_num")
            
            # Histogram
            if selected_num_col in plots.get('histograms', {}):
                st.plotly_chart(plots['histograms'][selected_num_col], use_container_width=True)
            
            # Boxplot
            if selected_num_col in plots.get('boxplots', {}):
                st.plotly_chart(plots['boxplots'][selected_num_col], use_container_width=True)
            
            # Correlation heatmap
            if 'correlation' in plots:
                st.markdown("---")
                st.subheader("🔗 Correlation Heatmap")
                st.plotly_chart(plots['correlation'], use_container_width=True)
        
        # Categorical columns visualizations
        if column_types['categorical']:
            st.markdown("---")
            st.subheader("📋 Categorical Columns")
            
            selected_cat_col = st.selectbox("Select Categorical Column",
                                        column_types['categorical'],
                                        key="viz_cat")
            
            if selected_cat_col in plots.get('barcharts', {}):
                st.plotly_chart(plots['barcharts'][selected_cat_col], use_container_width=True)
        
        # All numerical columns distribution
        if len(column_types['numerical']) > 1:
            st.markdown("---")
            st.subheader("📊 All Numerical Distributions")
            
            cols = st.columns(min(3, len(column_types['numerical'])))
            for i, col in enumerate(column_types['numerical'][:6]):
                with cols[i % 3]:
                    fig = px.histogram(df, x=col, nbins=20, title=col)
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#e2e8f0',
                        height=250
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # ============ TAB 4: INSIGHTS ============
    with tab4:
        st.header("AI-Generated Insights")
        
        insights = generate_insights(df, column_types)
        
        for insight in insights:
            st.markdown(f"<div class='card'>{insight}</div>", unsafe_allow_html=True)
        
        # Additional detailed insights
        st.markdown("---")
        
        # Outliers detailed
        st.subheader("📉 Outlier Analysis")
        outliers = detect_outliers(df, column_types['numerical'])
        
        if outliers:
            outlier_data = []
            for col, info in outliers.items():
                outlier_data.append({
                    'Column': col,
                    'Outlier Count': info['count'],
                    'Percentage': f"{info['percentage']:.1f}%",
                    'Lower Bound': f"{info['bounds'][0]:.2f}",
                    'Upper Bound': f"{info['bounds'][1]:.2f}"
                })
            st.dataframe(pd.DataFrame(outlier_data), use_container_width=True)
        else:
            st.success("No significant outliers detected!")
        
        # Data quality score
        st.markdown("---")
        st.subheader("📈 Data Quality Score")
        
        score = 100
        score -= min(30, df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
        score -= min(20, df.duplicated().sum() / len(df) * 100)
        
        if outliers:
            outlier_pct = sum([o['percentage'] for o in outliers.values()]) / len(outliers)
            score -= min(20, outlier_pct)
        
        score = max(0, int(score))
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = score,
                title = {'text': "Data Quality Score", 'font': {'size': 24, 'color': '#e2e8f0'}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickcolor': '#e2e8f0'},
                    'bar': {'color': "#6366f1" if score >= 70 else "#f59e0b" if score >= 50 else "#ef4444"},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.3)'},
                        {'range': [50, 70], 'color': 'rgba(245, 158, 11, 0.3)'},
                        {'range': [70, 100], 'color': 'rgba(16, 185, 129, 0.3)'}
                    ],
                }
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e2e8f0',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # ============ TAB 5: PREDICTIONS ============
    with tab5:
        st.header("Predictive Modeling")
        
        if not column_types['numerical']:
            st.warning("Need numerical columns to build prediction models!")
            return
        
        st.subheader("Select Target Variable")
        target_column = st.selectbox("Choose target column to predict", 
                                    column_types['numerical'],
                                    key="target")
        
        if target_column:
            feature_cols = [col for col in column_types['numerical'] if col != target_column]
            
            if not feature_cols:
                st.warning("Need at least one feature column!")
                return
            
            st.subheader("Select Features")
            selected_features = st.multiselect("Choose feature columns",
                                            feature_cols,
                                            default=feature_cols[:min(3, len(feature_cols))])
            
            if selected_features and st.button("Build Prediction Model"):
                with st.spinner("Building model..."):
                    model_result = build_prediction_model(df, column_types, target_column)
                    
                    if model_result:
                        st.success("Model built successfully!")
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Model Type", model_result['type'].title())
                        with col2:
                            if model_result['type'] == 'regression':
                                st.metric("R² Score", f"{model_result['score']:.4f}")
                            else:
                                st.metric("Accuracy", f"{model_result['score']:.4f}")
                        
                        if model_result['type'] == 'regression':
                            st.info(f"Mean Squared Error: {model_result['mse']:.2f}")
                        
                        # Feature importance
                        st.markdown("---")
                        st.subheader("Feature Importance")
                        
                        importance_df = pd.DataFrame({
                            'Feature': list(model_result['feature_importance'].keys()),
                            'Importance': list(model_result['feature_importance'].values())
                        }).sort_values('Importance', ascending=True)
                        
                        fig = px.bar(importance_df, x='Importance', y='Feature',
                                    title="Feature Importance",
                                    color='Importance',
                                    color_continuous_scale='Viridis',
                                    orientation='h')
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_color='#e2e8f0'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Make predictions
                        st.markdown("---")
                        st.subheader("Make Predictions")
                        
                        pred_inputs = {}
                        for feat in selected_features:
                            pred_inputs[feat] = st.number_input(f"Enter {feat}", 
                                                                value=float(df[feat].mean()))
                        
                        if st.button("Predict"):
                            pred_df = pd.DataFrame([pred_inputs])
                            prediction = model_result['model'].predict(pred_df)[0]
                            st.success(f"Predicted {target_column}: {prediction:.2f}")
                    else:
                        st.error("Could not build model. Not enough data or features!")
    
    # ============ TAB 6: DATA FILTER ============
    with tab6:
        st.header("Data Filter")
        
        st.subheader("Filter Data")
        
        # Initialize filtered dataframe
        df_filtered = df.copy()
        
        # Filter by numerical columns
        if column_types['numerical']:
            st.markdown("##### Numerical Filters")
            for col in column_types['numerical'][:5]:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                
                col1, col2, col3 = st.columns([1, 2, 2])
                with col1:
                    st.write(f"**{col}**")
                with col2:
                    range_vals = st.slider(f"Select range for {col}", 
                                        min_val, max_val, (min_val, max_val),
                                        key=f"slider_{col}")
                with col3:
                    filter_type = st.selectbox(f"Filter type for {col}",
                                            ["Keep in range", "Keep outside range"],
                                            key=f"type_{col}")
                
                if filter_type == "Keep in range":
                    df_filtered = df_filtered[(df_filtered[col] >= range_vals[0]) & 
                                            (df_filtered[col] <= range_vals[1])]
                else:
                    df_filtered = df_filtered[(df_filtered[col] < range_vals[0]) | 
                                            (df_filtered[col] > range_vals[1])]
        
        # Filter by categorical columns
        if column_types['categorical']:
            st.markdown("##### Categorical Filters")
            for col in column_types['categorical'][:3]:
                unique_vals = df[col].unique()
                selected_vals = st.multiselect(f"Select values for {col}",
                                               unique_vals,
                                               default=unique_vals,
                                               key=f"multi_{col}")
                df_filtered = df_filtered[df_filtered[col].isin(selected_vals)]
        
        # Display filtered results
        st.markdown("---")
        st.subheader("Filtered Data Preview")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original Rows", len(df))
        with col2:
            st.metric("Filtered Rows", len(df_filtered))
        
        st.dataframe(df_filtered.head(20), use_container_width=True)
        
        # Download filtered data
        if st.button("Download Filtered Data"):
            csv = df_filtered.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="filtered_data.csv",
                mime="text/csv"
            )


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()

