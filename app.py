"""
Streamlit Web Application for ML Classification Models
Assignment 2 - M.Tech (AIML/DSE)

This app provides an interactive interface for:
- Dataset upload
- Model selection
- Evaluation metrics display
- Confusion matrix visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# ML Models
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Page configuration
st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)


class MLStreamlitApp:
    """
    Streamlit Application for ML Classification
    """
    
    def __init__(self):
        self.models = self.initialize_models()
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def initialize_models(self):
        """Initialize all classification models"""
        return {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
            'K-Nearest Neighbors (KNN)': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
        }
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate all evaluation metrics"""
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'AUC': 0.0,  # Will be calculated below
            'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'F1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'MCC': matthews_corrcoef(y_true, y_pred)
        }
        
        # AUC Score
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    metrics['AUC'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    metrics['AUC'] = roc_auc_score(y_true, y_pred_proba, 
                                                   multi_class='ovr', average='weighted')
            except:
                metrics['AUC'] = 0.0
        
        return metrics
    
    def plot_confusion_matrix(self, cm, title):
        """Plot confusion matrix"""
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
        ax.set_title(f'Confusion Matrix - {title}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        return fig


def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 ML Classification Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize app
    app = MLStreamlitApp()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("---")
        
        # Dataset upload
        st.subheader("📁 Upload Dataset")
        uploaded_file = st.file_uploader(
            "Upload CSV file (Test Data)", 
            type=['csv'],
            help="Upload your test dataset in CSV format"
        )
        
        st.markdown("---")
        
        # Model selection
        st.subheader("🔍 Select Model")
        selected_model = st.selectbox(
            "Choose a classification model:",
            list(app.models.keys()),
            help="Select the model you want to evaluate"
        )
        
        st.markdown("---")
        
        # About
        with st.expander("ℹ️ About"):
            st.write("""
            **ML Classification Dashboard**
            
            This application demonstrates:
            - Multiple classification models
            - Comprehensive evaluation metrics
            - Interactive visualizations
            
            **Models Available:**
            - Logistic Regression
            - Decision Tree
            - K-Nearest Neighbors
            - Naive Bayes
            - Random Forest
            - XGBoost
            """)
    
    # Main content
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
            
            # Display dataset preview
            with st.expander("👀 View Dataset Preview"):
                st.dataframe(df.head(10), use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", df.shape[0])
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    st.metric("Features", df.shape[1] - 1)
            
            st.markdown("---")
            
            # Prepare data
            X = df.iloc[:, :-1].copy()
            y = df.iloc[:, -1].copy()
            
            # Encode categorical variables
            categorical_columns = X.select_dtypes(include=['object']).columns
            
            if len(categorical_columns) > 0:
                st.info(f"🔄 Encoding {len(categorical_columns)} categorical column(s): {', '.join(categorical_columns)}")
                for col in categorical_columns:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    app.label_encoders[col] = le
            
            # Encode target if categorical
            if y.dtype == 'object':
                le_target = LabelEncoder()
                y = le_target.fit_transform(y.astype(str))
                app.label_encoders['target'] = le_target
            
            # Convert to numeric
            X = X.apply(pd.to_numeric, errors='coerce')
            y = pd.to_numeric(y, errors='coerce')
            
            # Check for any remaining NaN values
            if X.isna().any().any() or y.isna().any():
                st.warning("⚠️ Found missing values after encoding. Removing rows with missing data...")
                valid_idx = ~(X.isna().any(axis=1) | y.isna())
                X = X[valid_idx]
                y = y[valid_idx]
                st.info(f"✅ Cleaned dataset size: {len(X)} rows")
            
            # Convert to numpy arrays
            X = X.values
            y = y.values
            
            # Scale features
            X_scaled = app.scaler.fit_transform(X)
            
            # Train selected model
            st.subheader(f"🎯 Training: {selected_model}")
            
            with st.spinner(f"Training {selected_model}..."):
                model = app.models[selected_model]
                model.fit(X_scaled, y)
                
                # Predictions
                y_pred = model.predict(X_scaled)
                y_pred_proba = model.predict_proba(X_scaled) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                metrics = app.calculate_metrics(y, y_pred, y_pred_proba)
            
            st.success(f"✅ Model trained successfully!")
            
            # Display Metrics
            st.markdown("---")
            st.subheader("📊 Evaluation Metrics")
            
            # Metrics in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("AUC Score", f"{metrics['AUC']:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Precision", f"{metrics['Precision']:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Recall", f"{metrics['Recall']:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("F1 Score", f"{metrics['F1']:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("MCC Score", f"{metrics['MCC']:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Metrics table
            st.markdown("---")
            metrics_df = pd.DataFrame([metrics])
            st.table(metrics_df.style.format("{:.4f}"))
            
            # Confusion Matrix
            st.markdown("---")
            st.subheader("🔥 Confusion Matrix")
            
            cm = confusion_matrix(y, y_pred)
            fig = app.plot_confusion_matrix(cm, selected_model)
            st.pyplot(fig)
            
            # Classification Report
            st.markdown("---")
            st.subheader("📋 Classification Report")
            
            report = classification_report(y, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='Blues').format("{:.2f}"), 
                        use_container_width=True)
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                st.markdown("---")
                st.subheader("🌟 Feature Importance")
                
                importance = model.feature_importances_
                feature_names = [f"Feature {i+1}" for i in range(len(importance))]
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(importance_df['Feature'][:10], importance_df['Importance'][:10])
                ax.set_xlabel('Importance', fontsize=12)
                ax.set_title('Top 10 Feature Importances', fontsize=14, fontweight='bold')
                ax.invert_yaxis()
                st.pyplot(fig)
            
        except Exception as e:
            st.error(f"❌ Error processing dataset: {str(e)}")
            st.info("Please ensure your CSV has features in all columns except the last, which should be the target variable.")
    
    else:
        # Instructions when no file uploaded
        st.info("👈 Please upload a CSV file from the sidebar to begin")
        
        st.markdown("---")
        st.subheader("📝 Instructions")
        
        st.markdown("""
        1. **Upload Dataset**: Use the sidebar to upload your test dataset (CSV format)
        2. **Select Model**: Choose a classification model from the dropdown
        3. **View Results**: Explore evaluation metrics, confusion matrix, and classification report
        
        **Dataset Requirements:**
        - CSV format
        - Last column should be the target variable
        - All other columns are features
        - Numeric data preferred
        """)
        
        st.markdown("---")
        st.subheader("📊 Sample Metrics Table")
        
        sample_metrics = {
            'Accuracy': [0.8500],
            'AUC': [0.9200],
            'Precision': [0.8300],
            'Recall': [0.8400],
            'F1': [0.8350],
            'MCC': [0.7100]
        }
        st.table(pd.DataFrame(sample_metrics).style.format("{:.4f}"))


if __name__ == "__main__":
    main()