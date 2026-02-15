"""
Machine Learning Classification Models Implementation
Assignment 2 - M.Tech (AIML/DSE)
Author - Rushikesh Kailash Iname
This script implements 6 classification models with comprehensive evaluation metrics.
Dataset: Heart Disease Prediction Dataset
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ML Models
from sklearn.model_selection import train_test_split
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

import joblib
import matplotlib.pyplot as plt
import seaborn as sns


class MLClassificationPipeline:
    """
    Complete ML Classification Pipeline for Heart Disease Prediction
    """
    
    def __init__(self, data_path=None, use_uci=False):
        """
        Initialize the pipeline with dataset path or UCI fetch option
        
        Parameters:
        - data_path: Path to CSV file (if use_uci=False)
        - use_uci: If True, fetch from UCI repository directly
        """
        self.data_path = data_path
        self.use_uci = use_uci
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.results = {}
        
    def load_and_preprocess_data(self):
        """
        Load dataset and perform preprocessing
        
        Dataset: Heart Disease UCI Dataset
        Features: 13 features
        Target: Binary classification (0: No disease, 1: Disease)
        """
        # Load data
        if self.use_uci:
            try:
                from ucimlrepo import fetch_ucirepo
                print("Fetching dataset from UCI ML Repository...")
                
                # Fetch the dataset (Heart Disease dataset, id=45)
                heart_disease = fetch_ucirepo(id=45)
                
                # Access the features and target data as pandas DataFrames
                X = heart_disease.data.features
                y = heart_disease.data.targets
                
                # Combine the features (X) and targets (y) into a single DataFrame
                df = pd.concat([X, y], axis=1)
                
                # Handle missing values if any
                df = df.dropna()
                
                print("Dataset fetched successfully from UCI!")
            except ImportError:
                print("ERROR: ucimlrepo package not found!")
                print("Install it using: pip install ucimlrepo")
                print("Falling back to local CSV file...")
                self.use_uci = False
            except Exception as e:
                print(f"ERROR fetching from UCI: {e}")
                print("Falling back to local CSV file...")
                self.use_uci = False
                
        if not self.use_uci:
            if self.data_path is None:
                raise ValueError("data_path must be provided when use_uci=False")
            df = pd.read_csv(self.data_path)
        
        print(f"\nDataset Shape: {df.shape}")
        print(f"\nColumns: {df.columns.tolist()}")
        print(f"\nData types:\n{df.dtypes}")
        
        # Handle missing values
        df = df.dropna()
        print(f"\nShape after removing missing values: {df.shape}")
        
        # Separate features and target
        X = df.iloc[:, :-1].copy()  # All columns except last
        y = df.iloc[:, -1].copy()   # Last column is target
        
        # Encode categorical variables in features
        print("\nEncoding categorical variables...")
        categorical_columns = X.select_dtypes(include=['object']).columns
        
        if len(categorical_columns) > 0:
            print(f"Found categorical columns: {categorical_columns.tolist()}")
            for col in categorical_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
                print(f"  - Encoded '{col}': {le.classes_}")
        else:
            print("No categorical columns found in features.")
        
        # Encode target variable if it's categorical
        if y.dtype == 'object':
            print("\nEncoding target variable...")
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))
            self.label_encoders['target'] = le_target
            print(f"  - Target classes: {le_target.classes_}")
        
        # Ensure all features are numeric
        X = X.apply(pd.to_numeric, errors='coerce')
        y = pd.to_numeric(y, errors='coerce')
        
        # Remove any rows with NaN after conversion
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        print(f"\nFinal dataset shape: {X.shape}")
        print(f"\nClass Distribution:\n{pd.Series(y).value_counts()}")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"\nTraining Set Size: {self.X_train.shape[0]}")
        print(f"Test Set Size: {self.X_test.shape[0]}")
        
        return self
    
    def initialize_models(self):
        """
        Initialize all 6 classification models
        """
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
        }
        
        print("\nModels Initialized:")
        for name in self.models.keys():
            print(f"  - {name}")
        
        return self
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate all 6 evaluation metrics
        """
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'F1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'MCC': matthews_corrcoef(y_true, y_pred)
        }
        
        # AUC Score (requires probability predictions)
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:  # Binary classification
                    metrics['AUC'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:  # Multi-class
                    metrics['AUC'] = roc_auc_score(y_true, y_pred_proba, 
                                                   multi_class='ovr', average='weighted')
            except Exception as e:
                print(f"Warning: Could not calculate AUC - {e}")
                metrics['AUC'] = 0.0
        else:
            metrics['AUC'] = 0.0
        
        return metrics
    
    def train_and_evaluate(self):
        """
        Train all models and evaluate them
        """
        print("\n" + "="*80)
        print("TRAINING AND EVALUATION")
        print("="*80)
        
        for name, model in self.models.items():
            print(f"\n{'='*80}")
            print(f"Training: {name}")
            print(f"{'='*80}")
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
            
            # Store results
            self.results[name] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'metrics': metrics,
                'confusion_matrix': confusion_matrix(self.y_test, y_pred),
                'classification_report': classification_report(self.y_test, y_pred)
            }
            
            # Print metrics
            print(f"\nMetrics for {name}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name:12s}: {value:.4f}")
        
        return self
    
    def get_results_dataframe(self):
        """
        Create a comparison dataframe of all models
        """
        results_data = []
        
        for name, result in self.results.items():
            row = {
                'ML Model Name': name,
                'Accuracy': result['metrics']['Accuracy'],
                'AUC': result['metrics']['AUC'],
                'Precision': result['metrics']['Precision'],
                'Recall': result['metrics']['Recall'],
                'F1': result['metrics']['F1'],
                'MCC': result['metrics']['MCC']
            }
            results_data.append(row)
        
        df_results = pd.DataFrame(results_data)
        
        # Reorder columns to match assignment requirement
        column_order = ['ML Model Name', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
        df_results = df_results[column_order]
        
        return df_results
    
    def print_comparison_table(self):
        """
        Print the comparison table
        """
        df_results = self.get_results_dataframe()
        
        print("\n" + "="*100)
        print("MODEL COMPARISON TABLE")
        print("="*100)
        print(df_results.to_string(index=False))
        print("="*100)
        
        return df_results
    
    def save_models(self, output_dir='models'):
        """
        Save all trained models
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save scaler
        joblib.dump(self.scaler, f'{output_dir}/scaler.pkl')
        print(f"Saved: scaler.pkl")
        
        # Save label encoders
        if self.label_encoders:
            joblib.dump(self.label_encoders, f'{output_dir}/label_encoders.pkl')
            print(f"Saved: label_encoders.pkl")
        
        # Save each model
        for name, result in self.results.items():
            model_filename = name.lower().replace(' ', '_') + '.pkl'
            joblib.dump(result['model'], f'{output_dir}/{model_filename}')
            print(f"Saved: {model_filename}")
        
        print(f"\nAll models saved to '{output_dir}/' directory")
    
    def plot_confusion_matrices(self):
        """
        Plot confusion matrices for all models
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, (name, result) in enumerate(self.results.items()):
            cm = result['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{name}\nAccuracy: {result["metrics"]["Accuracy"]:.4f}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("\nConfusion matrices saved as 'confusion_matrices.png'")
        plt.close()


def main():
    """
    Main execution function
    """
    print("="*100)
    print("MACHINE LEARNING CLASSIFICATION ASSIGNMENT")
    print("Dataset: Heart Disease UCI Dataset")
    print("="*100)
    
    # CHOOSE ONE OF THE FOLLOWING OPTIONS:
    
    # Option 1: Load from local CSV file (RECOMMENDED)
    # Download from: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
    #data_path = 'heart.csv'
    #pipeline = MLClassificationPipeline(data_path=data_path, use_uci=False)
    
    # Option 2: Fetch from UCI directly (requires: pip install ucimlrepo)
    pipeline = MLClassificationPipeline(use_uci=True)
    
    try:
        # Execute pipeline
        pipeline.load_and_preprocess_data()
        pipeline.initialize_models()
        pipeline.train_and_evaluate()
        
        # Display results
        df_results = pipeline.print_comparison_table()
        
        # Save results to CSV
        df_results.to_csv('model_comparison_results.csv', index=False)
        print("\nResults saved to 'model_comparison_results.csv'")
        
        # Plot confusion matrices
        pipeline.plot_confusion_matrices()
        
        # Save models
        pipeline.save_models()
        
        print("\n" + "="*100)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*100)
        
        return pipeline, df_results
        
    except Exception as e:
        print(f"\n{'='*100}")
        print(f"ERROR: {str(e)}")
        print(f"{'='*100}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    pipeline, results = main()
