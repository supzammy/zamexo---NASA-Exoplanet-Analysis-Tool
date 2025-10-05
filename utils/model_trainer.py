"""
Lightweight model training for Streamlit Cloud deployment.
This module trains the model on startup if it doesn't exist.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

def get_sample_training_data():
    """Create sample training data for demonstration purposes."""
    # This creates a realistic but synthetic dataset for demo purposes
    # In production, this would load from NASA's KOI table
    
    np.random.seed(42)  # For reproducibility
    n_samples = 1000
    
    # Generate synthetic features based on typical exoplanet characteristics
    data = {
        'koi_period': np.random.lognormal(2, 1, n_samples),  # Log-normal distribution for periods
        'koi_duration': np.random.gamma(2, 2, n_samples),   # Gamma distribution for durations
        'koi_depth': np.random.exponential(1000, n_samples), # Exponential for depths (ppm)
        'koi_impact': np.random.beta(2, 2, n_samples),      # Beta distribution for impact parameter
        'koi_prad': np.random.lognormal(0, 1, n_samples),   # Log-normal for planet radius
        'koi_srad': np.random.normal(1, 0.3, n_samples),    # Normal for stellar radius
        'koi_smass': np.random.normal(1, 0.2, n_samples),   # Normal for stellar mass
        'koi_slogg': np.random.normal(4.4, 0.3, n_samples), # Normal for stellar log g
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic target classes based on feature combinations
    # This mimics the patterns found in real KOI data
    targets = []
    for _, row in df.iterrows():
        # Simple heuristics to create realistic classifications
        score = 0
        
        # Period check (most exoplanets have periods < 1000 days)
        if 0.5 < row['koi_period'] < 500:
            score += 1
        
        # Duration vs period ratio check
        if 0.01 < row['koi_duration'] / (row['koi_period'] * 24) < 0.2:
            score += 1
            
        # Depth check (typical range)
        if 10 < row['koi_depth'] < 10000:
            score += 1
            
        # Impact parameter check
        if row['koi_impact'] < 1.5:
            score += 1
            
        # Assign classes based on score
        if score >= 3:
            targets.append('CANDIDATE')
        elif score >= 2:
            targets.append('FALSE POSITIVE')
        else:
            targets.append('FALSE POSITIVE')
    
    df['koi_disposition'] = targets
    
    # Add some confirmed planets (highest quality candidates)
    confirmed_mask = (df['koi_disposition'] == 'CANDIDATE') & (np.random.random(len(df)) < 0.2)
    df.loc[confirmed_mask, 'koi_disposition'] = 'CONFIRMED'
    
    return df

def train_lightweight_model():
    """Train a lightweight model for Streamlit Cloud deployment."""
    print("Training lightweight model for Streamlit Cloud...")
    
    # Get training data
    df = get_sample_training_data()
    
    # Define feature columns
    feature_cols = [
        'koi_period', 'koi_duration', 'koi_depth', 'koi_impact',
        'koi_prad', 'koi_srad', 'koi_smass', 'koi_slogg'
    ]
    
    # Prepare features and targets
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df['koi_disposition']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train lightweight RandomForest
    model = RandomForestClassifier(
        n_estimators=50,  # Reduced for faster training
        max_depth=10,     # Reduced for smaller model size
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model trained successfully! Accuracy: {accuracy:.3f}")
    
    # Create models directory
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Save model
    joblib.dump(model, models_dir / 'model.joblib')
    
    # Save feature columns
    with open(models_dir / 'feature_cols.json', 'w') as f:
        json.dump(feature_cols, f)
    
    # Save feature statistics
    feature_stats = {
        col: {
            'mean': float(X[col].mean()),
            'std': float(X[col].std()),
            'median': float(X[col].median())
        }
        for col in feature_cols
    }
    
    with open(models_dir / 'feature_stats.json', 'w') as f:
        json.dump(feature_stats, f)
    
    # Save feature importances
    importances = {
        feature_cols[i]: float(importance)
        for i, importance in enumerate(model.feature_importances_)
    }
    
    with open(models_dir / 'importances.json', 'w') as f:
        json.dump(importances, f)
    
    # Save metrics
    metrics = {
        'accuracy': float(accuracy),
        'n_samples': len(df),
        'n_features': len(feature_cols),
        'model_type': 'RandomForestClassifier'
    }
    
    with open(models_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f)
    
    print("Model artifacts saved successfully!")
    return model, feature_cols, feature_stats, importances

if __name__ == "__main__":
    train_lightweight_model()