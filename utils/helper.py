"""
Utility functions for data processing and predictions
"""

import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
FEATURE_INFO_PATH = os.path.join(BASE_DIR, 'feature_info.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'mobile_data.csv')

def load_model():
    """
    Load the trained model and scaler
    """
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_info = joblib.load(FEATURE_INFO_PATH) if os.path.exists(FEATURE_INFO_PATH) else None
        return model, scaler, feature_info
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

def predict_addiction_risk(features):
    """
    Predict addiction risk based on input features
    features: dict with keys: age, gender, screen_time_hours, social_media_hours,
              gaming_hours, sleep_hours, stress_level
    """
    model, scaler, feature_info = load_model()
    
    if model is None or scaler is None:
        return {'error': 'Model not loaded properly'}
    
    # Extract features in correct order
    feature_order = ['age', 'gender', 'screen_time_hours', 'social_media_hours', 
                     'gaming_hours', 'sleep_hours', 'stress_level']
    
    features_list = [float(features[col]) for col in feature_order]
    
    # Scale features
    features_array = np.array(features_list).reshape(1, -1)
    features_scaled = scaler.transform(features_array)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    return {
        'risk': int(prediction),
        'risk_level': 'High Risk' if prediction == 1 else 'Low Risk',
        'probability_low': probabilities[0],
        'probability_high': probabilities[1]
    }

def get_dashboard_stats():
    """
    Calculate statistics for the dashboard
    """
    try:
        df = pd.read_csv(DATA_PATH)
        
        stats = {
            'total_users': len(df),
            'high_risk_count': int(df['addiction_risk'].sum()),
            'low_risk_count': int(len(df) - df['addiction_risk'].sum()),
            'high_risk_percentage': round((df['addiction_risk'].sum() / len(df)) * 100, 1),
            'avg_screen_time': round(df['screen_time_hours'].mean(), 1),
            'avg_social_media': round(df['social_media_hours'].mean(), 1),
            'avg_gaming': round(df['gaming_hours'].mean(), 1),
            'avg_sleep': round(df['sleep_hours'].mean(), 1),
            'avg_stress': round(df['stress_level'].mean(), 1)
        }
        
        # Risk distribution data for chart
        risk_distribution = {
            'low_risk': stats['low_risk_count'],
            'high_risk': stats['high_risk_count']
        }
        
        # Screen time categories
        df['screen_category'] = pd.cut(df['screen_time_hours'], 
                                        bins=[0, 3, 6, 10], 
                                        labels=['Low (<3h)', 'Medium (3-6h)', 'High (>6h)'])
        screen_distribution = df['screen_category'].value_counts().to_dict()
        
        # Sleep vs addiction data
        sleep_vs_addiction = df.groupby('sleep_hours')['addiction_risk'].mean().to_dict()
        
        # Gender distribution
        gender_counts = df['gender'].value_counts().to_dict()
        
        # Social media by risk level
        social_by_risk = df.groupby('addiction_risk')['social_media_hours'].mean().to_dict()
        gaming_by_risk = df.groupby('addiction_risk')['gaming_hours'].mean().to_dict()
        
        return {
            'stats': stats,
            'risk_distribution': risk_distribution,
            'screen_distribution': screen_distribution,
            'sleep_vs_addiction': sleep_vs_addiction,
            'gender_counts': gender_counts,
            'social_by_risk': social_by_risk,
            'gaming_by_risk': gaming_by_risk
        }
        
    except Exception as e:
        print(f"Error getting dashboard stats: {e}")
        return None