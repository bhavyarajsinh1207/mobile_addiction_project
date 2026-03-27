"""
Main Flask Application for Mobile Phone Addiction Risk Prediction
"""

from flask import Flask, render_template, request, jsonify, url_for
import os
import sys
import traceback
import json
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Add the project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'

# Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'mobile_data.csv')
USER_DATA_FILE = os.path.join(BASE_DIR, 'data', 'user_submissions.json')

# Global variables for model and scaler
model = None
scaler = None
feature_columns = None

def load_model_artifacts():
    """Load the trained model and scaler"""
    global model, scaler, feature_columns
    
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print("✓ Model loaded successfully")
        else:
            print("⚠ Model not found. Please run model/train_model.py first")
            
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            print("✓ Scaler loaded successfully")
            
        # Load feature info if available
        feature_info_path = os.path.join(BASE_DIR, 'feature_info.pkl')
        if os.path.exists(feature_info_path):
            feature_info = joblib.load(feature_info_path)
            feature_columns = feature_info.get('feature_columns', 
                ['age', 'gender', 'screen_time_hours', 'social_media_hours', 
                 'gaming_hours', 'sleep_hours', 'stress_level'])
            print(f"✓ Feature columns loaded: {feature_columns}")
        else:
            feature_columns = ['age', 'gender', 'screen_time_hours', 'social_media_hours', 
                               'gaming_hours', 'sleep_hours', 'stress_level']
            
    except Exception as e:
        print(f"Error loading model artifacts: {e}")

def load_user_submissions():
    """Load user submitted data from JSON file"""
    if os.path.exists(USER_DATA_FILE):
        try:
            with open(USER_DATA_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    return []

def save_user_submission(data):
    """Save user submission to JSON file"""
    submissions = load_user_submissions()
    
    # Add timestamp and ID to submission
    data['submitted_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data['id'] = len(submissions) + 1
    
    submissions.append(data)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(USER_DATA_FILE), exist_ok=True)
    
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(submissions, f, indent=2)
    
    return True

def update_dataset_with_submissions():
    """Merge user submissions with original dataset for updated analytics"""
    try:
        if not os.path.exists(DATA_PATH):
            return None
            
        original_df = pd.read_csv(DATA_PATH)
        submissions = load_user_submissions()
        
        if submissions and len(submissions) > 0:
            submissions_df = pd.DataFrame(submissions)
            submission_cols = ['age', 'gender', 'screen_time_hours', 'social_media_hours', 
                              'gaming_hours', 'sleep_hours', 'stress_level', 'addiction_risk']
            
            available_cols = [col for col in submission_cols if col in submissions_df.columns]
            if available_cols:
                submissions_df = submissions_df[available_cols]
                combined_df = pd.concat([original_df, submissions_df], ignore_index=True)
                return combined_df
        
        return original_df
            
    except Exception as e:
        print(f"Error updating dataset: {e}")
        return None

def get_dashboard_stats(use_combined_data=True):
    """Calculate statistics for the dashboard"""
    try:
        if use_combined_data:
            df = update_dataset_with_submissions()
            if df is None:
                return None
        else:
            if not os.path.exists(DATA_PATH):
                return None
            df = pd.read_csv(DATA_PATH)
        
        df['screen_category'] = pd.cut(df['screen_time_hours'], 
                                        bins=[0, 3, 6, 24], 
                                        labels=['Low (<3h)', 'Medium (3-6h)', 'High (>6h)'])
        
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
        
        risk_distribution = {
            'low_risk': stats['low_risk_count'],
            'high_risk': stats['high_risk_count']
        }
        
        screen_distribution = df['screen_category'].value_counts().to_dict()
        
        sleep_vs_addiction = {}
        sleep_hours_sorted = sorted(df['sleep_hours'].unique())[:10]
        for sleep_hour in sleep_hours_sorted:
            subset = df[df['sleep_hours'] == sleep_hour]
            sleep_vs_addiction[str(sleep_hour)] = round(subset['addiction_risk'].mean(), 2)
        
        gender_counts = {
            '0': int((df['gender'] == 0).sum()),
            '1': int((df['gender'] == 1).sum())
        }
        
        social_by_risk = {
            '0': round(df[df['addiction_risk'] == 0]['social_media_hours'].mean(), 1) if len(df[df['addiction_risk'] == 0]) > 0 else 0,
            '1': round(df[df['addiction_risk'] == 1]['social_media_hours'].mean(), 1) if len(df[df['addiction_risk'] == 1]) > 0 else 0
        }
        
        gaming_by_risk = {
            '0': round(df[df['addiction_risk'] == 0]['gaming_hours'].mean(), 1) if len(df[df['addiction_risk'] == 0]) > 0 else 0,
            '1': round(df[df['addiction_risk'] == 1]['gaming_hours'].mean(), 1) if len(df[df['addiction_risk'] == 1]) > 0 else 0
        }
        
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
        traceback.print_exc()
        return None

def predict_addiction_risk(features_dict):
    """Make prediction using the loaded model"""
    global model, scaler, feature_columns
    
    if model is None or scaler is None:
        return {'error': 'Model not loaded. Please train the model first.'}
    
    try:
        feature_order = ['age', 'gender', 'screen_time_hours', 'social_media_hours', 
                         'gaming_hours', 'sleep_hours', 'stress_level']
        
        features_list = [float(features_dict[col]) for col in feature_order]
        
        features_array = np.array(features_list).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        return {
            'risk': int(prediction),
            'risk_level': 'High Risk' if prediction == 1 else 'Low Risk',
            'probability_low': round(probabilities[0], 4),
            'probability_high': round(probabilities[1], 4)
        }
        
    except Exception as e:
        return {'error': f'Prediction error: {str(e)}'}

# Load model on startup
load_model_artifacts()

# Routes
@app.route('/')
def home():
    """Home page route"""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction form and result page"""
    if request.method == 'GET':
        return render_template('predict.html')
    
    try:
        # Get form data
        form_data = {
            'age': request.form.get('age'),
            'gender': request.form.get('gender'),
            'screen_time_hours': request.form.get('screen_time'),
            'social_media_hours': request.form.get('social_media'),
            'gaming_hours': request.form.get('gaming_hours'),
            'sleep_hours': request.form.get('sleep_hours'),
            'stress_level': request.form.get('stress_level')
        }
        
        # Validation - check for empty fields
        for key, value in form_data.items():
            if value is None or value == '':
                return render_template('predict.html', 
                                     error=f"Please fill in the {key.replace('_', ' ')} field.")
        
        # Convert to appropriate types
        try:
            features = {
                'age': float(form_data['age']),
                'gender': float(form_data['gender']),
                'screen_time_hours': float(form_data['screen_time_hours']),
                'social_media_hours': float(form_data['social_media_hours']),
                'gaming_hours': float(form_data['gaming_hours']),
                'sleep_hours': float(form_data['sleep_hours']),
                'stress_level': float(form_data['stress_level'])
            }
        except ValueError:
            return render_template('predict.html', 
                                 error="Please enter valid numeric values.")
        
        # Additional validation
        if features['age'] < 10 or features['age'] > 80:
            return render_template('predict.html', 
                                 error="Age should be between 10 and 80 years.")
        if features['gender'] not in [0, 1]:
            return render_template('predict.html', 
                                 error="Gender must be 0 (Female) or 1 (Male).")
        if features['screen_time_hours'] < 0 or features['screen_time_hours'] > 24:
            return render_template('predict.html', 
                                 error="Screen time should be between 0 and 24 hours.")
        if features['sleep_hours'] < 0 or features['sleep_hours'] > 24:
            return render_template('predict.html', 
                                 error="Sleep hours should be between 0 and 24 hours.")
        if features['stress_level'] < 1 or features['stress_level'] > 10:
            return render_template('predict.html', 
                                 error="Stress level should be between 1 and 10.")
        
        # Validate social media + gaming <= screen time
        if features['social_media_hours'] + features['gaming_hours'] > features['screen_time_hours'] + 0.5:
            return render_template('predict.html', 
                                 error="Social media and gaming hours cannot exceed total screen time.")
        
        # Make prediction
        result = predict_addiction_risk(features)
        
        if 'error' in result:
            return render_template('predict.html', error=result['error'])
        
        # Add input features to result for display
        result['input_data'] = features
        
        # Render result page with the result
        return render_template('result.html', result=result)
        
    except Exception as e:
        traceback.print_exc()
        return render_template('predict.html', 
                             error=f"An unexpected error occurred: {str(e)}")

@app.route('/dashboard')
def dashboard():
    """Analytics dashboard page"""
    try:
        stats_data = get_dashboard_stats(use_combined_data=True)
        
        if stats_data is None:
            return render_template('dashboard.html', 
                                 error="Data not available. Please ensure data file exists.")
        
        return render_template('dashboard.html', data=stats_data)
        
    except Exception as e:
        traceback.print_exc()
        return render_template('dashboard.html', 
                             error=f"Error loading dashboard: {str(e)}")

# API endpoints for data storage
@app.route('/api/submit-data', methods=['POST'])
def submit_user_data():
    """API endpoint to save user submitted data"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        required_fields = ['age', 'gender', 'screen_time_hours', 'social_media_hours', 
                          'gaming_hours', 'sleep_hours', 'stress_level', 'addiction_risk']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Validate data ranges
        if data['age'] < 10 or data['age'] > 80:
            return jsonify({'error': 'Age must be between 10 and 80'}), 400
        if data['gender'] not in [0, 1]:
            return jsonify({'error': 'Gender must be 0 (Female) or 1 (Male)'}), 400
        if data['screen_time_hours'] < 0 or data['screen_time_hours'] > 24:
            return jsonify({'error': 'Screen time must be between 0 and 24 hours'}), 400
        if data['sleep_hours'] < 0 or data['sleep_hours'] > 24:
            return jsonify({'error': 'Sleep hours must be between 0 and 24 hours'}), 400
        if data['stress_level'] < 1 or data['stress_level'] > 10:
            return jsonify({'error': 'Stress level must be between 1 and 10'}), 400
        
        if data['social_media_hours'] + data['gaming_hours'] > data['screen_time_hours'] + 0.5:
            return jsonify({'error': 'Social media and gaming hours cannot exceed total screen time'}), 400
        
        save_user_submission(data)
        
        return jsonify({
            'success': True,
            'message': 'Data saved successfully! Thank you for contributing.',
            'id': len(load_user_submissions())
        }), 200
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-dashboard-data')
def get_dashboard_data():
    """API endpoint to get updated dashboard data"""
    try:
        stats_data = get_dashboard_stats(use_combined_data=True)
        
        if stats_data is None:
            return jsonify({'error': 'No data available'}), 404
        
        return jsonify(stats_data)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-submissions')
def get_submissions():
    """API endpoint to get all user submissions"""
    try:
        submissions = load_user_submissions()
        return jsonify({
            'success': True,
            'submissions': submissions,
            'total': len(submissions)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        required_fields = ['age', 'gender', 'screen_time_hours', 
                          'social_media_hours', 'gaming_hours', 
                          'sleep_hours', 'stress_level']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        result = predict_addiction_risk(data)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    """404 error handler"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    """500 error handler"""
    return render_template('500.html'), 500

# Run the app
if __name__ == '__main__':
    os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)
    
    if not os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'w') as f:
            json.dump([], f)
        print("✓ Created empty submissions file")
    
    print("\n" + "="*60)
    print("Mobile Phone Addiction Risk Predictor")
    print("="*60)
    print(f"Model loaded: {'✓' if model else '✗'}")
    print(f"Scaler loaded: {'✓' if scaler else '✗'}")
    print(f"Data file: {'✓' if os.path.exists(DATA_PATH) else '✗'}")
    print(f"Submissions file: {'✓' if os.path.exists(USER_DATA_FILE) else '✗'}")
    print("="*60)
    print("\nStarting Flask application...")
    print("Access the application at: http://localhost:5000")
    print("Press Ctrl+C to stop\n")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)