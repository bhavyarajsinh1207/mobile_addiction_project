"""
Model Training Script for Mobile Phone Addiction Prediction
Trains a Logistic Regression model using the dataset and saves it as model.pkl
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import the preprocessor module
try:
    from model.preprocess import DataPreprocessor, create_sample_dataset
except ImportError:
    print("Warning: Could not import DataPreprocessor. Using basic preprocessing.")
    DataPreprocessor = None

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'mobile_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
PREPROCESSOR_PATH = os.path.join(BASE_DIR, 'preprocessor.pkl')
FEATURE_INFO_PATH = os.path.join(BASE_DIR, 'feature_info.pkl')


def check_data_file():
    """
    Check if data file exists, if not create a sample dataset
    """
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found at {DATA_PATH}")
        print("Creating sample dataset...")
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'age': np.random.randint(15, 65, n_samples),
            'gender': np.random.randint(0, 2, n_samples),
            'screen_time_hours': np.random.uniform(2, 12, n_samples),
            'social_media_hours': np.random.uniform(0, 8, n_samples),
            'gaming_hours': np.random.uniform(0, 6, n_samples),
            'sleep_hours': np.random.uniform(4, 9, n_samples),
            'stress_level': np.random.randint(1, 11, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Ensure social_media + gaming doesn't exceed screen_time
        df['social_media_hours'] = df['social_media_hours'].clip(upper=df['screen_time_hours'] * 0.8)
        df['gaming_hours'] = df['gaming_hours'].clip(upper=df['screen_time_hours'] - df['social_media_hours'])
        
        # Generate target based on rules
        addiction_risk = []
        for _, row in df.iterrows():
            score = 0
            if row['screen_time_hours'] > 6:
                score += 2
            if row['social_media_hours'] > 3:
                score += 1
            if row['gaming_hours'] > 2:
                score += 1
            if row['sleep_hours'] < 6:
                score += 1
            if row['stress_level'] > 7:
                score += 1
            addiction_risk.append(1 if score >= 3 else 0)
        
        df['addiction_risk'] = addiction_risk
        df.to_csv(DATA_PATH, index=False)
        print(f"Sample dataset created with {n_samples} samples at {DATA_PATH}")
        print(f"Target distribution:\n{df['addiction_risk'].value_counts()}")
        return True
    
    return True


def load_and_preprocess_data_basic():
    """
    Load the dataset and preprocess features (basic version without preprocessor)
    """
    # Load the dataset
    df = pd.read_csv(DATA_PATH)
    
    print(f"Dataset loaded: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Features (X) - all columns except addiction_risk
    feature_columns = ['age', 'gender', 'screen_time_hours', 'social_media_hours', 
                       'gaming_hours', 'sleep_hours', 'stress_level']
    
    # Check if all feature columns exist
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")
    
    X = df[feature_columns].copy()
    y = df['addiction_risk'].copy()
    
    # Basic data cleaning
    print("\nPerforming basic data cleaning...")
    
    # Check for missing values
    if X.isnull().any().any():
        print("Missing values found. Filling with median...")
        X = X.fillna(X.median())
    
    if y.isnull().any():
        print("Missing values in target. Dropping rows...")
        mask = y.notnull()
        X = X[mask]
        y = y[mask]
    
    # Check for invalid values
    X = X.clip(lower=0)
    X['age'] = X['age'].clip(10, 80)
    X['screen_time_hours'] = X['screen_time_hours'].clip(0, 24)
    X['social_media_hours'] = X['social_media_hours'].clip(0, 24)
    X['gaming_hours'] = X['gaming_hours'].clip(0, 24)
    X['sleep_hours'] = X['sleep_hours'].clip(0, 24)
    X['stress_level'] = X['stress_level'].clip(1, 10)
    X['gender'] = X['gender'].clip(0, 1)
    
    return X, y, feature_columns


def load_and_preprocess_data_advanced():
    """
    Load and preprocess data using the DataPreprocessor class
    """
    if DataPreprocessor is None:
        return None
    
    try:
        preprocessor = DataPreprocessor(data_path=DATA_PATH)
        df = preprocessor.load_data()
        df_clean = preprocessor.clean_data(df)
        df_engineered = preprocessor.feature_engineering(df_clean)
        X, y = preprocessor.prepare_features(df_engineered)
        
        # Save preprocessor for later use
        preprocessor.save_preprocessor(PREPROCESSOR_PATH)
        
        return X, y, preprocessor.feature_columns, preprocessor
        
    except Exception as e:
        print(f"Advanced preprocessing failed: {e}")
        print("Falling back to basic preprocessing...")
        return None


def train_model(use_advanced_preprocessing=False):
    """
    Train the Logistic Regression model
    
    Parameters:
    -----------
    use_advanced_preprocessing : bool
        Whether to use advanced preprocessing with feature engineering
    """
    print("="*60)
    print("MOBILE PHONE ADDICTION PREDICTION - MODEL TRAINING")
    print("="*60)
    
    # Check if data file exists
    if not check_data_file():
        print("Error: Could not create or find data file.")
        return None, None, None
    
    # Load and preprocess data
    print("\n" + "="*60)
    print("STEP 1: LOADING AND PREPROCESSING DATA")
    print("="*60)
    
    X = None
    y = None
    feature_columns = None
    preprocessor = None
    
    if use_advanced_preprocessing:
        result = load_and_preprocess_data_advanced()
        if result:
            X, y, feature_columns, preprocessor = result
            print(f"\nAdvanced preprocessing completed!")
            print(f"Features used: {feature_columns}")
    
    if X is None:
        # Fall back to basic preprocessing
        X, y, feature_columns = load_and_preprocess_data_basic()
        print(f"\nBasic preprocessing completed!")
        print(f"Features used: {feature_columns}")
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Feature columns: {feature_columns}")
    print(f"Target distribution:")
    print(f"  Low Risk (0): {(y == 0).sum()} ({((y == 0).sum() / len(y) * 100):.1f}%)")
    print(f"  High Risk (1): {(y == 1).sum()} ({((y == 1).sum() / len(y) * 100):.1f}%)")
    
    # Split the data
    print("\n" + "="*60)
    print("STEP 2: SPLITTING DATA")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Test set size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    print(f"Training target distribution: {y_train.value_counts().to_dict()}")
    print(f"Test target distribution: {y_test.value_counts().to_dict()}")
    
    # Standardize the features
    print("\n" + "="*60)
    print("STEP 3: FEATURE SCALING")
    print("="*60)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features standardized using StandardScaler")
    print(f"Scaler mean: {scaler.mean_}")
    print(f"Scaler scale: {scaler.scale_}")
    
    # Train Logistic Regression model
    print("\n" + "="*60)
    print("STEP 4: TRAINING LOGISTIC REGRESSION MODEL")
    print("="*60)
    
    model = LogisticRegression(
        random_state=42, 
        max_iter=1000,
        C=1.0,
        solver='lbfgs'
    )
    model.fit(X_train_scaled, y_train)
    
    print("Model training completed!")
    print(f"Model coefficients: {model.coef_[0]}")
    print(f"Model intercept: {model.intercept_[0]}")
    
    # Evaluate the model
    print("\n" + "="*60)
    print("STEP 5: MODEL EVALUATION")
    print("="*60)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nModel Performance:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  AUC Score: {auc_score:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"\nConfusion Matrix Interpretation:")
    print(f"  True Negatives (Low Risk correctly identified): {cm[0,0]}")
    print(f"  False Positives (Low Risk incorrectly as High): {cm[0,1]}")
    print(f"  False Negatives (High Risk incorrectly as Low): {cm[1,0]}")
    print(f"  True Positives (High Risk correctly identified): {cm[1,1]}")
    
    # Cross-validation
    print("\n" + "="*60)
    print("STEP 6: CROSS-VALIDATION")
    print("="*60)
    
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"5-Fold Cross-Validation Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Save the model and scaler
    print("\n" + "="*60)
    print("STEP 7: SAVING MODEL AND ARTIFACTS")
    print("="*60)
    
    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"✓ Model saved to: {MODEL_PATH}")
    
    # Save scaler
    joblib.dump(scaler, SCALER_PATH)
    print(f"✓ Scaler saved to: {SCALER_PATH}")
    
    # Save feature info
    feature_info = {
        'feature_columns': feature_columns,
        'accuracy': accuracy,
        'auc_score': auc_score,
        'model_type': 'LogisticRegression',
        'coefficients': model.coef_[0].tolist(),
        'intercept': model.intercept_[0],
        'cv_mean_accuracy': cv_scores.mean(),
        'cv_std_accuracy': cv_scores.std()
    }
    joblib.dump(feature_info, FEATURE_INFO_PATH)
    print(f"✓ Feature info saved to: {FEATURE_INFO_PATH}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return model, scaler, feature_info


def predict_risk(model, scaler, features):
    """
    Make a prediction for a single user
    
    Parameters:
    -----------
    model : sklearn model
        Trained Logistic Regression model
    scaler : StandardScaler
        Fitted scaler
    features : list or numpy array
        Feature values in the correct order
        
    Returns:
    --------
    dict
        Prediction results
    """
    # Convert to numpy array if needed
    if isinstance(features, list):
        features_array = np.array(features).reshape(1, -1)
    else:
        features_array = features.reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features_array)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    return {
        'risk': int(prediction),
        'risk_level': 'High Risk' if prediction == 1 else 'Low Risk',
        'probability_low': probability[0],
        'probability_high': probability[1],
        'confidence': max(probability[0], probability[1])
    }


def test_prediction(model, scaler, feature_columns):
    """
    Test the model with sample inputs
    """
    print("\n" + "="*60)
    print("TESTING MODEL WITH SAMPLE INPUTS")
    print("="*60)
    
    # Sample test cases
    test_cases = [
        {
            'name': 'Low Risk User',
            'features': [25, 0, 3.5, 1.5, 0.5, 8.0, 3],
            'expected': 'Low Risk'
        },
        {
            'name': 'High Risk User',
            'features': [20, 1, 8.0, 5.0, 2.5, 5.0, 9],
            'expected': 'High Risk'
        },
        {
            'name': 'Moderate User',
            'features': [22, 0, 6.0, 3.0, 1.5, 6.5, 6],
            'expected': 'Varies'
        }
    ]
    
    for test in test_cases:
        result = predict_risk(model, scaler, test['features'])
        print(f"\n{test['name']}:")
        print(f"  Features: {dict(zip(feature_columns, test['features']))}")
        print(f"  Prediction: {result['risk_level']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Low Risk Probability: {result['probability_low']:.2%}")
        print(f"  High Risk Probability: {result['probability_high']:.2%}")
        print(f"  Expected: {test['expected']}")
        print(f"  {'✓ Correct' if result['risk_level'] == test['expected'] or test['expected'] == 'Varies' else '✗ Incorrect'}")


if __name__ == "__main__":
    # Train the model
    model, scaler, feature_info = train_model(use_advanced_preprocessing=False)
    
    if model is not None:
        # Test the model with sample inputs
        test_prediction(model, scaler, feature_info['feature_columns'])
        
        print("\n" + "="*60)
        print("MODEL READY FOR DEPLOYMENT")
        print("="*60)
        print("\nTo use the model in your Flask app:")
        print("  from utils.helper import load_model, predict_addiction_risk")
        print("  model, scaler, info = load_model()")
        print("  result = predict_addiction_risk(your_input_data)")
    else:
        print("\n❌ Model training failed. Please check the errors above.")