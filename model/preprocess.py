"""
Data Preprocessing Module for Mobile Phone Addiction Prediction
Handles data cleaning, feature engineering, and preprocessing for the Logistic Regression model
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Data Preprocessor class for handling all preprocessing operations
    """
    
    def __init__(self, data_path=None):
        """
        Initialize the preprocessor
        
        Parameters:
        -----------
        data_path : str, optional
            Path to the dataset CSV file
        """
        self.data_path = data_path
        self.scaler = None
        self.feature_columns = None
        self.target_column = 'addiction_risk'
        self.categorical_features = ['gender']
        self.numerical_features = ['age', 'screen_time_hours', 'social_media_hours', 
                                   'gaming_hours', 'sleep_hours', 'stress_level']
        
    def load_data(self, data_path=None):
        """
        Load the dataset from CSV file
        
        Parameters:
        -----------
        data_path : str, optional
            Path to the dataset CSV file
            
        Returns:
        --------
        pandas.DataFrame
            Loaded dataset
        """
        path = data_path or self.data_path
        if path is None:
            raise ValueError("Data path not provided")
        
        try:
            df = pd.read_csv(path)
            print(f"Data loaded successfully from {path}")
            print(f"Dataset shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            return df
        except FileNotFoundError:
            print(f"Error: File not found at {path}")
            raise
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def explore_data(self, df):
        """
        Explore and display basic information about the dataset
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        """
        print("\n" + "="*50)
        print("DATA EXPLORATION")
        print("="*50)
        
        print("\nFirst 5 rows:")
        print(df.head())
        
        print("\nDataset Info:")
        print(df.info())
        
        print("\nStatistical Summary:")
        print(df.describe())
        
        print("\nMissing Values:")
        print(df.isnull().sum())
        
        print("\nTarget Variable Distribution:")
        print(df[self.target_column].value_counts())
        print(f"High Risk Percentage: {(df[self.target_column].sum() / len(df) * 100):.2f}%")
        
    def clean_data(self, df):
        """
        Clean the dataset by handling missing values and outliers
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
            
        Returns:
        --------
        pandas.DataFrame
            Cleaned dataframe
        """
        print("\n" + "="*50)
        print("DATA CLEANING")
        print("="*50)
        
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Check for missing values
        missing_before = df_clean.isnull().sum().sum()
        if missing_before > 0:
            print(f"Missing values found: {missing_before}")
            
            # Fill numerical missing values with median
            for col in self.numerical_features:
                if col in df_clean.columns and df_clean[col].isnull().any():
                    median_val = df_clean[col].median()
                    df_clean[col].fillna(median_val, inplace=True)
                    print(f"Filled missing values in {col} with median: {median_val}")
            
            # Fill categorical missing values with mode
            for col in self.categorical_features:
                if col in df_clean.columns and df_clean[col].isnull().any():
                    mode_val = df_clean[col].mode()[0]
                    df_clean[col].fillna(mode_val, inplace=True)
                    print(f"Filled missing values in {col} with mode: {mode_val}")
        
        # Check for outliers using IQR method
        print("\nOutlier Detection:")
        for col in self.numerical_features:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
                
                if len(outliers) > 0:
                    print(f"  {col}: {len(outliers)} outliers detected")
                    # Cap outliers instead of removing them
                    df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
                    print(f"    Capped outliers to range [{lower_bound:.2f}, {upper_bound:.2f}]")
                else:
                    print(f"  {col}: No outliers detected")
        
        # Validate data ranges
        print("\nData Range Validation:")
        validation_rules = {
            'age': (10, 80),
            'screen_time_hours': (0, 24),
            'social_media_hours': (0, 24),
            'gaming_hours': (0, 24),
            'sleep_hours': (0, 24),
            'stress_level': (1, 10),
            'gender': (0, 1)
        }
        
        for col, (min_val, max_val) in validation_rules.items():
            if col in df_clean.columns:
                invalid = df_clean[(df_clean[col] < min_val) | (df_clean[col] > max_val)]
                if len(invalid) > 0:
                    print(f"  Warning: {col} has {len(invalid)} values outside range [{min_val}, {max_val}]")
                    # Clip values to valid range
                    df_clean[col] = df_clean[col].clip(min_val, max_val)
        
        missing_after = df_clean.isnull().sum().sum()
        print(f"\nMissing values after cleaning: {missing_after}")
        
        return df_clean
    
    def feature_engineering(self, df):
        """
        Create new features from existing ones for better prediction
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with engineered features
        """
        print("\n" + "="*50)
        print("FEATURE ENGINEERING")
        print("="*50)
        
        df_engineered = df.copy()
        
        # 1. Screen time ratio (screen_time / total hours awake)
        # Assuming 16 waking hours per day
        df_engineered['screen_time_ratio'] = df_engineered['screen_time_hours'] / 16
        print("Created feature: screen_time_ratio")
        
        # 2. Social media proportion of screen time
        df_engineered['social_media_proportion'] = df_engineered['social_media_hours'] / (df_engineered['screen_time_hours'] + 0.001)
        print("Created feature: social_media_proportion")
        
        # 3. Gaming proportion of screen time
        df_engineered['gaming_proportion'] = df_engineered['gaming_hours'] / (df_engineered['screen_time_hours'] + 0.001)
        print("Created feature: gaming_proportion")
        
        # 4. Sleep deficiency (recommended 8 hours)
        df_engineered['sleep_deficiency'] = 8 - df_engineered['sleep_hours']
        df_engineered['sleep_deficiency'] = df_engineered['sleep_deficiency'].clip(lower=0)
        print("Created feature: sleep_deficiency")
        
        # 5. Screen time to sleep ratio
        df_engineered['screen_sleep_ratio'] = df_engineered['screen_time_hours'] / (df_engineered['sleep_hours'] + 0.001)
        print("Created feature: screen_sleep_ratio")
        
        # 6. Total digital engagement (screen_time + social_media + gaming weighted)
        # Gaming and social media are weighted more as they're more addictive
        df_engineered['digital_engagement_score'] = (
            df_engineered['screen_time_hours'] * 0.3 +
            df_engineered['social_media_hours'] * 0.4 +
            df_engineered['gaming_hours'] * 0.3
        )
        print("Created feature: digital_engagement_score")
        
        # 7. Stress to sleep ratio
        df_engineered['stress_sleep_ratio'] = df_engineered['stress_level'] / (df_engineered['sleep_hours'] + 0.001)
        print("Created feature: stress_sleep_ratio")
        
        print(f"\nTotal features after engineering: {len(df_engineered.columns)}")
        
        return df_engineered
    
    def prepare_features(self, df):
        """
        Prepare features for model training
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
            
        Returns:
        --------
        tuple
            (X, y) - Features and target arrays
        """
        print("\n" + "="*50)
        print("FEATURE PREPARATION")
        print("="*50)
        
        # Define all features to use
        all_features = self.numerical_features + self.categorical_features + [
            'screen_time_ratio', 'social_media_proportion', 'gaming_proportion',
            'sleep_deficiency', 'screen_sleep_ratio', 'digital_engagement_score',
            'stress_sleep_ratio'
        ]
        
        # Filter to only features that exist in dataframe
        available_features = [f for f in all_features if f in df.columns]
        self.feature_columns = available_features
        
        # Separate features and target
        X = df[available_features].copy()
        y = df[self.target_column].copy()
        
        print(f"Features selected: {available_features}")
        print(f"Feature count: {len(available_features)}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def scale_features(self, X_train, X_test):
        """
        Scale features using StandardScaler
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        X_test : array-like
            Test features
            
        Returns:
        --------
        tuple
            (X_train_scaled, X_test_scaled) - Scaled features
        """
        print("\n" + "="*50)
        print("FEATURE SCALING")
        print("="*50)
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Features scaled using StandardScaler")
        print(f"Scaler mean: {self.scaler.mean_}")
        print(f"Scaler scale: {self.scaler.scale_}")
        
        return X_train_scaled, X_test_scaled
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target
        test_size : float
            Proportion of test data
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        print("\n" + "="*50)
        print("DATA SPLITTING")
        print("="*50)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
        print(f"Test set size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
        print(f"Training target distribution:\n{y_train.value_counts()}")
        print(f"Test target distribution:\n{y_test.value_counts()}")
        
        return X_train, X_test, y_train, y_test
    
    def validate_input(self, input_data):
        """
        Validate user input for prediction
        
        Parameters:
        -----------
        input_data : dict
            User input data
            
        Returns:
        --------
        tuple
            (is_valid, error_message)
        """
        # Define validation rules
        validation_rules = {
            'age': (10, 80, 'Age must be between 10 and 80'),
            'gender': (0, 1, 'Gender must be 0 (Female) or 1 (Male)'),
            'screen_time_hours': (0, 24, 'Screen time must be between 0 and 24 hours'),
            'social_media_hours': (0, 24, 'Social media hours must be between 0 and 24'),
            'gaming_hours': (0, 24, 'Gaming hours must be between 0 and 24'),
            'sleep_hours': (0, 24, 'Sleep hours must be between 0 and 24'),
            'stress_level': (1, 10, 'Stress level must be between 1 and 10')
        }
        
        for field, (min_val, max_val, error_msg) in validation_rules.items():
            if field not in input_data:
                return False, f"Missing field: {field}"
            
            value = input_data[field]
            try:
                value = float(value)
            except (ValueError, TypeError):
                return False, f"Invalid value for {field}: must be a number"
            
            if value < min_val or value > max_val:
                return False, error_msg
        
        # Additional logical validation
        screen_time = float(input_data['screen_time_hours'])
        social_media = float(input_data['social_media_hours'])
        gaming = float(input_data['gaming_hours'])
        
        if social_media + gaming > screen_time + 0.5:  # Allow small rounding error
            return False, "Social media and gaming hours cannot exceed total screen time"
        
        return True, None
    
    def preprocess_for_prediction(self, input_data):
        """
        Preprocess single user input for prediction
        
        Parameters:
        -----------
        input_data : dict
            User input data
            
        Returns:
        --------
        numpy.ndarray
            Preprocessed features ready for prediction
        """
        # Validate input
        is_valid, error_msg = self.validate_input(input_data)
        if not is_valid:
            raise ValueError(error_msg)
        
        # Convert to dataframe
        df = pd.DataFrame([input_data])
        
        # Add engineered features
        df = self.feature_engineering(df)
        
        # Select only the features used during training
        if self.feature_columns is None:
            raise ValueError("Feature columns not set. Please fit the preprocessor first.")
        
        # Ensure all required features are present
        for col in self.feature_columns:
            if col not in df.columns:
                if col in self.numerical_features:
                    df[col] = 0  # Default value for missing numerical features
                else:
                    # For engineered features, we need to compute them
                    # This should not happen if feature_engineering is called
                    raise ValueError(f"Required feature {col} not found")
        
        # Select features in correct order
        X = df[self.feature_columns].values
        
        # Scale features
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            raise ValueError("Scaler not fitted. Please fit the preprocessor first.")
        
        return X_scaled
    
    def get_feature_importance_info(self, model):
        """
        Get feature importance information from the trained model
        
        Parameters:
        -----------
        model : sklearn model
            Trained model with coef_ attribute
            
        Returns:
        --------
        pandas.DataFrame
            Feature importance rankings
        """
        if hasattr(model, 'coef_'):
            coefficients = model.coef_[0]
            importance_df = pd.DataFrame({
                'Feature': self.feature_columns,
                'Coefficient': coefficients,
                'Absolute_Importance': np.abs(coefficients)
            })
            importance_df = importance_df.sort_values('Absolute_Importance', ascending=False)
            return importance_df
        else:
            print("Model does not have coefficient attribute")
            return None
    
    def save_preprocessor(self, path):
        """
        Save the preprocessor (scaler and feature columns) to disk
        
        Parameters:
        -----------
        path : str
            Path to save the preprocessor
        """
        import joblib
        
        preprocessor_data = {
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features
        }
        
        joblib.dump(preprocessor_data, path)
        print(f"Preprocessor saved to {path}")
    
    def load_preprocessor(self, path):
        """
        Load the preprocessor from disk
        
        Parameters:
        -----------
        path : str
            Path to load the preprocessor from
        """
        import joblib
        
        preprocessor_data = joblib.load(path)
        self.scaler = preprocessor_data['scaler']
        self.feature_columns = preprocessor_data['feature_columns']
        self.numerical_features = preprocessor_data.get('numerical_features', self.numerical_features)
        self.categorical_features = preprocessor_data.get('categorical_features', self.categorical_features)
        print(f"Preprocessor loaded from {path}")


def create_sample_dataset(output_path='data/mobile_data.csv', n_samples=100):
    """
    Create a larger sample dataset for demonstration purposes
    
    Parameters:
    -----------
    output_path : str
        Path to save the generated dataset
    n_samples : int
        Number of samples to generate
    """
    np.random.seed(42)
    
    # Generate synthetic data
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
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Sample dataset created with {n_samples} samples at {output_path}")
    print(f"Target distribution:\n{df['addiction_risk'].value_counts()}")


if __name__ == "__main__":
    # Example usage
    print("Testing Data Preprocessor...")
    
    # Create sample dataset if needed
    create_sample_dataset()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(data_path='data/mobile_data.csv')
    
    # Load and explore data
    df = preprocessor.load_data()
    preprocessor.explore_data(df)
    
    # Clean data
    df_clean = preprocessor.clean_data(df)
    
    # Feature engineering
    df_engineered = preprocessor.feature_engineering(df_clean)
    
    # Prepare features
    X, y = preprocessor.prepare_features(df_engineered)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    # Scale features
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    print("\n" + "="*50)
    print("PREPROCESSING COMPLETE")
    print("="*50)
    print(f"Final training data shape: {X_train_scaled.shape}")
    print(f"Final test data shape: {X_test_scaled.shape}")
    print(f"Features used: {preprocessor.feature_columns}")