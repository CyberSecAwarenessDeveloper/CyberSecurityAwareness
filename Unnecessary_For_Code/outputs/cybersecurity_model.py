import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_data(file_path):
    try:
        # Load the Excel file
        df = pd.read_excel(file_path)
        
        # Drop non-relevant columns that shouldn't be used in the analysis
        columns_to_drop = ['Id', 'Start time', 'Completion time', 'Email', 'Name']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
        
        # Handle missing values
        # For categorical columns, fill with the most frequent value
        # For numerical columns, fill with the median
        for column in df.columns:
            if df[column].dtype == 'object':
                df[column] = df[column].fillna(df[column].mode()[0] if not df[column].mode().empty else 'Unknown')
            else:
                df[column] = df[column].fillna(df[column].median() if not pd.isna(df[column].median()) else 0)
        
        # Check if the dataframe has rows
        if df.empty:
            raise ValueError("After preprocessing, no data remains. Please check your dataset.")
        
        # Convert categorical variables to numerical using Label Encoding
        le = LabelEncoder()
        categorical_columns = df.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            df[column] = le.fit_transform(df[column])
        
        logging.info(f"Data shape after preprocessing: {df.shape}")
        logging.info("Data columns: %s", df.columns)
        return df
    except Exception as e:
        logging.error(f"Error loading and preprocessing data: {e}")
        raise

def prepare_features_and_target(df, target_column):
    # Define your target variable
    target = df[target_column]
    
    # Remove target variable from features
    features = df.drop(target_column, axis=1)
    
    # Scale the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    return features_scaled, target, features  # Return original features for column names

def train_and_evaluate_model(X, y, features):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Print model evaluation metrics
    print("\nModel Evaluation:")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': features.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Visualize feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
    plt.title('Top 10 Most Important Features for Cybersecurity Awareness')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    return rf_model

def make_recommendations(model, features):
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': features.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nRecommendations for Improving Cybersecurity Awareness:")
    for _, row in feature_importance.head(5).iterrows():
        print(f"\n- Focus on improving {row['feature']}: This factor has a {row['importance']:.2%} impact on security awareness")

def main():
    file_path = 'Cybersecurity Awareness Survey.xlsx'  # Parameterize file path
    target_column = 'How would you rate your cybersecurity knowledge based on your experience and security habits?.Rate your cybersecurity knowledge.'
    
    logging.info("Loading and preprocessing data...")
    df = load_and_preprocess_data(file_path)
    
    # Check if target column exists
    if target_column not in df.columns:
        logging.error(f"Target column '{target_column}' not found in dataset. Available columns: {df.columns}")
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
    
    logging.info("Preparing features and target variables...")
    X, y, features_df = prepare_features_and_target(df, target_column)  # Get original features for column names
    
    logging.info("Training and evaluating the model...")
    model = train_and_evaluate_model(X, y, features_df)
    
    logging.info("Generating recommendations...")
    make_recommendations(model, features_df)

if __name__ == "__main__":
    main() 