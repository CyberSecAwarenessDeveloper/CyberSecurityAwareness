import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

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
        
        # Convert categorical variables to numerical using Label Encoding
        le = LabelEncoder()
        categorical_columns = df.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            df[column] = le.fit_transform(df[column])
        
        logging.info(f"Data shape after preprocessing: {df.shape}")
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

def explore_data(df):
    # Basic statistics
    print("\nDataset Shape:", df.shape)
    print("\nData Types:\n", df.dtypes)
    
    # Distribution of target variable
    target_col = 'How would you rate your cybersecurity knowledge based on your experience and security habits?.Rate your cybersecurity knowledge.'
    print(f"\nDistribution of {target_col}:")
    print(df[target_col].value_counts(normalize=True) * 100)
    
    # Correlation with target
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    correlations = df[numerical_cols].corr()[target_col].sort_values(ascending=False)
    print("\nTop Correlations with Target Variable:")
    print(correlations.head(10))
    
    # Visualize distributions
    plt.figure(figsize=(10, 6))
    sns.histplot(df[target_col], kde=True)
    plt.title('Distribution of Cybersecurity Knowledge Levels')
    plt.savefig('knowledge_distribution.png')
    
    return correlations

def compare_models(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models to test
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Neural Network': MLPClassifier(max_iter=1000, random_state=42)
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        print(f"{name} Accuracy: {accuracy:.4f}")
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values())
    plt.title('Model Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    
    # Return best model
    best_model_name = max(results, key=results.get)
    return models[best_model_name], best_model_name

def engineer_features(df):
    # Copy the dataframe to avoid modifying the original
    df_engineered = df.copy()
    
    # Create aggregate features (examples based on cybersecurity domain):
    # These are placeholder examples - you'll need to adjust based on your actual columns
    
    # Example: Count how many security best practices someone follows
    best_practice_cols = [col for col in df.columns if 'practice' in col.lower() or 'habit' in col.lower()]
    if best_practice_cols:
        df_engineered['security_practice_score'] = df[best_practice_cols].sum(axis=1)
    
    # Example: Create risk awareness score
    risk_cols = [col for col in df.columns if 'risk' in col.lower() or 'threat' in col.lower()]
    if risk_cols:
        df_engineered['risk_awareness_score'] = df[risk_cols].sum(axis=1)
    
    # Add to engineer_features to capture more patterns
    awareness_cols = [col for col in df.columns if 'aware' in col.lower() or 'knowledge' in col.lower()]
    if awareness_cols:
        df_engineered['awareness_score'] = df[awareness_cols].sum(axis=1)
    
    # Add other domain-specific aggregations as needed
    
    return df_engineered

def make_recommendations(model, features):
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': features.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nRecommendations for Improving Cybersecurity Awareness:")
    for _, row in feature_importance.head(5).iterrows():
        print(f"\n- Focus on improving {row['feature']}: This factor has a {row['importance']:.2%} impact on security awareness")

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
    from sklearn.metrics import classification_report, confusion_matrix
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

def predict_awareness(new_data, model_file='cybersecurity_awareness_model.pkl'):
    # Load the model
    model = joblib.load(model_file)
    # Preprocess new data (similar to how training data was processed)
    # Make predictions
    predictions = model.predict(new_data)
    return predictions

def main():
    file_path = 'Cybersecurity Awareness Survey.xlsx'
    target_column = 'How would you rate your cybersecurity knowledge based on your experience and security habits?.Rate your cybersecurity knowledge.'
    
    logging.info("Loading and preprocessing data...")
    df = load_and_preprocess_data(file_path)
    
    # Explore the data
    logging.info("Exploring data...")
    explore_data(df)
    
    # Engineer features
    logging.info("Engineering features...")
    df_engineered = engineer_features(df)
    
    # Check if target column exists
    if target_column not in df_engineered.columns:
        logging.error(f"Target column '{target_column}' not found in dataset.")
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
    
    logging.info("Preparing features and target variables...")
    X, y, features_df = prepare_features_and_target(df_engineered, target_column)
    
    # Compare different models
    logging.info("Comparing different models...")
    best_model, model_name = compare_models(X, y)
    logging.info(f"Best model: {model_name}")
    
    # Evaluate the best model in detail
    logging.info(f"Training and evaluating the best model ({model_name})...")
    final_model = train_and_evaluate_model(X, y, features_df)
    
    logging.info("Generating recommendations...")
    make_recommendations(final_model, features_df)
    
    # Save the model
    joblib.dump(final_model, 'cybersecurity_awareness_model.pkl')

    # Add to main() after preprocessing
    logging.info(f"Training model on {len(df.columns)} questions/columns: {df.columns}")
