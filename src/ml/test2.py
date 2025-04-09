import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

def load_and_preprocess_data(file_path):
    try:
        # Load data
        print(f"Attempting to load awareness survey data from {file_path}...")
        df = pd.read_excel(file_path)
        
        print(f"Survey data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
        print("Column headers:", df.columns.tolist()[:5], "...")
        
        # Check for expected target column
        target_col = 'How would you rate your cybersecurity knowledge based on your experience and security habits?.Rate your cybersecurity knowledge.'
        if target_col not in df.columns:
            print(f"Warning: Expected target column '{target_col}' not found.")
            print("Available columns:", df.columns.tolist())
            
            # Try to find a suitable target column
            knowledge_cols = [col for col in df.columns if 'knowledge' in col.lower()]
            if knowledge_cols:
                target_col = knowledge_cols[0]
                print(f"Using '{target_col}' as target column instead.")
            else:
                print("No suitable target column found. The model may not train correctly.")
        
        # Handle datetime columns - convert to useful features or drop
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        for col in datetime_cols:
            # Try to extract potentially useful features from datetime
            try:
                # Extract day of week, hour of day as features
                df[f'{col}_day'] = df[col].dt.dayofweek
                df[f'{col}_hour'] = df[col].dt.hour
                
                # If we have start and completion time, calculate duration
                if 'Start time' in df.columns and 'Completion time' in df.columns:
                    df['duration_minutes'] = (df['Completion time'] - df['Start time']).dt.total_seconds() / 60
            except:
                print(f"Could not extract features from datetime column {col}")
            
            # Drop the original datetime column
            df = df.drop(columns=[col])
        
        # Handle missing values in numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # Handle missing values in categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna('Unknown')
            # Convert categorical variables to numeric
            df[col] = df[col].astype('category').cat.codes
        
        print(f"Data preprocessing completed successfully. Ready for model training.")
        return df
        
    except Exception as e:
        print(f"Error loading awareness data: {str(e)}")
        return None

def load_cve_data(file_path):
    # Load CVE dataset
    df = pd.read_csv(file_path)
    
    # Handle missing values
    df = df.fillna({'description': '', 'severity': 'unknown'})
    
    # Convert severity to numerical if needed
    severity_map = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1, 'unknown': 0}
    if 'severity' in df.columns:
        df['severity_score'] = df['severity'].map(lambda x: severity_map.get(x.lower(), 0) if isinstance(x, str) else 0)
    
    return df

def load_text_threats(file_path):
    # Load text threats dataset
    df = pd.read_csv(file_path)
    
    # Handle missing values in text columns
    text_columns = [col for col in df.columns if df[col].dtype == 'object']
    for col in text_columns:
        df[col] = df[col].fillna('')
    
    # If there's a label column, ensure it's properly formatted
    if 'label' in df.columns:
        # Convert text labels to binary/numeric if needed
        df['label'] = df['label'].map({'threat': 1, 'benign': 0}) if df['label'].dtype == 'object' else df['label']
    
    return df

def load_malware_data(file_path):
    # Load malware attacks dataset
    df = pd.read_csv(file_path)
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna('unknown')
    
    # Encode any categorical variables if needed
    # For common malware types or attack vectors
    if 'malware_type' in df.columns:
        df['malware_type'] = df['malware_type'].astype('category').cat.codes
    
    if 'is_malicious' in df.columns and df['is_malicious'].dtype == 'object':
        df['is_malicious'] = df['is_malicious'].map({'yes': 1, 'no': 0, True: 1, False: 0})
    
    return df

def load_security_dataset(dataset_type):
    try:
        if dataset_type == "awareness":
            file_path = 'Cybersecurity Awareness Survey.xlsx'
            print(f"Loading awareness data from {file_path}...")
            return load_and_preprocess_data(file_path)
        elif dataset_type == "cve":
            file_path = 'cve_dataset.csv'
            print(f"Loading CVE data from {file_path}...")
            return load_cve_data(file_path)
        elif dataset_type == "text_threats":
            file_path = 'text_threats.csv'
            print(f"Loading text threats data from {file_path}...")
            return load_text_threats(file_path)
        elif dataset_type == "malware":
            file_path = 'malware_attacks.csv'
            print(f"Loading malware data from {file_path}...")
            return load_malware_data(file_path)
    except FileNotFoundError:
        print(f"Warning: Dataset file for '{dataset_type}' not found. Skipping this model.")
        return None

def compare_models(X, y):
    # Split data
    from sklearn.model_selection import train_test_split
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
    from sklearn.metrics import accuracy_score
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        print(f"{name} Accuracy: {accuracy:.4f}")
    
    # Visualize results
    import matplotlib.pyplot as plt
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

def train_and_evaluate_model(X, y, features):
    # Split the data into training and testing sets
    from sklearn.model_selection import train_test_split
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
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
    plt.title('Top 10 Most Important Features for Cybersecurity Awareness')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    return rf_model

models = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Neural Network': MLPClassifier()
}

# Commented out loose code that uses undefined X, y
# best_model, model_name = compare_models(X, y)
# final_model = train_and_evaluate_model(X, y, features_df)
# joblib.dump(final_model, 'cybersecurity_model.pkl')

# Create prediction function
def predict_cybersecurity_risk(new_data, model_file='cybersecurity_model.pkl'):
    model = joblib.load(model_file)
    return model.predict(new_data)

def extract_email_features(emails_df):
    # Simple feature extraction from emails
    vectorizer = TfidfVectorizer(max_features=2000)
    X = vectorizer.fit_transform(emails_df['content'] if 'content' in emails_df.columns else emails_df.iloc[:,0])
    return X

# For text processing
def process_text_data(text_df):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(text_df['text_column'])
    return X, vectorizer

def train_vulnerability_model(cve_data):
    # Feature engineering specific to CVE data
    # Check available columns to determine features and target
    print("CVE dataset columns:", cve_data.columns.tolist())
    
    # Try to identify target variable
    target_col = None
    if 'label' in cve_data.columns:
        target_col = 'label'
    elif 'severity_score' in cve_data.columns:
        target_col = 'severity_score'
    elif 'severity' in cve_data.columns:
        # Create severity score if not already present
        severity_map = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1, 'unknown': 0}
        cve_data['severity_score'] = cve_data['severity'].map(
            lambda x: severity_map.get(str(x).lower(), 0) if x is not None else 0
        )
        target_col = 'severity_score'
    else:
        # If no suitable target column found, use the first numeric column
        numeric_cols = cve_data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            target_col = numeric_cols[0]
            print(f"No standard target column found, using {target_col} as target")
        else:
            # Last resort - create a dummy target
            print("No suitable target column found. Creating a dummy target for demonstration.")
            cve_data['dummy_target'] = 1
            target_col = 'dummy_target'
    
    # Identify features - drop ID columns and the target
    drop_cols = ['id', 'ID', 'Unnamed: 0', target_col]
    features = [col for col in cve_data.columns if col not in drop_cols]
    
    # Handle text columns if present
    text_cols = cve_data.select_dtypes(include=['object']).columns
    for col in text_cols:
        if col in features:
            # Convert text to categorical codes
            cve_data[col] = cve_data[col].astype('category').cat.codes
    
    # Handle datetime columns
    datetime_cols = cve_data.select_dtypes(include=['datetime64']).columns
    for col in datetime_cols:
        if col in features:
            # Extract year and month as features
            cve_data[f'{col}_year'] = cve_data[col].dt.year
            cve_data[f'{col}_month'] = cve_data[col].dt.month
            # Drop original datetime column
            features.remove(col)
            cve_data = cve_data.drop(col, axis=1)
    
    # Prepare final X and y
    X = cve_data[features]
    y = cve_data[target_col]
    
    print(f"Training vulnerability model with {len(features)} features and target '{target_col}'")
    
    # Check if target is numeric or categorical
    is_regression = False
    if pd.api.types.is_numeric_dtype(y):
        # Check if there are many unique values (likely regression)
        if len(y.unique()) > 10:
            is_regression = True
            print(f"Target '{target_col}' appears to be continuous. Using regression model.")
    
    # Train appropriate model
    if is_regression:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
    return model

def train_text_threat_model(text_data):
    # Print columns for debugging
    print("Text threats dataset columns:", text_data.columns.tolist())
    
    # Find a suitable text column
    text_columns = text_data.select_dtypes(include=['object']).columns
    if len(text_columns) == 0:
        print("Warning: No text columns found in text_threats dataset. Cannot train model.")
        return None
    
    # Choose the best text column - look for 'text', 'content', 'description', etc.
    priority_text_cols = ['text', 'content', 'description', 'message', 'summary']
    selected_text_col = None
    
    for col in priority_text_cols:
        if col in text_columns:
            selected_text_col = col
            break
    
    if not selected_text_col:
        # Just use the first text column
        selected_text_col = text_columns[0]
    
    print(f"Using '{selected_text_col}' as text feature")
    
    # Find or create a target column
    target_col = None
    priority_target_cols = ['label', 'target', 'class', 'is_threat', 'is_malicious', 'threat', 'malicious']
    
    for col in priority_target_cols:
        if col in text_data.columns:
            target_col = col
            print(f"Found target column: '{target_col}'")
            break
    
    # Try to infer a target from other columns if not found
    if not target_col:
        # Look for columns like 'label_1', 'label_2'
        label_cols = [col for col in text_data.columns if 'label' in col.lower()]
        if label_cols:
            # Use the first label column
            target_col = label_cols[0]
            print(f"Using '{target_col}' as target column")
        else:
            # If no target found, create a binary classification dummy target
            # This is just for demonstration - in real use, you'd need actual labels
            print("No suitable target column found. Creating a binary classification target for demonstration.")
            
            # Try to create a more meaningful dummy target using text analysis
            # For example, looking for security-related keywords in the text
            security_keywords = ['attack', 'threat', 'malware', 'virus', 'breach', 'hack', 
                                'phishing', 'vulnerability', 'exploit', 'malicious']
            
            def has_security_keywords(text):
                if isinstance(text, str):
                    return 1 if any(keyword in text.lower() for keyword in security_keywords) else 0
                return 0
            
            text_data['dummy_target'] = text_data[selected_text_col].apply(has_security_keywords)
            print(f"Created dummy target based on security keyword presence in text")
            print(f"Found {text_data['dummy_target'].sum()} potential threats out of {len(text_data)} records")
            
            target_col = 'dummy_target'
    
    # Text vectorization
    try:
        X, vectorizer = process_text_data(text_data.rename(columns={selected_text_col: 'text_column'}))
        y = text_data[target_col]
        
        # Convert target to numeric if it's categorical
        if not pd.api.types.is_numeric_dtype(y):
            print(f"Converting categorical target '{target_col}' to numeric")
            text_data[target_col] = text_data[target_col].astype('category').cat.codes
            y = text_data[target_col]
        
        print(f"Training text threat model with '{selected_text_col}' as feature and '{target_col}' as target")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Save the vectorizer for future use
        joblib.dump(vectorizer, 'text_vectorizer.pkl')
        print("Text vectorizer saved as 'text_vectorizer.pkl'")
        
        return model
    except Exception as e:
        print(f"Error training text threat model: {str(e)}")
        return None

def train_malware_detection_model(malware_data):
    # Print columns for debugging
    print("Malware dataset columns:", malware_data.columns.tolist())
    
    # Try to identify target variable
    priority_targets = ['is_malicious', 'malicious', 'label', 'target', 'class', 'Target Variable', 'malware']
    
    target_col = None
    for col in priority_targets:
        if col in malware_data.columns:
            target_col = col
            print(f"Found target column: '{target_col}'")
            break
    
    if not target_col:
        # Look for columns with 'target' in the name
        target_cols = [col for col in malware_data.columns if 'target' in col.lower()]
        if target_cols:
            target_col = target_cols[0] 
            print(f"Using '{target_col}' as target")
        else:
            # If no suitable target column found, check for numeric columns
            numeric_cols = malware_data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                # Use the last numeric column as target (often this is the class label)
                target_col = numeric_cols[-1]
                print(f"No standard target column found, using '{target_col}' as target")
            else:
                # Last resort - create a dummy target based on packet characteristics
                print("No suitable target column found. Creating a security-based target for demonstration.")
                
                # Create a heuristic-based target looking at networking patterns
                # This is just for demonstration and should be replaced with actual labeled data
                if ('Protocol' in malware_data.columns and 'Packet Size' in malware_data.columns and 
                    'Source Port' in malware_data.columns):
                    
                    def is_suspicious(row):
                        # Very simplistic rules for demonstration
                        if row['Protocol'].lower() == 'tcp' and row['Packet Size'] > 1000:
                            return 1
                        if row['Source Port'] in [25, 80, 443, 8080] and row['Packet Size'] < 100:
                            return 1
                        return 0
                    
                    malware_data['security_target'] = malware_data.apply(is_suspicious, axis=1)
                    target_col = 'security_target'
                    print(f"Created security-based target. Found {malware_data[target_col].sum()} suspicious packets.")
                else:
                    malware_data['dummy_target'] = 1
                    target_col = 'dummy_target'
    
    # Identify features - drop ID columns and the target
    drop_cols = ['id', 'ID', target_col]
    X = malware_data.drop(drop_cols, axis=1, errors='ignore')
    
    # Handle any object columns by converting to categorical
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype('category').cat.codes
    
    y = malware_data[target_col]
    
    # Convert target to numeric if it's categorical
    if not pd.api.types.is_numeric_dtype(y):
        print(f"Converting categorical target '{target_col}' to numeric")
        malware_data[target_col] = malware_data[target_col].astype('category').cat.codes
        y = malware_data[target_col]
    
    print(f"Training malware detection model with {X.shape[1]} features and target '{target_col}'")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def train_awareness_model(awareness_data):
    print("Awareness dataset columns:", awareness_data.columns.tolist()[:5], "...")
    
    # Define target column
    target_column = 'How would you rate your cybersecurity knowledge based on your experience and security habits?.Rate your cybersecurity knowledge.'
    
    # Check if the target column exists
    if target_column not in awareness_data.columns:
        # Look for alternative columns
        knowledge_cols = [col for col in awareness_data.columns if 'knowledge' in col.lower()]
        if knowledge_cols:
            target_column = knowledge_cols[0]
            print(f"Using '{target_column}' as target instead of default target")
        else:
            # Use the last column as target
            target_column = awareness_data.columns[-1]
            print(f"No suitable target column found. Using '{target_column}' as target")
    
    print(f"Using target column: '{target_column}'")
    
    # Engineer features
    df_engineered = awareness_data.copy()
    
    # Handle datetime columns - they cause problems with model training
    # Identify datetime columns
    datetime_cols = df_engineered.select_dtypes(include=['datetime64']).columns.tolist()
    if datetime_cols:
        print(f"Removing datetime columns for compatibility: {datetime_cols}")
        df_engineered = df_engineered.drop(columns=datetime_cols)
    
    # Also check for ID, email, name columns we should exclude
    exclude_patterns = ['id', 'email', 'name', 'time', 'date']
    cols_to_exclude = []
    for col in df_engineered.columns:
        if any(pattern in col.lower() for pattern in exclude_patterns) and col != target_column:
            cols_to_exclude.append(col)
    
    if cols_to_exclude:
        print(f"Excluding non-predictive columns: {cols_to_exclude}")
        df_engineered = df_engineered.drop(columns=cols_to_exclude, errors='ignore')
    
    # Handle any remaining non-numeric columns
    for col in df_engineered.select_dtypes(include=['object']).columns:
        if col != target_column:  # Don't convert target if it's categorical
            df_engineered[col] = df_engineered[col].astype('category').cat.codes
    
    # Ensure target is also numeric
    if df_engineered[target_column].dtype == 'object':
        df_engineered[target_column] = df_engineered[target_column].astype('category').cat.codes
    
    # Prepare data
    X = df_engineered.drop(target_column, axis=1)
    y = df_engineered[target_column]
    
    print(f"Training awareness model with {X.shape[1]} features...")
    
    try:
        # Compare models and get best
        best_model, model_name = compare_models(X, y)
        print(f"Best model for awareness: {model_name}")
        
        # Train and evaluate full model
        final_model = train_and_evaluate_model(X, y, X)
        
        return final_model
    except Exception as e:
        print(f"Error during awareness model training: {str(e)}")
        # Fallback to a simple model if comparison fails
        print("Falling back to basic Random Forest model...")
        try:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            return model
        except Exception as e2:
            print(f"Final error: {str(e2)}")
            print("Could not train model with this dataset. Please check data format.")
            return None

def build_security_suite():
    print("\n==== Starting Cybersecurity Model Training ====")
    print("Checking for available datasets...")
    
    # Dictionary to store trained models
    models = {}
    
    # Load and train with awareness data if available
    awareness_data = load_security_dataset("awareness")
    if awareness_data is not None:
        print("Training awareness model...")
        model = train_awareness_model(awareness_data)
        if model is not None:
            models["awareness"] = model
            joblib.dump(models["awareness"], 'awareness_model.pkl')
            print("Awareness model saved as 'awareness_model.pkl'")
        else:
            print("Failed to train awareness model.")
    
    # Load and train with CVE data if available
    cve_data = load_security_dataset("cve")
    if cve_data is not None:
        print("Training vulnerability model...")
        model = train_vulnerability_model(cve_data)
        if model is not None:
            models["vulnerability"] = model
            joblib.dump(models["vulnerability"], 'vulnerability_model.pkl')
            print("Vulnerability model saved as 'vulnerability_model.pkl'")
        else:
            print("Failed to train vulnerability model.")
    
    # Load and train with text threats data if available
    text_threats_data = load_security_dataset("text_threats")
    if text_threats_data is not None:
        print("Training text threat model...")
        model = train_text_threat_model(text_threats_data)
        if model is not None:
            models["threat"] = model
            joblib.dump(models["threat"], 'threat_model.pkl')
            print("Text threat model saved as 'threat_model.pkl'")
        else:
            print("Failed to train text threat model.")
    
    # Load and train with malware data if available
    malware_data = load_security_dataset("malware")
    if malware_data is not None:
        print("Training malware detection model...")
        model = train_malware_detection_model(malware_data)
        if model is not None:
            models["malware"] = model
            joblib.dump(models["malware"], 'malware_model.pkl')
            print("Malware model saved as 'malware_model.pkl'")
        else:
            print("Failed to train malware model.")
    
    if not models:
        print("\nNo models were successfully trained. Please check your datasets:")
        print("1. CVE dataset: https://www.kaggle.com/datasets/andrewkronser/cve-common-vulnerabilities-and-exposures")
        print("2. Text-based threats: https://www.kaggle.com/datasets/ramoliyafenil/text-based-cyber-threat-detection/data")
        print("3. Malware attacks: https://www.kaggle.com/datasets/zunxhisamniea/cyber-threat-data-for-new-malware-attacks")
        print("\nSave the CSV files to the same directory as this script with the following names:")
        print("- cve_dataset.csv")
        print("- text_threats.csv")
        print("- malware_attacks.csv")
    else:
        print(f"\nSuccessfully trained {len(models)} models!")
    
    return models

def main():
    print("Building comprehensive cybersecurity AI suite...")
    models = build_security_suite()
    
    if models:
        print("\nAll available models trained and saved successfully!")
        print(f"Models trained: {', '.join(models.keys())}")
    else:
        print("\nNo models were trained. Please add datasets and try again.")
    
    return models

if __name__ == "__main__":
    main()

# Load models
awareness_model = joblib.load('awareness_model.pkl')
vulnerability_model = joblib.load('vulnerability_model.pkl')
threat_model = joblib.load('threat_model.pkl')
malware_model = joblib.load('malware_model.pkl')

# For text-based predictions, you'll also need the vectorizer
text_vectorizer = joblib.load('text_vectorizer.pkl')

# Make predictions with new data
# Example: awareness_prediction = awareness_model.predict(new_survey_data)
