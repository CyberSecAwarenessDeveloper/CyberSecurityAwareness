import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def analyze_model(model_name, model):
    """Analyze a model and print relevant information"""
    print(f"\n{'=' * 50}")
    print(f"MODEL: {model_name}")
    print(f"{'=' * 50}")
    print(f"Type: {type(model).__name__}")
    print(f"Parameters: {model.get_params()}")
    
    # Check if it's a classifier or regressor
    if hasattr(model, 'classes_'):
        print(f"Number of classes: {len(model.classes_)}")
        print(f"Classes: {model.classes_}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        print("\nTop 5 Feature Importances:")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        for i in range(min(5, len(importances))):
            print(f"  Feature {indices[i]}: {importances[indices[i]]:.4f}")
        
        # Create feature importance plot
        plt.figure(figsize=(10, 6))
        plt.title(f'Feature Importances for {model_name}')
        plt.bar(range(min(10, len(importances))), 
                importances[indices][:10], 
                align='center')
        plt.xticks(range(min(10, len(importances))), indices[:10])
        plt.tight_layout()
        plt.savefig(f'{model_name}_feature_importance.png')
        print(f"Feature importance plot saved as {model_name}_feature_importance.png")

# List of available model files
print("Looking for model files...")
model_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
print(f"Found {len(model_files)} model files: {model_files}")

# Open and analyze each model
for model_file in model_files:
    try:
        model = joblib.load(model_file)
        model_name = model_file.replace('.pkl', '')
        
        # Skip vectorizers
        if 'vectorizer' in model_file:
            print(f"\nSkipping analysis of {model_file} (vectorizer)")
            print(f"Type: {type(model).__name__}")
            continue
        
        analyze_model(model_name, model)
        
    except Exception as e:
        print(f"\nError analyzing {model_file}: {str(e)}")

print("\nAnalysis complete!")

print("\nPrediction Examples:")
print("====================")

# Do you want to make predictions? Uncomment the sections below and replace with your actual data
print("\nTo make predictions, uncomment and adapt the code below:")
print("For example, to predict cybersecurity awareness level:")
print("""
# Load awareness model
awareness_model = joblib.load('awareness_model.pkl')

# Create example data with the right number of features
# For demonstration only - replace with actual features from your model
# This creates a dataframe with 21 features (columns) and 1 sample (row)
example_data = pd.DataFrame(
    [[0, 1, 0, 2, 1, 3, 2, 1, 0, 1, 2, 3, 1, 0, 2, 1, 0, 3, 2, 1, 0]],
    columns=[f'feature_{i}' for i in range(21)]
)

# Make prediction
prediction = awareness_model.predict(example_data)
print(f"Predicted awareness level: {prediction}")
""")

print("\nTo analyze text for security threats:")
print("""
# Load text model and vectorizer
threat_model = joblib.load('threat_model.pkl')
text_vectorizer = joblib.load('text_vectorizer.pkl')

# Example text data
example_texts = ["This email claims I won a lottery I never entered",
                "Please download this attachment to see your invoice"]

# Transform text using the vectorizer
transformed_text = text_vectorizer.transform(example_texts)

# Make prediction
predictions = threat_model.predict(transformed_text)
print(f"Predicted threat classes: {predictions}")
""")

print("\nTo assess vulnerability severity:")
print("""
# Load vulnerability model
vulnerability_model = joblib.load('vulnerability_model.pkl')

# Example vulnerability data with the right number of features
# For demonstration only - replace with actual features from your model
example_vuln_data = pd.DataFrame(
    [[1, 0, 1, 0, 1, 0, 2, 3, 1, 2, 0]], 
    columns=[f'feature_{i}' for i in range(11)]
)

# Make prediction (will return a continuous severity score)
prediction = vulnerability_model.predict(example_vuln_data)
print(f"Predicted vulnerability severity: {prediction}")
""")

# COMMENTED OUT non-working examples:
"""
# Load model
awareness_model = joblib.load('awareness_model.pkl')

# Prepare new data (must have same features in same order as training data)
new_data = pd.DataFrame({...})  # Fill with appropriate values

# Make prediction
prediction = awareness_model.predict(new_data)
print(f"Predicted awareness level: {prediction}")

# Load model and vectorizer
threat_model = joblib.load('threat_model.pkl')
text_vectorizer = joblib.load('text_vectorizer.pkl')

# Prepare new text data
new_text = ["This is a suspicious email asking for password"]

# Transform text using the vectorizer
transformed_text = text_vectorizer.transform(new_text)

# Make prediction
prediction = threat_model.predict(transformed_text)
print(f"Predicted threat class: {prediction}")

# Load model
vulnerability_model = joblib.load('vulnerability_model.pkl')

# Prepare new vulnerability data
new_vuln_data = pd.DataFrame({...})  # Fill with appropriate values

# Make prediction (will return a continuous severity score)
prediction = vulnerability_model.predict(new_vuln_data)
print(f"Predicted vulnerability severity: {prediction}")
"""
