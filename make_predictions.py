import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Cybersecurity AI Models - Prediction Examples")
print("============================================")

# Example 1: Awareness Model Predictions
print("\n1. Awareness Model Predictions:")
print("-------------------------------")
try:
    # Load awareness model
    awareness_model = joblib.load('awareness_model.pkl')
    
    # Get feature names from the model
    if hasattr(awareness_model, 'feature_names_in_'):
        feature_names = awareness_model.feature_names_in_
        print(f"Awareness model features: {', '.join(feature_names[:5])}... (total: {len(feature_names)})")
        
        # Create example data with actual feature names
        # For demonstration, creating dummy data with the right column names
        example_data = pd.DataFrame(columns=feature_names)
        
        # Fill with random demo values (adjust based on your actual data)
        example_data.loc[0] = np.random.randint(0, 4, size=len(feature_names))
        
        # Make prediction
        prediction = awareness_model.predict(example_data)
        print(f"Predicted awareness level: {prediction[0]}")
        
        # Awareness level interpretation (based on your classes)
        awareness_levels = {
            0: "Very Low",
            1: "Low", 
            2: "Medium",
            3: "High", 
            4: "Very High"
        }
        
        print(f"Interpretation: {awareness_levels.get(prediction[0], 'Unknown')} cybersecurity awareness")
        
        # Add prediction probabilities
        proba = awareness_model.predict_proba(example_data)[0]
        print("\nProbability for each awareness level:")
        for i, prob in enumerate(proba):
            print(f"  Level {i} ({awareness_levels.get(i, 'Unknown')}): {prob:.2f}")
    else:
        print("Could not get feature names from the model.")
except Exception as e:
    print(f"Error with awareness prediction: {str(e)}")

# Example 2: Text Threat Analysis
print("\n2. Text Threat Analysis:")
print("----------------------")
try:
    # Load text model and vectorizer
    threat_model = joblib.load('threat_model.pkl')
    text_vectorizer = joblib.load('text_vectorizer.pkl')
    
    # Example text data - this works because text data is transformed by the vectorizer
    example_texts = [
        "This email claims I won a lottery I never entered",
        "Please download this attachment to see your invoice",
        "Your account password needs to be reset immediately",
        "Meeting notes from yesterday's conference call"
    ]
    
    # Transform text using the vectorizer
    transformed_text = text_vectorizer.transform(example_texts)
    
    # Make prediction
    predictions = threat_model.predict(transformed_text)
    
    # Show results
    print("Text threat analysis results:")
    for i, text in enumerate(example_texts):
        print(f"\nText: {text}")
        print(f"Predicted threat class: {predictions[i]}")
except Exception as e:
    print(f"Error with text threat prediction: {str(e)}")

# Example 3: Vulnerability Severity Assessment
print("\n3. Vulnerability Severity Assessment:")
print("-----------------------------------")
try:
    # Load vulnerability model
    vulnerability_model = joblib.load('vulnerability_model.pkl')
    
    # Get feature names from the model
    if hasattr(vulnerability_model, 'feature_names_in_'):
        vulnerability_features = vulnerability_model.feature_names_in_
        print(f"Vulnerability model features: {', '.join(vulnerability_features[:5])}... (total: {len(vulnerability_features)})")
        
        # Create example data with the right feature names
        example_vuln_data = pd.DataFrame(columns=vulnerability_features)
        
        # Fill with random demo values for 3 examples
        for i in range(3):
            example_vuln_data.loc[i] = np.random.randint(0, 4, size=len(vulnerability_features))
        
        # Make predictions
        predictions = vulnerability_model.predict(example_vuln_data)
        
        # Display results
        print("Vulnerability severity predictions:")
        for i, pred in enumerate(predictions):
            print(f"Vulnerability {i+1}: Severity Score {pred:.2f}")
            
            # Categorize severity based on CVSS standard ranges
            if pred < 4.0:
                severity = "Low"
            elif pred < 7.0:
                severity = "Medium"
            elif pred < 9.0:
                severity = "High"
            else:
                severity = "Critical"
                
            print(f"  Severity Category: {severity}")
    else:
        print("Could not get feature names from the vulnerability model.")
except Exception as e:
    print(f"Error with vulnerability prediction: {str(e)}")

# Example 4: Malware Detection
print("\n4. Malware Detection:")
print("------------------")
try:
    # Load malware model
    malware_model = joblib.load('malware_model.pkl')
    
    # Get feature names from the model
    if hasattr(malware_model, 'feature_names_in_'):
        malware_features = malware_model.feature_names_in_
        print(f"Malware model features: {', '.join(malware_features[:5])}... (total: {len(malware_features)})")
        
        # Create example data with the right feature names
        example_packets = pd.DataFrame(columns=malware_features)
        
        # Fill with random demo values for 3 examples
        for i in range(3):
            example_packets.loc[i] = np.random.randint(0, 4, size=len(malware_features))
        
        # Make predictions
        predictions = malware_model.predict(example_packets)
        
        # Display results
        print("Malware detection results:")
        for i, pred in enumerate(predictions):
            print(f"Packet {i+1}: Classified as Type {pred}")
    else:
        print("Could not get feature names from the malware model.")
except Exception as e:
    print(f"Error with malware prediction: {str(e)}")

print("\nAll predictions complete!") 