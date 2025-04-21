import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Paths
base_path = os.path.join(os.path.dirname(__file__), )
vectorizer_path = os.path.join(base_path, "text_vectorizer.pkl")
model_output_path = os.path.join(base_path, "awareness_model.pkl")

# Example training data (replace or load yours)
# Label should be 'cyber' or 'non-cyber'
data = pd.DataFrame({
    "text": [
        "How can I secure my Wi-Fi?",
        "My password got hacked",
        "Flowers in the garden",
        "Best place to buy shoes",
        "Phishing emails look legit",
        "Clicking unknown links",
        "I got a virus on my laptop",
        "What are CVEs?",
        "Cybersecurity training tips",
        "Secure email providers"
    ],
    "label": [
        "cyber", "cyber", "non-cyber", "non-cyber", "cyber",
        "cyber", "cyber", "cyber", "cyber", "cyber"
    ]
})

# Load existing vectorizer
vectorizer = joblib.load(vectorizer_path)

# Transform text
X = vectorizer.transform(data["text"])
y = data["label"]

# Train classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, model_output_path)
print(f"✅ awareness_model.pkl retrained and saved to {model_output_path}")
