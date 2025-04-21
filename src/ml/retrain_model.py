# scripts/retrain_awareness_model.py

import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Load the existing vectorizer
vectorizer_path = os.path.join("models", "text_vectorizer.pkl")
vectorizer = joblib.load(vectorizer_path)

# 2. Your labeled dataset (replace with your actual data)
texts = [
    "phishing email pretending to be from a bank",
    "always update your passwords regularly",
    "machine learning is fun",
    "never share your personal information",
    "click here to win a free iPhone",
    "book review about a fantasy novel",
    "enable two-factor authentication",
    "this article is about cooking pasta",
    "ransomware locks your files",
    "this document explains quantum physics"
]

# Labels: 1 = cyber-related, 0 = not cyber-related
labels = [1, 1, 0, 1, 1, 0, 1, 0, 1, 0]

# 3. Vectorize the text using the current vectorizer
X = vectorizer.transform(texts)
y = labels

# 4. Split for optional evaluation (not mandatory here)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Save the model
model_path = os.path.join("models", "awareness_model.pkl")
joblib.dump(model, model_path)

print(f"✅ Model retrained and saved to: {model_path}")
