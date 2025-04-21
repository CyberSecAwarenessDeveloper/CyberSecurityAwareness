# train_classifier_cyber_or_not.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv("data/cyber_classification_dataset.csv")

# Clean dataset: remove rows with labels that occur only once
df = df.groupby("label").filter(lambda x: len(x) > 1)

# Extract features and labels
X = df["text"]
y = df["label"]

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "models/cyber_or_not_classifier.pkl")
joblib.dump(vectorizer, "models/cyber_or_not_vectorizer.pkl")

print("✅ Model and vectorizer saved.")
