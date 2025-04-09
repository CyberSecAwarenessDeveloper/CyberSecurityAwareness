# src/ml/predictor.py
from src.ml.load_models import load_all_models

# Load all models once
models = load_all_models()

# Helper: transform input if vectorizer is needed
def vectorize_input(text):
    vectorizer = models.get("text_vectorizer")
    return vectorizer.transform([text]) if vectorizer else [text]

def predict_awareness(text):
    vectorized = vectorize_input(text)
    result = models["awareness"].predict(vectorized)
    return result[0]

def predict_malware(text):
    vectorized = vectorize_input(text)
    result = models["malware"].predict(vectorized)
    return result[0]

def predict_threat(text):
    vectorized = vectorize_input(text)
    result = models["threat"].predict(vectorized)
    return result[0]

def predict_vulnerability(text):
    vectorized = vectorize_input(text)
    result = models["vulnerability"].predict(vectorized)
    return result[0]

# Optional: unified dispatcher
def predict_by_category(text, category):
    category = category.lower()
    if category == "awareness":
        return predict_awareness(text)
    elif category == "malware":
        return predict_malware(text)
    elif category == "threat":
        return predict_threat(text)
    elif category == "vulnerability":
        return predict_vulnerability(text)
    else:
        return "Unknown category"
