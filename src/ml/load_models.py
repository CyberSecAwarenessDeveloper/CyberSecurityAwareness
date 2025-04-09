# src/ml/load_models.py
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\src\\ml", "")

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_all_models():
    models_path = os.path.join(BASE_DIR, "models")
    return {
        "awareness": load_pickle(os.path.join(models_path, "awareness_model.pkl")),
        "text_vectorizer": load_pickle(os.path.join(models_path, "text_vectorizer.pkl")),
        "threat": load_pickle(os.path.join(models_path, "threat_model.pkl")),
        "vulnerability": load_pickle(os.path.join(models_path, "vulnerability_model.pkl")),
        "malware": load_pickle(os.path.join(models_path, "malware_model.pkl")),
    }
