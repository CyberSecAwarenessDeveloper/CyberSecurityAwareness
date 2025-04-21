import os
import pandas as pd
import numpy as np
import json
import csv
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import traceback
import warnings

warnings.filterwarnings("ignore")

DATASET_DIR = "data/allDataSets"
OUTPUT_DIR = "models/trained_pipelines"
REPORT_DIR = "models/training_reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

FALLBACK_LABELS = ['label', 'target', 'category', 'classification']

def is_text_column(series):
    return series.dtype == object and series.str.len().mean() > 5

def is_categorical(series):
    return (
        (series.dtype == object or pd.api.types.is_categorical_dtype(series))
        and 1 < series.nunique() < 50
    )

def detect_delimiter(file_path):
    with open(file_path, 'r', encoding='ISO-8859-1') as f:
        sample = f.read(2048)
        return csv.Sniffer().sniff(sample).delimiter

def safe_read_csv(file_path):
    try:
        return pd.read_csv(file_path)
    except UnicodeDecodeError:
        try:
            return pd.read_csv(file_path, encoding="ISO-8859-1")
        except Exception:
            return None
    except pd.errors.ParserError:
        try:
            delimiter = detect_delimiter(file_path)
            return pd.read_csv(file_path, delimiter=delimiter, encoding="ISO-8859-1", engine="python", on_bad_lines='skip')
        except Exception:
            return None

def train_model_on_file(file_path):
    name = os.path.splitext(os.path.basename(file_path))[0]
    df = safe_read_csv(file_path)

    if df is None:
        print(f"❌ [{name}] Skipped: Could not read file due to encoding or structure issues.")
        return

    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(axis=0, how='all', inplace=True)

    text_cols = [col for col in df.columns if is_text_column(df[col])]
    if not text_cols:
        print(f"❌ [{name}] Skipped: No valid text columns found.")
        return

    label_cols = [col for col in df.columns if is_categorical(df[col])]
    if not label_cols:
        label_cols = [col for col in df.columns if col.lower() in FALLBACK_LABELS]
    if not label_cols:
        print(f"❌ [{name}] Skipped: No suitable label columns found.")
        return

    X = df[text_cols].astype(str).agg(" ".join, axis=1)

    for label_col in label_cols:
        y = df[label_col].astype(str)
        combined = pd.concat([X, y], axis=1)
        combined = combined[~combined[label_col].isin(["", "nan", "none", "NaN", "Null"])]
        X_clean = combined.iloc[:, 0]
        y_clean = combined.iloc[:, 1]

        if len(X_clean) < 10 or len(set(y_clean)) <= 1:
            print(f"⚠️ [{name}::{label_col}] Skipped: Not enough samples or only one label.")
            continue

        try:
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y_clean)

            X_train, X_val, y_train, y_val = train_test_split(
                X_clean, y_encoded, test_size=0.2, random_state=42
            )

            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(max_features=5000)),
                ("clf", LogisticRegression(solver="lbfgs", max_iter=500))
            ])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)
            acc = accuracy_score(y_val, y_pred)

            output_filename = f"{name}__{label_col}_pipeline.pkl"
            joblib.dump(pipeline, os.path.join(OUTPUT_DIR, output_filename))

            report = {
                "dataset": name,
                "label_column": label_col,
                "samples": len(X_clean),
                "classes": list(label_encoder.classes_),
                "accuracy": round(acc, 4)
            }
            with open(os.path.join(REPORT_DIR, f"{name}__{label_col}_report.json"), "w") as f:
                json.dump(report, f, indent=2)

            print(f"✅ [{name}::{label_col}] Model trained. Accuracy: {acc:.4f} | Samples: {len(X_clean)} | Labels: {label_encoder.classes_}")

        except Exception as e:
            print(f"❌ [{name}::{label_col}] Failed: {e}")
            traceback.print_exc()

def main():
    for file in os.listdir(DATASET_DIR):
        if file.endswith(".csv"):
            train_model_on_file(os.path.join(DATASET_DIR, file))

if __name__ == "__main__":
    main()
