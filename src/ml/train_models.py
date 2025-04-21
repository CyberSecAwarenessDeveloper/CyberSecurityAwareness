import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Paths
datasets = {
    #"cyber_classification_dataset": "data/cyber_classification_dataset.csv",
    #"awareness_dataset": "data/cyber-threat-intelligence_all.csv",
    #"cve_dataset": "data/cve_dataset.csv",
    "cybersecurity_attacks": "data/cybersecurity_attacks.csv",
    "malware_dataset": "data/malware_dataset.csv",
    #"phishing_dataset": "data/phishing_dataset.csv",
    #"phishing_domain_dataset": "data/phishing_domain_dataset.csv",
    "text_threats": "data/text_threats.csv",
}

output_dir = "models"
os.makedirs(output_dir, exist_ok=True)

def train_model_text(name, df):
    df = df[['text', 'label']].dropna()
    df = df[df['text'].str.strip() != '']
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    pipe = Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    joblib.dump(pipe, f"{output_dir}/{name}_pipeline.pkl")
    print(f"[{name}] ✅ Trained & saved — Accuracy: {acc:.2f}")

def train_model_numeric(name, df, label_col):
    df = df.dropna(subset=[label_col])
    X = df.drop(columns=[label_col])
    y = df[label_col]
    X = X.select_dtypes(include=['float64', 'int64'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=300, max_depth=30, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    joblib.dump(pipe, f"{output_dir}/{name}_pipeline.pkl")
    print(f"[{name}] ✅ Trained & saved — Accuracy: {acc:.2f}")

for name, path in datasets.items():
    try:
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines='skip')
    except Exception as e:
        print(f"[{name}] ❌ Load error: {e}")
        continue

    try:
        if name == "awareness_dataset":
            if 'text' not in df.columns and 'Category' in df.columns:
                df = df.rename(columns={'Category': 'label'})
                df['text'] = df['label']
            train_model_text(name, df)
            continue

        if name == "cve_dataset" and 'summary' in df.columns and 'cwe_name' in df.columns:
            df = df.rename(columns={'summary': 'text', 'cwe_name': 'label'})
            train_model_text(name, df)
            continue

        if name == "phishing_dataset" and 'Type' in df.columns:
            df = df.rename(columns={'Type': 'label'})
            train_model_numeric(name, df, 'label')
            continue

        if name == "phishing_domain_dataset" and 'phishing' in df.columns:
            df = df.rename(columns={'phishing': 'label'})
            train_model_numeric(name, df, 'label')
            continue

        if name == "cybersecurity_attacks" and 'Attack Type' in df.columns:
            df = df.rename(columns={'Attack Type': 'label'})
            train_model_numeric(name, df, 'label')
            continue

        if name == "malware_dataset" and 'Target Variable' in df.columns:
            df = df.rename(columns={'Target Variable': 'label'})
            train_model_numeric(name, df, 'label')
            continue

        if 'text' in df.columns and 'label' in df.columns:
            train_model_text(name, df)
        else:
            print(f"[{name}] ❌ Missing required columns")

    except Exception as e:
        print(f"[{name}] ❌ Error during training: {e}")
