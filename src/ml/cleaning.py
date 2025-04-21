import os
import pandas as pd
import chardet
import unicodedata
import re

INPUT_DIR = "data"
OUTPUT_DIR = os.path.join(INPUT_DIR, "cleaned")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Common bad character replacements
CHARACTER_FIXES = {
    "â€“": "-",     # en-dash
    "â€”": "—",     # em-dash
    "â€": "\"",    # right double quote
    "â€œ": "\"",    # left double quote
    "â€˜": "'",     # left single quote
    "â€™": "'",     # right single quote
    "â€¦": "...",   # ellipsis
    "â€": "\"",     # generic quote
    "Ã©": "é",
    "Ã¨": "è",
    "Ã¤": "ä",
    "Ã¶": "ö",
    "Ã¼": "ü",
    "ÃŸ": "ß",
    "Ã ": "à",
    "Ã": "à",
    # Add more as needed
}

def fix_text(text):
    if not isinstance(text, str):
        return text
    for bad, good in CHARACTER_FIXES.items():
        text = text.replace(bad, good)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\x00-\x7F]+", lambda x: x.group(0), text)
    return text

def clean_csv_file(filepath):
    with open(filepath, 'rb') as f:
        rawdata = f.read()
        encoding_guess = chardet.detect(rawdata)['encoding']

    try:
        df = pd.read_csv(filepath, encoding=encoding_guess)
    except Exception as e:
        print(f"❌ Failed to read {filepath}: {e}")
        return

    df_cleaned = df.map(fix_text)

    output_path = os.path.join(OUTPUT_DIR, os.path.basename(filepath))
    try:
        df_cleaned.to_csv(output_path, index=False, encoding="utf-8")
        print(f"✅ Cleaned: {os.path.basename(filepath)} → {output_path}")
    except Exception as e:
        print(f"❌ Failed to write cleaned file: {filepath}: {e}")

def main():
    for file in os.listdir(INPUT_DIR):
        if file.endswith(".csv"):
            clean_csv_file(os.path.join(INPUT_DIR, file))

if __name__ == "__main__":
    main()
