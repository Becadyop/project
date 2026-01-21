# feature_extraction.py
# ---------------------
# This script extracts features from the Amazon reviews dataset.
# Steps:
# 1. Load dataset
# 2. Clean text (if not already cleaned)
# 3. Extract lexical features (length, word count, etc.)
# 4. Extract sentiment polarity features (using VADER from nltk)
# 5. TF-IDF vectorization
# 6. Combine features

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# -------------------------------
# 1. Load dataset
# -------------------------------
print("✅ Script started", flush=True)

df = pd.read_csv(r"C:\Users\arun\Desktop\ECHO FEELING\Echo_Feeling\data\text\amazon_reviews.csv", encoding="utf-8")

print("Columns available:", df.columns)

# -------------------------------
# 2. Clean text (create cleaned_text if missing)
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)   # keep only letters and spaces
    return " ".join(text.split())

if "cleaned_text" not in df.columns:
    df["cleaned_text"] = df["verified_reviews"].apply(clean_text)

# -------------------------------
# 3. Lexical features
# -------------------------------
df["char_count"] = df["cleaned_text"].apply(len)
df["word_count"] = df["cleaned_text"].apply(lambda x: len(x.split()))
df["avg_word_length"] = df["cleaned_text"].apply(
    lambda x: (sum(len(w) for w in x.split())/len(x.split())) if len(x.split()) > 0 else 0
)

# -------------------------------
# 4. Sentiment polarity features (VADER)
# -------------------------------
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

df["polarity"] = df["cleaned_text"].apply(lambda x: sia.polarity_scores(x)["compound"])
df["neg"] = df["cleaned_text"].apply(lambda x: sia.polarity_scores(x)["neg"])
df["neu"] = df["cleaned_text"].apply(lambda x: sia.polarity_scores(x)["neu"])
df["pos"] = df["cleaned_text"].apply(lambda x: sia.polarity_scores(x)["pos"])

# -------------------------------
# 5. TF-IDF features
# -------------------------------
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # limit to top 5000 features
X_tfidf = tfidf_vectorizer.fit_transform(df["cleaned_text"])

print("TF-IDF shape:", X_tfidf.shape)

# -------------------------------
# 6. Combine features
# -------------------------------
extra_features = df[["char_count", "word_count", "avg_word_length", "polarity", "neg", "neu", "pos"]]

print("\nSample of extracted features:")
print(extra_features.head())

print("\nFirst 10 TF-IDF feature names:")
print(tfidf_vectorizer.get_feature_names_out()[:10])

# -------------------------------
# Final summary
# -------------------------------
print("\n✅ Feature extraction finished successfully!")
print("Dataset shape:", df.shape)
print("Feature matrix ready for modeling.")
