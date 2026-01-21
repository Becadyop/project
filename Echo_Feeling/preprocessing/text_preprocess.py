# text_preprocess.py
# ------------------
# This script preprocesses the Amazon reviews dataset for sentiment analysis.
# Steps:
# 1. Load dataset
# 2. Clean text (lowercase, remove punctuation, normalize whitespace)
# 3. Convert ratings into sentiment labels (positive, neutral, negative)
# 4. Encode labels into integers
# 5. Vectorize text (Bag-of-Words)
# 6. Split into train/test sets

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# 1. Load dataset
# -------------------------------
# df = pd.read_csv("data/text/amazon_reviews.csv", encoding="utf-8")
df = pd.read_csv(r"C:\Users\arun\Desktop\ECHO FEELING\Echo_Feeling\data\text\amazon_reviews.csv", encoding="utf-8")


print("First 5 rows of dataset:")
print(df.head())
print("Columns available:", df.columns)

# -------------------------------
# 2. Clean text
# -------------------------------
def clean_text(text):
    """
    Lowercase text, remove non-alphabetic characters,
    and normalize whitespace.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)   # keep only letters and spaces
    return " ".join(text.split())

# Apply cleaning to the 'verified_reviews' column
df["cleaned_text"] = df["verified_reviews"].apply(clean_text)

# -------------------------------
# 3. Convert ratings into sentiment labels
# -------------------------------
def label_sentiment(rating):
    """
    Map numeric rating to sentiment:
    - rating >= 4 → positive
    - rating == 3 → neutral
    - rating <= 2 → negative
    """
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"

df["sentiment"] = df["rating"].apply(label_sentiment)

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

# -------------------------------
# 4. Encode labels
# -------------------------------
encoder = LabelEncoder()
y = encoder.fit_transform(df["sentiment"])  
# positive → 2, neutral → 1, negative → 0

print("\nLabel encoding mapping:", dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))

# -------------------------------
# 5. Vectorize text
# -------------------------------
# Option A: Bag-of-Words
count_vectorizer = CountVectorizer()
X_count = count_vectorizer.fit_transform(df["cleaned_text"])

# Option B: TF-IDF (better for weighting informative words)
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(df["cleaned_text"])

print("\nVectorization complete.")
print("Bag-of-Words shape:", X_count.shape)
print("TF-IDF shape:", X_tfidf.shape)

# -------------------------------
# 6. Train/Test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

print("\nTrain/Test split complete.")
print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])

# -------------------------------
# Final summary
# -------------------------------
print("\n✅ Text preprocessing finished successfully!")
print("Dataset shape:", df.shape)
