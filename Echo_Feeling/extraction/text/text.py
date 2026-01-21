# feature_extraction.py
# ---------------------
# Extract features from Amazon reviews dataset
# Updated: Optimized for low-end PCs (4GB RAM, Ryzen 3 CPU) by using DistilBERT, smaller batches, and memory-efficient processing.
# Updated: Save output files in the 'extraction' folder under 'text' (e.g., data\text\extraction\$.

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import DistilBertTokenizer, DistilBertModel  # Changed to DistilBERT for lower memory usage
import torch
import os
import numpy as np

print("✅ Script started", flush=True)

# -------------------------------
# 1. Load dataset
# -------------------------------
# Updated: Added check for file existence
input_path = r"C:\Users\arun\Desktop\ECHO FEELING\Echo_Feeling\data\text\amazon_reviews.csv"
if not os.path.exists(input_path):
    raise FileNotFoundError(f"Dataset not found at {input_path}. Please check the path.")
df = pd.read_csv(input_path, encoding="utf-8")

# Added: Check for required column
if "verified_reviews" not in df.columns:
    raise ValueError(f"Column 'verified_reviews' not found. Available columns: {df.columns.tolist()}")

# Added: Warn if dataset is large (potential RAM issue)
if len(df) > 5000:
    print("⚠️ Warning: Dataset has more than 5000 rows. Processing may be slow or exceed RAM on low-end PCs. Consider reducing dataset size.")

print("Columns available:", df.columns)

# -------------------------------
# 2. Clean text (create cleaned_text)
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)   # keep only letters and spaces
    return " ".join(text.split())

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
# Updated: Reduced max_features to 2000 for lower memory usage (adjust if needed)
tfidf_vectorizer = TfidfVectorizer(max_features=2000)
X_tfidf = tfidf_vectorizer.fit_transform(df["cleaned_text"])
print("TF-IDF shape:", X_tfidf.shape)

# -------------------------------
# 6. BERT embeddings (Optimized for low-end PC)
# -------------------------------
print("Loading DistilBERT model...")  # Using DistilBERT instead of full BERT for speed and memory
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
model.eval()  # Set to evaluation mode

# Updated: Smaller batch size (4) to fit 4GB RAM; process in chunks if needed
def get_bert_embeddings_batched(texts, batch_size=4):  # Reduced batch_size for lower RAM
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        # Handle empty texts
        batch_texts = [text if text.strip() else "empty" for text in batch_texts]
        inputs = tokenizer(batch_texts, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # Use [CLS] token embeddings (DistilBERT has 768 dims like BERT-base)
        batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy().astype(np.float32)  # Use float32 for memory savings
        embeddings.extend(batch_embeddings)
        # Optional: Free memory after each batch
        del inputs, outputs
        torch.cuda.empty_cache() if torch.cuda.is_available() else None  # Though we're on CPU
    return embeddings

print("Extracting DistilBERT embeddings (batched, optimized for low RAM)...")
bert_embeddings = get_bert_embeddings_batched(df["cleaned_text"].tolist())
bert_df = pd.DataFrame(bert_embeddings, columns=[f"bert_{i}" for i in range(768)])
print("DistilBERT embeddings shape:", bert_df.shape)

# -------------------------------
# 7. Combine features
# -------------------------------
extra_features = df[["char_count", "word_count", "avg_word_length", "polarity", "neg", "neu", "pos"]]

print("\nSample of extracted features:")
print(extra_features.head())

print("\nFirst 10 TF-IDF feature names:")
print(tfidf_vectorizer.get_feature_names_out()[:10])

# -------------------------------
# 8. Save extracted features
# -------------------------------
# Convert TF-IDF sparse matrix to DataFrame
tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Combine extra features, TF-IDF, and BERT embeddings
combined_features_df = pd.concat([extra_features.reset_index(drop=True), tfidf_df, bert_df], axis=1)

# Updated: Save in the 'extraction' folder under 'text' (e.g., data\text\extraction\$
output_features_path = r"C:\Users\arun\Desktop\ECHO FEELING\Echo_Feeling\extraction\text\extracted_features.csv"
os.makedirs(os.path.dirname(output_features_path), exist_ok=True)
combined_features_df.to_csv(output_features_path, index=False)
print("✅ Combined features (including DistilBERT) saved to 'extraction\\text\\extracted_features.csv'")

# Optionally, save the full DataFrame (including original and cleaned text) to another CSV in the same folder
output_reviews_path = r"C:\Users\arun\Desktop\ECHO FEELING\Echo_Feeling\extraction\text\processed_reviews.csv"
os.makedirs(os.path.dirname(output_reviews_path), exist_ok=True)
df.to_csv(output_reviews_path, index=False)
print("✅ Processed reviews (with cleaned text and lexical/sentiment features) saved to 'extraction\\text\\extracted_features.csv'")

# -------------------------------
# Final summary
# -------------------------------
print("\n✅ Feature extraction finished successfully!")
print("Dataset shape:", df.shape)
print("Feature matrix ready for modeling.")