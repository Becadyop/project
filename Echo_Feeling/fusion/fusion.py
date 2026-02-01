import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model, Model
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import emoji  # For emoji extraction; install via pip install emoji if not already

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# 1. Load Saved Extracted Data from Text, Emoji, and Sticker Codes
# -------------------------------
# Text features (from text extraction code: extracted_features.csv)
text_features_path = r"C:\Users\arun\Desktop\ECHO FEELING\Echo_Feeling\extraction\text\extracted_features.csv"
if not os.path.exists(text_features_path):
    raise FileNotFoundError(f"Text features not found at {text_features_path}. Run the text extraction code first.")
text_features_df = pd.read_csv(text_features_path)

# Processed reviews (from text extraction code: processed_reviews.csv for base data and labels)
processed_reviews_path = r"C:\Users\arun\Desktop\ECHO FEELING\Echo_Feeling\extraction\text\processed_reviews.csv"
if not os.path.exists(processed_reviews_path):
    raise FileNotFoundError(f"Processed reviews not found at {processed_reviews_path}. Run the text extraction code first.")
reviews_df = pd.read_csv(processed_reviews_path)

# Emoji features (from emoji extraction code: emojis_extracted.csv)
emoji_features_path = r"C:\Users\arun\Desktop\ECHO FEELING\Echo_Feeling\extraction\emojis\emojis_extracted.csv"
if not os.path.exists(emoji_features_path):
    raise FileNotFoundError(f"Emoji features not found at {emoji_features_path}. Run the emoji extraction code first.")
emoji_features_df = pd.read_csv(emoji_features_path).set_index('Emoji')

# Sticker model (from sticker code: sticker_cnn_model.keras, saved in the emojis extraction folder)
sticker_model_path = r"C:\Users\arun\Desktop\ECHO FEELING\Echo_Feeling\extraction\sticker\sticker_cnn_model.keras"
if not os.path.exists(sticker_model_path):
    raise FileNotFoundError(f"Sticker model not found at {sticker_model_path}. Run the sticker code first (ensure it's saved in the emojis extraction folder).")
sticker_model = load_model(sticker_model_path)

# Create a feature extractor for stickers (output from Dense(128) layer before Dropout)
feature_extractor = Model(inputs=sticker_model.input, outputs=sticker_model.layers[-3].output)  # Dense(128)

# -------------------------------
# 2. Build Multimodal Dataset from Saved Data
# -------------------------------
# Use reviews_df as base. Add emoji and sticker columns if not present (populate with actual data or dummies).
# For real fusion, extract emojis from text (e.g., using regex) and link sticker paths based on sentiment or manually.
if 'emoji' not in reviews_df.columns:
    # Extract emojis from text using emoji library
    def extract_emojis(text):
        return ','.join([c for c in str(text) if c in emoji.EMOJI_DATA])
    reviews_df['emoji'] = reviews_df['verified_reviews'].apply(extract_emojis)
    reviews_df['emoji'] = reviews_df['emoji'].apply(lambda x: x if x else '😂')  # Default if none

if 'sticker_path' not in reviews_df.columns:
    # Assign sticker paths based on sentiment (positive/negative folders)
    # Assuming sticker data has folders like 'positive', 'negative' in r"C:\Users\arun\Desktop\ECHO FEELING\Echo_Feeling\data\sticker"
    sticker_base = r"C:\Users\arun\Desktop\ECHO FEELING\Echo_Feeling\data\sticker"
    def assign_sticker_path(sentiment):
        if sentiment == 'Positive':
            return os.path.join(sticker_base, 'positive', 'sample.jpg')  # Replace with actual file name
        else:
            return os.path.join(sticker_base, 'negative', 'sample.jpg')  # Replace with actual file name
    # Derive sentiment from polarity
    reviews_df['sticker_path'] = reviews_df['polarity'].apply(lambda p: assign_sticker_path('Positive' if p > 0 else 'Negative'))

# Create label from rating (assuming binary sentiment)
if 'rating' in reviews_df.columns:
    reviews_df['label'] = reviews_df['rating'].apply(lambda x: 1 if x > 3 else 0)  # 1: positive, 0: negative
else:
    reviews_df['label'] = 1  # Dummy

df = reviews_df[['verified_reviews', 'emoji', 'sticker_path', 'label']].copy()
df.columns = ['text', 'emoji', 'sticker_path', 'label']

# -------------------------------
# 3. Feature Extraction Functions
# -------------------------------
# Text features: Retrieve from pre-extracted text_features_df
def get_text_features(index):
    return text_features_df.iloc[index].values

# Emoji features: From emoji_features_df
def get_emoji_features(emojis):
    if pd.isna(emojis) or not emojis:
        return np.zeros(3)
    emoji_list = str(emojis).split(',')
    features = []
    for e in emoji_list:
        if e in emoji_features_df.index:
            row = emoji_features_df.loc[e]
            dom_sent = {'Negative': 0, 'Neutral': 1, 'Positive': 2}[row['Dominant_Sentiment']]
            features.append([row['Sentiment_Score'], row['Sentiment_Ratio'], dom_sent])
    if features:
        return np.mean(features, axis=0)
    return np.zeros(3)

# Sticker features: Extract using the CNN model
def get_sticker_features(sticker_path, img_height=224, img_width=224):
    if not os.path.exists(sticker_path):
        return np.zeros(128)
    img = Image.open(sticker_path).convert("RGB").resize((img_height, img_width))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    features = feature_extractor.predict(img_array).flatten()
    return features

# -------------------------------
# 4. Fuse Features
# -------------------------------
fused_features = []
for idx, row in df.iterrows():
    text_feat = get_text_features(idx)
    emoji_feat = get_emoji_features(row['emoji'])
    sticker_feat = get_sticker_features(row['sticker_path'])
    fused = np.concatenate([text_feat, emoji_feat, sticker_feat])
    fused_features.append(fused)

X = np.array(fused_features)
y = df['label'].values

print("Fused feature shape:", X.shape)
print("Labels shape:", y.shape)

# -------------------------------
# 5. Train/Test Split and Train Fusion Model
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------
# 6. Save Fused Data for Training
# -------------------------------
fused_output_path = r"C:\Users\arun\Desktop\ECHO FEELING\Echo_Feeling\fusion\fused_sentiment_data.csv"
os.makedirs(os.path.dirname(fused_output_path), exist_ok=True)
fused_df.to_csv(fused_output_path, index=False)
print(f"✅ Fused data saved to {fused_output_path}. Use this CSV to train your sentiment analysis model.")

# Optional: Save the trained fusion model in the same fusion folder
import joblib
model_output_path = r"C:\Users\arun\Desktop\ECHO FEELING\Echo_Feeling\fusion\fused_sentiment_model.pkl"
joblib.dump(clf, model_output_path)
print(f"Trained fusion model saved as '{model_output_path}'")
