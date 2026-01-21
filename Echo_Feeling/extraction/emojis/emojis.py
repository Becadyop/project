import pandas as pd

# 1. Load emoji dataset into a DataFrame
emoji_df = pd.read_csv(
    r"C:\Users\arun\Desktop\ECHO FEELING\Echo_Feeling\data\emoji\Emoji_Sentiment_Data_v1.0.csv",
    encoding="utf-8"
)

# Optional: strip whitespace from column names
emoji_df.columns = emoji_df.columns.str.strip()

# 2. Create sentiment features (as in the original code)
emoji_df["Sentiment_Score"]   = emoji_df["Positive"] - emoji_df["Negative"]
emoji_df["Sentiment_Ratio"]   = (emoji_df["Positive"] + 1) / (emoji_df["Negative"] + 1)  # avoid divide by zero
emoji_df["Dominant_Sentiment"] = emoji_df[["Negative","Neutral","Positive"]].idxmax(axis=1)

# 3. Create the extracted DataFrame with Emoji and sentiment features
emojis_extracted_df = emoji_df[["Emoji", "Sentiment_Score", "Sentiment_Ratio", "Dominant_Sentiment"]]

# 4. Save the extracted file to the same path under the name emojis_extracted in CSV format
emojis_extracted_df.to_csv(
    r"C:\Users\arun\Desktop\ECHO FEELING\Echo_Feeling\extraction\emojis\emojis_extracted.csv",
    index=False
)

# Optional: Show first few rows of the extracted DataFrame
print(emojis_extracted_df.head())