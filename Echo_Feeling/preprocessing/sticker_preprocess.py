import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# 1. Define dataset path
# -------------------------------
main_folder = r"C:\Users\user\Desktop\Projects\EchoFeeling\data\sticker"

img_height, img_width = 64, 64   # resize all stickers
X, y = [], []

# -------------------------------
# 2. Load images and labels
# -------------------------------
for label in os.listdir(main_folder):
    path = os.path.join(main_folder, label)
    if os.path.isdir(path):
        for file in os.listdir(path):
            if file.lower().endswith(".png"):
                img_path = os.path.join(path, file)
                img = Image.open(img_path).convert("RGB")
                img = img.resize((img_height, img_width))
                X.append(np.array(img))
                y.append(label)

X = np.array(X) / 255.0   # normalize pixel values
print("Dataset shape:", X.shape)

# -------------------------------
# 3. Encode labels
# -------------------------------
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
print("Label mapping:", dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))

# -------------------------------
# 4. Train/Test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])