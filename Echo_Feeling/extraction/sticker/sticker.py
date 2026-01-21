import os
import numpy as np
import pandas as pd  # Added for potential future use
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
import warnings
warnings.filterwarnings("ignore")  # Suppress general warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings/logs

# -------------------------------
# 1. Define dataset path (UPDATE THIS TO YOUR IMAGE FOLDER)
# -------------------------------
# Replace with the path to your main folder containing subfolders (each subfolder is a class/label)
main_folder = r"C:\Users\arun\Desktop\ECHO FEELING\Echo_Feeling\data\sticker"  # Example: adjust to your actual path

img_height, img_width = 224, 224  # Increased for transfer learning (MobileNetV2 expects 224x224)
X, y = [], []

# -------------------------------
# 2. Load images and labels from directories
# -------------------------------
if not os.path.exists(main_folder):
    raise ValueError(f"Main folder '{main_folder}' does not exist. Please check the path.")

for label in os.listdir(main_folder):
    path = os.path.join(main_folder, label)
    if os.path.isdir(path):
        for file in os.listdir(path):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):  # Support common image formats
                img_path = os.path.join(path, file)
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize((img_height, img_width))
                    X.append(np.array(img))
                    y.append(label)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")

if not X:
    raise ValueError("No images found in the specified folder. Check your dataset structure.")

X = np.array(X) / 255.0   # Normalize pixel values
print("Dataset shape:", X.shape)

# -------------------------------
# 3. Encode labels
# -------------------------------
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
num_classes = len(encoder.classes_)
y_categorical = to_categorical(y_encoded, num_classes=num_classes)
print("Label mapping:", dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))
print("Number of classes:", num_classes)

# -------------------------------
# 4. Train/Test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])

# -------------------------------
# 5. Data Augmentation (Improves generalization and reduces overfitting)
# -------------------------------
datagen = ImageDataGenerator(
    rotation_range=30,      # Increased rotation
    width_shift_range=0.3,  # Increased shift
    height_shift_range=0.3, # Increased shift
    horizontal_flip=True,   # Flip images horizontally
    zoom_range=0.2,         # Added zoom
    fill_mode='nearest'     # Fill empty pixels
)

# -------------------------------
# 6. Compute class weights to handle imbalance
# -------------------------------
class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weights = dict(enumerate(class_weights))

# -------------------------------
# 7. Build Transfer Learning Model (MobileNetV2 for better performance on small datasets)
# -------------------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False  # Freeze base layers initially

# Add custom head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# -------------------------------
# 8. Train the Model (With augmentation, early stopping, and class weights)
# -------------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)  # Increased patience

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=16),  # Smaller batch size
    epochs=50,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    class_weight=class_weights,
    verbose=1
)

# Optional: Fine-tune (unfreeze some base layers after initial training)
base_model.trainable = True
for layer in base_model.layers[:-20]:  # Freeze all but last 20 layers
    layer.trainable = False
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])  # Lower LR for fine-tuning

history_fine = model.fit(
    datagen.flow(X_train, y_train, batch_size=16),
    epochs=20,  # Fewer epochs for fine-tuning
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    class_weight=class_weights,
    verbose=1
)

# -------------------------------
# 9. Evaluate the Model
# -------------------------------
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Detailed per-class report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=encoder.classes_, zero_division=0))

# -------------------------------
# 10. Save the Model
# -------------------------------
model.save('sticker_cnn_model.keras')
print("Model saved as 'sticker_cnn_model.keras'")

# -------------------------------
# 11. Prediction Function (For Testing on New Images)
# -------------------------------
# Function to predict on a single image
def predict_sticker(image_path, model, encoder, img_height=224, img_width=224):
    img = Image.open(image_path).convert("RGB").resize((img_height, img_width))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    class_name = encoder.inverse_transform([predicted_class])[0]
    confidence = np.max(predictions)
    return class_name, confidence

# Example usage (uncomment and replace with your image path to test)
# image_path = r"C:\path\to\your\sticker.jpg"  # Replace with actual path
# predicted_label, conf = predict_sticker(image_path, model, encoder)
# print(f"Predicted: {predicted_label} with confidence {conf:.2f}")