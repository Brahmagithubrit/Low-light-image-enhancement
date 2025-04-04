# train.py
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

# Image loader
def load_images(path):
    data = []
    for file in os.listdir(path):
        if file.endswith(('jpg', 'jpeg', 'png')):
            img = cv2.imread(os.path.join(path, file))
            img = cv2.resize(img, (256, 256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            data.append(img / 255.0)
    return np.array(data)

print("\n🚀 Loading dataset...")
low_dir = 'lol_dataset/our485/low'
high_dir = 'lol_dataset/our485/high'
low_imgs = load_images(low_dir)
high_imgs = load_images(high_dir)

print(f"Loaded {len(low_imgs)} low-light images and {len(high_imgs)} high-light images.")

# Split
train_low, val_low, train_high, val_high = train_test_split(low_imgs, high_imgs, test_size=0.1, random_state=42)
print(f"Training data: {len(train_low)} images | Validation data: {len(val_low)} images")

# Model
input_layer = Input(shape=(256, 256, 3))
x = Conv2D(64, (9, 9), activation='relu', padding='same')(input_layer)
x = Conv2D(32, (1, 1), activation='relu', padding='same')(x)
x = Conv2D(3, (5, 5), activation='sigmoid', padding='same')(x)
model = Model(inputs=input_layer, outputs=x)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mse', metrics=['accuracy'])

# Train
print("\n📚 Training model...")
history = model.fit(train_low, train_high, validation_data=(val_low, val_high), epochs=20, batch_size=8)

# Save
model.save("complex_model.h5")

# Report
val_loss, val_accuracy = model.evaluate(val_low, val_high)
print("\n📈 Training Complete")
print("-----------------------------------------")
print(f"✅ Final Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"🔁 Final Validation Loss: {val_loss:.4f}")
print("Model saved as 'complex_model.h5'")