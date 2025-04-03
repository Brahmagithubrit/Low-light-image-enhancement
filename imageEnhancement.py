import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, Input, Add
from tensorflow.keras.models import Model

# Ensure TensorFlow is installed correctly
print("TensorFlow Version:", tf.__version__)

# Function to filter image files
def get_image_files(directory):
    return [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Directories
low_image_dir = r'E:/Low-Light-Image-Enhancement2/lol_dataset/our485/low'
high_image_dir = r'E:/Low-Light-Image-Enhancement2/lol_dataset/our485/high'
low_image_dir_test = r'E:/Low-Light-Image-Enhancement2/lol_dataset/eval15/low'
high_image_dir_test = r'E:/Low-Light-Image-Enhancement2/lol_dataset/eval15/high'

# Get image files
low_image_files = get_image_files(low_image_dir)
high_image_files = get_image_files(high_image_dir)
low_image_files_testing = get_image_files(low_image_dir_test)
high_image_files_testing = get_image_files(high_image_dir_test)

# Function to load images
def load_images(directory, file_list):
    images = []
    for file_name in file_list:
        file_path = os.path.join(directory, file_name)
        image = cv2.imread(file_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
    return images

# Load images
low_light_images = load_images(low_image_dir, low_image_files)
high_light_images = load_images(high_image_dir, high_image_files)

# Resize & Normalize images
image_size = (256, 256)

def preprocess_images(images):
    images_resized = [cv2.resize(img, image_size) for img in images]
    return np.array(images_resized) / 255.0  # Normalize

low_light_images = preprocess_images(low_light_images)
high_light_images = preprocess_images(high_light_images)

# Split data
train_low, val_low, train_high, val_high = train_test_split(
    low_light_images, high_light_images, test_size=0.2, random_state=42
)

# Model Definition
def build_model(input_shape):
    input_layer = Input(shape=input_shape)

    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)

    merged = Add()([conv1, conv2, conv3])  # Now all layers have 64 filters

    output_layer = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(merged)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Build and compile the model
input_shape = (256, 256, 3)
model = build_model(input_shape)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Train the model
history = model.fit(train_low, train_high, epochs=20, validation_data=(val_low, val_high))

# Save the model
model.save('enhancement_model.h5')
print("Model saved successfully!")

# Plot training history
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
