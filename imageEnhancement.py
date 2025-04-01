import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, add
from keras.layers import Input
from keras.models import Model

# Function to filter image files
def get_image_files(directory):
    try:
        return [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    except Exception as e:
        print(f"Error accessing directory {directory}: {e}")
        return []

# Directories
low_image_dir = r'E:/Low-Light-Image-Enhancement/'
high_image_dir = r'E:/Low-Light-Image-Enhancement/lol_dataset/our485/high'
low_image_dir_test = r'E:/Low-Light-Image-Enhancement'
high_image_dir_test = r'E:/Low-Light-Image-Enhancement/lol_dataset/eval15/high'

# Get image files
low_image_files = get_image_files(low_image_dir)
high_image_files = get_image_files(high_image_dir)
low_image_files_testing = get_image_files(low_image_dir_test)
high_image_files_testing = get_image_files(high_image_dir_test)

# Print file counts
print("Length of low light images:", len(low_image_files))
print("Length of high light images:", len(high_image_files))
print("Length of low light images for testing:", len(low_image_files_testing))
print("Length of high light images for testing:", len(high_image_files_testing))

# Function to load images
def load_images(directory, file_list):
    images = []
    for file_name in file_list:
        file_path = os.path.join(directory, file_name)
        image = cv2.imread(file_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
        else:
            print(f"Failed to load image: {file_path}")
    return images

# Load images
low_light_images = load_images(low_image_dir, low_image_files)
high_light_images = load_images(high_image_dir, high_image_files)
low_light_images_testing = load_images(low_image_dir_test, low_image_files_testing)
high_light_images_testing = load_images(high_image_dir_test, high_image_files_testing)

# Check if lists are empty before proceeding
if not low_light_images or not high_light_images:
    print("No valid training images found. Exiting.")
    exit()

if not low_light_images_testing or not high_light_images_testing:
    print("No valid testing images found. Exiting.")
    exit()

# Display the first image pair
plt.subplot(1, 2, 1)
plt.imshow(low_light_images[0])
plt.title("Low Light Image")
plt.subplot(1, 2, 2)
plt.imshow(high_light_images[0])
plt.title("High Light Image")
plt.show()

# Data Preprocessing
new_image_size = (256, 256)

# Function to resize images
def resize_image(image, new_image_size):
    return cv2.resize(image, new_image_size)

# Resize images
low_light_images_resized = [resize_image(image, new_image_size) for image in low_light_images]
high_light_images_resized = [resize_image(image, new_image_size) for image in high_light_images]
low_light_images_resized_testing = [resize_image(image, new_image_size) for image in low_light_images_testing]
high_light_images_resized_testing = [resize_image(image, new_image_size) for image in high_light_images_testing]

# Normalize images
def normalize_image(image):
    return image / 255.0

low_light_images_normalized = [normalize_image(image) for image in low_light_images_resized]
high_light_images_normalized = [normalize_image(image) for image in high_light_images_resized]
low_light_images_normalized_testing = [normalize_image(image) for image in low_light_images_resized_testing]
high_light_images_normalized_testing = [normalize_image(image) for image in high_light_images_resized_testing]

# Split data into training and validation
train_low, val_low, train_high, val_high = train_test_split(
    low_light_images_normalized, high_light_images_normalized, test_size=0.2, random_state=42
)

# Convert to numpy arrays
train_low = np.array(train_low)
train_high = np.array(train_high)
val_low = np.array(val_low)
val_high = np.array(val_high)
test_low = np.array(low_light_images_normalized_testing)
test_high = np.array(high_light_images_normalized_testing)

# Print data shapes
print("Shape of training low light images:", train_low.shape)
print("Shape of training high light images:", train_high.shape)
print("Shape of validation low light images:", val_low.shape)
print("Shape of validation high light images:", val_high.shape)
print("Shape of test low light images:", test_low.shape)
print("Shape of test high light images:", test_high.shape)

# Model definition
def instantiate_model(input_shape):
    input_layer = Input(shape=input_shape)

    # Branch 1
    branch1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
    branch1 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch1)
    branch1 = Conv2D(64, (2, 2), activation='relu', padding='same')(branch1)

    # Branch 2
    branch2 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    branch2 = Conv2D(64, (2, 2), activation='relu', padding='same')(branch2)
    branch2_0 = Conv2D(64, (2, 2), activation='relu', padding='same')(branch2)

    # Branch 3
    branch3 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
    branch3 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch3)
    branch3 = Conv2D(64, (2, 2), activation='relu', padding='same')(branch3)

    # Merge branches
    merged = add([branch1, branch2, branch2_0, branch3])

    # Output branch
    output_branch = Conv2D(3, (3, 3), activation='relu', padding='same')(merged)

    model = Model(inputs=input_layer, outputs=output_branch)
    return model

# Instantiate and compile the model
input_shape = (256, 256, 3)
model = instantiate_model(input_shape)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Train the model
history = model.fit(train_low, train_high, epochs=20, validation_data=(val_low, val_high))

# Evaluate the model
loss, accuracy = model.evaluate(test_low, test_high)
print("Loss on test data:", loss)
print("Accuracy on test data:", accuracy)

# Save the model
model.save('model.h5')
print("Model saved successfully")

# Plot training history
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.show()

# Make predictions
predictions = model.predict(test_low)

# Display predictions
for i in range(len(test_low)):
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 3, 1)
    plt.imshow(predictions[i])
    plt.title("Predicted Image")
    plt.subplot(1, 3, 2)
    plt.imshow(test_low[i])
    plt.title("Low Light Image")
    plt.subplot(1, 3, 3)
    plt.imshow(test_high[i])
    plt.title("High Light Image")
    plt.show()



