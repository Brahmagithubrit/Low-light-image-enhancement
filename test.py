# test.py
import os
import cv2
import numpy as np
import tensorflow as tf

# Load test data
def load_images(path):
    data = []
    for file in os.listdir(path):
        if file.endswith(('jpg', 'jpeg', 'png')):
            img = cv2.imread(os.path.join(path, file))
            img = cv2.resize(img, (256, 256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            data.append(img / 255.0)
    return np.array(data)

print("🧪 Loading test dataset...")
test_dir = 'lol_dataset/eval15/low'
ground_truth_dir = 'lol_dataset/eval15/high'

test_imgs = load_images(test_dir)
gt_imgs = load_images(ground_truth_dir)

print(f"🔍 Total test images: {len(test_imgs)}")

# Load trained model
model = tf.keras.models.load_model('enhancement_model.h5')

# Predict
print("📸 Enhancing test images...")
preds = model.predict(test_imgs)

# Evaluate
accuracy = 0
for i in range(len(preds)):
    diff = np.mean(np.abs(preds[i] - gt_imgs[i]))
    if diff < 0.1:
        accuracy += 1
final_accuracy = (accuracy / len(preds)) * 100

print("\n📊 TEST RESULTS")
print("-----------------------------------------")
print(f"✅ Test Accuracy (approx.) : {final_accuracy:.2f}%")
print("🧠 Note: Accuracy calculated based on mean difference threshold < 0.1")
print("-----------------------------------------")

# Save outputs
output_dir = "test_output"
os.makedirs(output_dir, exist_ok=True)
for i, img in enumerate(preds):
    out_img = (img * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, f"enhanced_{i}.png"), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

print(f"📂 Enhanced test images saved to: {output_dir}")
