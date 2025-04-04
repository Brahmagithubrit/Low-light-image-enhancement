import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split

# Paths
low_dir = r'E:/Low-Light-Image-Enhancement2/lol_dataset/our485/low'
high_dir = r'E:/Low-Light-Image-Enhancement2/lol_dataset/our485/high'
pretrained_model_path = 'zerodcepp_pretrained.h5'

# Load images
def load_images(folder):
    data = []
    for file in os.listdir(folder):
        if file.lower().endswith(('jpg', 'jpeg', 'png')):
            path = os.path.join(folder, file)
            img = cv2.imread(path)
            img = cv2.resize(img, (256, 256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            data.append(img / 255.0)
    return np.array(data)

low_imgs = load_images(low_dir)
high_imgs = load_images(high_dir)

train_low, val_low, train_high, val_high = train_test_split(low_imgs, high_imgs, test_size=0.1, random_state=42)

# Load pre-trained model
model = load_model(pretrained_model_path, compile=False)

# VGG perceptual model
vgg = VGG19(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
vgg_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
vgg.trainable = False
mse = MeanSquaredError()

def perceptual_loss(y_true, y_pred):
    return mse(vgg_model(y_true), vgg_model(y_pred))

def hybrid_loss(y_true, y_pred):
    ssim_val = tf.image.ssim(y_true, y_pred, max_val=1.0)
    return 0.5 * perceptual_loss(y_true, y_pred) + 0.5 * (1 - tf.reduce_mean(ssim_val))

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=hybrid_loss)

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
    ModelCheckpoint('best_finetuned_model.h5', save_best_only=True)
]

# Train
model.fit(train_low, train_high, validation_data=(val_low, val_high), epochs=25, batch_size=8, callbacks=callbacks)

# Predict
preds = model.predict(val_low)

# Metrics without skimage
def calc_psnr(img1, img2):
    mse_val = np.mean((img1 - img2) ** 2)
    if mse_val == 0:
        return 100
    return 20 * np.log10(1.0 / np.sqrt(mse_val))

def calc_ssim(img1, img2):
    img1 = (img1 * 255).astype(np.uint8)
    img2 = (img2 * 255).astype(np.uint8)
    return tf.image.ssim(tf.convert_to_tensor(img1[np.newaxis], dtype=tf.float32),
                         tf.convert_to_tensor(img2[np.newaxis], dtype=tf.float32),
                         max_val=255.0).numpy()[0]

psnr_total, ssim_total, garbage_total = 0, 0, 0
for pred, gt in zip(preds, val_high):
    psnr_total += calc_psnr(pred, gt)
    ssim_total += calc_ssim(pred, gt)
    garbage_total += np.mean(np.abs(pred - gt))

n = len(preds)
avg_psnr = psnr_total / n
avg_ssim = ssim_total / n
avg_garbage = garbage_total / n
proxy_accuracy = max(0, min(1, (avg_ssim + (avg_psnr / 50)) / 2)) * 100

# Report
print("\n📊 QUALITY REPORT")
print("-" * 40)
print(f"🔍 PSNR Score      : {avg_psnr:.2f}")
print(f"🧠 SSIM Score      : {avg_ssim:.4f}")
print(f"♻️  Garbage Left   : {avg_garbage:.4f}")
print(f"✅ Proxy Accuracy  : {proxy_accuracy:.2f}%")
print("-" * 40)

# Save sample outputs
output_dir = "enhanced_samples"
os.makedirs(output_dir, exist_ok=True)
for i in range(min(5, len(preds))):
    out_img = (preds[i] * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, f"enhanced_{i}.png"), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

print(f"\n✅ Sample enhanced images saved to: {output_dir}")
