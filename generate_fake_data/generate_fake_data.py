import cv2
import os
import random
import numpy as np

#  ABSOLUTE PATHS (important)
real_dir = "D:/fake-logo-detection-cnn/dataset/real"
fake_dir = "D:/fake-logo-detection-cnn/dataset/fake"

os.makedirs(fake_dir, exist_ok=True)

# ❗ Clear old fake images (IMPORTANT)
for f in os.listdir(fake_dir):
    os.remove(os.path.join(fake_dir, f))

count = 0

for img_name in os.listdir(real_dir):
    img_path = os.path.join(real_dir, img_name)

    img = cv2.imread(img_path)
    if img is None:
        continue

    img = cv2.resize(img, (128, 128))
    fake = img.copy()


    # 1. Rotation
    if random.random() > 0.5:
        angle = random.randint(-25, 25)
        h, w = fake.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        fake = cv2.warpAffine(fake, M, (w, h))

    # 2. Crop + resize (distortion)
    if random.random() > 0.5:
        h, w = fake.shape[:2]
        crop = random.randint(5, 20)
        fake = fake[crop:h-crop, crop:w-crop]
        fake = cv2.resize(fake, (128, 128))

    # 3. Mild blur
    if random.random() > 0.5:
        fake = cv2.GaussianBlur(fake, (3, 3), 0)

    # 4. Slight noise (fixed version)
    if random.random() > 0.5:
        noise = np.zeros_like(fake, dtype=np.int16)
        cv2.randn(noise, 0, 10)
        fake = cv2.add(fake.astype(np.int16), noise)
        fake = np.clip(fake, 0, 255).astype(np.uint8)

    # Save
    save_path = os.path.join(fake_dir, f"fake_{count}.jpg")
    cv2.imwrite(save_path, fake)
    count += 1

print(f" Fake images generated: {count}")