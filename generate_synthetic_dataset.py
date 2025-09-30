import cv2
import numpy as np
import os
import random

# Parameters
classes = ["wheat", "rice", "corn", "barley"]
img_size = 128
num_train = 100
num_val = 30
num_test = 30

base_dir = "data/raw"

def create_dirs():
    for split in ["train", "validation", "test"]:
        for cls in classes:
            path = os.path.join(base_dir, split, cls)
            os.makedirs(path, exist_ok=True)

def generate_seed_image(cls_name):
    # blank canvas
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    img.fill(255)  # white background

    # random ellipse to simulate a seed
    center = (random.randint(20, img_size-20), random.randint(20, img_size-20))
    axes = (random.randint(10, 25), random.randint(5, 15))
    angle = random.randint(0, 360)
    
    # different colors for classes
    color_dict = {
        "wheat": (139, 69, 19),
        "rice": (245, 222, 179),
        "corn": (255, 215, 0),
        "barley": (205, 133, 63)
    }
    color = color_dict.get(cls_name, (0,0,0))

    cv2.ellipse(img, center, axes, angle, 0, 360, color, -1)
    return img

def save_images(num_images, split):
    for cls in classes:
        for i in range(num_images):
            img = generate_seed_image(cls)
            filename = f"{cls}_{i}.jpg"
            path = os.path.join(base_dir, split, cls, filename)
            cv2.imwrite(path, img)

if __name__ == "__main__":
    create_dirs()
    save_images(num_train, "train")
    save_images(num_val, "validation")
    save_images(num_test, "test")
    print("Synthetic dataset generated successfully!")
