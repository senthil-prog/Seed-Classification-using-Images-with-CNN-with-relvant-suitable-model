import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------
# Load model
# ------------------------------
model = tf.keras.models.load_model("saved_models/seed_cnn.h5")

# Seed classes (must match training order)
class_names = ['barley', 'corn', 'rice', 'wheat']

# ------------------------------
# Function to predict a single image
# ------------------------------
def predict_image(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(128, 128))  # same size as training
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array, verbose=0)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = 100 * np.max(predictions)

    # Show result
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class} ({confidence:.2f}%)")
    plt.axis("off")
    plt.show()

# ------------------------------
# Predict all images in test folder
# ------------------------------
if __name__ == "__main__":
    test_dir = "data/raw/test"  # path to your test dataset

    for class_folder in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_folder)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                print(f"Classifying: {img_file}")
                predict_image(img_path)
