import tensorflow as tf 
from utils.data_loader import load_data
import os

# Base directory (contains train, validation, test folders)
BASE_DIR = "data/raw"

# Load datasets
_, _, test_ds, class_names = load_data(BASE_DIR)

# Load trained model
model = tf.keras.models.load_model("saved_models/seed_cnn.h5")

# Evaluate model
loss, acc = model.evaluate(test_ds)
print(f"✅ Test Accuracy: {acc*100:.2f}%")
