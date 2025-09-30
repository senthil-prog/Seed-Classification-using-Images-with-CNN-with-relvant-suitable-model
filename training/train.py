import os
from models.cnn_model import build_cnn_model
from utils.data_loader import load_data
from utils.visualize import plot_training

# ------------------------------
# Base directory containing train/validation/test subfolders
# ------------------------------
BASE_DIR = "data/raw"

# ------------------------------
# Load data
# ------------------------------
train_ds, val_ds, test_ds, class_names = load_data(BASE_DIR)
print("Seed classes:", class_names)
num_classes = len(class_names)

# ------------------------------
# Build CNN model
# ------------------------------
model = build_cnn_model(num_classes=num_classes)

# ------------------------------
# Train the model
# ------------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15
)

# ------------------------------
# Save the trained model
# ------------------------------
os.makedirs("saved_models", exist_ok=True)
model.save("saved_models/seed_cnn.h5")
print("Model saved successfully!")

# ------------------------------
# Plot training history
# ------------------------------
plot_training(history)
