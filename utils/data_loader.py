import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory

def load_data(data_dir, img_size=(128, 128), batch_size=32):
    train_ds = image_dataset_from_directory(
        f"{data_dir}/train",
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical"
    )
    val_ds = image_dataset_from_directory(
        f"{data_dir}/validation",
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical"
    )
    test_ds = image_dataset_from_directory(
        f"{data_dir}/test",
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical"
    )

    # Save class names
    class_names = train_ds.class_names

    # Normalize datasets
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

    return train_ds, val_ds, test_ds, class_names
