############################ MODEL BUILDING #####################################

# Models will be built/declared here, then exported for training in model_train.py.

# Load the prepared/processed data from data_preparation:
from data_preparation import export_data
# Need X_train and y_train to determine input/output layer shapes
X_train, X_val, X_test, y_train, y_val, y_test, unique_labels = export_data() 
# input_shape=X_train[0].shape for CNN input layer 

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D


model = Sequential([
    # Model taken from 'Reyes and Sanchez. Heliyon 10 (2024)'. More parameters than previous models.
    # Block 1
    Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=X_train[0].shape),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    # Block 2
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    # Block 3
    Conv2D(256, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    # Block 4
    Conv2D(256, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    # Block 5
    Conv2D(512, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    # Output block
    Flatten(),
    Dense(512, activation="relu"),
    Dropout(0.5),
    Dense(4, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy", #Good for One-Hot encoded labels.
    metrics=["accuracy"]
)

model.summary()

def export_model():
    return model