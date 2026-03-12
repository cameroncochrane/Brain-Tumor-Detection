############################ MODEL BUILDING #####################################

# Models will be built/declared here, then exported for training in model_train.py.

# Load the prepared/processed data from data_preparation:
from data_preparation import export_data
# Need X_train and y_train to determine input/output layer shapes
X_train, X_val, X_test, y_train, y_val, y_test = export_data() 

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D

# Tuned model 1 (trained model 3):

model = Sequential([
    Conv2D(32, (3,3), padding="same", activation="relu", input_shape=X_train[0].shape),
    BatchNormalization(),
    Conv2D(32, (3,3), padding="same", activation="relu"),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(64, (3,3), padding="same", activation="relu"),
    BatchNormalization(),
    Conv2D(64, (3,3), padding="same", activation="relu"),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(128, (3,3), padding="same", activation="relu"),
    BatchNormalization(),
    Conv2D(128, (3,3), padding="same", activation="relu"),
    MaxPooling2D((2,2)),
    Dropout(0.3),

    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),
    Dense(4, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

def export_model():
    return model