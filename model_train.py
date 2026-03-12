############################ MODEL TRAINING #####################################


# Models created in 'model_build.py' will be trained here. Use this script to
# - Declare callbacks/early stopping
# - Declare training metrics e.g accuracy, cross entropy etc.
# - Run training/save models


# Load the prepared/processed data from data_preparation...
from data_preparation import export_data, export_datagen
# and return model built/compiled in model_build.py:
from model_build import export_model

X_train, X_val, X_test, y_train, y_val, y_test = export_data()

import tensorflow
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

callbacks = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
]

model = export_model()
datagen = export_datagen()

model.summary()