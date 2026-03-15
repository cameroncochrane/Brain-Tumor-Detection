############################ MODEL TRAINING #####################################


# Models created in 'model_build.py' will be trained here. Use this script to
# - Declare callbacks/early stopping
# - Declare training metrics e.g accuracy, cross entropy etc.
# - Run training/save models

# Custom Imports:
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
# Load the prepared/processed data from data_preparation...
from data_preparation import export_data, export_datagen
# and return model built/compiled in model_build.py:
from model_build import export_model
from save_load_models import *

X_train, X_val, X_test, y_train, y_val, y_test = export_data()

import tensorflow
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# callbacks = [
    # EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    #ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
#]

model = export_model()
datagen = export_datagen()

model.summary()

trained_model = model.fit(datagen.flow(X_train, y_train, batch_size=32, shuffle = True), 
                            #callback=callbacks
                            epochs=100,
                            validation_data=(X_val, y_val))

training_history = trained_model

# Saving the model and its history
save_model_path = "models/large_model_da_1e5_100epoch_1.keras"
save_model(save_model_path,model) # We save the whole model as trained_model created by model.fit() is only a history object.

save_history_path = "models/history/large_model_da_1e5_100epoch_1.pkl"
save_history(save_history_path,training_history)


# If model has any of the following, the abbreviation used in the name:
# - ReduceLearningRateOnPlateau = rlrp
# - If no rlrp is shown, the learning rate is shown as e.g. 1e4 = 1x10-4
# - Data Augmentation = da
# - Early Stopping = es

# Number of epochs trained on is given...

# Best model so far = 

#################################################################################################################
# Need to retrain the models so that history is also saved (do as a pkl file), with the model as a keras file. 
# This will result in two files for each model. Functions have been made to do this.
#################################################################################################################


# If 'large' in the file name, it has the following structure:
#model = Sequential([
#    # Model taken from 'Reyes and Sanchez. Heliyon 10 (2024)'. More parameters than previous models.
#    # Block 1
#    Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=X_train[0].shape),
#    MaxPooling2D(pool_size=(2, 2)),
#    BatchNormalization(),
#
#    # Block 2
#    Conv2D(128, (3, 3), activation="relu", padding="same"),
#    MaxPooling2D(pool_size=(2, 2)),
#    BatchNormalization(),
#
#    # Block 3
#    Conv2D(256, (3, 3), activation="relu", padding="same"),
#    MaxPooling2D(pool_size=(2, 2)),
#    BatchNormalization(),
#
#    # Block 4
#    Conv2D(256, (3, 3), activation="relu", padding="same"),
#    MaxPooling2D(pool_size=(2, 2)),
#    BatchNormalization(),
#
#    # Block 5
#    Conv2D(512, (3, 3), activation="relu", padding="same"),
#    MaxPooling2D(pool_size=(2, 2)),
#    BatchNormalization(),
#
#    # Output block
#    Flatten(),
#    Dense(512, activation="relu"),
#    Dropout(0.5),
#    Dense(4, activation="softmax")
#])

