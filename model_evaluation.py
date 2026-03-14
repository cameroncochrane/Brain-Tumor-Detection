import pickle
import os
import sys

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D


sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from save_load_models import load_model, load_history
from plot_and_evaluation import plot_training_and_validation, plot_confusion_matrix
from data_preparation import export_data


X_train, X_val, X_test, y_train, y_val, y_test = export_data()

saved_model_path = "models/large_model_da_1e4_10epoch_1.keras"
model = load_model(saved_model_path)

saved_model_history_path = "models/history/large_model_da_1e4_10epoch_1.pkl"
history = load_history(saved_model_history_path)

plot_training_and_validation(history)

y_pred = model.predict(X_test)
y_pred = y_pred.argmax(axis=1)

plot_confusion_matrix(y_test,y_pred)


