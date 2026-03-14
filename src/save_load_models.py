import sys
import os
import pickle
import tensorflow as tf

def save_model(file_path, model):
    """
    Save a trained TensorFlow/Keras model to disk using TensorFlow's built-in save functionality.

    Parameters
    ----------
    file_path : str
        Destination path for the output model. Parent directories are
        created automatically if they do not exist.
    model : tf.keras.Model
        Trained TensorFlow/Keras model to save.

    Returns
    -------
    None
        This function writes the model to disk and prints a confirmation message.

    Notes
    -----
    This function uses TensorFlow's model.save(), which is recommended for Keras models.
    """

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    model.save(file_path)
    print(f"Saved TensorFlow model to '{file_path}'")


def save_history(file_path, history):
    """
    Save the training history of a model to disk as a pickle file.

    Parameters
    ----------
    file_path : str
        Destination path for the output history file. Parent directories are
        created automatically if they do not exist.
    history : dict or keras.callbacks.History
        The history object or its .history dictionary to save.

    Returns
    -------
    None
        This function writes the history to disk and prints a confirmation message.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # If a keras.callbacks.History object is passed, extract the .history dict
    if hasattr(history, 'history'):
        history = history.history #This is a dictionary...
    with open(file_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"Saved training history to '{file_path}'")


def load_model(file_path):
    """
    Loads a TensorFlow/Keras model from the specified file path.

    Parameters
    ----------
    file_path : str
        The path to the model file to be loaded.

    Returns
    -------
    tf.keras.Model
        The loaded TensorFlow/Keras model.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No model found at '{file_path}'")
    return tf.keras.models.load_model(file_path)

def load_history(file_path):
    """
    Loads the training history of a model from a pickle file.

    Parameters
    ----------
    file_path : str
        The path to the history pickle file to be loaded.

    Returns
    -------
    dict
        The loaded training history.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No history file found at '{file_path}'")
    with open(file_path, 'rb') as f:
        history = pickle.load(f)
    return history