import sys
import os
import pickle

def save_model(file_path, model):
    """
    Save a Python object (e.g., trained model/history) to disk using pickle.

    Parameters
    ----------
    file_path : str
        Destination path for the output `.pkl` file. Parent directories are
        created automatically if they do not exist.
    model : object
        Python object to serialize and save.

    Returns
    -------
    None
        This function writes the object to disk and prints a confirmation message.

    Notes
    -----
    This function uses `pickle`, so load the file only in trusted environments.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Saved model to '{file_path}'")


def load_model(file_path):
    """
    Loads a machine learning model from the specified file path.
    Parameters:
        file_path (str): The path to the model file to be loaded.
    Returns:
        object: The loaded model object.
    Raises:
        FileNotFoundError: If the specified file does not exist.
        pickle.UnpicklingError: If the file cannot be unpickled.
    """

    return pickle.load(open(file_path, "rb"))