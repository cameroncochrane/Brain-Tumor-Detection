import os
from PIL import Image
import numpy as np
import tensorflow as tf

# Load local model helper from same folder
from save_load_models import load_model as _load_model


DEFAULT_IMG_SIZE = (128, 128)


def load_model_from_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return _load_model(path)


def preprocess_pil_image(pil_image, img_size=DEFAULT_IMG_SIZE):
    """Convert a PIL image to model-ready array: grayscale, resized, scaled.

    Returns a numpy array shaped (1, H, W, 1), dtype float32, values in [0,1].
    """
    # Ensure RGB -> convert to L (grayscale)
    img = pil_image.convert('L')
    img = img.resize(img_size, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    # Add channel and batch dims
    arr = arr[..., np.newaxis]  # (H, W, 1)
    arr = arr[np.newaxis, ...]  # (1, H, W, 1)
    arr /= 255.0
    return arr


def predict_image(model, processed_array, class_names=None, top_k=3):
    """Run model prediction and return top-k classes with probabilities.

    processed_array: (1, H, W, C) float32
    class_names: list of display names for classes (ordered)
    """
    preds = model.predict(processed_array)
    if preds.ndim == 2 and preds.shape[0] == 1:
        probs = preds[0]
    else:
        probs = np.asarray(preds).reshape(-1)

    # Get top-k indices
    top_idx = np.argsort(probs)[::-1][:top_k]
    results = []
    for i in top_idx:
        name = None
        if class_names is not None and i < len(class_names):
            name = class_names[i]
        results.append({'class_index': int(i), 'class_name': name, 'probability': float(probs[i])})
    return results


def load_class_names_from_data():
    """Attempt to retrieve class names by importing data_preparation.export_data().

    This will run the dataset preparation code, which is acceptable for local
    usage where the data folder exists. If this is not desired, pass `class_names`
    directly to `predict_image` from another source.
    """
    try:
        # Import locally to avoid top-level cost if not used
        from data_preparation import export_data
        _, _, _, _, _, _, class_names = export_data()
        return class_names
    except Exception:
        return None
