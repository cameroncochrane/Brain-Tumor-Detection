import os
from PIL import Image
import numpy as np
import pandas as pd

# /c:/Users/cochr/OneDrive/Coding_and_DS/Int-Elligence Internship November 2025/Projects/Brain Tumor Detection/data_import.py

ALLOWED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif'}

def load_images_to_df(base_dir='Misc data', nested_dir='BTC MRI Data Bhuvaji'):
    """
    Walks base_dir expecting structure:
      Misc data/
        BTC MRI Data Bhuvaji/
          Training/<label>/*.jpg
          Testing/<label>/*.jpg
    Returns a pandas DataFrame with columns: image (numpy array), label (subfolder name), split ('training'|'testing'), path.
    """
    records = []
    root = os.path.join(base_dir, nested_dir) if nested_dir else base_dir
    if not os.path.isdir(root):
        # fallback to base_dir if nested_dir not present
        if os.path.isdir(base_dir):
            root = base_dir
        else:
            return pd.DataFrame(records)

    for entry in sorted(os.listdir(root)):
        split_dir = os.path.join(root, entry)
        if not os.path.isdir(split_dir):
            continue
        entry_l = entry.lower()
        if entry_l not in ('training', 'testing'):
            continue
        split = entry_l  # normalized split name
        for label in sorted(os.listdir(split_dir)):
            label_dir = os.path.join(split_dir, label)
            if not os.path.isdir(label_dir):
                continue
            for fname in sorted(os.listdir(label_dir)):
                path = os.path.join(label_dir, fname)
                if not os.path.isfile(path):
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext not in ALLOWED_EXTS:
                    continue
                try:
                    img = Image.open(path).convert('RGB')
                    arr = np.asarray(img)
                except Exception:
                    continue
                records.append({'image': arr, 'label': label, 'split': split, 'path': path})
    return pd.DataFrame(records)

if __name__ == "__main__":
    df = load_images_to_df('Misc data', 'BTC MRI Data Bhuvaji')
    print(f"Loaded {len(df)} images")
    # example: df.iloc[0]['image'] -> numpy array, df['label'] -> labels