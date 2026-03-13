import pickle
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from save_load_models import load_model


saved_model_path_pkl = "models/"
model = load_model()