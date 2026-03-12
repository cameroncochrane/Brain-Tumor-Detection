# Base Imports:
import os
import sys
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle

# Custom Imports:
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from data_import import *
##################################################################################################

raw_df = load_images_to_df('data', 'BTC MRI Data Bhuvaji')#Load raw data into a single df
data = raw_df.drop(['split','path'],axis=1) #Drop unecessary columns

data['image'] = data['image'].apply(lambda x: x[:,:,0]) #Re-shape

# 1) Train-test-val split:
X = data['image']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, stratify=y, random_state=13) # Create train and test sets.
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.1, stratify=y_train, random_state=13) # Create val set from train sets.

# 2) Resize to a fixed size, then add a channel dimension (grayscale => 1 channel)
IMG_SIZE = (128, 128)

def resize_stack_with_channel(x_series, size=IMG_SIZE):
	arr = np.stack([
		np.array(Image.fromarray(img).resize(size, Image.BILINEAR))
		for img in x_series
	])
	return arr[..., np.newaxis]  # (N, H, W, 1)

X_train = resize_stack_with_channel(X_train)
X_val = resize_stack_with_channel(X_val)
X_test = resize_stack_with_channel(X_test)

# 3) Normalise the pixel values + make sure dataype = float32:
X_train = X_train.astype("float32") / 255.0
X_val = X_val.astype("float32") / 255.0
X_test  = X_test.astype("float32") / 255.0

# 4) Encode the labels:
label_encoder = LabelEncoder()

y_train_int = label_encoder.fit_transform(y_train)
y_val_int = label_encoder.transform(y_val)
y_test_int = label_encoder.transform(y_test)

num_classes = len(label_encoder.classes_)

y_train = to_categorical(y_train_int, num_classes=num_classes)
y_val = to_categorical(y_val_int, num_classes=num_classes)
y_test = to_categorical(y_test_int, num_classes=num_classes)

class_names = label_encoder.classes_.tolist()

# 5) Data augmentation (ImageDataGenerator):
    # Helps to combat overfitting
    # Introduces slight variablilities in newly generated training images.
datagen = ImageDataGenerator( #If the augmentation is too agressive, early training can be dominated by 'distored' digits, causing poor predictions)
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
)
datagen.fit(X_train) #Apply only to the training set, not the validation (or test) set.

print(X_train)

def export_data():
	# Export train, val and test data to be used in external scripts
	return X_train, X_val, X_test, y_train, y_val, y_test

def export_datagen():
	# Export the fitted image datagen that may be needed during model training.
	return datagen
