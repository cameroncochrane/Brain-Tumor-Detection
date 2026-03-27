######### BRAIN TUMOR DETECTION APP ############################

import streamlit as st
import os
import sys
from PIL import Image

# Ensure local modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model_utils import (
	load_model_from_path,
	preprocess_pil_image,
	predict_image,
	load_class_names_from_data,
)


# Path to the model file (use the model suggested in the original script)
MODEL_PATH = os.path.join('models', 'large_model_da_rlrp_1e4_100epoch_1.keras')


def get_model(path=MODEL_PATH):
	try:
		# Streamlit cache/resource wrapper for model loading
		if hasattr(st, 'cache_resource'):
			@st.cache_resource
			def _load():
				return load_model_from_path(path)
			return _load()
		else:
			@st.cache(allow_output_mutation=True)
			def _load():
				return load_model_from_path(path)
			return _load()
	except Exception as e:
		st.error(f"Failed to load model: {e}")
		return None


def main():
	st.title('Brain Tumor Classification')
	st.write('Upload a brain MRI image and the model will classify it.')

	# Model availability check
	model = None
	if os.path.exists(MODEL_PATH):
		with st.spinner('Loading model...'):
			model = get_model(MODEL_PATH)
	else:
		st.warning(f"Model not found at {MODEL_PATH}. Please place a Keras model there.")

	uploaded = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff'])
	if uploaded is not None:
		try:
			image = Image.open(uploaded)
		except Exception as e:
			st.error(f'Cannot open image: {e}')
			return

		st.image(image, caption='Uploaded image', use_container_width=True)

		# Preprocess
		processed = preprocess_pil_image(image)

		if model is None:
			st.info('Model not loaded — showing preprocessed image only.')
			return

		# Attempt to load class names from data_preparation; optional
		class_names = load_class_names_from_data()

		with st.spinner('Running prediction...'):
			try:
				results = predict_image(model, processed, class_names=class_names, top_k=3)
			except Exception as e:
				st.error(f'Prediction failed: {e}')
				return

		# Display results
		st.subheader('Predictions')
		for r in results:
			label = r['class_name'] if r['class_name'] is not None else f"Class {r['class_index']}"
			prob = r['probability']
			st.write(f"- **{label}** — {prob:.3f}")


if __name__ == '__main__':
	main()
