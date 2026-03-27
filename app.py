######### BRAIN TUMOR DETECTION APP ############################

import streamlit as st
import os
import sys
import io
import time
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

	# Initialize session history
	if 'history' not in st.session_state:
		st.session_state['history'] = []
	if 'viewing_history' not in st.session_state:
		st.session_state['viewing_history'] = None

	# Sidebar: history navigation
	st.sidebar.header('History')
	history = st.session_state['history']
	history_options = ['Current Upload'] + [f"{i+1}: {h['filename']}" for i, h in enumerate(history)]
	# Use a radio list so items are visible in the sidebar; preserve selection by index
	selected_idx = (st.session_state['viewing_history'] + 1) if st.session_state['viewing_history'] is not None else 0
	if selected_idx < 0 or selected_idx >= len(history_options):
		selected_idx = 0
	selection = st.sidebar.radio('View previous classification', history_options, index=selected_idx)
	if selection != 'Current Upload':
		idx = int(selection.split(':')[0]) - 1
		st.session_state['viewing_history'] = idx
	else:
		st.session_state['viewing_history'] = None

	# Clear history button
	if st.sidebar.button('Clear history'):
		st.session_state['history'] = []
		st.session_state['viewing_history'] = None

	uploaded = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff'])
	# If viewing an item from history, display it instead of current upload
	if st.session_state['viewing_history'] is not None:
		idx = st.session_state['viewing_history']
		# protect against stale index
		if idx < 0 or idx >= len(st.session_state['history']):
			st.session_state['viewing_history'] = None
		else:
			entry = st.session_state['history'][idx]
			try:
				image = Image.open(io.BytesIO(entry['bytes']))
			except Exception as e:
				st.error(f"Cannot open history image: {e}")
				return

			st.image(image, caption=f"History: {entry['filename']}", use_container_width=True)
			st.subheader('Predictions (history)')
			for r in entry['results']:
				label = r.get('class_name') if r.get('class_name') is not None else f"Class {r.get('class_index')}"
				prob = r.get('probability', 0.0)
				st.write(f"- **{label}** — {prob:.3f}")

			# Allow renaming and deletion from the sidebar while viewing
			new_name = st.sidebar.text_input('Rename image', value=entry['filename'], key=f"rename_input_{idx}")
			if st.sidebar.button('Save name', key=f"save_name_{idx}"):
				chosen = new_name.strip() if isinstance(new_name, str) and new_name.strip() else entry['filename']
				st.session_state['history'][idx]['filename'] = chosen
				# Rerun (via query params) so the updated name appears in the sidebar list immediately
				try:
					st.experimental_set_query_params(_refresh=str(time.time()))
				except Exception:
					pass

			if st.sidebar.button('Delete image', key=f"delete_{idx}"):
				st.session_state['history'].pop(idx)
				st.session_state['viewing_history'] = None
				try:
					st.experimental_set_query_params(_refresh=str(time.time()))
				except Exception:
					pass
			return

	if uploaded is not None:
		try:
			image = Image.open(uploaded)
		except Exception as e:
			st.error(f'Cannot open image: {e}')
			return

		st.image(image, caption='Uploaded image', use_container_width=True)

		# Get image bytes early to check for duplicates and reuse results
		try:
			img_bytes = uploaded.getvalue()
		except Exception:
			buf = io.BytesIO()
			image.save(buf, format='PNG')
			img_bytes = buf.getvalue()

		# Check if this image already exists in history
		existing_entry = None
		for h in st.session_state['history']:
			if h.get('bytes') == img_bytes:
				existing_entry = h
				break

		if existing_entry is not None:
			results = existing_entry['results']
			st.info('Loaded previous results from session history.')
		else:
			# Preprocess and predict only for new images
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

			# Save new classification to session history (use uploaded file name)
			chosen_name = getattr(uploaded, 'name', f"upload_{len(st.session_state['history'])+1}")
			entry = {
				'bytes': img_bytes,
				'filename': chosen_name,
				'results': results,
			}
			st.session_state['history'].append(entry)

		# Display results (either loaded from history or freshly computed)
		st.subheader('Predictions')
		for r in results:
			label = r['class_name'] if r.get('class_name') is not None else f"Class {r.get('class_index')}"
			prob = r.get('probability', 0.0)
			st.write(f"- **{label}** — {prob:.3f}")


if __name__ == '__main__':
	main()
