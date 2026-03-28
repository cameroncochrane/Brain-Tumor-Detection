# Brain Tumor Detection

**Overview**
- **Purpose:** Interactive Streamlit app for classifying brain MRI images using the provided Keras models.
- **Main app:** [app.py](app.py) — upload images, view predictions, and manage a per-session history of classified images.

**Quick Start (app only)**
- **Create & activate venv:** Use your platform's Python 3.8+ virtual environment tooling.
- **Install dependencies:** `pip install -r requirements.txt` (see [requirements.txt](requirements.txt)).
- **Run the app:** `streamlit run app.py` then open the shown Local URL in a browser.

**Using the app**
- **Upload:** Use the uploader on the main page to add an MRI image (jpg, png, bmp, tif).
- **Immediate classification:** Uploaded images are preprocessed and classified; the result appears on a dedicated history page.
- **History sidebar:** The sidebar contains a visible list of session-classified images. Select items to revisit saved predictions.
- **Rename / Delete:** When viewing a history item, use the sidebar controls to rename (persistent for the session) or delete the entry.
- **Refresh:** Use the sidebar Refresh button if the UI appears unresponsive — this forces an update of sidebar items and page content.

**Project layout (important files/folders)**
- **app.py:** Main Streamlit application (UI + session history).
- **requirements.txt:** Python dependencies for running the app.
- **models/**: Trained Keras models used by the app. The default model path is set in `app.py` — replace or add models here.
- **data/**: Example datasets and the original MRI folders used for training/testing.
- **src/**: Helper modules used by the app and training code (data import, model utilities, save/load helpers). See [src/model_utils.py](src/model_utils.py).
- **dissemination/**: Deliverables (PowerPoint, Excel) showing how the model was selected and evaluated. Open these to review analysis and reporting artifacts.

**Configuration**
- **MODEL_PATH:** The app uses the path set in `app.py` (variable `MODEL_PATH`). Point this to a different Keras model file in the `models/` folder if needed.

**Notes & Tips**
- The history is session-scoped (stored in Streamlit session state) and is not persisted across browser reloads.
- If you plan to run training or heavy inference, consider installing a GPU-enabled TensorFlow package and adjust `requirements.txt` accordingly.

**Troubleshooting**
- If you see duplicate widget ID errors, ensure only one `file_uploader` exists in `app.py` (it should by default).
- If model loading fails, confirm the model file exists at the path shown in `app.py` or update `MODEL_PATH`.

**License**
- See the LICENSE file in the repository.

