import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from save_load_models import load_model, load_history
from plot_and_evaluation import plot_model_comparison_dashboard
from data_preparation import export_data


X_train, X_val, X_test, y_train, y_val, y_test, unique_labels = export_data()

# Leave this list empty to compare every matched .keras/.pkl pair.
# Add explicit model stems to compare only specific runs.
SELECTED_MODEL_STEMS = ['large_model_da_1e4_100epoch_1','large_model_da_1e5_100epoch_1']


def discover_model_history_pairs(models_dir="models", histories_dir="models/history", selected_stems=None):
	model_root = Path(models_dir)
	history_root = Path(histories_dir)

	if not model_root.exists():
		raise FileNotFoundError(f"Models directory not found: {models_dir}")
	if not history_root.exists():
		raise FileNotFoundError(f"History directory not found: {histories_dir}")

	selected_stems = set(selected_stems or [])
	model_files = sorted(model_root.glob("*.keras"))
	pairs = []

	for model_path in model_files:
		stem = model_path.stem
		if selected_stems and stem not in selected_stems:
			continue

		history_path = history_root / f"{stem}.pkl"
		if not history_path.exists():
			print(f"Skipping '{stem}' because history file is missing.")
			continue

		pairs.append(
			{
				"name": stem,
				"model_path": str(model_path),
				"history_path": str(history_path),
			}
		)

	if not pairs:
		raise ValueError("No valid model/history pairs were found to evaluate.")

	return pairs


model_specs = discover_model_history_pairs(selected_stems=SELECTED_MODEL_STEMS)
comparison_results = []

for spec in model_specs:
	print(f"Evaluating: {spec['name']}")
	model = load_model(spec["model_path"])
	history = load_history(spec["history_path"])

	y_pred = model.predict(X_test, verbose=0)

	comparison_results.append(
		{
			"name": spec["name"],
			"history": history,
			"y_true": y_test,
			"y_pred": y_pred,
		}
	)

plot_model_comparison_dashboard(comparison_results, labels=unique_labels)


