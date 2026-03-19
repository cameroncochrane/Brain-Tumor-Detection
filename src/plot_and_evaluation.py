import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


def plot_training_and_validation(history):

    plt.figure(figsize=(8, 5))
    plt.plot(history['loss'], marker='o', label='Training Loss')
    plt.plot(history['val_loss'], marker='o', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.show()


def plot_confusion_matrix(y_test, y_pred, labels=None):
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_test = np.argmax(y_test, axis=1)

    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)

    label_indices = np.unique(np.concatenate([y_test, y_pred]))
    display_labels = labels if labels is not None else label_indices

    cm = confusion_matrix(y_test, y_pred, labels=label_indices)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title("Confusion Matrix")
    plt.show()


def _to_class_indices(values):
    values = np.asarray(values)
    if values.ndim > 1 and values.shape[1] > 1:
        return np.argmax(values, axis=1)
    return values.reshape(-1)


def _resolve_display_labels(label_indices, labels):
    if labels is None:
        return label_indices

    labels = np.asarray(labels)
    if len(labels) == len(label_indices):
        return labels

    # If class ids are integer encoded, map ids directly into full label list.
    if np.issubdtype(label_indices.dtype, np.integer):
        max_index = int(np.max(label_indices))
        if len(labels) > max_index:
            return labels[label_indices]

    raise ValueError("Provided labels do not align with the class indices in predictions.")


def plot_model_comparison_dashboard(model_results, labels=None, max_confusion_cols=3):
    """
    Render a single evaluation window for any number of models.

    Parameters
    ----------
    model_results : list[dict]
        Each item must contain:
        - 'name': model name shown in legends/titles
        - 'history': training history dict with keys like 'loss' and 'val_loss'
        - 'y_true': ground-truth labels (one-hot or class indices)
        - 'y_pred': model predictions (one-hot/probabilities or class indices)
    labels : array-like, optional
        Class display labels for confusion matrices.
    max_confusion_cols : int
        Maximum confusion matrix columns in the dashboard grid.
    """
    if not model_results:
        raise ValueError("model_results must contain at least one model entry.")

    y_pairs = []
    for result in model_results:
        y_true = _to_class_indices(result["y_true"])
        y_pred = _to_class_indices(result["y_pred"])
        y_pairs.append((y_true, y_pred))

    label_indices = np.unique(
        np.concatenate([np.concatenate((y_true, y_pred)) for y_true, y_pred in y_pairs])
    )
    display_labels = _resolve_display_labels(label_indices, labels)

    model_count = len(model_results)
    cm_cols = min(max_confusion_cols, model_count)
    cm_rows = ceil(model_count / cm_cols)
    history_cols = cm_cols
    history_rows = ceil(model_count / history_cols)
    report_cols = cm_cols
    report_rows = ceil(model_count / report_cols)

    fig_height = (4 * history_rows) + (4 * cm_rows) + (5 * report_rows) + 2
    fig = plt.figure(figsize=(6 * cm_cols, fig_height), constrained_layout=True)
    grid = fig.add_gridspec(history_rows + cm_rows + report_rows, cm_cols)

    for idx, result in enumerate(model_results):
        history_row = idx // history_cols
        history_col = idx % history_cols
        history_ax = fig.add_subplot(grid[history_row, history_col])

        history = result.get("history", {})
        name = result.get("name", f"Model {idx + 1}")

        loss = history.get("loss", [])
        val_loss = history.get("val_loss", [])

        if len(loss) > 0:
            history_ax.plot(np.arange(1, len(loss) + 1), loss, marker='o', label="Training")
        if len(val_loss) > 0:
            history_ax.plot(
                np.arange(1, len(val_loss) + 1),
                val_loss,
                marker='o',
                linestyle='--',
                label="Val"
            )

        history_ax.set_xlabel("Epoch")
        history_ax.set_ylabel("Loss")
        history_ax.set_title(f"{name} - Loss")
        history_ax.grid(True)
        history_ax.legend(loc="best", fontsize=8)

    history_total_slots = history_rows * history_cols
    for empty_idx in range(model_count, history_total_slots):
        history_row = empty_idx // history_cols
        history_col = empty_idx % history_cols
        ax = fig.add_subplot(grid[history_row, history_col])
        ax.axis("off")

    for idx, result in enumerate(model_results):
        row = history_rows + (idx // cm_cols)
        col = idx % cm_cols
        ax = fig.add_subplot(grid[row, col])

        y_true, y_pred = y_pairs[idx]
        cm = confusion_matrix(y_true, y_pred, labels=label_indices)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False, values_format='d')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.set_title(result.get("name", f"Model {idx + 1}"))

    total_slots = cm_rows * cm_cols
    for empty_idx in range(model_count, total_slots):
        row = history_rows + (empty_idx // cm_cols)
        col = empty_idx % cm_cols
        ax = fig.add_subplot(grid[row, col])
        ax.axis("off")

    for idx, result in enumerate(model_results):
        row = history_rows + cm_rows + (idx // report_cols)
        col = idx % report_cols
        ax = fig.add_subplot(grid[row, col])

        y_true, y_pred = y_pairs[idx]
        report = classification_report(
            y_true,
            y_pred,
            labels=label_indices,
            target_names=[str(name) for name in display_labels],
            zero_division=0,
        )

        ax.axis("off")
        ax.set_title(f"{result.get('name', f'Model {idx + 1}')} - Classification Report", fontsize=12)
        ax.text(
            0.5,
            1.0,
            report,
            va="top",
            ha="center",
            multialignment="left",
            family="monospace",
            fontsize=12,
            transform=ax.transAxes,
        )

    report_total_slots = report_rows * report_cols
    for empty_idx in range(model_count, report_total_slots):
        row = history_rows + cm_rows + (empty_idx // report_cols)
        col = empty_idx % report_cols
        ax = fig.add_subplot(grid[row, col])
        ax.axis("off")

    fig.suptitle("Model Evaluation Dashboard", fontsize=14)
    plt.show()