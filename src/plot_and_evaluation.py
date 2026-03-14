import matplotlib.pyplot as plt
import tensorflow
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    plt.figure(figsize=(6, 5))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    
    plt.show()