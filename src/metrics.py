from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.preprocessing import preprocess
from src.train_model import train
from src.visualize import log_confusion_matrix

import matplotlib.pyplot as plt
import mlflow

def log_confusion_matrix(y_true, y_pred, model=None, labels=None):
    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=labels)
    disp.plot()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()
