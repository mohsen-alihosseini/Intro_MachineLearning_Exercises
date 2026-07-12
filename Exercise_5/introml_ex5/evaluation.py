import numpy as np


def confusion_matrix(y_true, y_pred):
    """
    Compute a confusion matrix using NumPy only.

    Requirements:
        - convert both inputs to NumPy arrays
        - check that they have the same one-dimensional shape
        - infer the number of classes from the largest observed label
        - count how often every pair (true_label, pred_label) occurs
        - return an integer matrix of shape (num_classes, num_classes)
    """
    # Convert inputs to NumPy arrays.
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Validate the shapes.
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Input to confusion matrix must be oone dimensional")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    # Handle the empty-input case.
    if y_true.shape[0] == 0:
        return np.zeros((0, 0), dtype=int)
    # Infer the number of classes.
    max_label = int(np.max([y_true.max(), y_pred.max()]))
    classes = max_label + 1
    # Initialize the confusion matrix.
    cm = np.zeros((classes, classes), dtype=int)
    # Fill the confusion matrix.
    for true, pred in zip(y_true, y_pred):
        cm[int(true), int(pred)] += 1
    return cm
