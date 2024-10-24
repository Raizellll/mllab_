"""
metrics of classification
"""

import numpy as np


def accuracy_score(y_true, y_pred):
    """Accuracy classification score.
    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in y_true.
    Read more in the :ref:`User Guide <accuracy_score>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.

    Returns
    -------
    score : accuracy
    """
    acc = -1
    #########################################################################
    # TODO:                                                                 #
    # Calculate the accuracy.                                               #
    #########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Check if the shapes of y_true and y_pred match
    if y_true.shape != y_pred.shape:
        raise ValueError("The shape of y_true and y_pred must be the same.")
    
    # Calculate accuracy
    correct = (y_true == y_pred).sum()
    total = len(y_true)
    acc = correct / total
    
    # Ensure acc is a float between 0 and 1
    acc = float(acc)
    if not (0 <= acc <= 1):
        raise ValueError("Accuracy score must be between 0 and 1.")
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return acc
