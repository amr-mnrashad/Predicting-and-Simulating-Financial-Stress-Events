import numpy as np
from xgboost import XGBClassifier

def custom_loss(y_true, y_pred):
    """
    Custom loss function for XGBoost.
    """
    y_pred = 1.0 / (1.0 + np.exp(-y_pred))  # Converting log-odds to probabilities
    grad = -(custom_loss.weight_positive * y_true * (1 - y_pred) - custom_loss.weight_negative * (1 - y_true) * y_pred)
    hess = (custom_loss.weight_positive * y_true + custom_loss.weight_negative * (1 - y_true)) * y_pred * (1 - y_pred)
    return grad, hess

def set_custom_loss_weights(weight_positive, weight_negative):
    """
    Set the weights for the custom loss function globally.
    """
    custom_loss.weight_positive = weight_positive
    custom_loss.weight_negative = weight_negative

class CustomXGBClassifier(XGBClassifier):
    """
    Custom XGBClassifier class that allows for setting custom weights for the custom_loss function.
    """
    def __init__(self, weight_positive, weight_negative, **kwargs):
        super().__init__(**kwargs)
        self.weight_positive = weight_positive
        self.weight_negative = weight_negative

    def fit(self, X, y, **kwargs):
        # Setting global weights for custom_loss
        set_custom_loss_weights(self.weight_positive, self.weight_negative)
        # Using the custom loss function
        self.set_params(objective=custom_loss)
        return super().fit(X, y, **kwargs)