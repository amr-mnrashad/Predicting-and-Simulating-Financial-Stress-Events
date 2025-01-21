import pandas as pd
import numpy as np
from typing import List
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay

def feature_importances_to_df(model, feature_names: List[str]):
    """
    Saves model feature importances as a pandas DataFrame with feature names and importance percentages to a CSV file.

    Args:
        model: Trained model with a `feature_importances_` attribute (e.g., XGBoost or similar).
        feature_names (List[str]): List of feature names corresponding to the model's input features.

    Returns:
        Feature Importance DataFrame
    """
    # ExtractING feature importances and format them
    feature_importances = {
        "Feature": feature_names,
        "Importance (%)": [float(importance) * 100 for importance in model.feature_importances_]
    }

    df_importances = pd.DataFrame(feature_importances)
    df_importances = df_importances.sort_values(by="Importance (%)", ascending=False)

    return df_importances

def create_classification_report_general_evaluation(model, X_val, y_val):
    """
    Generates and prints a classification report.

    Args:
        model: Trained model for predictions.
        X_val: Validation set features.
        y_val: Validation set true labels.

    Returns:
        report: Dictionary representation of the classification report.
    """
    predictions = model.predict(X_val)
    report = classification_report(y_val, predictions)
    print("Classification Report:")
    print(report)
    return report

def plot_roc_auc(xgb_model, X_test, y_test):
    """
    Plots the ROC-AUC curve for the given XGBoost model and test set.

    Args:
        xgb_model: Trained XGBoost model.
        X_test (pd.DataFrame or np.ndarray): Test feature set.
        y_test (pd.Series or np.ndarray): True labels for the test set.
    """
    # Predicting probabilities for the positive class
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

    # Calculating the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Plotting the ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkblue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.fill_between(fpr, tpr, alpha=0.2, color='darkblue')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def plot_confusion_matrix(xgb_model, X_test, y_test):
    """
    Plots the confusion matrix for the given XGBoost model and test set.

    Args:
        xgb_model: Trained XGBoost model.
        X_test (pd.DataFrame or np.ndarray): Test feature set.
        y_test (pd.Series or np.ndarray): True labels for the test set.
    """
    # Predicting the class labels
    y_pred = xgb_model.predict(X_test)

    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap='Blues', values_format='d', ax=ax)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(False)
    plt.show()

def plot_shap_values(xgb_model, X_train, max_display):
    """
    Plots SHAP values for the given XGBoost model and training data.

    Args:
        xgb_model: Trained XGBoost model.
        X_train (pd.DataFrame or np.ndarray): Training feature set.
        max_display (int): Maximum number of features to display in the SHAP summary plot.
    """
    # Creating a SHAP explainer for the model
    explainer = shap.Explainer(xgb_model, X_train)

    # Calculating SHAP values
    shap_values = explainer(X_train)

    # Summary plot (bar chart for mean absolute SHAP values)
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_train, max_display=max_display)

def plot_histograms(best_model, X_test, y_test):
    """
    Plots histograms of predicted probabilities for stress and non-stress events.
    Args:
    best_model: The trained model used for prediction.
    X_test: The test set features.
    y_test: The test set target values.
    """
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # Converting y_test to a 1-dimensional array
    y_test_1d = y_test.values.ravel()

    # Histogram of predicted probabilities
    plt.hist(y_prob[y_test_1d == 1], bins=20, alpha=0.7, label='Stress Events')
    plt.hist(y_prob[y_test_1d == 0], bins=20, alpha=0.7, label='Non-Stress Events')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def model_accuracy(data, best_model, target_col):
    """
    Evaluates the trained model on the simulated data and visualizes the results.
    Args:
    simulated_data: The synthetic data generated for evaluation.
    best_model: The trained model to evaluate.
    Outputs:
    Model Accuracy: The accuracy of the model on the test set
    """
    # Dropping the target column from the test set and the train set
    X_test = data.drop(columns=[target_col])
    y_test = data[[target_col]]

    accuracy = best_model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}")

def plot_partial_dependency(best_model, df, features):
    """
    Plots the partial dependence plots for the given features.
    Args:
    best_model: The trained model to evaluate.
    df: The test set features.
    features: The features for which to plot the partial dependence plots.
    """
    # Extracting the training features from the model
    training_features = best_model.get_booster().feature_names
    df = df[training_features]
    
    for i, feature_name in enumerate(df.columns):
        if feature_name in features:
            fig, ax = plt.subplots(figsize=(10, 6))
            PartialDependenceDisplay.from_estimator(
                best_model, df, [i], percentiles=(0.01, 0.99), grid_resolution=50, ax=ax
            )
            ax.set_title(f"Partial Dependence Plot for Feature {feature_name}", fontsize=16)
            ax.set_xlabel(feature_name, fontsize=14)
            ax.set_ylabel("Partial Dependence", fontsize=14)
            plt.show()

def model_evaluation(data, best_model, target_col):
    """
    Evaluates the trained model on the simulated data and visualizes the results.
    Args:
    simulated_data: The synthetic data generated for evaluation.
    best_model: The trained model to evaluate.
    """
    # Dropping the target column from the test set and the train set
    X_test = data.drop(columns=[target_col])
    y_test = data[[target_col]]

    # visualizing the classification report
    cr = create_classification_report_general_evaluation(best_model, X_test, y_test)

    # visualizing the roc_auc plot
    plot_roc_auc(best_model, X_test, y_test)

    # visualizing the confusion matrix
    plot_confusion_matrix(best_model, X_test, y_test)

    # visualizing the histogram of predicted probabilities
    plot_histograms(best_model, X_test, y_test)