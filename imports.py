# Data manipulation and visualization libraries
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
# Set the renderer to notebook
import plotly.io as pio
import seaborn as sns
# CatBoost and XGBoost libraries
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
# Scikit-learn cross-validation, logistic regression, and evaluation metrics
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
# Label Encoder
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

import aggregation_functions
import customized_classifier
import evaluation_functions
import reusable_functions
import scaling_functions
import visualization_functions
from customized_classifier import *

pio.renderers.default = "notebook"

plt.style.use('fivethirtyeight')
pd.options.display.max_columns = 500
color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']

import math
import pickle
from typing import List

import plotly.graph_objects as go
import shap
import xgboost as xgb
from joblib import dump, load
from plotly.subplots import make_subplots
from scipy.stats import boxcox, pointbiserialr, zscore
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV, mutual_info_classif
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, auc,
                             classification_report, confusion_matrix, f1_score,
                             make_scorer, roc_auc_score, roc_curve)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from skopt import BayesSearchCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBClassifier

# Suppress all warnings
warnings.filterwarnings("ignore")
