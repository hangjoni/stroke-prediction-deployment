# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pickle

import xgboost
import imblearn
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE

import sklearn
import skopt
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, KBinsDiscretizer, PolynomialFeatures
from sklearn import tree
from sklearn.model_selection import StratifiedKFold, train_test_split, LearningCurveDisplay
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import confusion_matrix #classification_report, roc_curve, auc, accuracy_score

from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

from sklearn.base import BaseEstimator, TransformerMixin




# to surpress warnings from imblearn
import numpy as np 
np.int = int
# to surpress warnings from skopt
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.simplefilter("ignore", category=UserWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# use a consistent random state for hyper parameters tuning
RANDOM_STATE=1001

# Utilities to plot feature importances. Needed since training was done on a pipeline, consisting of transformers and a predictor 
def get_feature_names(column_transformer):
    """Get feature names from all transformers."""
    output_features = []

    for name, pipe, features in column_transformer.transformers_:
        if name == 'remainder':
            continue
        if hasattr(pipe, 'get_feature_names_out'):
            # For transformers with a get_feature_names_out method
            transformed_features = pipe.get_feature_names_out(features)
            output_features.extend(transformed_features)
        else:
            # For transformers without a get_feature_names_out method
            output_features.extend(features)

    return output_features


def show_feature_importance(trained_pipeline, clf_name):
    # Get feature names
    feature_names = get_feature_names(trained_pipeline.named_steps['columntransformer'])

    # Get feature importances
    feature_importances = trained_pipeline.named_steps[clf_name].feature_importances_

    # Create a DataFrame for easy viewing
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

    # Sort the DataFrame by importance
    feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

    # print(feature_importance_df)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.gca().invert_yaxis()  # Invert the y-axis to have the most important feature on top
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()


# Experiment with feature engineering

# Custom transformer for binning
class CustomBinner(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins=4, strategy='quantile'):
        self.n_bins = n_bins
        self.strategy = strategy
        self.binners = {}
    
    def fit(self, X, y=None):
        for column in X.columns:
            binner = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy=self.strategy)
            self.binners[column] = binner.fit(X[[column]])
        return self
    
    def transform(self, X):
        X_binned = X.copy()
        for column in X.columns:
            X_binned[column] = self.binners[column].transform(X[[column]])
        return X_binned
    
    def get_feature_names_out(self, input_features=None):
        print(input_features)
        if input_features is None:
            input_features = self.binners.keys()
        return [f"{column}_binned" for column in input_features]

# Custom transformer for polynomial features
class CustomPoly(BaseEstimator, TransformerMixin):
    def __init__(self, degree=2, include_bias=False):
        self.degree = degree
        self.include_bias = include_bias
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=self.include_bias)
    
    def fit(self, X, y=None):
        self.poly.fit(X)
        return self
    
    def transform(self, X):
        return self.poly.transform(X)
    
    def get_feature_names_out(self, input_features=None):
        # Generate polynomial feature names. This requires input_features to be specified.
        if input_features is None:
            raise ValueError("input_features is required to get feature names.")
        feature_names = self.poly.get_feature_names_out(input_features)
        return feature_names

# Processing pipelines, one without feature engineering and one with
def processing(min_freq=1):
    return ColumnTransformer([
        ('numericals', StandardScaler(), numericals),
        ('binaries', OrdinalEncoder(), binaries),
        ('categoricals', OneHotEncoder(handle_unknown='ignore', min_frequency=min_freq), categoricals)
    ], remainder='drop')


def processing2(min_freq=1):    
    return ColumnTransformer([
        ('numericals', Pipeline([
            ('scaler', StandardScaler()),
            ('poly', CustomPoly(degree=2)),
        ]), numericals),
        ('binaries', OrdinalEncoder(), binaries),
        ('categoricals', OneHotEncoder(handle_unknown='ignore', min_frequency=min_freq), categoricals),
        ('binned', Pipeline([
            ('binner', CustomBinner(n_bins=4, strategy='quantile')),
        ]), numericals),
    ], remainder='drop')

