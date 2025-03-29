import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

class NaiveCellMeanModel(BaseEstimator, RegressorMixin):
    """
    Naive model that predicts the average target value for a single spatial cell.
    Should be used per (row, col) location.
    """
    def __init__(self):
        self.fitted_ = False

    def fit(self, X, y):
        df = pd.DataFrame(X)
        df["target"] = y
        self.row = df["row"].iloc[0]
        self.col = df["col"].iloc[0]
        self.mean_ = df["target"].mean()
        self.fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, attributes=["mean_"])
        return np.full(len(X), self.mean_)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


class NaivePerCellModel(BaseEstimator, RegressorMixin):
    """
    Naive model that stores a NaiveCellMeanModel per (row, col) cell.
    Predicts the average target value for each cell based on training data.
    """
    def __init__(self):
        self.models_ = {}
        self.fitted_ = False

    def fit(self, X, y):
        df = pd.DataFrame(X)
        df["target"] = y
        for (r, c), group in df.groupby(["row", "col"]):
            model = NaiveCellMeanModel()
            model.fit(group, group["target"])
            self.models_[(r, c)] = model
        self.fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, attributes=["models_"])

        if isinstance(X, pd.DataFrame):
            if "row" not in X.columns or "col" not in X.columns:
                raise ValueError("DataFrame must contain 'row' and 'col' columns.")
            df = X
        elif isinstance(X, np.ndarray):
            if X.shape[1] != 2:
                raise ValueError(f"Expected ndarray with 2 columns (row, col), got shape {X.shape}")
            df = pd.DataFrame(X, columns=["row", "col"])
        else:
            raise TypeError("X must be a DataFrame or ndarray with 2 columns.")

        return np.array([
            self.models_.get((row, col), NaiveCellMeanModel()).predict([[row, col]])[0]
            for row, col in zip(df["row"], df["col"])
        ])


    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self
