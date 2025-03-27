# src/models/poisson_cell.py
from sklearn.linear_model import PoissonRegressor
import numpy as np

class PoissonPerCellModel:
    """
    Poisson regression model per spatial cell.

    If use_timestep=True: uses a time-dependent covariate (may break exchangeability).
    If use_timestep=False: uses intercept only (stationary model, CP-compatible).
    """
    def __init__(self, use_timestep=True):
        self.models_ = {}  # (row, col) -> model
        self.use_timestep = use_timestep

    def fit(self, X, y):
        for key in zip(X["row"], X["col"]):
            self.models_[key] = None

        for (r, c) in self.models_:
            mask = (X["row"] == r) & (X["col"] == c)
            y_cell = y[mask]
            if self.use_timestep:
                X_cell = X.loc[mask, ["timestep"]].values.reshape(-1, 1)
            else:
                X_cell = np.ones((len(y_cell), 1))  # intercept only

            model = PoissonRegressor(alpha=0, fit_intercept=True)
            model.fit(X_cell, y_cell)
            self.models_[(r, c)] = model

    def predict(self, X):
        preds = []
        for row, col, t in zip(X["row"], X["col"], X["timestep"]):
            key = (row, col)
            model = self.models_.get(key)
            if model:
                if self.use_timestep:
                    pred = model.predict([[t]])[0]
                else:
                    pred = model.predict([[1]])[0]  # intercept only
            else:
                pred = 0.0
            preds.append(pred)
        return np.array(preds)
