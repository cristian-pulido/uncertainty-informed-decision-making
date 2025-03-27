import pandas as pd
import numpy as np

class NaiveCellMeanModel:
    """
    Naive model that predicts the average target value per (row, col) cell.
    """

    def __init__(self):
        self.cell_means = {}

    def fit(self, X, y):
        df = pd.DataFrame(X)
        df["target"] = y
        grouped = df.groupby(["row", "col"])["target"].mean()
        self.cell_means = grouped.to_dict()

    def predict(self, X):
        df = pd.DataFrame(X)
        return np.array([
            self.cell_means.get((row, col), 0) for row, col in zip(df["row"], df["col"])
        ])
