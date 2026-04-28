import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

class NaivePerCellModel(BaseEstimator, RegressorMixin):
    """
    Predice la media observada por celda (row,col) a partir del entrenamiento.
    Si en predict aparece una celda no vista, usa un fallback configurable.
    
    Parameters
    ----------
    row_col : tuple(str, str), default=("row","col")
        Nombres de columnas para fila y columna.
    fallback : {"global","nan","zero"}, default="global"
        Estrategia para celdas no vistas en el fit:
          - "global": usa la media global de entrenamiento
          - "nan": devuelve np.nan
          - "zero": devuelve 0.0
    """
    def __init__(self, row_col=("row","col"), fallback="global"):
        self.row_col = row_col
        self.fallback = fallback

    def fit(self, X, y):
        df = pd.DataFrame(X).copy()
        rcol, ccol = self.row_col

        # Validaciones básicas
        if rcol not in df.columns or ccol not in df.columns:
            raise ValueError(f"X must contain '{rcol}' and '{ccol}' columns.")

        y_arr = np.asarray(y)  # evita problemas de index
        if len(df) != len(y_arr):
            raise ValueError(f"X and y have different lengths: {len(df)} vs {len(y_arr)}")

        df["_target"] = y_arr

        # Media por celda
        g = df.groupby([rcol, ccol], sort=False)["_target"].mean()
        # Guardamos como dict para lookup rápido
        self.cell_mean_ = g.to_dict()

        # Media global para fallback
        self.global_mean_ = float(df["_target"].mean()) if len(df) else 0.0

        # Guarda tipos para asegurar consistencia
        self._fitted_rows_ = df[rcol].dtype
        self._fitted_cols_ = df[ccol].dtype

        self.n_cells_ = len(self.cell_mean_)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, attributes=["cell_mean_", "global_mean_", "is_fitted_"])
        df = pd.DataFrame(X)
        rcol, ccol = self.row_col

        if rcol not in df.columns or ccol not in df.columns:
            raise ValueError(f"X must contain '{rcol}' and '{ccol}' columns.")

        # Asegurar tipos comparables (por si en predict llegan como int/str mezclados)
        r = df[rcol].astype(self._fitted_rows_, copy=False)
        c = df[ccol].astype(self._fitted_cols_, copy=False)

        # Pred por lookup; vectorizado con get en bucle (rápido para tamaños típicos)
        if self.fallback == "global":
            default = self.global_mean_
        elif self.fallback == "zero":
            default = 0.0
        elif self.fallback == "nan":
            default = np.nan
        else:
            raise ValueError("fallback must be one of {'global','nan','zero'}")

        out = np.empty(len(df), dtype=float)
        cm = self.cell_mean_
        for i, (ri, ci) in enumerate(zip(r, c)):
            out[i] = cm.get((ri, ci), default)
        return out

    # Opcional: para compatibilidad sklearn
    def get_params(self, deep=True):
        return {"row_col": self.row_col, "fallback": self.fallback}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
