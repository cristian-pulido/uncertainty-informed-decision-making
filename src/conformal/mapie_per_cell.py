import numpy as np
import pandas as pd
from mapie.regression import MapieRegressor
from sklearn.dummy import DummyRegressor
from tqdm import tqdm
from sklearn.base import clone

def apply_conformal_mapie_per_cell(X_cal, y_cal, X_test, 
                                   base_estimator=None, 
                                   alpha=0.1, 
                                   method="plus", 
                                   grid_size=(40, 40), 
                                   random_state=None, 
                                   prefit=False):
    """
    Apply conformal prediction using MAPIE individually per spatial cell.

    Parameters:
    - X_cal, y_cal: Calibration data. If prefit=True, this should only contain calibration data.
      If prefit=False, it should include training + calibration data for internal splitting.
    - X_test: Test data (must include 'row', 'col', 'timestep')
    - base_estimator: callable or pre-instantiated model. If callable, a new instance is created per cell.
    - alpha: significance level (e.g., 0.1 for 90% coverage)
    - method: MAPIE method (e.g., 'plus')
    - grid_size: (rows, cols) tuple
    - random_state: optional seed for reproducibility
    - prefit: if True, assumes base_estimator is already fitted externally.

    Returns:
    - y_pred_grid: grid of predicted means (rows x cols)
    - y_lower_grid: grid of lower bounds
    - y_upper_grid: grid of upper bounds
    """
    # Ensure inputs are DataFrames
    X_cal = pd.DataFrame(X_cal)
    X_test = pd.DataFrame(X_test)

    rows, cols = grid_size
    y_pred_grid = np.zeros((rows, cols))
    y_lower_grid = np.zeros((rows, cols))
    y_upper_grid = np.zeros((rows, cols))

    for r in tqdm(range(rows), desc="Processing rows"):
        for c in range(cols):
            # Filtrar datos de la celda
            mask_cal = (X_cal["row"] == r) & (X_cal["col"] == c)
            mask_test = (X_test["row"] == r) & (X_test["col"] == c)

            if not mask_test.any() or not mask_cal.any():
                continue

            X_cal_cell = X_cal.loc[mask_cal].copy()
            y_cal_cell = y_cal.loc[mask_cal].copy()
            X_test_cell = X_test.loc[mask_test].copy()

            # Seleccionar modelo según configuración
            if prefit:
                model = base_estimator
            else:
                model = base_estimator() if callable(base_estimator) else DummyRegressor(strategy="mean")
                model.fit(X_cal_cell, y_cal_cell)

            cv_mode = "prefit" if prefit else "split"
            mapie = MapieRegressor(estimator=model, method=method, cv=cv_mode, random_state=random_state)
            mapie.fit(X_cal_cell, y_cal_cell)
            y_pred_cell, y_interval_cell = mapie.predict(X_test_cell, alpha=alpha)

            y_pred_grid[r, c] = np.mean(y_pred_cell)
            y_lower_grid[r, c] = np.mean(y_interval_cell[:, 0])
            y_upper_grid[r, c] = np.mean(y_interval_cell[:, 1])

    return y_pred_grid, y_lower_grid, y_upper_grid