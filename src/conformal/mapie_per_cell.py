import numpy as np
import pandas as pd
from mapie.regression import SplitConformalRegressor, JackknifeAfterBootstrapRegressor, TimeSeriesRegressor
from mapie.subsample import BlockBootstrap
from sklearn.dummy import DummyRegressor
from tqdm import tqdm
from sklearn.base import clone


def apply_conformal_mapie_per_cell(
    X_train, y_train,
    X_cal, y_cal,
    X_test,
    base_estimator=None,
    alpha=0.1,
    conformalizer="jackknife",  # "split", "jackknife" or "time_serie"
    grid_size=(40, 40),
    method="plus",
    agg_function="mean",
    random_state=None
):
    """
    Apply conformal prediction using MAPIE individually per spatial cell.

    This function fits and applies conformal regressors for each (row, col) spatial cell
    independently, using either the Split Conformal or Jackknife-After-Bootstrap method.

    Parameters:
    - X_train, y_train: Training data. Must include columns 'row' and 'col'.
    - X_cal, y_cal: Calibration data for conformalization. Must include 'row' and 'col'.
    - X_test: Test data (must include 'row', 'col', and optionally 'timestep').
    - base_estimator: A callable that returns an untrained estimator, or a pre-instantiated estimator.
                      If None, a DummyRegressor is used.
    - alpha: Significance level (e.g., 0.1 for 90% prediction intervals).
    - conformalizer: Conformal method to use. One of {"split", "jackknife","time_serie"}.
    - grid_size: Tuple (rows, cols) defining the spatial grid dimensions.
    - method: Aggregation method for JackknifeAfterBootstrap ("plus" or "minmax") or TimeSeriesRegressor ("enbpi").
              Only relevant if conformalizer is "jackknife".
    - method: Aggregation method for predictions across bootstrap samples ("mean" or "median"). Only relevant if conformalizer is "time_serie".
    - random_state: Optional random seed for reproducibility.

    Returns:
    - y_pred_grid: 2D numpy array (rows x cols) of predicted means.
    - y_lower_grid: 2D numpy array of lower bounds of prediction intervals.
    - y_upper_grid: 2D numpy array of upper bounds of prediction intervals.

    Notes:
    - Each cell is processed independently. If any of the train/cal/test splits for a cell are empty, it is skipped.
    - For "split", training and calibration sets must be disjoint; both are used separately via `.fit()` and `.conformalize()`.
    - For "jackknife", training and calibration data are concatenated and passed together to `.fit_conformalize()`.
    - The function returns averaged predictions per cell if multiple test samples exist in a cell.
    """

    X_train = pd.DataFrame(X_train)
    X_cal = pd.DataFrame(X_cal)
    X_test = pd.DataFrame(X_test)

    rows, cols = grid_size
    y_pred_grid = np.zeros((rows, cols))
    y_lower_grid = np.zeros((rows, cols))
    y_upper_grid = np.zeros((rows, cols))

    conformal_classes = {
        "split": SplitConformalRegressor,
        "jackknife": JackknifeAfterBootstrapRegressor,
        "time_serie": TimeSeriesRegressor,
    }

    if conformalizer not in conformal_classes:
        raise ValueError(f"Conformalizer '{conformalizer}' not supported. Choose from: {list(conformal_classes.keys())}")

    ConformalClass = conformal_classes[conformalizer]

    for r in tqdm(range(rows), desc="Processing rows"):
        for c in range(cols):
            # Masks
            mask_train = (X_train["row"] == r) & (X_train["col"] == c)
            mask_cal = (X_cal["row"] == r) & (X_cal["col"] == c)
            mask_test = (X_test["row"] == r) & (X_test["col"] == c)

            if not mask_test.any() or not mask_cal.any() or not mask_train.any():
                continue

            # Extract datasets
            X_train_cell = X_train.loc[mask_train].drop(columns=["row", "col", "timestep"], errors="ignore")
            y_train_cell = y_train.loc[mask_train]

            X_cal_cell = X_cal.loc[mask_cal].drop(columns=["row", "col", "timestep"], errors="ignore")
            y_cal_cell = y_cal.loc[mask_cal]

            X_test_cell = X_test.loc[mask_test].drop(columns=["row", "col", "timestep"], errors="ignore")

            # Select or fit model
            model = base_estimator() if callable(base_estimator) else DummyRegressor(strategy="mean")

            if conformalizer == "split":
                mapie = ConformalClass(estimator=model, confidence_level=1 - alpha, prefit=False)
                mapie.fit(X_train_cell, y_train_cell)
                mapie.conformalize(X_cal_cell, y_cal_cell)

            elif conformalizer == "jackknife": 
                if method not in ["plus","minmax"]:
                    raise ValueError(f"method={method} is not compatible with JackknifeAfterBootstrapRegressor.")
                X_comb = pd.concat([X_train_cell, X_cal_cell])
                y_comb = pd.concat([y_train_cell, y_cal_cell])
                mapie = ConformalClass(
                    estimator=model,
                    confidence_level=1 - alpha,
                    method=method,
                    random_state=random_state
                )
                mapie.fit_conformalize(X_comb, y_comb)

            else:
                if agg_function not in ["mean","median"]:
                    raise ValueError(f"method={method} is not compatible with TimeSeriesRegressor.")
                X_comb = pd.concat([X_train_cell, X_cal_cell])
                y_comb = pd.concat([y_train_cell, y_cal_cell])
                cv = BlockBootstrap(n_resamplings=15, n_blocks=3, overlapping=False, random_state=random_state)
                mapie_ts = TimeSeriesRegressor(
                    estimator=model,
                    method="enbpi",
                    cv=cv,
                    agg_function="mean",
                    n_jobs=-1,
                )
                mapie_ts.fit(X_comb, y_comb)

            # Predict

            if conformalizer=="time_serie":
                y_pred_cell, y_interval_cell  = mapie_ts.predict(
                                                    X_test_cell,
                                                    confidence_level=1-alpha,
                                                    ensemble=True
                                                )
            else:
                y_pred_cell, y_interval_cell = mapie.predict_interval(X_test_cell)
            y_pred_grid[r, c] = np.mean(y_pred_cell)
            y_lower_grid[r, c] = np.mean(y_interval_cell[:, 0])
            y_upper_grid[r, c] = np.mean(y_interval_cell[:, 1])

    return y_pred_grid, y_lower_grid, y_upper_grid