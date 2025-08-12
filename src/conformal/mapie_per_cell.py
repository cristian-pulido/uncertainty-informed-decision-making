import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from mapie.regression import CrossConformalRegressor, SplitConformalRegressor, JackknifeAfterBootstrapRegressor, TimeSeriesRegressor
from mapie.subsample import Subsample, BlockBootstrap
from sklearn.dummy import DummyRegressor
from tqdm import tqdm
from .interval_prediction_gen import MinMaxIntervalRegressor,GaussianHomoskedasticPIRegressor,ResidualBootstrapPIRegressor,PoissonIntervalRegressor
from sklearn.base import clone

def apply_conformal_mapie_per_cell(
    X_train, y_train,
    X_cal, y_cal,
    X_test, y_test,
    base_estimator=None,
    alpha=0.1,
    conformalizer="jackknife",  # "split", "jackknife", or "time_serie" "cross"
    grid_size=(40, 40),
    method="plus",              # Only for "jackknife"
    agg_function="mean",        # Only for "time_serie"
    random_state=None,
    return_models=False         # Whether to return trained MAPIE objects
):
    """
    Apply conformal prediction per spatial cell using MAPIE with 3D outputs.

    Returns tensors (timestep x row x col) of predictions and intervals,
    and optionally the fitted conformal models per cell.

    Parameters
    ----------
    X_train, y_train : pandas DataFrame or compatible Training data.
        Must include columns 'row' and 'col'.
    X_cal, y_cal : pandas DataFrame or compatible 
        Calibration data for conformalization. Must include 'row' and 'col'.
    X_test : pandas DataFrame Test data. 
        Must include 'row' and 'col'. Optionally includes 'timestep'.
    base_estimator : callable or estimator, 
        optional Callable that returns an untrained estimator (e.g. DummyRegressor), or a fitted estimator. If None, a DummyRegressor is used.
    alpha : float 
        Significance level (e.g., 0.1 for 90% prediction intervals).
    conformalizer : str Method to use for conformal prediction.
         One of {"split", "jackknife", "time_serie"}.
    grid_size : tuple of int
        Tuple (rows, cols) defining the spatial grid size.
    method : str
        Aggregation method for JackknifeAfterBootstrap ("plus", "minmax").
    agg_function : str
        Aggregation function for TimeSeriesRegressor ("mean", "median").
    random_state : int, optional
        Random seed for reproducibility.
    return_models : bool
        If True, also return a (rows x cols) array of fitted MAPIE objects (or None if not trained).

    Returns
    -------
    y_pred_tensor : np.ndarray of shape (T, R, C)
    y_lower_tensor : np.ndarray of shape (T, R, C)
    y_upper_tensor : np.ndarray of shape (T, R, C)
    y_true_tensor : np.ndarray of shape (T, R, C)
    models_grid : np.ndarray of shape (R, C), optional
        If return_models=True. Each cell contains the fitted MAPIE model or None.
    """
    X_train = pd.DataFrame(X_train)
    X_cal = pd.DataFrame(X_cal)
    X_test = pd.DataFrame(X_test)

    rows, cols = grid_size
    T = X_test["timestep"].nunique()
    

    y_pred_tensor = np.full((T, rows, cols), np.nan)
    y_lower_tensor = np.full((T, rows, cols), np.nan)
    y_upper_tensor = np.full((T, rows, cols), np.nan)
    y_true_tensor = np.full((T, rows, cols), np.nan)

    if return_models:
        models_grid = np.empty((rows, cols), dtype=object)

    conformal_classes = {
        "split": SplitConformalRegressor,
        "jackknife": JackknifeAfterBootstrapRegressor,
        "time_serie": TimeSeriesRegressor,
        "min_max": MinMaxIntervalRegressor,
        "cross": CrossConformalRegressor,
        "GaussianHomos": GaussianHomoskedasticPIRegressor,
        "ResidualBootstrap": ResidualBootstrapPIRegressor,
        "Poisson": PoissonIntervalRegressor
    }

    if conformalizer not in conformal_classes:
        raise ValueError(f"Conformalizer '{conformalizer}' not supported. Choose from: {list(conformal_classes.keys())}")

    ConformalClass = conformal_classes[conformalizer]

    for r in tqdm(range(rows), desc="Processing rows"):
        for c in range(cols):
            mask_train = (X_train["row"] == r) & (X_train["col"] == c)
            mask_cal = (X_cal["row"] == r) & (X_cal["col"] == c)
            mask_test = (X_test["row"] == r) & (X_test["col"] == c)

            if not mask_test.any() or not mask_cal.any() or not mask_train.any():
                continue

            X_train_cell = X_train.loc[mask_train].drop(columns=["row", "col", "timestep"], errors="ignore")
            y_train_cell = y_train.loc[mask_train]

            X_cal_cell = X_cal.loc[mask_cal].drop(columns=["row", "col", "timestep"], errors="ignore")
            y_cal_cell = y_cal.loc[mask_cal]

            X_test_cell = X_test.loc[mask_test].drop(columns=["row", "col", "timestep"], errors="ignore")
            y_test_cell = y_test.loc[mask_test]
            t_test_cell = X_test.loc[mask_test, "timestep"].values


            # if callable(base_estimator):
            #     model = base_estimator()
            # elif base_estimator is not None:
            #     model = clone(base_estimator)
            # else:
            #     model = DummyRegressor(strategy="mean")

            model = base_estimator() if callable(base_estimator) else DummyRegressor(strategy="mean")

            if conformalizer in ["split", "min_max", "GaussianHomos", "ResidualBootstrap","Poisson"]:
                mapie = ConformalClass(estimator=model, confidence_level=1 - alpha, prefit=False)
                mapie.fit(X_train_cell, y_train_cell)
                mapie.conformalize(X_cal_cell, y_cal_cell)

            elif conformalizer in ["jackknife", "cross"]:
                if method not in ["plus", "minmax"]:
                    raise ValueError(f"method={method} is not compatible with JackknifeAfterBootstrapRegressor or CrossConformalRegressor.")
                X_comb = pd.concat([X_train_cell, X_cal_cell])
                y_comb = pd.concat([y_train_cell, y_cal_cell])

                
                keys = {
                        "estimator": model,
                        "confidence_level": 1 - alpha,
                        "method": method,
                        "n_jobs": -1                      
                    }

                if conformalizer == "cross":
                    cv = TimeSeriesSplit(n_splits=10)
                    keys["cv"] = cv

                else:
                    keys["resampling"]=50
                    keys["random_state"] = random_state
                    keys["aggregation_method"]=agg_function


                mapie = ConformalClass(
                    **keys
                )
                mapie.fit_conformalize(X_comb, y_comb)

            elif conformalizer == "time_serie":
                if method not in ["aci", "enbpi"]:
                    raise ValueError(f"method={method} is not compatible with TimeSeriesRegressor.")
                if agg_function not in ["mean", "median"]:
                    raise ValueError(f"agg_function={agg_function} is not compatible with TimeSeriesRegressor.")
                X_comb = pd.concat([X_train_cell, X_cal_cell])
                y_comb = pd.concat([y_train_cell, y_cal_cell])
                cv = BlockBootstrap(n_resamplings=200, length=7, overlapping=True, random_state=random_state)
                mapie = ConformalClass(
                    estimator=model,
                    method=method,
                    cv=cv,
                    agg_function=agg_function,
                    n_jobs=-1,
                )
                mapie.fit(X_comb, y_comb)

            if conformalizer == "time_serie":
                y_pred_cell, y_interval_cell = mapie.predict(
                    X_test_cell,
                    confidence_level = 1 - alpha,
                    ensemble=True
                )
            else:
                y_pred_cell, y_interval_cell = mapie.predict_interval(X_test_cell)

            # Insert into tensors by timestep
            
            y_pred_tensor[:, r, c] = y_pred_cell
            y_lower_tensor[:, r, c] = np.clip(y_interval_cell[:,0].flatten(),min=0)  # clip lower bound
            y_upper_tensor[:, r, c] = y_interval_cell[:,1].flatten()
            y_true_tensor[:, r, c] =  y_test_cell.values

            if return_models:
                models_grid[r, c] = mapie

    if return_models:
        return y_pred_tensor, y_lower_tensor, y_upper_tensor, y_true_tensor, models_grid
    else:
        return y_pred_tensor, y_lower_tensor, y_upper_tensor, y_true_tensor
