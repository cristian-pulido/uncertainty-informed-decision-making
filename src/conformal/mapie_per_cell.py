import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from mapie.regression import CrossConformalRegressor, SplitConformalRegressor, JackknifeAfterBootstrapRegressor, TimeSeriesRegressor
from mapie.subsample import Subsample, BlockBootstrap
from sklearn.dummy import DummyRegressor
from tqdm import tqdm
from .interval_prediction_gen import MinMaxIntervalRegressor,GaussianHomoskedasticPIRegressor,ResidualBootstrapPIRegressor,PoissonIntervalRegressor, MAEIntervalRegressor
from sklearn.base import clone

from sklearn.base import clone, is_regressor

conformal_classes = {
    "split": SplitConformalRegressor,
    "jackknife": JackknifeAfterBootstrapRegressor,
    "time_serie": TimeSeriesRegressor,
    "min_max": MinMaxIntervalRegressor,
    "cross": CrossConformalRegressor,
    "GaussianHomos": GaussianHomoskedasticPIRegressor,
    "ResidualBootstrap": ResidualBootstrapPIRegressor,
    "Poisson": PoissonIntervalRegressor,
    "MAE": MAEIntervalRegressor
}

def reconstruct_tensors(X_test, y_pred, y_pi, y_true, grid_size, t_col="timestep"):
    rows, cols = grid_size
    T = X_test[t_col].nunique()
    y_pred_tensor  = np.full((T, rows, cols), np.nan)
    y_lower_tensor = np.full((T, rows, cols), np.nan)
    y_upper_tensor = np.full((T, rows, cols), np.nan)
    y_true_tensor  = np.full((T, rows, cols), np.nan)

    # Asumimos X_test tiene columnas ['row','col',t_col] alineadas con y_pred/y_pi/y_true
    # y_pi shape: (n, 2) con [low, up]
    t_index = X_test[t_col].values
    r = X_test["row"].values
    c = X_test["col"].values

    # Mapea cada fila a índice temporal 0..T-1
    _, t_codes = np.unique(t_index, return_inverse=True)

    y_pred_tensor[t_codes, r, c]  = y_pred
    y_lower_tensor[t_codes, r, c] = np.clip(y_pi[:,0], 0, None)
    y_upper_tensor[t_codes, r, c] = y_pi[:,1]
    y_true_tensor[t_codes, r, c]  = y_true.values if hasattr(y_true, "values") else y_true
    return y_pred_tensor, y_lower_tensor, y_upper_tensor, y_true_tensor



def _build_estimator(base_estimator):
    """
    Devuelve un estimador NUEVO (no entrenado) para cada celda.

    base_estimator puede ser:
    - None -> DummyRegressor(mean)
    - callable -> factory que retorna un estimador
    - estimator sklearn -> se clona
    """
    if base_estimator is None:
        return DummyRegressor(strategy="mean")

    if callable(base_estimator):
        est = base_estimator()
        if est is None:
            raise ValueError("base_estimator factory devolvió None.")
        return est

    # Si es una instancia sklearn, clónala
    try:
        return clone(base_estimator)
    except Exception as e:
        raise TypeError(
            "base_estimator debe ser None, un callable (factory) o una instancia sklearn clonable."
        ) from e



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

            X_train_cell = X_train.loc[mask_train].drop(columns=["row", "col"], errors="ignore")
            y_train_cell = y_train.loc[mask_train]

            X_cal_cell   = X_cal.loc[mask_cal].drop(columns=["row", "col"], errors="ignore")
            y_cal_cell = y_cal.loc[mask_cal]

            X_test_cell  = X_test.loc[mask_test].drop(columns=["row", "col"], errors="ignore")
            y_test_cell = y_test.loc[mask_test]
            t_test_cell = X_test.loc[mask_test, "timestep"].values


            # if callable(base_estimator):
            #     model = base_estimator()
            # elif base_estimator is not None:
            #     model = clone(base_estimator)
            # else:
            #     model = DummyRegressor(strategy="mean")

            model = _build_estimator(base_estimator)


            if conformalizer in ["split", "min_max", "GaussianHomos", "ResidualBootstrap","Poisson","MAE"]:
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
    

def apply_conformal_mapie_global(
    X_train, y_train,
    X_cal, y_cal,
    X_test, y_test,
    base_estimator=None,
    alpha=0.1,
    conformalizer="jackknife",
    grid_size=(40, 40),
    method="plus",
    agg_function="mean",
    random_state=None,
):
    """
    Versión GLOBAL: entrena un solo modelo + un solo conformalizador
    usando todos los datos (todas las celdas) y luego reconstruye tensores.
    """
    X_train = pd.DataFrame(X_train).copy()
    X_cal   = pd.DataFrame(X_cal).copy()
    X_test  = pd.DataFrame(X_test).copy()

    # Aseguramos que existan las columnas necesarias
    for col in ["row", "col", "timestep"]:
        if col not in X_train.columns or col not in X_cal.columns or col not in X_test.columns:
            raise ValueError(f"Column '{col}' is required in X_train, X_cal, and X_test")

    # Separar features de identificadores espaciales/temporales
    feat_cols = [c for c in X_train.columns if c not in ["row", "col", "timestep"]]

    X_train_feat = X_train[feat_cols]
    X_cal_feat   = X_cal[feat_cols]
    X_test_feat  = X_test[feat_cols]

    # Definir modelo base
    if callable(base_estimator):
        model = base_estimator()
    elif base_estimator is not None:
        model = clone(base_estimator)
    else:
        model = DummyRegressor(strategy="mean")

    if conformalizer not in conformal_classes:
        raise ValueError(f"Conformalizer '{conformalizer}' not supported. Choose from: {list(conformal_classes.keys())}")

    ConformalClass = conformal_classes[conformalizer]

    # --- Construir y entrenar el conformalizador global ---
    if conformalizer in ["split", "min_max", "GaussianHomos", "ResidualBootstrap", "Poisson", "MAE"]:
        mapie = ConformalClass(estimator=model, confidence_level=1 - alpha, prefit=False)
        mapie.fit(X_train_feat, y_train)
        mapie.conformalize(X_cal_feat, y_cal)

    elif conformalizer in ["jackknife", "cross"]:
        if method not in ["plus", "minmax"]:
            raise ValueError(f"method={method} is not compatible with JackknifeAfterBootstrapRegressor or CrossConformalRegressor.")
        
        X_comb = pd.concat([X_train_feat, X_cal_feat])
        y_comb = pd.concat([y_train, y_cal])

        keys = dict(
            estimator=model,
            confidence_level=1 - alpha,
            method=method,
            n_jobs=-1
        )

        if conformalizer == "cross":
            cv = TimeSeriesSplit(n_splits=10)
            keys["cv"] = cv
        else:
            keys["resampling"] = 50
            keys["random_state"] = random_state
            keys["aggregation_method"] = agg_function

        mapie = ConformalClass(**keys)
        mapie.fit_conformalize(X_comb, y_comb)

    elif conformalizer == "time_serie":
        if method not in ["aci", "enbpi"]:
            raise ValueError(f"method={method} is not compatible with TimeSeriesRegressor.")
        if agg_function not in ["mean", "median"]:
            raise ValueError(f"agg_function={agg_function} is not compatible with TimeSeriesRegressor.")

        X_comb = pd.concat([X_train_feat, X_cal_feat])
        y_comb = pd.concat([y_train, y_cal])

        cv = BlockBootstrap(
            n_resamplings=200,
            length=7,
            overlapping=True,
            random_state=random_state,
        )

        mapie = ConformalClass(
            estimator=model,
            method=method,
            cv=cv,
            agg_function=agg_function,
            n_jobs=-1,
        )
        mapie.fit(X_comb, y_comb)

    # --- Predicción e intervalos sobre TODO X_test ---
    if conformalizer == "time_serie":
        y_pred, y_interval = mapie.predict(
            X_test_feat,
            confidence_level=1 - alpha,
            ensemble=True
        )
    else:
        y_pred, y_interval = mapie.predict_interval(X_test_feat)

    # --- Reconstruir tensores (T, R, C) ---
    y_pred_tensor, y_lower_tensor, y_upper_tensor, y_true_tensor = reconstruct_tensors(
        X_test,
        y_pred,
        y_interval,
        y_test,
        grid_size=grid_size,
        t_col="timestep",
    )

    return y_pred_tensor, y_lower_tensor, y_upper_tensor, y_true_tensor




def apply_conformal_mapie_pooled(
    X_train, y_train, X_cal, y_cal, X_test, y_test,
    base_estimator, conformalizer="split", alpha=0.1,
    method="plus", agg_function="mean", random_state=None,
    grid_size=(40,40)
):
    X_train = pd.DataFrame(X_train); X_cal = pd.DataFrame(X_cal); X_test = pd.DataFrame(X_test)
    # Quita columnas índice, pero CONSÉRVALAS en copias para rearmar
    drop_cols = ["row","col","timestep"]
    Xtr = X_train.drop(columns=drop_cols, errors="ignore")
    Xca = X_cal.drop(columns=drop_cols, errors="ignore")
    Xte = X_test.drop(columns=drop_cols, errors="ignore")

    # Modelo base
    model = base_estimator() if callable(base_estimator) else (clone(base_estimator) if base_estimator is not None else DummyRegressor(strategy="mean"))

    ConformalClass = conformal_classes[conformalizer]
    if conformalizer in ["split","min_max","GaussianHomos","ResidualBootstrap","Poisson","MAE"]:
        mapie = ConformalClass(estimator=model, confidence_level=1-alpha, prefit=False)
        mapie.fit(Xtr, y_train)
        mapie.conformalize(Xca, y_cal)
        y_pred, y_pi = mapie.predict_interval(Xte)

    elif conformalizer in ["jackknife","cross"]:
        if method not in ["plus","minmax"]:
            raise ValueError("method inválido.")
        X_comb = pd.concat([Xtr, Xca], axis=0)
        y_comb = pd.concat([y_train, y_cal], axis=0)
        keys = dict(estimator=model, confidence_level=1-alpha, method=method, n_jobs=-1)
        if conformalizer == "cross":
            keys["cv"] = TimeSeriesSplit(n_splits=10)
        else:
            keys.update(resampling=50, random_state=random_state, aggregation_method=agg_function)
        mapie = ConformalClass(**keys)
        mapie.fit_conformalize(X_comb, y_comb)
        y_pred, y_pi = mapie.predict_interval(Xte)

    elif conformalizer == "time_serie":
        cv = BlockBootstrap(n_resamplings=200, length=7, overlapping=True, random_state=random_state)
        mapie = ConformalClass(estimator=model, method=method, cv=cv,
                               agg_function=agg_function, n_jobs=-1)
        mapie.fit(pd.concat([Xtr,Xca]), pd.concat([y_train,y_cal]))
        y_pred, y_pi = mapie.predict(Xte, confidence_level=1-alpha, ensemble=True)

    return reconstruct_tensors(X_test, y_pred, y_pi, y_test, grid_size)


def mondrian_intervals_from_split(model, X_tr, y_tr, X_ca, y_ca, X_te, groups_ca, groups_te, alpha=0.1):
    # 1) Fit puntual
    model.fit(X_tr, y_tr)
    mu_ca = model.predict(X_ca)
    resid_ca = np.abs(y_ca - mu_ca)

    # 2) Cuantiles por grupo (no conformidad)
    q_dict = {}
    for g, r in pd.Series(resid_ca).groupby(groups_ca):
        q_dict[g] = r.quantile(1 - alpha)

    # 3) Predicción + intervalos por grupo
    mu_te = model.predict(X_te)
    q_te = np.array([q_dict.get(g, np.quantile(resid_ca, 1-alpha)) for g in groups_te])
    lower = np.clip(mu_te - q_te, 0, None)
    upper = mu_te + q_te
    return mu_te, np.c_[lower, upper]
