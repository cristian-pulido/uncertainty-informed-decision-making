# src/evaluation/temporal_evaluation.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from src.evaluation.spatial_metrics import pai, pei, pei_star
from src.utils.spatial_processing import predictions_to_grid, define_hotspot_by_crimes

def evaluate_temporal_rmse_mae(X_test, y_test, y_pred):
    df = pd.DataFrame(X_test).copy()
    df["y_true"] = y_test
    df["y_pred"] = y_pred

    results = []
    for t, group in df.groupby("timestep"):
        rmse = root_mean_squared_error(group["y_true"], group["y_pred"])
        mae = mean_absolute_error(group["y_true"], group["y_pred"])
        results.append((t, rmse, mae))

    df_result = pd.DataFrame(results, columns=["timestep", "rmse", "mae"])
    return {
        "rmse_mean": df_result["rmse"].mean(),
        "rmse_std": df_result["rmse"].std(),
        "mae_mean": df_result["mae"].mean(),
        "mae_std": df_result["mae"].std(),
        "per_timestep": df_result
    }

def evaluate_temporal_spatial_metrics(
    X_test, 
    y_test, 
    y_pred, 
    grid_size, 
    hotspot_percentage,
    hotspot_masks_pred=None,
):
    """
    Evaluate spatial-temporal metrics (PAI, PEI, PEI*) optionally using provided hotspot masks.

    Parameters
    ----------
    X_test : pd.DataFrame
        DataFrame containing "timestep", "row", "col" columns.
    y_test : array-like
        True values.
    y_pred : array-like
        Predicted values.
    grid_size : tuple
        Tuple of grid dimensions (rows, cols).
    hotspot_percentage : float
        Percentage of crimes to define hotspots.
    hotspot_masks_pred : list of np.ndarray (bool), optional
        Precomputed hotspot masks for each timestep. If None, hotspots are computed internally.

    Returns
    -------
    dict
        Dictionary with mean and std of PAI, PEI, PEI*, and results per timestep.
    """
    df = pd.DataFrame(X_test).copy()
    df["y_true"] = y_test
    df["y_pred"] = y_pred

    results = []
    timesteps = sorted(df["timestep"].unique())

    

    for idx, t in enumerate(timesteps):
        group = df[df["timestep"] == t]
        grid_true, grid_pred = predictions_to_grid(
            group[["row", "col"]],
            group["y_true"],
            group["y_pred"],
            grid_size,
            aggregate=False,
        )

        # If hotspot masks are provided, use them directly
        if hotspot_masks_pred is not None:
            mask_pred = hotspot_masks_pred
        else:
            mask_pred = define_hotspot_by_crimes(grid_pred, hotspot_percentage)

       

        # Optimal hotspot from true data always computed internally
        mask_opt = define_hotspot_by_crimes(grid_true, hotspot_percentage)

        results.append({
            "timestep": t,
            "pai": pai(grid_true, mask_pred),
            "pei": pei(grid_true, mask_pred, mask_opt),
            "pei_star": pei_star(grid_true, mask_pred)
        })

    df_result = pd.DataFrame(results)
    return {
        "pai_mean": df_result["pai"].mean(),
        "pai_std": df_result["pai"].std(),
        "pei_mean": df_result["pei"].mean(),
        "pei_std": df_result["pei"].std(),
        "pei_star_mean": df_result["pei_star"].mean(),
        "pei_star_std": df_result["pei_star"].std(),
        "per_timestep": df_result
    }


def evaluate_spatial_metrics_over_coverages(X_test, y_test, y_pred, grid_size, coverages):
    results = []
    for cov in coverages:
        spatial_scores = evaluate_temporal_spatial_metrics(X_test, y_test, y_pred, grid_size, cov)
        results.append({
            "coverage": cov,
            "pai_mean": spatial_scores["pai_mean"],
            "pai_std": spatial_scores["pai_std"],
            "pei_mean": spatial_scores["pei_mean"],
            "pei_std": spatial_scores["pei_std"],
            "pei_star_mean": spatial_scores["pei_star_mean"],
            "pei_star_std": spatial_scores["pei_star_std"]
        })

    return pd.DataFrame(results)
