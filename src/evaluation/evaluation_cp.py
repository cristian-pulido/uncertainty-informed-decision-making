import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
from src.utils.spatial_processing import grid3d_to_dataframe_with_index, predictions_to_grid

def compute_miscoverage_per_cell(y_true_grid, y_min_grid, y_max_grid, reference_X):
    """
    Compute per-sample interval errors: both a binary miscoverage flag and the distance to the interval bounds
    when the true value lies outside the predicted interval.

    Parameters
    ----------
    y_true_grid : np.ndarray of shape (T, R, C)
        Ground truth values for each timestep and spatial cell.

    y_min_grid : np.ndarray of shape (R, C)
        Lower bound of the prediction interval (constant over time).

    y_max_grid : np.ndarray of shape (R, C)
        Upper bound of the prediction interval (constant over time).

    reference_X : pd.DataFrame
        DataFrame with 'timestep', 'row', and 'col' columns, used to align the resulting DataFrame with the original input.

    Returns
    -------
    pd.DataFrame
        A DataFrame aligned with reference_X that includes:
        - 'not_in_interval' : 1 if the true value lies outside the interval, 0 otherwise.
        - 'distance_to_interval' : distance from the true value to the closest bound when outside the interval (0 if inside).
    """

    n_in_interval = ~((y_true_grid >= y_min_grid) & (y_true_grid <= y_max_grid))
    df1 = grid3d_to_dataframe_with_index(n_in_interval, reference_X, "not_in_interval")

    error = np.zeros_like(n_in_interval).astype(float)

    error[(y_min_grid - y_true_grid) > 0]+=(y_min_grid-y_true_grid)[(y_min_grid - y_true_grid) > 0]
    error[(y_max_grid - y_true_grid) < 0]+=(y_true_grid-y_max_grid)[(y_max_grid - y_true_grid) < 0]
    df2 = grid3d_to_dataframe_with_index(error, reference_X, "distance_to_interval")

    return pd.concat([reference_X,df1,df2],axis=1)


def compute_overall_miscoverage(miscoverage):
    """
    Aggregate per-sample miscoverage and interval error distance into per-cell and overall metrics.

    Parameters
    ----------
    miscoverage : pd.DataFrame
        DataFrame containing at least the following columns:
        - 'row', 'col': spatial coordinates
        - 'not_in_interval': binary flag (1 if outside interval, 0 otherwise)
        - 'distance_to_interval': float, distance to nearest bound (0 if inside)

    Returns
    -------
    miscoverage_grid : np.ndarray
        Grid of average miscoverage per cell.

    overall_m_coverage : float
        Overall mean miscoverage across all cells and timesteps.

    std_m_coverage : float
        Standard deviation of miscoverage across all samples.

    avg_error_outside : float
        Mean distance to interval, considering only samples outside the interval.

    std_error_outside : float
        Standard deviation of the distance to interval for outside samples.
    """
    grid_size = np.array(miscoverage[["row", "col"]].max()) + 1

    # Compute miscoverage grid
    group_time = miscoverage.groupby(["row", "col"]).agg({"not_in_interval": "mean"}).reset_index()
    miscoverage_grid, _ = predictions_to_grid(
        group_time,
        group_time["not_in_interval"].values,
        group_time["not_in_interval"].values,
        grid_size,
        aggregate=False
    )

    # Overall miscoverage
    overall_m_coverage = miscoverage["not_in_interval"].mean()
    std_m_coverage = miscoverage["not_in_interval"].std()

    # Distance to interval (only when outside)
    outside = miscoverage[miscoverage["not_in_interval"] == 1]
    avg_error_outside = outside["distance_to_interval"].mean()
    std_error_outside = outside["distance_to_interval"].std()

    return (
        miscoverage_grid,
        overall_m_coverage,
        std_m_coverage,
        avg_error_outside,
        std_error_outside
    )


def compute_interval_width_per_sample(y_min_grid, y_max_grid, reference_X):
    """
    Compute interval width per sample from temporal 3D prediction grid and map it to the reference index.

    Parameters:
    - y_min_grid: np.ndarray of shape (T, R, C), lower bounds
    - y_max_grid: np.ndarray of shape (T, R, C), upper bounds
    - reference_X: pd.DataFrame with 'timestep', 'row', 'col', used to align the results

    Returns:
    - pd.DataFrame with 'interval_width' column and same index as reference_X
    """
    width_grid = y_max_grid - y_min_grid
    df = grid3d_to_dataframe_with_index(width_grid, reference_X, "interval_width")
    return pd.merge(reference_X, df, left_index=True, right_index=True)


def compute_overall_interval_width(interval_df):
    """
    Aggregate interval width per cell and compute overall statistics.

    Parameters:
    - interval_df: pd.DataFrame with ['row', 'col', 'interval_width']

    Returns:
    - width_grid: np.ndarray of shape (R, C), average width per cell
    - overall_width: float, global mean of interval width
    - std_width: float, global standard deviation of interval width
    """
    grid_size = np.array(interval_df[["row", "col"]].max()) + 1

    grouped = interval_df.groupby(["row", "col"]).agg({"interval_width": "mean"}).reset_index()

    width_grid, _ = predictions_to_grid(
        grouped,
        grouped["interval_width"].values,
        grouped["interval_width"].values,
        grid_size,
        aggregate=False
    )

    overall_width = interval_df["interval_width"].mean()
    std_width = interval_df["interval_width"].std()

    return width_grid, overall_width, std_width


def compute_spatiotemporal_confidence(df_metrics):
    """
    Compute a per-cell, per-time confidence score based only on the predicted interval width.

    This function is suitable for real-world deployment, where ground truth values are not available.

    Parameters:
    - df_metrics: pd.DataFrame with columns:
        ["timestep", "row", "col", "Interval Width"]

    Returns:
    - df_confidence: pd.DataFrame with added 'Confidence' column
        Confidence is defined as 1 - normalized_interval_width ∈ [0, 1]
    """

    df = df_metrics.copy()

    # Normalize interval width
    width_min = df["Interval Width"].min()
    width_max = df["Interval Width"].max()
    df["width_norm"] = (df["Interval Width"] - width_min) / (width_max - width_min + 1e-8)

    # Confidence purely from interval width
    df["Confidence"] = 1 - df["width_norm"]
    df["Confidence"] = df["Confidence"].clip(lower=0.0, upper=1.0)

    return df