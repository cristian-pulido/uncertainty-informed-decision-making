import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
from src.utils.spatial_processing import grid3d_to_dataframe_with_index, predictions_to_grid

def compute_misscoverage_per_cell(y_true_grid, y_min_grid, y_max_grid, reference_X):
    """
    Compute a per-sample misscoverage flag (1 if the true value is NOT within the predicted interval).

    Parameters:
    - y_true_grid: np.ndarray of shape (T, R, C), true values in 3D spatial-temporal format
    - y_min_grid: np.ndarray of shape (R, C), lower bound of prediction interval
    - y_max_grid: np.ndarray of shape (R, C), upper bound of prediction interval
    - reference_X: pd.DataFrame with original row, col, timestep, and index structure

    Returns:
    - pd.DataFrame with same index as reference_X and a new column 'not_in_interval' (0 or 1)
    """
    n_in_interval = ~((y_true_grid >= y_min_grid) & (y_true_grid <= y_max_grid))
    df = grid3d_to_dataframe_with_index(n_in_interval, reference_X, "not_in_interval")
    return pd.merge(reference_X, df, left_index=True, right_index=True)


def compute_overall_misscoverge(miss_covergare):
    """
    Aggregate per-sample misscoverage into per-cell and overall metrics.

    Parameters:
    - miss_covergare: pd.DataFrame with columns ['row', 'col', 'not_in_interval']

    Returns:
    - misscoverage_grid: np.ndarray of shape (R, C), average misscoverage per cell
    - overall_m_coverage: float, overall mean misscoverage
    - std_m_coverage: float, overall std deviation of misscoverage
    """
    grid_size = np.array(miss_covergare[["row", "col"]].max()) + 1

    group_time = miss_covergare.groupby(["row", "col"]).agg({"not_in_interval": "mean"}).reset_index()

    misscoverage_grid, _ = predictions_to_grid(
        group_time,
        group_time["not_in_interval"].values,
        group_time["not_in_interval"].values,
        grid_size,
        aggregate=False
    )

    overall_m_coverage = miss_covergare["not_in_interval"].mean()
    std_m_coverage = miss_covergare["not_in_interval"].std()

    return misscoverage_grid, overall_m_coverage, std_m_coverage



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


def compute_spatiotemporal_confidence(df_metrics, lambda_param=0.5):
    """
    Compute a per-cell, per-time confidence score based on interval width and coverage.

    Parameters:
    - df_metrics: pd.DataFrame with columns:
        ["timestep", "row", "col", "Interval Width", "Outside Interval"]
    - lambda_param: float, trade-off between width and coverage

    Returns:
    - df_confidence: pd.DataFrame with added 'Confidence' column
    """

    df = df_metrics.copy()

    # Normalize interval width (relative scale)
    width_min = df["Interval Width"].min()
    width_max = df["Interval Width"].max()
    df["width_norm"] = (df["Interval Width"] - width_min) / (width_max - width_min)

    # Compute confidence score: 1 = full confidence, 0 = no confidence
    df["Confidence"] = 1 - df["Outside Interval"].astype(float) - lambda_param * df["width_norm"]
    df["Confidence"] = df["Confidence"].clip(lower=0.0, upper=1.0)

    return df