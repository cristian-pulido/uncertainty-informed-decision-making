import numpy as np
import pandas as pd
from src.utils.spatial_processing import define_hotspot_by_crimes
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def classify_hotspots(grid_true, grid_pred, hotspot_percentage):
    """
    Classify cells (2D or 3D) based on hotspot alignment between true and predicted values.

    Parameters
    ----------
    grid_true : np.ndarray
        Ground truth crime counts. Either 3D (T, R, C) or 2D (R, C).
    grid_pred : np.ndarray
        Predicted crime counts or scores. Must be the same shape as grid_true.
    hotspot_percentage : float
        Percentage (0-1) used to define top cells as hotspots.

    Returns
    -------
    cell_type : np.ndarray
        Array of strings ("Both", "GT-only", "Pred-only", "Neither") with same shape as input.
    """
    assert grid_true.shape == grid_pred.shape, "grid_true and grid_pred must have the same shape"

    if grid_true.ndim == 2:
        # Single timestep case
        rows, cols = grid_true.shape
        pred_mask = define_hotspot_by_crimes(grid_pred, hotspot_percentage)
        true_mask = define_hotspot_by_crimes(grid_true, hotspot_percentage)

        cell_type = np.full((rows, cols), "Neither", dtype=object)
        cell_type[true_mask & pred_mask] = "Both"
        cell_type[true_mask & ~pred_mask] = "GT-only"
        cell_type[~true_mask & pred_mask] = "Pred-only"

        return cell_type

    elif grid_true.ndim == 3:
        # Multiple timesteps case
        timesteps, rows, cols = grid_true.shape
        cell_type = np.full((timesteps, rows, cols), "Neither", dtype=object)

        for t in range(timesteps):
            pred_mask = define_hotspot_by_crimes(grid_pred[t], hotspot_percentage)
            true_mask = define_hotspot_by_crimes(grid_true[t], hotspot_percentage)

            cell_type[t][true_mask & pred_mask] = "Both"
            cell_type[t][true_mask & ~pred_mask] = "GT-only"
            cell_type[t][~true_mask & pred_mask] = "Pred-only"

        return cell_type

    else:
        raise ValueError("Input arrays must be 2D or 3D (T, R, C)")


def hotspot_priority(
    conf_grid,
    base_grid,
    base_type="frequency",  # {"frequency", "binary", "continuous"}
    freq_thresh=0.25,
    conf_thresh=0.75
):
    """
    Classify grid cells based on confidence and either frequency, binary prediction, or predicted crime magnitude.

    Parameters:
    - conf_grid: np.ndarray (R x C), confidence per cell in [0,1]
    - base_grid: np.ndarray (R x C)
        - If base_type="frequency", this is a normalized frequency [0,1]
        - If base_type="binary", this is a boolean hotspot prediction mask
        - If base_type="continuous", this is raw predicted crime counts (will be normalized)
    - base_type: str, one of {"frequency", "binary", "continuous"}
    - freq_thresh: float, threshold to consider high frequency / high prediction (default 0.25)
    - conf_thresh: float, threshold to consider high confidence (default 0.75)

    Returns:
    - category_grid: np.ndarray (R x C), integer code:
        0: Priority
        1: Critical
        2: Under Surveillance
        3: Low Interest
    - legend: dict of int → label
    """

    if base_type == "frequency":
        high_risk = base_grid >= freq_thresh

    elif base_type == "binary":
        high_risk = base_grid.astype(bool)

    elif base_type == "continuous":
        norm_pred = (base_grid - base_grid.min()) / (base_grid.max() - base_grid.min())
        high_risk = norm_pred >= freq_thresh

    else:
        raise ValueError("Invalid base_type. Choose from 'frequency', 'binary', or 'continuous'.")

    high_conf = conf_grid >= conf_thresh

    category_grid = np.zeros_like(conf_grid, dtype=int)
    category_grid[np.where(high_risk & high_conf)] = 0  # Priority
    category_grid[np.where(high_risk & ~high_conf)] = 1  # Critical
    category_grid[np.where(~high_risk & ~high_conf)] = 2  # Under Surveillance
    category_grid[np.where(~high_risk & high_conf)] = 3  # Low Interest

    legend = {
        0: "Priority (high crime, high confidence)",
        1: "Critical (high crime, low confidence)",
        2: "Under Surveillance (low crime, low confidence)",
        3: "Low Interest (low crime, high confidence)"
    }

    return category_grid, legend



def plot_hotspot_priority_map(category_grid, legend, title="Hotspot Priority Map"):
    """
    Plot the categorized hotspot priority map.

    Parameters:
    - category_grid: 2D array with category codes (output of classify_cells)
    - legend: dict mapping category codes to descriptions
    - title: str, plot title
    """
    # Define category colors: from high to low priority
    colors = ["red", "orange", "yellow", "green"]
    cmap = mcolors.ListedColormap(colors)
    bounds = [0, 1, 2, 3, 4]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(category_grid, cmap=cmap, norm=norm, origin="lower")

    # Custom colorbar with labels from the legend
    cbar = plt.colorbar(im, ax=ax, ticks=[0.5, 1.5, 2.5, 3.5], shrink=0.5)
    cbar.ax.set_yticklabels([legend[i] for i in range(4)])
    cbar.ax.invert_yaxis()  # Optional: reverse to show priority at top

    ax.set_title(title)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    plt.tight_layout()
    plt.show()
