import numpy as np
import pandas as pd
from src.utils.spatial_processing import define_hotspot_by_crimes
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def classify_temporal_hotspots(grid_true, grid_pred, hotspot_percentage):
    """
    Classify cells over time based on true and predicted hotspot alignment.

    Parameters:
    - grid_true: np.ndarray (timesteps, rows, cols), ground truth crime counts
    - grid_pred: np.ndarray (rows, cols), predicted hotspot mask (aggregated or static)
    - hotspot_percentage: float, percentage used to define hotspots

    Returns:
    - df_metrics: pd.DataFrame with columns:
        timestep, row, col, misscoverage, interval_width, cell_type
    """
    timesteps, rows, cols = grid_true.shape
    cell_type_sequence = np.full((timesteps, rows, cols), "Neither", dtype=object)

    pred_mask = define_hotspot_by_crimes(grid_pred, hotspot_percentage)

    for t in range(timesteps):
        true_mask = define_hotspot_by_crimes(grid_true[t], hotspot_percentage)

        both = true_mask & pred_mask
        gt_only = true_mask & (~pred_mask)
        pred_only = (~true_mask) & pred_mask

        cell_type_sequence[t][both] = "Both"
        cell_type_sequence[t][gt_only] = "GT-only"
        cell_type_sequence[t][pred_only] = "Pred-only"

    df_metrics = pd.DataFrame({
        "timestep": np.repeat(np.arange(timesteps), rows * cols),
        "row": np.tile(np.repeat(np.arange(rows), cols), timesteps),
        "col": np.tile(np.arange(cols), timesteps * rows),
        "cell_type": cell_type_sequence.flatten()
    })

    return df_metrics


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
        0: "Priority (high risk, high confidence)",
        1: "Critical (high risk, low confidence)",
        2: "Under Surveillance (low risk, low confidence)",
        3: "Low Interest (low risk, high confidence)"
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
