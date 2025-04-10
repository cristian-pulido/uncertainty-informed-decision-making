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


def hotspot_priority(freq_grid, conf_grid, freq_thresh=0.25, conf_thresh=0.75):
    """
    Classify grid cells based on hotspot frequency and prediction confidence.

    Parameters:
    - freq_grid: 2D array with normalized hotspot frequency per cell (0 to 1)
    - conf_grid: 2D array with confidence values per cell (0 to 1)
    - freq_thresh: threshold for high hotspot frequency (default=0.25)
    - conf_thresh: threshold for high confidence (default=0.75)

    Returns:
    - category_grid: 2D array with integer codes representing the cell category
    - legend: dictionary mapping code -> human-readable description
    """
    category_grid = np.zeros_like(freq_grid)

    high_freq = freq_grid >= freq_thresh
    high_conf = conf_grid >= conf_thresh

    category_grid[np.where(high_freq & high_conf)] = 0  # Priority
    category_grid[np.where(high_freq & ~high_conf)] = 1  # Critical
    category_grid[np.where(~high_freq & ~high_conf)] = 2  # Under Surveillance
    category_grid[np.where(~high_freq & high_conf)] = 3  # Low Interest

    legend = {
        0: "Priority (frequent, high confidence)",
        1: "Critical (frequent, low confidence)",
        2: "Under Surveillance (infrequent, low confidence)",
        3: "Low Interest (infrequent, high confidence)"
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
