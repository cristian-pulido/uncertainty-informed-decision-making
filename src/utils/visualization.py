import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def compare_prediction_maps(grids, titles, vmin=None, vmax=None, figsize=(14, 4)):
    fig, axes = plt.subplots(1, len(grids), figsize=figsize)
    for ax, grid, title in zip(axes, grids, titles):
        sns.heatmap(grid, cmap="YlGnBu", ax=ax, vmin=vmin, vmax=vmax, cbar=True)
        ax.set_title(title)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def compare_hotspot_masks(masks, titles, ncols=3, figsize=(12, 4)):
    n = len(masks)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n > 1 else [axes]

    for i in range(n):
        sns.heatmap(masks[i].astype(int), cmap="YlOrRd", linewidths=0.5, linecolor='gray', cbar=False, ax=axes[i])
        axes[i].set_title(titles[i])
        axes[i].invert_yaxis()
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def plot_hotspot_masks_over_days(mask_dict, ncols=4, figsize=(14, 8)):
    """
    Visualize multiple hotspot masks by day.
    
    Parameters:
    - mask_dict: dict {day_label: np.ndarray mask}
    - ncols: int, number of columns in the subplot grid
    - figsize: tuple, figure size
    """
    titles = list(mask_dict.keys())
    masks = list(mask_dict.values())
    n = len(masks)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n > 1 else [axes]

    for i in range(n):
        sns.heatmap(masks[i].astype(int), cmap="YlOrRd", linewidths=0.5, linecolor='gray', cbar=False, ax=axes[i])
        axes[i].set_title(titles[i])
        axes[i].invert_yaxis()
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()



def plot_prediction_interval_map(pred, lower, upper, title="Prediction with Interval", cmap="YlOrRd", vmin=None, vmax=None):
    """
    Plot prediction, lower, and upper interval maps side by side.

    Parameters:
    - pred: 2D array, prediction mean
    - lower: 2D array, lower bound
    - upper: 2D array, upper bound
    - title: str, global title for the figure
    - cmap: colormap
    - vmin, vmax: optional color scale bounds
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    vmin = vmin if vmin is not None else min(pred.min(), lower.min(), upper.min())
    vmax = vmax if vmax is not None else max(pred.max(), lower.max(), upper.max())

    titles = ["Prediction", "Lower Bound", "Upper Bound"]
    data = [pred, lower, upper]

    for ax, arr, t in zip(axes, data, titles):
        sns.heatmap(arr, cmap=cmap, cbar=True, ax=ax, vmin=vmin, vmax=vmax, square=True, linewidths=0.3, linecolor="gray")
        ax.set_title(t)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()