import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.ticker import PercentFormatter
import pathlib


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



def plot_prediction_interval_map(pred, lower, upper, title="Prediction with Interval", cmap="YlOrRd", vmin=None, vmax=None,titles = ["Prediction", "Lower Bound", "Upper Bound"]):
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

    data = [pred, lower, upper]

    for ax, arr, t in zip(axes, data, titles):
        sns.heatmap(arr, cmap=cmap, cbar=True, ax=ax, vmin=vmin, vmax=vmax, square=True, linewidths=0, linecolor="gray")
        ax.set_title(t)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_grid_map(grid, title="Grid Map", cmap="viridis", label="Value", vmin=None, vmax=None, figsize=(6, 5)):
    """
    Visualize a single 2D grid (e.g., coverage or interval width).

    Parameters:
    - grid: 2D array (rows x cols)
    - title: str
    - cmap: str
    - label: str, label for the colorbar
    - vmin, vmax: color limits
    - figsize: tuple
    """
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.imshow(grid, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    fig.colorbar(cax, ax=ax, label=label)
    plt.show()


################################################################################


def plot_geospatial_data_maps(
    gdf_base,
    df_list,
    columns=None,
    titles=None,
    cmap="YlOrRd",
    share_colorbar=True,
    suptitle=None,
    figsize=(12, 5),
    edgecolor="black",
    vmin=None,
    vmax=None,
    colorbar_labels=None,
    percent_format=None,
    save_path=None 
):
    """
    Plot multiple prediction maps side by side with optional shared or individual colorbars.

    Parameters
    ----------
    gdf_base : GeoDataFrame
        Base geometry to merge with each df
    df_list : list of DataFrames
        Each must contain a 'beat' column and the data to plot
    columns : list of str
        Column names to visualize (one per df)
    titles : list of str
        Titles for each subplot
    cmap : str
        Matplotlib colormap
    share_colorbar : bool
        Whether to use a single shared colorbar
    suptitle : str
        Title for the entire figure
    figsize : tuple
        Figure size
    edgecolor : str
        Color of borders
    vmin, vmax : float or list of float
        Global or per-map color scale min/max
    colorbar_labels : list of str
        Custom labels for the colorbars
    percent_format : list of bool
        Whether to use percent format per map (or single bool if shared)
    save_path : str
        If given, saves figure to this path
    """

    n = len(df_list)
    if columns is None or len(columns) != n:
        raise ValueError("Must provide 'columns': one per DataFrame.")

    if percent_format is None:
        percent_format = [False] * n
    elif isinstance(percent_format, bool):
        percent_format = [percent_format] * n

    if not share_colorbar and (vmin is None or isinstance(vmin, (int, float))):
        vmin = [vmin] * n
    if not share_colorbar and (vmax is None or isinstance(vmax, (int, float))):
        vmax = [vmax] * n

    fig, axes = plt.subplots(1, n, figsize=figsize, constrained_layout=True)
    if n == 1:
        axes = [axes]

    # Shared normalization
    norm = None
    if share_colorbar:
        global_min = min(df[col].min() for df, col in zip(df_list, columns)) if vmin is None else vmin
        global_max = max(df[col].max() for df, col in zip(df_list, columns)) if vmax is None else vmax
        norm = mcolors.Normalize(vmin=global_min, vmax=global_max)

    for i, (df, col, ax) in enumerate(zip(df_list, columns, axes)):
        merged = gdf_base.merge(df, how="left", left_on="beat_num", right_on="beat")

        if share_colorbar:
            merged.plot(column=col, cmap=cmap, ax=ax, edgecolor=edgecolor, legend=False, norm=norm)
        else:
            this_vmin = vmin[i] if isinstance(vmin, list) else None
            this_vmax = vmax[i] if isinstance(vmax, list) else None
            fmt = PercentFormatter(decimals=0) if percent_format[i] else None
            merged.plot(
                column=col,
                cmap=cmap,
                ax=ax,
                edgecolor=edgecolor,
                legend=True,
                vmin=this_vmin,
                vmax=this_vmax,
                legend_kwds={
                    "label": colorbar_labels[i] if colorbar_labels else col,
                    "format": fmt
                }
            )

        ax.set_title(titles[i] if titles else f"Map {i+1}")
        ax.set_xticks([])
        ax.set_yticks([])

    # Shared colorbar
    if share_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []
        cbar = fig.colorbar(sm, ax=axes, shrink=0.8)
        cbar.set_label(colorbar_labels[0] if colorbar_labels else columns[0])
        if percent_format[0]:
            cbar.ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))

    if suptitle:
        plt.suptitle(suptitle, fontsize=14)

    if save_path:
        format_ = pathlib.Path(save_path).suffix[1:]
        fig.savefig(save_path, format=format_, bbox_inches="tight", dpi=300)

    plt.show()