import numpy as np
import pandas as pd

def aggregate_by_cell(X, y, grid_size, aggfunc="sum"):
    """
    Aggregate values (e.g., counts or predictions) into a 2D spatial grid.

    Parameters:
    - X: DataFrame with 'row' and 'col' columns
    - y: Series or array of values to aggregate
    - grid_size: tuple (rows, cols)
    - aggfunc: aggregation function ("sum", "mean", etc.)

    Returns:
    - grid: np.ndarray with shape (rows, cols)
    """
    df = X.copy()
    df["count"] = y
    grouped = df.groupby(["row", "col"])["count"].agg(aggfunc)
    grid = np.zeros(grid_size)
    for (r, c), val in grouped.items():
        grid[r, c] = val
    return grid
