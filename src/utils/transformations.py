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


def ensure_full_grid(df, time_steps, grid_size):
    """
    Ensure that all combinations of (timestep, row, col) exist in the dataframe.
    Missing combinations are filled with count = 0.

    Parameters:
    - df: DataFrame with 'timestep', 'row', 'col', and 'count' columns.
    - time_steps: int, number of timesteps.
    - grid_size: tuple (rows, cols)

    Returns:
    - DataFrame with complete grid structure filled with 0s where missing.
    """
    rows, cols = grid_size
    full_index = pd.MultiIndex.from_product(
        [range(time_steps), range(rows), range(cols)],
        names=["timestep", "row", "col"]
    )

    df = df.groupby(["timestep", "row", "col"], as_index=False).agg({"count": "sum"})
    df_full = df.set_index(["timestep", "row", "col"]).reindex(full_index, fill_value=0).reset_index()

    return df_full
