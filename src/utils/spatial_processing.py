import numpy as np
import pandas as pd

def predictions_to_grid(X, y_true, y_pred, grid_size, aggregate=True):
    """
    Convert predictions and true values into spatial grids.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame with 'row', 'col' (and optionally 'timestep') columns.
    y_true : np.array
    y_pred : np.array
    grid_size : tuple
        (rows, cols)
    aggregate : bool
        If True, sum across time. If False and X has 'timestep', return 3D grid [time, row, col].

    Returns
    -------
    grid_true, grid_pred : np.array
    """
    rows, cols = grid_size
    has_time = "timestep" in X.columns

    if not has_time:
        # No tiempo, crear 2D directamente
        grid_true = np.zeros((rows, cols))
        grid_pred = np.zeros((rows, cols))
        for (row, col), yt, yp in zip(X[["row", "col"]].values, y_true, y_pred):
            grid_true[int(row), int(col)] += yt
            grid_pred[int(row), int(col)] += yp
        return grid_true, grid_pred

    # Si hay columna 'timestep'
    timesteps = X["timestep"].nunique()

    if aggregate:
        grid_true = np.zeros((rows, cols))
        grid_pred = np.zeros((rows, cols))
        for (_, row, col), yt, yp in zip(X[["row", "col"]].values, y_true, y_pred):
            grid_true[int(row), int(col)] += yt
            grid_pred[int(row), int(col)] += yp
    else:
        grid_true = np.zeros((timesteps, rows, cols))
        grid_pred = np.zeros((timesteps, rows, cols))
        # Reindex timesteps
        timestep_map = {t: i for i, t in enumerate(sorted(X["timestep"].unique()))}
        X_local = X.copy()
        X_local["timestep_idx"] = X["timestep"].map(timestep_map)

        for (t, row, col), yt, yp in zip(X_local[["timestep_idx", "row", "col"]].values, y_true, y_pred):
            grid_true[int(t), int(row), int(col)] = yt
            grid_pred[int(t), int(row), int(col)] = yp

    return grid_true, grid_pred



def grid3d_to_dataframe_with_index(grid, reference_X,name_colum):
    """
    Convierte un grid 3D a un DataFrame ordenado según reference_X.

    Parameters:
    - grid: np.ndarray (timesteps, rows, cols)
    - reference_X: pd.DataFrame con columnas ['timestep', 'row', 'col']

    Returns:
    - pd.Series alineada con reference_X
    """
    # Reindexar los timesteps para que coincidan con el orden del grid
    unique_timesteps = sorted(reference_X["timestep"].unique())
    timestep_map = {t: i for i, t in enumerate(unique_timesteps)}

    values = []
    for t, r, c in reference_X[["timestep", "row", "col"]].values:
        t_idx = timestep_map[t]
        values.append(grid[int(t_idx), int(r), int(c)])
    return pd.DataFrame(values, index=reference_X.index,columns=[name_colum])




def define_hotspot_by_cells(grid, hotspot_percentage):
    """
    Create binary mask of top hotspot cells based on predictions or true values.

    Parameters
    ----------
    grid : np.array
        Grid of predicted or actual counts.

    hotspot_percentage : float
        Fraction (between 0 and 1) defining hotspot area.

    Returns
    -------
    hotspot_mask : np.array (bool)
        Binary mask indicating hotspot cells.
    """
    total_cells = grid.size
    hotspot_size = int(total_cells * hotspot_percentage)
    flat_indices = np.argsort(grid.flatten())[::-1][:hotspot_size]
    mask = np.zeros_like(grid, dtype=bool)
    mask[np.unravel_index(flat_indices, grid.shape)] = True
    return mask


def define_hotspot_by_crimes(grid, crime_coverage_target=0.1):
    """
    Generate a binary hotspot mask that selects the minimum number of cells
    whose predicted values cover at least a given percentage of total predicted crimes.

    Parameters
    ----------
    grid : np.ndarray
        2D array representing predicted (or actual) crime intensity per spatial unit.

    crime_coverage_target : float
        Proportion (between 0 and 1) of total crimes that the hotspot area should capture.
        For example, 0.1 means the hotspot should include cells covering at least 10% of predicted crimes.

    Returns
    -------
    hotspot_mask : np.ndarray (bool)
        Binary mask (same shape as input grid) with True in hotspot cells
        that together sum to at least the specified percentage of total crimes.
    
    Notes
    -----
    - If multiple cells have the same predicted value around the threshold,
      the selection is based on order of appearance in the flattened array.
    - The function assumes non-negative values in the grid.
    """
    total_crimes = grid.sum()
    if total_crimes == 0:
        return np.zeros_like(grid, dtype=bool)

    # Flatten and sort grid values in descending order
    sorted_indices = np.argsort(grid.flatten())[::-1]
    sorted_values = grid.flatten()[sorted_indices]
    cumulative_sum = np.cumsum(sorted_values)

    # Determine how many top cells are needed to reach the desired crime coverage
    threshold = total_crimes * crime_coverage_target
    num_cells = np.searchsorted(cumulative_sum, threshold) + 1

    # Create binary hotspot mask
    hotspot_mask = np.zeros_like(grid, dtype=bool)
    hotspot_coords = np.unravel_index(sorted_indices[:num_cells], grid.shape)
    hotspot_mask[hotspot_coords] = True

    return hotspot_mask
