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


def define_hotspot_by_crimes(grid, crime_coverage_target=0.1, priority_grid=None):
    """
    Generate a binary hotspot mask selecting cells that cover at least a target percentage
    of total crimes. Optionally prioritize selection based on a priority grid.

    Parameters
    ----------
    grid : np.ndarray
        2D array representing predicted (or actual) crime intensity per spatial unit.

    crime_coverage_target : float
        Proportion (between 0 and 1) of total crimes that the hotspot area should capture.

    priority_grid : np.ndarray, optional
        2D array (same shape) of integers representing cell priority (lower = higher priority).
        If provided, cells are first sorted by priority, then by predicted crime within each priority.

    Returns
    -------
    hotspot_mask : np.ndarray of bool
        Binary mask with True in hotspot cells covering the target percentage of crimes.
    """
    total_crimes = grid.sum()
    if total_crimes == 0:
        return np.zeros_like(grid, dtype=bool)

    flat_grid = grid.flatten()

    if priority_grid is not None:
        flat_priority = priority_grid.flatten()
        dtype = [("priority", int), ("crime", float)]
        structured = np.array(list(zip(flat_priority, -flat_grid)), dtype=dtype)
        sorted_indices = np.argsort(structured, order=["priority", "crime"])
    else:
        sorted_indices = np.argsort(flat_grid)[::-1]  # Descending

    sorted_values = flat_grid[sorted_indices]
    cumulative_sum = np.cumsum(sorted_values)
    threshold = total_crimes * crime_coverage_target
    num_cells = np.searchsorted(cumulative_sum, threshold) + 1

    hotspot_mask = np.zeros_like(flat_grid, dtype=bool)
    hotspot_mask[sorted_indices[:num_cells]] = True

    return hotspot_mask.reshape(grid.shape)


def grid_to_dataframe(grid, beat_to_coord_map, value_name="value"):
    """
    Convert a 2D or 3D grid to a DataFrame with (timestep), row, col and beat identifiers.

    Parameters:
    - grid: np.ndarray, either (rows, cols) or (timesteps, rows, cols)
    - beat_to_coord_map: dict mapping (row, col) -> beat_id
    - value_name: str, name of the value column

    Returns:
    - df: pd.DataFrame with columns ['timestep' (if 3D), 'row', 'col', 'beat', value_name]
    """
    if grid.ndim == 2:
        rows, cols = grid.shape
        timestep_dim = False
        coords = [(r, c) for r in range(rows) for c in range(cols)]
        data = [grid[r, c] for r, c in coords]
    elif grid.ndim == 3:
        timesteps, rows, cols = grid.shape
        timestep_dim = True
        coords = [(t, r, c) for t in range(timesteps) for r in range(rows) for c in range(cols)]
        data = [grid[t, r, c] for t, r, c in coords]
    else:
        raise ValueError("Grid must be 2D or 3D.")

    # Build DataFrame
    if timestep_dim:
        df = pd.DataFrame(coords, columns=["timestep", "row", "col"])
    else:
        df = pd.DataFrame(coords, columns=["row", "col"])

    df[value_name] = data

    # Add beat column from mapping
    df["beat"] = df[["row", "col"]].apply(lambda x: beat_to_coord_map.get((x[0], x[1]), None), axis=1)
    
    return df