import numpy as np

def predictions_to_grid(X, y_true, y_pred, grid_size):
    """
    Convert predictions and true values into spatial grids.

    Parameters
    ----------
    X : pd.DataFrame or np.array
        DataFrame or array with ["row", "col"] columns.

    y_true : np.array or pd.Series
        True target values.

    y_pred : np.array or pd.Series
        Predicted values.

    grid_size : tuple
        Tuple (rows, cols) specifying the grid dimensions.

    Returns
    -------
    grid_true, grid_pred : np.array
        Spatial grids for true and predicted counts.
    """
    grid_true = np.zeros(grid_size)
    grid_pred = np.zeros(grid_size)

    for (_, row, col), y_val, pred_val in zip(X.values, y_true, y_pred):
        grid_true[row, col] += y_val
        grid_pred[row, col] += pred_val
        
    return grid_true, grid_pred

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
