import numpy as np

def compute_misscoverage_per_cell(y_true_grid, y_min_grid, y_max_grid):
    """
    Compute Misscoverage (i.e., proportion of times the true value not lies within the predicted interval)
    from 3D spatial-temporal grids.

    Parameters:
    - y_true_grid: np.ndarray of shape (T, R, C)
    - y_min_grid: np.ndarray of shape (R, C)
    - y_max_grid: np.ndarray of shape (R, C)

    Returns:
    - coverage_grid: np.ndarray of shape (R, C)
    - overall_coverage: float, mean coverage across all cells and timesteps
    """
    n_in_interval = ~((y_true_grid >= y_min_grid) & (y_true_grid <= y_max_grid))
    misscoverage_grid = n_in_interval.mean(axis=0)  # mean over time
    overall_m_coverage = n_in_interval.mean()
    std_m_coverage = n_in_interval.std()
    return misscoverage_grid, overall_m_coverage, std_m_coverage

def compute_interval_width_per_cell(y_min_grid, y_max_grid):
    """
    Compute average interval width for each cell.

    Parameters:
    - y_min_grid: np.ndarray of shape (R, C)
    - y_max_grid: np.ndarray of shape (R, C)

    Returns:
    - width_grid: np.ndarray of shape (R, C)
    - overall_width: float, mean width
    """
    width_grid = y_max_grid - y_min_grid
    overall_width = np.mean(width_grid)
    std_width  = width_grid.std()
    return width_grid, overall_width, std_width
