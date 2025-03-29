import numpy as np

def compute_coverage_per_cell(y_true, y_intervals, X_test, grid_size):
    """
    Compute the coverage (percentage of times the true value falls within the interval)
    for each spatial cell.

    Parameters:
    - y_true: array of true values (n_samples,)
    - y_intervals: array of shape (n_samples, 2) or (n_samples, n_alphas, 2)
    - X_test: DataFrame with 'row' and 'col' columns (same order as y_true)
    - grid_size: tuple (rows, cols)

    Returns:
    - coverage_grid: array (rows, cols) with coverage values [0, 1]
    - overall_coverage: mean coverage across all cells
    """
    coverage_count = np.zeros(grid_size)
    total_count = np.zeros(grid_size)

    y_lower = y_intervals[:, 0] if y_intervals.ndim == 2 else y_intervals[:, 0, 0]
    y_upper = y_intervals[:, 1] if y_intervals.ndim == 2 else y_intervals[:, 0, 1]

    in_interval = (y_true >= y_lower) & (y_true <= y_upper)

    for (r, c, covered) in zip(X_test["row"], X_test["col"], in_interval):
        total_count[r, c] += 1
        coverage_count[r, c] += int(covered)

    with np.errstate(divide='ignore', invalid='ignore'):
        coverage_grid = np.true_divide(coverage_count, total_count)
        coverage_grid[np.isnan(coverage_grid)] = 0.0

    overall_coverage = coverage_grid.mean()
    return coverage_grid, overall_coverage

def compute_interval_width_per_cell(y_intervals, X_test, grid_size):
    """
    Compute the average prediction interval width per spatial cell.

    Parameters:
    - y_intervals: array of shape (n_samples, 2) or (n_samples, n_alphas, 2)
    - X_test: DataFrame with 'row' and 'col'
    - grid_size: tuple (rows, cols)

    Returns:
    - width_grid: array (rows, cols)
    - overall_width: mean width across all cells
    """
    width = y_intervals[:, 1] - y_intervals[:, 0] if y_intervals.ndim == 2 else y_intervals[:, 0, 1] - y_intervals[:, 0, 0]

    width_sum = np.zeros(grid_size)
    count = np.zeros(grid_size)

    for (r, c, w) in zip(X_test["row"], X_test["col"], width):
        width_sum[r, c] += w
        count[r, c] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        width_grid = np.true_divide(width_sum, count)
        width_grid[np.isnan(width_grid)] = 0.0

    overall_width = width_grid.mean()
    return width_grid, overall_width
