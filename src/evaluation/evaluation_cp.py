import numpy as np
import pandas as pd
from src.evaluation.hotspot_classification import classify_hotspots

def compute_in_interval(y_min, y_max, y_true):
    """
    Returns a boolean array indicating whether y_true is within [y_min, y_max].
    """
    return (y_min <= y_true) & (y_max >= y_true)

def compute_mean_width(y_min, y_max):
    """
    Computes the width of prediction intervals.
    """
    return y_max - y_min

def compute_error_outside_interval(y_min, y_max, y_true):
    """
    Computes the error magnitude for predictions that fall outside the interval.
    Returns np.nan for values inside the interval.
    """
    error = np.full_like(y_true, np.nan)
    lower_miss = (y_min - y_true) > 0
    upper_miss = (y_max - y_true) < 0
    error[lower_miss] = (y_min - y_true)[lower_miss]
    error[upper_miss] = (y_true - y_max)[upper_miss]
    return error

def compute_mwi(widths, error, alpha=0.1):
    """
    Compute the Mean Winkler Interval Score (MWI) from flattened arrays.

    This metric combines the average interval width with a penalty term 
    proportional to the amount of error outside the prediction interval.

    Parameters
    ----------
    widths : np.ndarray
        1D array of interval widths.
    error : np.ndarray
        1D array of errors for values outside the prediction interval.
        Should be 0 when the value is within the interval.
    alpha : float
        Significance level (e.g., 0.1 for 90% prediction intervals).

    Returns
    -------
    mwi : float
        The Mean Winkler Interval Score.
    """
    widths = np.asarray(widths).flatten()
    error = np.asarray(error).flatten()

    if widths.shape != error.shape:
        raise ValueError("Widths and error arrays must have the same shape.")
    
    per_instance = widths + 2 / alpha * np.nan_to_num(error)

    return np.nanmean(per_instance)


def compute_cwc(widths, in_interval, alpha=0.1, eta=8, ref=10.0):
    """
    Compute the Coverage Width-based Criterion (CWC) from flattened arrays.

    Parameters¶
    ----------
    widths : np.ndarray
        1D array of interval widths.
    in_interval : np.ndarray
        1D boolean array indicating whether the true value falls within the interval.
    alpha : float
        Desired error level (e.g., 0.1 for 90% nominal coverage).
    eta : float
        Penalization parameter for coverage deviation.
    ref : float
        Reference maximum width used for normalization.

    Returns
    -------
    cwc : float
        Computed CWC score.
    """
    widths = np.asarray(widths).flatten()
    in_interval = np.asarray(in_interval).flatten()

    if widths.shape != in_interval.shape:
        raise ValueError("Widths and in_interval must have the same shape.")

    width_score = np.clip(widths, None, ref) / ref
    mean_width_score = np.nanmean(width_score)
    coverage = np.nanmean(in_interval)
    penalty = np.exp(-eta * (coverage - (1 - alpha))**2)

    return (1 - mean_width_score) * penalty


def compute_metric_by_group(metric_fn, metric_inputs, mask, axis=None,**kwargs):
    """
    Compute a metric by group mask over flattened or temporal slices.

    Parameters
    ----------
    metric_fn : callable
        Function to compute (e.g. compute_cwc or compute_mwi).
    metric_inputs : tuple of np.ndarray
        Input arrays to the function, must have matching shapes.
    mask : np.ndarray
        Grouping mask. Can be:
        - 2D (R, C) → static grouping over space.
        - 3D (T, R, C) → dynamic grouping over space and time.
    axis : int or None
        If None: aggregate over all time.
        If axis=0: aggregate per timestep (e.g., shape (T, R, C)).

    Returns
    -------
    dict
        Mapping group label to metric (float) or series (list of float).
    """
    mask = np.asarray(mask)
    results = {}

    # Get labels excluding nan
    unique_labels = np.unique(mask)

    for label in unique_labels:
        submask = mask == label

        if axis is None:
            # Expand 2D mask to 3D if needed
            if submask.ndim == 2:
                submask = np.broadcast_to(submask, metric_inputs[0].shape)

            # Apply same mask to all inputs
            inputs = [x[submask] for x in metric_inputs]
            results[str(label)] = metric_fn(*inputs,**kwargs)

        else:
            # axis=0: per timestep
            values = []
            for t in range(metric_inputs[0].shape[0]):
                m = submask[t] if submask.ndim == 3 else submask  # time slice or static
                inputs = [x[t][m] for x in metric_inputs]
                values.append(metric_fn(*inputs,**kwargs))
            results[str(label)] = values

    return results

def compute_full_metric_analysis(
    metric_fn,
    metric_inputs,
    grid_size,
    static_group_mask=None,
    dynamic_group_mask=None,
    **kwargs
):
    """
    Compute global, spatial, temporal and group-based evaluations for a given metric.

    Parameters
    ----------
    metric_fn : callable
        Function to compute the metric (e.g., compute_cwc, compute_mwi).
    metric_inputs : tuple of np.ndarray
        Tuple of arrays to be passed to the metric function, e.g. (widths, in_interval).
    grid_size : tuple
        Shape of the grid as (rows, cols).
    static_group_mask : np.ndarray, optional
        2D array (R, C) indicating static group categories.
    dynamic_group_mask : np.ndarray, optional
        3D array (T, R, C) indicating group per timestep.

    Returns
    -------
    dict
        Dictionary containing:
            - global: scalar
            - per_cell: 2D np.ndarray (R, C)
            - per_time: 1D np.ndarray (T,)
            - static_group: DataFrame
            - static_group_time: DataFrame
            - dynamic_group: DataFrame
            - dynamic_group_time: DataFrame
    """
    widths, in_interval = metric_inputs
    rows, cols = grid_size
    T = widths.shape[0]

    # Global metric
    global_metric = metric_fn(*[x.flatten() for x in metric_inputs],**kwargs)

    # Per-cell metric
    numering_cells = np.arange(rows * cols).reshape(rows, cols)
    per_cell_values = [
        metric_fn(*[x[:, numering_cells == n] for x in metric_inputs],**kwargs)
        for n in range(rows * cols)
    ]
    per_cell_matrix = np.array(per_cell_values).reshape(rows, cols)

    # Per-time metric
    per_time = np.array([
        metric_fn(*[x[t] for x in metric_inputs],**kwargs)
        for t in range(T)
    ])

    # Grouped by static categories
    static_group = None
    static_group_time = None
    if static_group_mask is not None:
        static_group = compute_metric_by_group(metric_fn, metric_inputs, static_group_mask,**kwargs)
        static_group = pd.DataFrame.from_dict(static_group, orient='index', columns=["Metric"]).reset_index().rename(columns={"index": "Cell Type"})

        static_group_time = compute_metric_by_group(metric_fn, metric_inputs, static_group_mask, axis=0,**kwargs)
        static_group_time = pd.DataFrame.from_dict(static_group_time)

    # Grouped by dynamic categories
    dynamic_group = None
    dynamic_group_time = None
    if dynamic_group_mask is not None:
        dynamic_group = compute_metric_by_group(metric_fn, metric_inputs, dynamic_group_mask,**kwargs)
        dynamic_group = pd.DataFrame.from_dict(dynamic_group, orient='index', columns=["Metric"]).reset_index().rename(columns={"index": "Cell Type"})

        dynamic_group_time = compute_metric_by_group(metric_fn, metric_inputs, dynamic_group_mask, axis=0,**kwargs)
        dynamic_group_time = pd.DataFrame.from_dict(dynamic_group_time)

    return {
        "global": global_metric,
        "per_cell": per_cell_matrix,
        "per_time": per_time,
        "static_group": static_group,
        "static_group_time": static_group_time,
        "dynamic_group": dynamic_group,
        "dynamic_group_time": dynamic_group_time,
    }


def base_analysis(y_min, y_max, y_pred, y_true, grid_size, alpha=0.1, hotspot_percentage=0.3, eta=8, ref=10 ):
    """
    Performs a full evaluation of prediction intervals and hotspot classification.

    This function computes core uncertainty-related metrics (coverage, interval width,
    error outside interval, MWI and CWC) along with hotspot classifications
    (both static and temporal), and returns all intermediate and final results.

    Parameters
    ----------
    y_min : np.ndarray
        Lower bounds of prediction intervals (T, R, C).
    y_max : np.ndarray
        Upper bounds of prediction intervals (T, R, C).
    y_pred : np.ndarray
        Predicted values or expected values (T, R, C).
    y_true : np.ndarray
        Ground truth values (T, R, C).
    grid_size : tuple of int
        Spatial grid size as (rows, cols).
    alpha : float
        Significance level of prediction intervals (e.g., 0.1 for 90%).
    hotspot_percentage : float
        Percentage of cells to classify as hotspots.
    eta : float
        Penalization parameter for CWC.
    ref : float
        Reference maximum width used in CWC normalization.

    Returns
    -------
    Tuple of:
        - in_interval : np.ndarray
            Boolean array (T, R, C), True if true value within prediction interval.
        - widths : np.ndarray
            Width of prediction intervals (T, R, C).
        - error : np.ndarray
            Error outside interval (T, R, C), NaN if inside.
        - overall_hs_class_grid : np.ndarray
            Static hotspot classification (R, C).
        - time_step_hs_class_grid : np.ndarray
            Temporal hotspot classification (T, R, C).
        - result_cwc : dict
            Full results for Coverage Width-Based Criterion.
        - result_mwi : dict
            Full results for Mean Winkler Interval Score.
        - coverage_results : dict
            Full results for coverage metric.
        - width_results : dict
            Full results for interval width.
        - error_results : dict
            Full results for error outside interval.
    """

    # Compute interval components
    in_interval = compute_in_interval(y_min, y_max, y_true)
    widths = compute_mean_width(y_min, y_max)
    error = compute_error_outside_interval(y_min, y_max, y_true)

    # Hotspot classifications
    overall_hs_class_grid = classify_hotspots(
        y_true.sum(axis=0), y_pred.sum(axis=0), hotspot_percentage
    )
    time_step_hs_class_grid = classify_hotspots(
        y_true, y_pred, hotspot_percentage
    )

    # Metric evaluations
    result_cwc = compute_full_metric_analysis(
        compute_cwc,
        (widths, in_interval),
        grid_size=grid_size,
        static_group_mask=overall_hs_class_grid,
        dynamic_group_mask=time_step_hs_class_grid,
        eta=eta,
        ref=ref,
        alpha=alpha
    )

    result_mwi = compute_full_metric_analysis(
        compute_mwi,
        (widths, error),
        grid_size=grid_size,
        static_group_mask=overall_hs_class_grid,
        dynamic_group_mask=time_step_hs_class_grid,
        alpha=alpha
    )

    coverage_results = compute_full_metric_analysis(
        lambda x, _: np.nanmean(x),
        (in_interval, in_interval),
        grid_size=grid_size,
        static_group_mask=overall_hs_class_grid,
        dynamic_group_mask=time_step_hs_class_grid
    )

    width_results = compute_full_metric_analysis(
        lambda x, _: np.nanmean(x),
        (widths, widths),
        grid_size=grid_size,
        static_group_mask=overall_hs_class_grid,
        dynamic_group_mask=time_step_hs_class_grid
    )

    error_results = compute_full_metric_analysis(
        lambda x, _: np.nanmean(x),
        (error, error),
        grid_size=grid_size,
        static_group_mask=overall_hs_class_grid,
        dynamic_group_mask=time_step_hs_class_grid
    )

    y_true_results = compute_full_metric_analysis(
        lambda x, _: np.nansum(x),
        (y_true, y_true),
        grid_size=grid_size,
        static_group_mask=overall_hs_class_grid,
        dynamic_group_mask=time_step_hs_class_grid
    )

    y_pred_results = compute_full_metric_analysis(
        lambda x, _: np.nansum(x),
        (y_pred, y_pred),
        grid_size=grid_size,
        static_group_mask=overall_hs_class_grid,
        dynamic_group_mask=time_step_hs_class_grid
    )

    return (
        in_interval,
        widths,
        error,
        overall_hs_class_grid,
        time_step_hs_class_grid,
        result_cwc,
        result_mwi,
        coverage_results,
        width_results,
        error_results,
        y_true_results,
        y_pred_results
    )