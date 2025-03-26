# spatial_metrics.py (updated according to White 2023)
import numpy as np

def pai(real_crimes, predicted_hotspot_mask):
    """
    Predictive Accuracy Index (PAI), as defined in White (2023).

    Parameters
    ----------
    real_crimes : 2D np.array
        Actual crime counts per spatial unit.

    predicted_hotspot_mask : 2D np.array (bool)
        Binary mask indicating predicted hotspot cells.

    Returns
    -------
    float
        PAI value (higher is better, unbounded).
    """
    n = real_crimes[predicted_hotspot_mask].sum()
    N = real_crimes.sum()

    a = predicted_hotspot_mask.sum()
    A = predicted_hotspot_mask.size

    return (n / N) / (a / A)

def pei(real_crimes, predicted_hotspot_mask, optimal_hotspot_mask):
    """
    Predictive Efficiency Index (PEI), general definition from White (2023).

    Parameters
    ----------
    real_crimes : 2D np.array
        Actual crime counts per spatial unit.

    predicted_hotspot_mask : 2D np.array (bool)
        Predicted hotspot area.

    optimal_hotspot_mask : 2D np.array (bool)
        Optimal possible hotspot area (highest crime density).

    Returns
    -------
    float
        PEI value.
    """
    n = real_crimes[predicted_hotspot_mask].sum()
    a_p = predicted_hotspot_mask.sum()

    n_star = real_crimes[optimal_hotspot_mask].sum()
    a_r = optimal_hotspot_mask.sum()

    if (n_star / a_r) == 0:
        return 0

    return (n / a_p) / (n_star / a_r)

def pei_star(real_crimes, predicted_hotspot_mask):
    """
    Adjusted Predictive Efficiency Index (PEI*), simplified operational definition (White, 2023).

    Assumes equal predicted and optimal areas.

    Parameters
    ----------
    real_crimes : 2D np.array
        Actual crime counts per spatial unit.

    predicted_hotspot_mask : 2D np.array (bool)
        Predicted hotspot cells.

    Returns
    -------
    float
        PEI* value (between 0 and 1).
    """
    area_size = predicted_hotspot_mask.sum()

    # Identify optimal cells (highest crime density)
    optimal_indices = np.unravel_index(
        np.argsort(-real_crimes, axis=None)[:area_size],
        real_crimes.shape
    )
    optimal_hotspot_mask = np.zeros_like(real_crimes, dtype=bool)
    optimal_hotspot_mask[optimal_indices] = True

    n = real_crimes[predicted_hotspot_mask].sum()
    n_star = real_crimes[optimal_hotspot_mask].sum()

    if n_star == 0:
        return 0

    return n / n_star
