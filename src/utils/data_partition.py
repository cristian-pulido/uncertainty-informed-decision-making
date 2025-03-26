import pandas as pd

def temporal_split(df, train_end, calibration_end=None, test_end=None):
    """
    Generic temporal split of dataset based on specified timestep indices.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the column ["timestep"].

    train_end : int
        Final timestep index for the training set (exclusive).

    calibration_end : int, optional
        Final timestep index for calibration set (exclusive).
        If None, no calibration set is returned.

    test_end : int, optional
        Final timestep index for test set (exclusive).
        If None, returns remaining timesteps after calibration.

    Returns
    -------
    tuple of pd.DataFrame
        (df_train, df_calibration, df_test) or subsets thereof based on provided indices.
    """
    df_sorted = df.sort_values(by="timestep")

    df_train = df_sorted[df_sorted["timestep"] < train_end]

    df_calibration, df_test = None, None

    if calibration_end:
        df_calibration = df_sorted[
            (df_sorted["timestep"] >= train_end) & (df_sorted["timestep"] < calibration_end)
        ]

    if test_end:
        start_test = calibration_end if calibration_end else train_end
        df_test = df_sorted[
            (df_sorted["timestep"] >= start_test) & (df_sorted["timestep"] < test_end)
        ]
    elif calibration_end:
        df_test = df_sorted[df_sorted["timestep"] >= calibration_end]

    if calibration_end is None and test_end is None:
        return df_train

    return df_train, df_calibration, df_test
