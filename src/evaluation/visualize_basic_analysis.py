import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import pathlib

def plot_temporal_metrics_summary(
    result_cwc,
    coverage_results,
    width_results,
    result_mwi,
    error_results,
    true_results,
    pred_results,
    colors=None,
    save_path=None 
):
    """
    Plots a 2x3 summary of temporal uncertainty metrics and crime trends.

    Parameters:
    - result_cwc, coverage_results, width_results, result_mwi, error_results: dicts with 'per_time' keys
    - grid_true: ndarray (timesteps x rows x cols) of observed crime
    - grid_pred: ndarray (timesteps x rows x cols) of predicted crime
    - colors: optional dict with color codes
    """

    if colors is None:
        colors = {
            "cwc": "#1f77b4",       # blue
            "coverage": "#2ca02c",  # green
            "width": "#ff7f0e",     # orange
            "crimes": "#d62728",    # red
            "pred": "#17becf",      # cyan
            "mwi": "#9467bd",       # purple
            "error": "#8c564b"      # brown
        }

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)

    # Plot 1: CWC
    sns.lineplot(ax=axes[0, 0], data=result_cwc["per_time"], label="CWC", color=colors["cwc"])
    axes[0, 0].set_ylabel("CWC Score")
    axes[0, 0].set_title("Coverage Width-Based Criterion")
    axes[0, 0].legend()

    # Plot 2: Coverage (%)
    sns.lineplot(ax=axes[0, 1], data=coverage_results["per_time"] * 100, label="Coverage", color=colors["coverage"])
    axes[0, 1].set_ylabel("Coverage (%)")
    axes[0, 1].set_title("Coverage Over Time")
    axes[0, 1].legend()

    # Plot 3: Interval Width
    sns.lineplot(ax=axes[0, 2], data=width_results["per_time"], label="Interval Width", color=colors["width"])
    axes[0, 2].set_ylabel("Width")
    axes[0, 2].set_title("Interval Width Over Time")
    axes[0, 2].legend()

    # Plot 4: Mean Winkler Interval
    sns.lineplot(ax=axes[1, 0], data=result_mwi["per_time"], label="MWI", color=colors["mwi"])
    axes[1, 0].set_ylabel("MWI Score")
    axes[1, 0].set_title("Mean Winkler Interval Score")
    axes[1, 0].legend()

    # Plot 5: Distance to Interval (Error)
    sns.lineplot(ax=axes[1, 1], data=error_results["per_time"], label="Distance to Interval", color=colors["error"])
    axes[1, 1].set_ylabel("Distance")
    axes[1, 1].set_title("Distance when Missed")
    axes[1, 1].legend()

    # Plot 6: Real vs Predicted Crimes
    sns.lineplot(ax=axes[1, 2], data=true_results["per_time"], label="Observed Crimes", color=colors["crimes"])
    sns.lineplot(ax=axes[1, 2], data=pred_results["per_time"], label="Predicted Crimes", color=colors["pred"])
    axes[1, 2].set_ylabel("Crime Count")
    axes[1, 2].set_title("Crime Trends")
    axes[1, 2].legend()

    # Final layout adjustments
    for ax in axes[1]:
        ax.set_xlabel("Timestep")

    fig.suptitle("Temporal Evaluation of Uncertainty and Crime Patterns", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        format_ = pathlib.Path(save_path).suffix[1:]
        fig.savefig(save_path, format=format_, bbox_inches="tight", dpi=300)

    plt.show()

    # Correlations
    print("Pearson Correlation with Total Observed Crimes:")
    print(f"CWC Score               : ρ = {pearsonr(result_cwc['per_time'], true_results['per_time'])[0]:.3f}")
    print(f"Coverage (%)            : ρ = {pearsonr(coverage_results['per_time'], true_results['per_time'])[0]:.3f}")
    print(f"Interval Width          : ρ = {pearsonr(width_results['per_time'], true_results['per_time'])[0]:.3f}")
    print(f"MWI                     : ρ = {pearsonr(result_mwi['per_time'], true_results['per_time'])[0]:.3f}")
    print(f"Distance when Missed    : ρ = {pearsonr(error_results['per_time'], true_results['per_time'])[0]:.3f}")
    print(f"Predicted Crime Count   : ρ = {pearsonr(pred_results['per_time'], true_results['per_time'])[0]:.3f}")

    print("\n" + "#"*50)
    print("Time Series Metrics (Mean ± Std):")
    print(f"CWC                     : {result_cwc['per_time'].mean():.2f} ± {result_cwc['per_time'].std():.2f}")
    print(f"MWI                     : {result_mwi['per_time'].mean():.2f} ± {result_mwi['per_time'].std():.2f}")
    print(f"Coverage (%)            : {coverage_results['per_time'].mean() * 100:.2f}% ± {coverage_results['per_time'].std() * 100:.2f}%")
    print(f"Interval Width          : {width_results['per_time'].mean():.2f} ± {width_results['per_time'].std():.2f}")
    print(f"Distance when Missed    : {error_results['per_time'].mean():.2f} ± {error_results['per_time'].std():.2f}")
    print(f"Observed Crimes         : {true_results['per_time'].mean():.2f} ± {true_results['per_time'].std():.2f}")
    print(f"Predicted Crimes        : {pred_results['per_time'].mean():.2f} ± {pred_results['per_time'].std():.2f}")



def plot_metric_radar_by_hotspot_type(
    results_list,
    metric_names,
    static=True,
    metric_limits=None,
    title="Comparison of Hotspot Types Across Metrics",
    figsize=(13, 8),
    show_tables=True,
    save_path=None
):
    """
    Plot radar charts of uncertainty metrics per hotspot type, and optionally display summary table.

    Parameters
    ----------
    results_list : list of dict
        List of results dictionaries (e.g., coverage_results, result_cwc, etc.)
        Each must contain keys 'static_group' or 'dynamic_group' depending on `static`.
    metric_names : list of str
        Names of metrics to use as subplot titles.
    static : bool, default=True
        Whether to use static classification of hotspot types. If False, uses dynamic.
    metric_limits : dict or None
        Dictionary of y-axis limits per metric (e.g., {"Coverage Score": (0.8, 1.0)}).
        If None, limits are computed automatically from data.
    title : str
        Super title of the plot.
    figsize : tuple
        Figure size in inches.
    show_tables : bool, default=True
        If True, prints and returns the combined summary DataFrame.
    
    Returns
    -------
    summary_df : pd.DataFrame or None
        Combined table of all metrics per hotspot type. Only returned if `show_tables=True`.
    """
    if len(results_list) != len(metric_names):
        raise ValueError("Length of results_list must match metric_names.")

    key = "static_group" if static else "dynamic_group"
    cell_types = results_list[0][key]["Cell Type"].values

    num_metrics = len(metric_names)
    ncols = 3
    nrows = int(np.ceil(num_metrics / ncols))
    fig, axes = plt.subplots(nrows, ncols, subplot_kw={"projection": "polar"}, figsize=figsize)
    axes = axes.flatten()

    angles = np.linspace(0, 2 * np.pi, len(cell_types), endpoint=False).tolist()
    angles += angles[:1]

    if metric_limits is None:
        metric_limits = {}
        for result, name in zip(results_list, metric_names):
            vals = result[key].set_index("Cell Type").loc[cell_types]["Metric"]
            min_val = np.floor(np.nanmin(vals) * 10) / 10
            max_val = np.ceil(np.nanmax(vals) * 10) / 10
            metric_limits[name] = (min_val, max_val)

    summary_df = pd.DataFrame(index=cell_types)
    for i, (result, name) in enumerate(zip(results_list, metric_names)):
        ax = axes[i]
        df = result[key].set_index("Cell Type").reindex(cell_types)
        values = df["Metric"].values.tolist()
        values += values[:1]

        ax.plot(angles, values, linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_title(name, size=10, pad=10)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(cell_types, fontsize=8)
        ax.set_ylim(*metric_limits[name])

        # Add column to summary table
        summary_df[name] = df["Metric"]

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        format_ = pathlib.Path(save_path).suffix[1:]
        fig.savefig(save_path, format=format_, bbox_inches="tight", dpi=300)
    plt.show()

    if show_tables:
        import IPython.display as disp
        disp.display(summary_df)

    return summary_df if show_tables else None


def plot_temporal_metrics_over_time(
    result_cwc,
    coverage_results,
    width_results,
    result_mwi,
    error_results,
    mode="static",  # "static" or "dynamic"
    overall_hs_class_df=None,
    time_step_hs_class_df=None,
    palette=None,
    save_path=None
):
    """
    Plot temporal behavior of uncertainty metrics (CWC, Coverage, Width, MWI, Distance to interval)
    and cell counts by hotspot type using static or dynamic classification.

    Parameters
    ----------
    result_cwc : dict
        Output of compute_full_metric_analysis for Coverage Width-Based Criterion.
    coverage_results : dict
        Output of compute_full_metric_analysis for coverage scores.
    width_results : dict
        Output of compute_full_metric_analysis for interval widths.
    result_mwi : dict
        Output of compute_full_metric_analysis for Mean Winkler Interval.
    error_results : dict
        Output of compute_full_metric_analysis for Distance to interval when no coverage.
    mode : str
        "static" or "dynamic". Determines how to compute cell counts.
    overall_hs_class_df : pd.DataFrame, optional
        Required if mode="static". Must contain 'cell_type'.
    time_step_hs_class_df : pd.DataFrame, optional
        Required if mode="dynamic". Must contain 'timestep' and 'cell_type'.
    palette : dict
        Dictionary mapping cell types to colors.
    save_path : str or Path, optional
        If provided, saves the figure to this path.
    """
    if mode == "static" and overall_hs_class_df is None:
        raise ValueError("overall_hs_class_df must be provided in static mode.")
    if mode == "dynamic" and time_step_hs_class_df is None:
        raise ValueError("time_step_hs_class_df must be provided in dynamic mode.")
    
    group_key = "static_group_time" if mode == "static" else "dynamic_group_time"

    cwc_df = result_cwc[group_key]
    coverage_df = coverage_results[group_key] * 100
    width_df = width_results[group_key]
    mwi_df = result_mwi[group_key]
    errors_df = error_results[group_key]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)

    # Plot 1: CWC
    for group in cwc_df.columns:
        axes[0, 0].plot(cwc_df.index, cwc_df[group], label=group, color=palette.get(group))
    axes[0, 0].set_title("Coverage Width-Based Criterion")
    axes[0, 0].set_ylabel("CWC Score")
    axes[0, 0].grid(True)

    # Plot 2: Coverage
    for group in coverage_df.columns:
        axes[0, 1].plot(coverage_df.index, coverage_df[group], color=palette.get(group))
    axes[0, 1].set_title("Coverage Score (%)")
    axes[0, 1].set_ylabel("Coverage (%)")
    axes[0, 1].grid(True)

    # Plot 3: Interval Width
    for group in width_df.columns:
        axes[0, 2].plot(width_df.index, width_df[group], color=palette.get(group))
    axes[0, 2].set_title("Average Interval Width")
    axes[0, 2].set_ylabel("Width")
    axes[0, 2].grid(True)

    # Plot 4: Cell count
    if mode == "static":
        group_sizes = overall_hs_class_df["cell_type"].value_counts().to_dict()
        for group, count in group_sizes.items():
            axes[1, 0].plot(coverage_df.index, [count] * len(coverage_df), color=palette.get(group))
        axes[1, 0].set_title("Number of Cells per Hotspot Type (Static)")
    else:
        cell_counts_time = time_step_hs_class_df.groupby(["timestep", "cell_type"]).size().unstack(fill_value=0)
        for group in cell_counts_time.columns:
            axes[1, 0].plot(cell_counts_time.index, cell_counts_time[group], color=palette.get(group))
        axes[1, 0].set_title("Number of Cells per Hotspot Type (Dynamic)")
    axes[1, 0].set_ylabel("Cell Count")
    axes[1, 0].grid(True)

    # Plot 5: MWI
    for group in mwi_df.columns:
        axes[1, 1].plot(mwi_df.index, mwi_df[group], color=palette.get(group))
    axes[1, 1].set_title("Mean Winkler Interval Score")
    axes[1, 1].set_ylabel("MWI")
    axes[1, 1].grid(True)

    # Plot 6: Distance to interval (scatter plot)
    for group in errors_df.columns:
        axes[1, 2].scatter(errors_df.index, errors_df[group], color=palette.get(group), s=10, alpha=0.7, label=group)
    axes[1, 2].set_title("Distance to Interval (when no coverage)")
    axes[1, 2].set_ylabel("Distance")
    axes[1, 2].grid(True)

    # Common x label
    for ax in axes.flat:
        ax.set_xlabel("Timestep")

    # Shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        format_ = pathlib.Path(save_path).suffix[1:]
        fig.savefig(save_path, format=format_, bbox_inches="tight", dpi=300)
    plt.show()

def plot_weighted_cwc_series(
    results_dict,
    title="Comparison of Weighted CWC Scores by Hotspot Assignment Strategy",
    save_path=None
):
    """
    Plots weighted CWC time series for static and dynamic hotspot assignment strategies using Seaborn.

    Parameters
    ----------
    results_dict : dict
        Must include:
            - 'global_value': float
            - 'static_weighted': float
            - 'static_series': pd.Series
            - 'dynamic_weighted': float
            - 'dynamic_series': pd.Series
    title : str
        Title for the plot.
    save_path : str or Path, optional
        If provided, saves the figure to the given path.
    """
    sns.set_theme(style="whitegrid")

    # Prepare DataFrame
    df_plot = pd.DataFrame({
        "Timestep": results_dict["static_series"].index,
        "Static Assignment": results_dict["static_series"].values,
        "Dynamic Assignment": results_dict["dynamic_series"].values
    }).melt(id_vars="Timestep", var_name="Assignment Strategy", value_name="Weighted CWC Score")

    # Plot
    plt.figure(figsize=(12, 5))
    sns.lineplot(
        data=df_plot,
        x="Timestep", y="Weighted CWC Score",
        hue="Assignment Strategy",
        linewidth=2.2,
        palette=["#1f77b4", "#2ca02c"]
    )
    plt.title(title, fontsize=14)
    plt.xlabel("Timestep")
    plt.ylabel("Weighted Hotspot CWC Score (WH-CWC)")
    plt.legend(title="Assignment Strategy", loc="best", frameon=True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    # Print summary
    print("=== 🔢 WH-CWC Scores Summary ===")
    print(f"🌐 Global CWC Score: {results_dict['global_value']:.3f}")
    print(f"📘 Static Assignment - Weighted Score: {results_dict['static_weighted']:.3f}")
    print(f"📗 Dynamic Assignment - Weighted Score: {results_dict['dynamic_weighted']:.3f}")
    print("\n📊 Static Assignment Series:")
    print(f"  Mean: {results_dict['static_series'].mean():.3f}")
    print(f"  Std: {results_dict['static_series'].std():.3f}")
    print("📊 Dynamic Assignment Series:")
    print(f"  Mean: {results_dict['dynamic_series'].mean():.3f}")
    print(f"  Std: {results_dict['dynamic_series'].std():.3f}")
