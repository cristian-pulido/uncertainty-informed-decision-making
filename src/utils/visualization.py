import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.ticker import PercentFormatter
import pathlib

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle, Wedge, FancyArrowPatch
import matplotlib.patheffects as pe


def compare_prediction_maps(grids, titles, vmin=None, vmax=None, figsize=(14, 4)):
    fig, axes = plt.subplots(1, len(grids), figsize=figsize)
    for ax, grid, title in zip(axes, grids, titles):
        sns.heatmap(grid, cmap="YlGnBu", ax=ax, vmin=vmin, vmax=vmax, cbar=True)
        ax.set_title(title)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def compare_hotspot_masks(masks, titles, ncols=3, figsize=(12, 4)):
    n = len(masks)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n > 1 else [axes]

    for i in range(n):
        sns.heatmap(masks[i].astype(int), cmap="YlOrRd", linewidths=0.5, linecolor='gray', cbar=False, ax=axes[i])
        axes[i].set_title(titles[i])
        axes[i].invert_yaxis()
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def plot_hotspot_masks_over_days(mask_dict, ncols=4, figsize=(14, 8)):
    """
    Visualize multiple hotspot masks by day.
    
    Parameters:
    - mask_dict: dict {day_label: np.ndarray mask}
    - ncols: int, number of columns in the subplot grid
    - figsize: tuple, figure size
    """
    titles = list(mask_dict.keys())
    masks = list(mask_dict.values())
    n = len(masks)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n > 1 else [axes]

    for i in range(n):
        sns.heatmap(masks[i].astype(int), cmap="YlOrRd", linewidths=0.5, linecolor='gray', cbar=False, ax=axes[i])
        axes[i].set_title(titles[i])
        axes[i].invert_yaxis()
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()



def plot_prediction_interval_map(pred, lower, upper, title="Prediction with Interval", cmap="YlOrRd", vmin=None, vmax=None,titles = ["Prediction", "Lower Bound", "Upper Bound"]):
    """
    Plot prediction, lower, and upper interval maps side by side.

    Parameters:
    - pred: 2D array, prediction mean
    - lower: 2D array, lower bound
    - upper: 2D array, upper bound
    - title: str, global title for the figure
    - cmap: colormap
    - vmin, vmax: optional color scale bounds
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    vmin = vmin if vmin is not None else min(pred.min(), lower.min(), upper.min())
    vmax = vmax if vmax is not None else max(pred.max(), lower.max(), upper.max())

    data = [pred, lower, upper]

    for ax, arr, t in zip(axes, data, titles):
        sns.heatmap(arr, cmap=cmap, cbar=True, ax=ax, vmin=vmin, vmax=vmax, square=True, linewidths=0, linecolor="gray")
        ax.set_title(t)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_grid_map(grid, title="Grid Map", cmap="viridis", label="Value", vmin=None, vmax=None, figsize=(6, 5)):
    """
    Visualize a single 2D grid (e.g., coverage or interval width).

    Parameters:
    - grid: 2D array (rows x cols)
    - title: str
    - cmap: str
    - label: str, label for the colorbar
    - vmin, vmax: color limits
    - figsize: tuple
    """
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.imshow(grid, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    fig.colorbar(cax, ax=ax, label=label)
    plt.show()


################################################################################


def plot_geospatial_data_maps(
    gdf_base,
    df_list,
    columns=None,
    titles=None,
    cmap="YlOrRd",
    share_colorbar=True,
    suptitle=None,
    figsize=(12, 5),
    edgecolor="black",
    vmin=None,
    vmax=None,
    colorbar_labels=None,
    percent_format=None,
    save_path=None 
):
    """
    Plot multiple prediction maps side by side with optional shared or individual colorbars.

    Parameters
    ----------
    gdf_base : GeoDataFrame
        Base geometry to merge with each df
    df_list : list of DataFrames
        Each must contain a 'beat' column and the data to plot
    columns : list of str
        Column names to visualize (one per df)
    titles : list of str
        Titles for each subplot
    cmap : str
        Matplotlib colormap
    share_colorbar : bool
        Whether to use a single shared colorbar
    suptitle : str
        Title for the entire figure
    figsize : tuple
        Figure size
    edgecolor : str
        Color of borders
    vmin, vmax : float or list of float
        Global or per-map color scale min/max
    colorbar_labels : list of str
        Custom labels for the colorbars
    percent_format : list of bool
        Whether to use percent format per map (or single bool if shared)
    save_path : str
        If given, saves figure to this path
    """

    n = len(df_list)
    if columns is None or len(columns) != n:
        raise ValueError("Must provide 'columns': one per DataFrame.")

    if percent_format is None:
        percent_format = [False] * n
    elif isinstance(percent_format, bool):
        percent_format = [percent_format] * n

    if not share_colorbar and (vmin is None or isinstance(vmin, (int, float))):
        vmin = [vmin] * n
    if not share_colorbar and (vmax is None or isinstance(vmax, (int, float))):
        vmax = [vmax] * n

    fig, axes = plt.subplots(1, n, figsize=figsize, constrained_layout=True)
    if n == 1:
        axes = [axes]

    # Shared normalization
    norm = None
    if share_colorbar:
        global_min = min(df[col].min() for df, col in zip(df_list, columns)) if vmin is None else vmin
        global_max = max(df[col].max() for df, col in zip(df_list, columns)) if vmax is None else vmax
        norm = mcolors.Normalize(vmin=global_min, vmax=global_max)

    for i, (df, col, ax) in enumerate(zip(df_list, columns, axes)):
        merged = gdf_base.merge(df, how="left", left_on="beat_num", right_on="beat")

        if share_colorbar:
            merged.plot(column=col, cmap=cmap, ax=ax, edgecolor=edgecolor, legend=False, norm=norm)
        else:
            this_vmin = vmin[i] if isinstance(vmin, list) else None
            this_vmax = vmax[i] if isinstance(vmax, list) else None
            fmt = PercentFormatter(decimals=0) if percent_format[i] else None
            merged.plot(
                column=col,
                cmap=cmap,
                ax=ax,
                edgecolor=edgecolor,
                legend=True,
                vmin=this_vmin,
                vmax=this_vmax,
                legend_kwds={
                    "label": colorbar_labels[i] if colorbar_labels else col,
                    "format": fmt
                }
            )

        ax.set_title(titles[i] if titles else f"Map {i+1}")
        ax.set_xticks([])
        ax.set_yticks([])

    # Shared colorbar
    if share_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []
        cbar = fig.colorbar(sm, ax=axes, shrink=0.8)
        cbar.set_label(colorbar_labels[0] if colorbar_labels else columns[0])
        if percent_format[0]:
            cbar.ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))

    if suptitle:
        plt.suptitle(suptitle, fontsize=14)

    if save_path:
        format_ = pathlib.Path(save_path).suffix[1:]
        fig.savefig(save_path, format=format_, bbox_inches="tight", dpi=300)

    plt.show()




# ---------- helpers ----------
def _prep_ids(gdf, df_pred, df_min, df_max, df_true=None, df_cov=None, cov_col="Coverage Score"):
    """
    Normalize key columns and dtypes across inputs.
    Returns copies to avoid mutating caller dataframes.
    """
    g = gdf.copy()
    if "beat_num" not in g.columns:
        raise KeyError("gdf must contain a 'beat_num' column.")
    g["beat_num"] = g["beat_num"].astype(str)

    p = df_pred.copy(); p["beat"] = p["beat"].astype(str)
    mn = df_min.copy(); mn["beat"] = mn["beat"].astype(str)
    mx = df_max.copy(); mx["beat"] = mx["beat"].astype(str)

    tr = None
    if df_true is not None:
        tr = df_true.copy(); tr["beat"] = tr["beat"].astype(str)

    cov = None
    if df_cov is not None:
        cov = df_cov.copy()
        if "beat" in cov.columns:
            cov = cov.dropna(subset=["beat"]).copy()
            cov["beat"] = cov["beat"].astype(str)
        if cov_col not in cov.columns:
            raise ValueError(f"df_cov is missing the '{cov_col}' column.")
    return g, p, mn, mx, tr, cov


def _centroids(gdf):
    """Return centroids and their x/y for each beat."""
    c = gdf.copy()
    c["centroid"] = c.geometry.centroid
    c["cx"] = c["centroid"].x
    c["cy"] = c["centroid"].y
    return c[["beat_num", "cx", "cy"]]


def _draw_ring_abs(
    ax, xy, r_in, r_out, cmap, norm, vmin_val, vmax_val,
    n_steps=64, alpha=0.95, inner_ls="--", outer_ls="-",
    lw=1.4, white_inner=True, banded=True, band_steps=5,
    zorder_fill=10, zorder_edge=11
):
    """
    Draw a colored annulus between r_in and r_out.
    - If banded=True: few thick bands.
    - Else: many thin bands (smooth gradient).
    Also draws a white disk at the center (if white_inner=True) and ring outlines.
    """
    # White center above the basemap, below color bands
    if white_inner and r_in > 0:
        ax.add_patch(Circle(xy, r_in, facecolor="white", edgecolor="none", zorder=zorder_fill - 1))

    if banded:
        radii  = np.linspace(r_out, r_in, band_steps + 1)
        values = np.linspace(vmax_val, vmin_val, band_steps)
        for j in range(band_steps):
            r0, r1 = radii[j], radii[j + 1]
            width = max(r0 - r1, 1e-9)
            w = Wedge(
                xy, r0, 0, 360, width=width,
                facecolor=cmap(norm(values[j])), edgecolor="none",
                alpha=alpha, zorder=zorder_fill
            )
            ax.add_patch(w)
    else:
        radii  = np.linspace(max(r_out, r_in + 1e-9), r_in, n_steps)
        values = np.linspace(vmax_val, vmin_val, n_steps)
        for rr, vv in zip(radii, values):
            ax.add_patch(
                Circle(xy, rr, facecolor=cmap(norm(vv)), edgecolor="none",
                       alpha=alpha, zorder=zorder_fill)
            )

    # Ring outlines on top
    ax.add_patch(Circle(xy, r_in,  fill=False, linestyle=inner_ls, linewidth=lw, zorder=zorder_edge))
    ax.add_patch(Circle(xy, r_out, fill=False, linestyle=outer_ls, linewidth=lw, zorder=zorder_edge))


def _draw_coverage_gauge(
    ax, xy, coverage_pct, r_in, r_out, span,
    linecolor="#1f77b4", lw=1.8, fontsize=11, bold=True,
    use_halo=True, add_tail_dot=False,  # dots ya no se usan
    arrow_scale=10, zorder=30,
    max_arc=2* np.pi,     #
    head_frac=0.06,    # fracción del arco usada para orientar cada flecha
):
    """
    Dibuja un ARCO que siempre inicia en 90° (vertical arriba) y se abre hacia
    la izquierda (CCW). La longitud del arco es proporcional a la cobertura.
    Flechas en ambos extremos del arco indican inicio y fin del recorrido.
    """
    cov = float(np.clip(coverage_pct, 0, 100))

    # ----- 1) Geometría del arco -----
    x0, y0 = xy
    r_line = r_in + 0.5 * (r_out - r_in)   # dibuja en el medio del anillo
    theta_start = np.pi / 2                # 90° fijo
    arc_len     = (cov / 100.0) * max_arc  # más cobertura -> arco más largo
    theta_end   = theta_start + arc_len    # hacia la izquierda (CCW)

    # puntos del arco (trayectoria circular)
    npts = max(12, int(60 * cov / 100.0))  # más puntos si el arco es más largo
    thetas = np.linspace(theta_start, theta_end, npts)
    xs = x0 + r_line * np.cos(thetas)
    ys = y0 + r_line * np.sin(thetas)

    # estilo halo
    pe_line = [pe.Stroke(linewidth=lw + 1.8, foreground="white"), pe.Normal()] if use_halo else None

    # ----- 2) Trazo del arco -----
    line = Line2D(xs, ys, linewidth=lw, color=linecolor, zorder=zorder)
    if pe_line is not None:
        line.set_path_effects(pe_line)
    ax.add_line(line)

    # ----- 3) Flechas en los extremos, tangentes al arco -----
    #   - flecha de INICIO en 90°, apuntando en sentido del recorrido (CCW)
    #   - flecha de FIN en theta_end, también tangente al arco
    dth = max(1e-3, head_frac * max(theta_end - theta_start, 1e-3))

    # start arrow: desde un punto "adelantado" hacia el inicio
    xs1, ys1 = (x0 + r_line * np.cos(theta_start + dth),
                y0 + r_line * np.sin(theta_start + dth))
    xst, yst = (x0 + r_line * np.cos(theta_start),
                y0 + r_line * np.sin(theta_start))
    ap0 = FancyArrowPatch((xs1, ys1), (xst, yst),
                          arrowstyle="-|>", mutation_scale=arrow_scale,
                          linewidth=lw, color=linecolor, shrinkA=0, shrinkB=0,
                          zorder=zorder)
    if pe_line is not None:
        ap0.set_path_effects(pe_line)
    ax.add_patch(ap0)

    # end arrow: desde un punto "antes" hacia el final
    xse, yse = (x0 + r_line * np.cos(theta_end - dth),
                y0 + r_line * np.sin(theta_end - dth))
    xe, ye = (x0 + r_line * np.cos(theta_end),
              y0 + r_line * np.sin(theta_end))
    ap1 = FancyArrowPatch((xse, yse), (xe, ye),
                          arrowstyle="-|>", mutation_scale=arrow_scale,
                          linewidth=lw, color=linecolor, shrinkA=0, shrinkB=0,
                          zorder=zorder)
    if pe_line is not None:
        ap1.set_path_effects(pe_line)
    ax.add_patch(ap1)

    # ----- 4) Texto de porcentaje (igual que antes) -----
    offset = max(0.05 * span, 0.15 * (r_out - r_in))
    xt = x0 - 3 * offset
    yt = y0
    txt_kw = dict(
        fontsize=fontsize, ha="right", va="center", color=linecolor,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=linecolor, lw=1.1, alpha=0.9),
        zorder=zorder + 1,
        fontweight="bold" if bold else None
    )
    ax.text(xt, yt, f"{cov:.0f}%", **txt_kw)




# ---------- main ----------
def plot_cp(
    gdf_base, df_pred, df_min, df_max,
    df_cov=None, cov_col="Coverage Score",
    *,
    cmap="YlOrRd", vmin=None, vmax=None,
    zoom_on="beat", zoom_value=None, neighbors_k=20,
    gradient_steps=64, rings_alpha=0.95,
    inner_ls="--", outer_ls="-", white_inner=True,
    banded=True, band_steps=5,
    figsize=(14, 7), width_ratios=(1.1, 1.0),
    title_left="Prediction", title_right="Zoom (Prediction intervals)",
    colorbar_label="Prediction Crimes", legend_loc="lower left",
    scale_circles=0.55, span_factor=0.32, r0_ratio=0.05,
    min_visible_ratio=0.12, min_thickness_policy="outward",
    reuse_config=None, annotate_values=True,
    show_coverage=True, coverage_place="outside", coverage_color="#1f77b4",
    save_path=None, show=True,
):
    """
    Plot a citywide choropleth and a zoom panel with per-beat prediction
    intervals as colored rings, plus an optional coverage gauge.

    Returns
    -------
    config : dict
        Fully-specified configuration used for the render (for reproducibility).
    """

    # ---------- 1) DATA PREP & COLOR NORMALIZATION (always first) ----------
    gdf, dpred, dmin, dmax, _unused_true, dcov = _prep_ids(
        gdf_base, df_pred, df_min, df_max, df_true=None, df_cov=df_cov, cov_col=cov_col
    )
    gdf = gdf.to_crs(epsg=26916)
    merged = gdf.merge(dpred, how="left", left_on="beat_num", right_on="beat")

    if "Prediction Crimes" not in merged.columns:
        raise KeyError("Expected column 'Prediction Crimes' in df_pred/merged.")
    if "Prediction Crimes" not in dmin.columns or "Prediction Crimes" not in dmax.columns:
        raise KeyError("Expected column 'Prediction Crimes' in df_min/df_max.")

    # Defaults from data (will be overridden by reuse_config if provided)
    vmin_data = min(merged["Prediction Crimes"].min(), dmin["Prediction Crimes"].min())
    vmax_data = max(merged["Prediction Crimes"].max(), dmax["Prediction Crimes"].max())
    vmin = vmin if vmin is not None else vmin_data
    vmax = vmax if vmax is not None else vmax_data
    _cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Centroids & values to choose zoom area
    cents = _centroids(gdf)
    mcent = cents.merge(merged[["beat_num", "Prediction Crimes"]], on="beat_num", how="left")

    # ---------- 2) ZOOM SELECTION (neigh needed by scaling) ----------
    if zoom_on == "beat":
        target = str(zoom_value) if zoom_value is not None else \
                 str(merged.sort_values("Prediction Crimes", ascending=False).iloc[0]["beat_num"])
        cx, cy = mcent.loc[mcent["beat_num"] == target, ["cx", "cy"]].iloc[0].tolist()
        d2 = (mcent["cx"] - cx) ** 2 + (mcent["cy"] - cy) ** 2
        neigh = mcent.assign(d2=d2).nsmallest(max(1, min(neighbors_k, len(mcent))), "d2")
        beats_sel = set(neigh["beat_num"])
        g_zoom = merged[merged["beat_num"].isin(beats_sel)]
    elif zoom_on == "xy":
        cx, cy = float(zoom_value[0]), float(zoom_value[1])
        d2 = (mcent["cx"] - cx) ** 2 + (mcent["cy"] - cy) ** 2
        neigh = mcent.assign(d2=d2).nsmallest(max(1, min(neighbors_k, len(mcent))), "d2")
        beats_sel = set(neigh["beat_num"])
        g_zoom = merged[merged["beat_num"].isin(beats_sel)]
    elif zoom_on == "bbox":
        minx, miny, maxx, maxy = zoom_value
        g_zoom = merged.cx[minx:maxx, miny:maxy]
        neigh = mcent[mcent["beat_num"].isin(g_zoom["beat_num"])]
    else:
        raise ValueError("zoom_on must be one of {'beat', 'xy', 'bbox'}.")

    # Optional coverage map
    cov_map = dict(zip(dcov["beat"], dcov[cov_col])) if dcov is not None else {}

    # ---------- 3) SCALING / CONFIG (after neigh exists) ----------
    if reuse_config is None:
        # derive a geometric scale from neighbor spacing
        xy = neigh[["cx", "cy"]].to_numpy()
        if len(xy) >= 2:
            D = np.sqrt(((xy[None, :, :] - xy[:, None, :]) ** 2).sum(-1))
            D = D[np.triu_indices_from(D, k=1)]
            geom_scale = np.median(D[np.isfinite(D)]) if np.any(np.isfinite(D)) else 1.0
        else:
            geom_scale = 1.0

        span = span_factor * geom_scale * scale_circles
        r0 = r0_ratio * span
        min_thickness_abs = min_visible_ratio * span

        config = {
            # scaling
            "mode": "absolute",
            "r0": float(r0),
            "span": float(span),
            "min_thickness_abs": float(min_thickness_abs),
            "min_thickness_policy": str(min_thickness_policy),
            # color scale
            "vmin": float(vmin), "vmax": float(vmax),
            "cmap": cmap if isinstance(cmap, str) else str(cmap),
            # layout
            "scale_circles": float(scale_circles),
            "span_factor": float(span_factor),
            "r0_ratio": float(r0_ratio),
            # ring rendering
            "banded": bool(banded),
            "band_steps": int(band_steps),
            "gradient_steps": int(gradient_steps),
            "rings_alpha": float(rings_alpha),
            "inner_ls": str(inner_ls),
            "outer_ls": str(outer_ls),
            "white_inner": bool(white_inner),
            # coverage styling
            "show_coverage": bool(show_coverage),
            "coverage_place": str(coverage_place),
            "coverage_color": str(coverage_color),
            # meta
            "neighbors_k": int(neighbors_k),
            "zoom_on": str(zoom_on),
            "zoom_value": zoom_value,
            "figsize": tuple(figsize),
            "width_ratios": tuple(width_ratios),
            "titles": {"left": title_left, "right": title_right},
            "legend_loc": str(legend_loc),
            "annotate_values": bool(annotate_values),
        }
    else:
        # reuse provided config (override plotting knobs BEFORE building norm/_cmap)
        config = dict(reuse_config)
        if config.get("mode", "absolute") != "absolute":
            raise ValueError("Reused config must be in 'absolute' mode'.")

        # required scalars
        r0   = float(config["r0"])
        span = float(config["span"])
        # optional scalars / switches
        min_thickness_abs    = float(config.get("min_thickness_abs", 0.10 * span))
        min_thickness_policy = config.get("min_thickness_policy", min_thickness_policy)

        # adopt color scale from config to guarantee identical colors
        vmin = float(config.get("vmin", vmin))
        vmax = float(config.get("vmax", vmax))
        cmap = config.get("cmap", cmap)
        _cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        # ring & coverage style
        banded         = bool(config.get("banded", banded))
        band_steps     = int(config.get("band_steps", band_steps))
        gradient_steps = int(config.get("gradient_steps", gradient_steps))
        rings_alpha    = float(config.get("rings_alpha", rings_alpha))
        inner_ls       = str(config.get("inner_ls", inner_ls))
        outer_ls       = str(config.get("outer_ls", outer_ls))
        white_inner    = bool(config.get("white_inner", white_inner))
        show_coverage  = bool(config.get("show_coverage", show_coverage))
        coverage_place = str(config.get("coverage_place", coverage_place))
        coverage_color = str(config.get("coverage_color", coverage_color))

    # ---------- 4) FIGURE ----------
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=figsize,
        gridspec_kw={"width_ratios": width_ratios},
        constrained_layout=True
    )

    # Left panel (citywide)
    merged.plot(column="Prediction Crimes", cmap=_cmap, ax=axL,
                edgecolor="black", linewidth=0.4, legend=False, norm=norm, zorder=0)
    axL.set_title(title_left); axL.set_xticks([]); axL.set_yticks([])
    sm = plt.cm.ScalarMappable(cmap=_cmap, norm=norm); sm._A = []
    cbar = fig.colorbar(sm, ax=axL, shrink=0.8, pad=0.02); cbar.set_label(colorbar_label)
    zx0, zy0, zx1, zy1 = g_zoom.total_bounds
    axL.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0,
                            fill=False, ec="black", lw=1.5, linestyle=":"))

    # Right panel (zoom)
    g_zoom.plot(column="Prediction Crimes", cmap=_cmap, ax=axR,
                edgecolor="black", linewidth=0.6, legend=False, norm=norm, zorder=0)
    axR.set_title(title_right); axR.set_xticks([]); axR.set_yticks([])
    axR.set_xlim(zx0, zx1); axR.set_ylim(zy0, zy1)

    # --- per-beat intervals in zoom
    mm = dmin.merge(dmax, on="beat", suffixes=("_min", "_max")).rename(columns={"beat": "beat_num"})
    aux = neigh.merge(mm, on="beat_num", how="left")

    def _rad_from_val(v):
        if vmax <= vmin:
            return r0 + 0.5 * span
        t = np.clip((v - vmin) / (vmax - vmin), 0.0, 1.0)
        return r0 + t * span

    for _, row in aux.iterrows():
        beat = str(row["beat_num"])
        v_lo_raw = float(row["Prediction Crimes_min"])
        v_hi_raw = float(row["Prediction Crimes_max"])
        lo, hi = min(v_lo_raw, v_hi_raw), max(v_lo_raw, v_hi_raw)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue

        r_in, r_out = _rad_from_val(lo), _rad_from_val(hi)
        if r_out - r_in < min_thickness_abs:
            if min_thickness_policy == "symmetric":
                mid = 0.5 * (r_in + r_out); half = 0.5 * min_thickness_abs
                r_in, r_out = mid - half, mid + half
            else:
                r_out = min(r_in + min_thickness_abs, r0 + span)

        cx_i = float(neigh.loc[neigh["beat_num"] == beat, "cx"].iloc[0])
        cy_i = float(neigh.loc[neigh["beat_num"] == beat, "cy"].iloc[0])

        _draw_ring_abs(
            axR, (cx_i, cy_i), r_in, r_out, _cmap, norm, lo, hi,
            n_steps=gradient_steps, alpha=rings_alpha,
            inner_ls=inner_ls, outer_ls=outer_ls, lw=1.3,
            white_inner=white_inner, banded=banded, band_steps=band_steps
        )

        if annotate_values:
            pred_val = float(merged.loc[merged["beat_num"] == beat, "Prediction Crimes"].iloc[0])
            txt = f"[{lo:.1f}, {hi:.1f}]  pred={pred_val:.1f}"
            axR.text(
                cx_i + 0.02 * span, cy_i, txt, fontsize=8, color="black",
                va="center", ha="left",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75)
            )

        if show_coverage and (beat in cov_map):
            cov = cov_map[beat]
            _draw_coverage_gauge(
                axR, (cx_i, cy_i), cov,
                r_in=r_in, r_out=r_out, span=span,
                # gap_center_deg=50,
                linecolor=coverage_color,
                lw=1.6, fontsize=9, bold=False,
                # start_offset=0.1,
                # end_offset=0.98,
                # curve_rad=0.05,
                arrow_scale=7, zorder=30
            )

    # Legend
    handles = [
        Line2D([0], [0], color="black", linestyle=inner_ls, lw=1.6, label="Lower (min)"),
        Line2D([0], [0], color="black", linestyle=outer_ls, lw=1.6, label="Upper (max)"),
        Line2D([0], [0], color="black", linestyle=":", lw=1.5, label="Zoom area (left)"),
    ]
    coverage_proxy = FancyArrowPatch((0, 0), (1, 0),
                                     arrowstyle="-|>", mutation_scale=12,
                                     linewidth=1.8, color=coverage_color)
    handles.append(coverage_proxy)
    labels = ["Lower (min)", "Upper (max)", "Zoom area (left)", "Coverage (%)"]
    axR.legend(handles=handles, labels=labels, loc=legend_loc, fontsize=8, frameon=True)

    if save_path:
        plt.gcf().savefig(save_path, bbox_inches="tight", dpi=300)
    if show:
        plt.show()

    return config