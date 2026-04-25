"""Small visualization helpers for Stage 3 feature/energy maps.

The Stage 3 analysis code imports this module from the repository root.  Keep
the functions dependency-light and tolerant of both torch tensors and numpy
arrays so they can be used from training/debug scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np


def _to_numpy_2d(values, name: str) -> np.ndarray:
    """Convert a tensor/array-like object into a finite 2D numpy array."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    arr = np.asarray(values, dtype=float)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D after squeeze, got shape {arr.shape}")
    return arr


def _finite_minmax(arrays: Iterable[np.ndarray]) -> Tuple[float, float]:
    vals = []
    for arr in arrays:
        finite = arr[np.isfinite(arr)]
        if finite.size:
            vals.append(finite)
    if not vals:
        return 0.0, 1.0
    merged = np.concatenate(vals)
    vmin = float(np.min(merged))
    vmax = float(np.max(merged))
    if np.isclose(vmin, vmax):
        pad = max(abs(vmin) * 0.05, 1e-6)
        return vmin - pad, vmax + pad
    return vmin, vmax


def _make_level_values(vmin: float, vmax: float, n_levels: int) -> np.ndarray:
    n_levels = max(3, int(n_levels))
    return np.linspace(vmin, vmax, n_levels)


def _build_mpl_cmap(cmap: Union[str, Sequence[str]]):
    """Accept a matplotlib cmap name or a custom color sequence."""
    import matplotlib.colors as mcolors

    if isinstance(cmap, str):
        return cmap
    colors = list(cmap)
    if len(colors) < 2:
        raise ValueError("Custom cmap color sequence must contain at least 2 colors.")
    return mcolors.LinearSegmentedColormap.from_list("custom_contour_cmap", colors)


def _draw_gt_marker(ax, gt_coords: Optional[Sequence[float]], shape: Tuple[int, int]):
    from matplotlib.patches import Circle

    if gt_coords is None:
        return
    nr, nc = float(gt_coords[0]), float(gt_coords[1])
    n_rows, n_cols = shape
    if not (-0.5 <= nr <= n_rows - 0.5 and -0.5 <= nc <= n_cols - 0.5):
        return

    max_dim = max(n_rows, n_cols)
    radii = [0.06 * max_dim, 0.12 * max_dim]
    for radius in radii:
        ax.add_patch(
            Circle(
                (nc, nr),
                radius=radius,
                fill=False,
                edgecolor="#f4e84a",
                linewidth=1.4,
                alpha=0.95,
                zorder=6,
            )
        )
    ax.scatter(
        [nc],
        [nr],
        marker="*",
        s=180,
        facecolor="#f4e84a",
        edgecolor="#1f1f1f",
        linewidth=1.2,
        zorder=7,
    )


def _normalize_for_argmode(field: np.ndarray, flow_mode: str) -> np.ndarray:
    """Return a scalar field where gradient ascent moves in the requested mode."""
    if str(flow_mode).lower() in {"descent", "min", "minimize"}:
        return -field
    return field


def _trace_gradient_path(
    field: np.ndarray,
    start_rc: Optional[Sequence[float]],
    n_steps: int = 18,
    step_size: Optional[float] = None,
    flow_mode: str = "ascent",
) -> Optional[np.ndarray]:
    """Trace a simple normalized gradient path on a 2D scalar grid."""
    n_rows, n_cols = field.shape
    if start_rc is None:
        flat_idx = int(np.nanargmin(field) if str(flow_mode).lower() in {"descent", "min", "minimize"} else np.nanargmax(field))
        row, col = np.unravel_index(flat_idx, field.shape)
        pos = np.array([float(row), float(col)], dtype=float)
    else:
        pos = np.array([float(start_rc[0]), float(start_rc[1])], dtype=float)

    if step_size is None:
        step_size = max(n_rows, n_cols) / 45.0

    drive = _normalize_for_argmode(np.nan_to_num(field, nan=np.nanmedian(field)), flow_mode)
    grad_r, grad_c = np.gradient(drive)
    path = [pos.copy()]

    for _ in range(max(1, int(n_steps))):
        r = int(np.clip(round(pos[0]), 0, n_rows - 1))
        c = int(np.clip(round(pos[1]), 0, n_cols - 1))
        direction = np.array([grad_r[r, c], grad_c[r, c]], dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            break
        pos = pos + direction / norm * float(step_size)
        pos[0] = np.clip(pos[0], 0, n_rows - 1)
        pos[1] = np.clip(pos[1], 0, n_cols - 1)
        path.append(pos.copy())

    return np.asarray(path, dtype=float) if len(path) > 1 else None


def _draw_flow_path(
    ax,
    field: np.ndarray,
    start_coords: Optional[Sequence[float]],
    flow_mode: str,
    color: str = "white",
):
    path = _trace_gradient_path(field, start_coords, flow_mode=flow_mode)
    if path is None:
        return
    ax.plot(
        path[:, 1],
        path[:, 0],
        color=color,
        linewidth=4.2,
        alpha=0.75,
        solid_capstyle="round",
        zorder=8,
    )
    ax.plot(
        path[:, 1],
        path[:, 0],
        color=color,
        linewidth=2.0,
        alpha=0.98,
        marker="o",
        markersize=2.2,
        zorder=9,
    )


def plot_contour(
    dist_ingp,
    dist_proj=None,
    gt_coords: Optional[Sequence[float]] = None,
    show_gt_marker: bool = True,
    crop_size: int = 0,
    with_flow: bool = False,
    flow_mode: str = "ascent",
    unified_scale: bool = False,
    save_path: Optional[str] = None,
    titles: Optional[Sequence[str]] = None,
    cmap: Union[str, Sequence[str]] = "coolwarm",
    n_fill_levels: int = 64,
    n_line_levels: int = 18,
    contour_line_color: Union[str, Sequence[str]] = "#2b2b2b",
    contour_line_width: float = 0.65,
    contour_line_alpha: float = 0.55,
    dpi: int = 220,
    show: bool = False,
    flow_start_coords: Optional[Sequence[float]] = None,
):
    """Plot one or two 2D fields as filled contour maps.

    Args:
        dist_ingp: First 2D map. Existing callers pass INGP energy here.
        dist_proj: Optional second 2D map. Existing callers pass projector energy.
        gt_coords: Ground-truth location as ``(row, col)`` in array index space.
        show_gt_marker: Whether to draw the yellow star and yellow rings.
        crop_size: If positive, crop a square around ``gt_coords`` before plotting.
        with_flow: Draw a simple gradient path on top of the first panel.
        flow_mode: ``"ascent"`` follows increasing values; ``"descent"`` follows
            decreasing values.
        unified_scale: Share color limits across both maps.
        save_path: PNG/PDF/SVG path. Parent directories are created.
        titles: Optional panel titles.
        cmap: Matplotlib colormap name or custom color sequence, e.g.
            ``["#ffffff", "#3a3a3a"]``.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    first = _to_numpy_2d(dist_ingp, "dist_ingp")
    fields = [first]
    if dist_proj is not None:
        fields.append(_to_numpy_2d(dist_proj, "dist_proj"))

    if crop_size and gt_coords is not None:
        crop_size = int(crop_size)
        nr, nc = int(round(float(gt_coords[0]))), int(round(float(gt_coords[1])))
        r0 = max(0, nr - crop_size)
        r1 = min(first.shape[0], nr + crop_size + 1)
        c0 = max(0, nc - crop_size)
        c1 = min(first.shape[1], nc + crop_size + 1)
        fields = [arr[r0:r1, c0:c1] for arr in fields]
        gt_plot = (float(gt_coords[0]) - r0, float(gt_coords[1]) - c0)
        flow_start_plot = None
        if flow_start_coords is not None:
            flow_start_plot = (float(flow_start_coords[0]) - r0, float(flow_start_coords[1]) - c0)
    else:
        gt_plot = gt_coords
        flow_start_plot = flow_start_coords

    n_panels = len(fields)
    if titles is None:
        titles = [None] * n_panels

    if unified_scale:
        shared_vmin, shared_vmax = _finite_minmax(fields)
    else:
        shared_vmin = shared_vmax = None

    mpl_cmap = _build_mpl_cmap(cmap)

    fig_w = 4.6 * n_panels
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 2.45), squeeze=False)
    axes = axes.ravel()

    for ax, field, title in zip(axes, fields, titles):
        n_rows, n_cols = field.shape
        x = np.arange(n_cols)
        y = np.arange(n_rows)
        xx, yy = np.meshgrid(x, y)

        if unified_scale:
            vmin, vmax = shared_vmin, shared_vmax
        else:
            vmin, vmax = _finite_minmax([field])
        fill_levels = _make_level_values(vmin, vmax, n_fill_levels)
        line_levels = _make_level_values(vmin, vmax, n_line_levels)

        ax.contourf(xx, yy, field, levels=fill_levels, cmap=mpl_cmap, vmin=vmin, vmax=vmax)
        ax.contour(
            xx,
            yy,
            field,
            levels=line_levels,
            colors=contour_line_color,
            linewidths=float(contour_line_width),
            alpha=float(contour_line_alpha),
        )
        if show_gt_marker:
            _draw_gt_marker(ax, gt_plot, field.shape)
        if with_flow:
            _draw_flow_path(ax, field, flow_start_plot, flow_mode=flow_mode)

        if title:
            ax.set_title(str(title), fontsize=10)
        ax.set_xlim(0, n_cols - 1)
        ax.set_ylim(n_rows - 1, 0)
        ax.set_aspect("auto")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#303030")

    fig.tight_layout(pad=0.35, w_pad=1.4)

    if save_path:
        path = Path(save_path)
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=int(dpi), bbox_inches="tight", pad_inches=0.02)
    if show:
        plt.show()
    plt.close(fig)
    return fig


def vis_griddata_in_3d_surface_interactive(
    griddata,
    p2save: Optional[str] = None,
    colorscale: Union[str, Sequence[str]] = "Viridis",
    show_axis_info: bool = True,
):
    """Save a 2D grid as an interactive Plotly 3D surface."""
    import plotly.graph_objects as go

    data = _to_numpy_2d(griddata, "griddata")
    n_rows, n_cols = data.shape
    x = np.arange(n_cols)
    y = np.arange(n_rows)

    fig = go.Figure(
        data=[
            go.Surface(
                x=x,
                y=y,
                z=data,
                colorscale=colorscale,
                colorbar=dict(title="Value"),
                hovertemplate="col=%{x}<br>row=%{y}<br>value=%{z:.5f}<extra></extra>",
            )
        ]
    )
    axis_cfg = dict(visible=bool(show_axis_info))
    fig.update_layout(
        width=900,
        height=700,
        margin=dict(l=0, r=0, b=0, t=30),
        scene=dict(
            xaxis=dict(title="NC", **axis_cfg),
            yaxis=dict(title="NR", autorange="reversed", **axis_cfg),
            zaxis=dict(title="Value", **axis_cfg),
            aspectmode="manual",
            aspectratio=dict(x=1, y=max(n_rows / max(n_cols, 1), 0.2), z=0.45),
        ),
    )

    if p2save:
        path = Path(p2save)
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    return fig


def vis_girddata_in_3d_surface(
    griddata,
    p2save: Optional[str] = None,
    cmap: Union[str, Sequence[str]] = "viridis",
    show_axis_info: bool = True,
    dpi: int = 180,
):
    """Save a 2D grid as a static matplotlib 3D surface.

    The function name keeps the historical typo used by old scripts.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    data = _to_numpy_2d(griddata, "griddata")
    n_rows, n_cols = data.shape
    x = np.arange(n_cols)
    y = np.arange(n_rows)
    xx, yy = np.meshgrid(x, y)

    mpl_cmap = _build_mpl_cmap(cmap)

    fig = plt.figure(figsize=(7.0, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(xx, yy, data, cmap=mpl_cmap, linewidth=0, antialiased=True)
    fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08)

    if show_axis_info:
        ax.set_xlabel("NC")
        ax.set_ylabel("NR")
        ax.set_zlabel("Value")
    else:
        ax.set_axis_off()
    ax.invert_yaxis()
    fig.tight_layout()

    if p2save:
        path = Path(p2save)
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=int(dpi), bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
    return fig


__all__ = [
    "plot_contour",
    "vis_griddata_in_3d_surface_interactive",
    "vis_girddata_in_3d_surface",
]
