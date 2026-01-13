"""Utility modules for examples."""
from .plotting_utils import compute_metrics
from .plotting_style import (
    COLORS,
    PALETTES,
    setup_plot_style,
    plot_with_uncertainty,
    get_figure_size,
    format_axes,
    create_custom_colormap,
)
from .synthetic_functions import synthetic

__all__ = [
    "compute_metrics",
    "COLORS",
    "PALETTES",
    "setup_plot_style",
    "plot_with_uncertainty",
    "get_figure_size",
    "format_axes",
    "create_custom_colormap",
    "synthetic",
]
