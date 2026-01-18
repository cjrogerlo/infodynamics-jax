"""
Plotting style configuration.
"""
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from typing import Dict, Optional, Tuple, Literal
import platform


# Color scheme
COLORS: Dict[str, str] = {
    'train': '#1f77b4',  # Blue
    'test': '#ff7f0e',   # Orange
    'true': '#2ca02c',   # Green
    'primary': '#1f77b4',  # Blue (for SMC)
    'secondary': '#d62728',  # Red (for ML-II)
    'tertiary': '#9467bd',  # Purple
}

# Color palettes
PALETTES: Dict[str, list] = {
    'main': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    'cool': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'],
    'warm': ['#FF6B6B', '#FFA07A', '#FFD93D', '#6BCF7F', '#4ECDC4'],
}


def setup_plot_style():
    """Setup matplotlib style with Chinese font support."""
    plt.style.use('default')
    matplotlib.rcParams['figure.dpi'] = 100
    matplotlib.rcParams['savefig.dpi'] = 100
    matplotlib.rcParams['font.size'] = 10
    matplotlib.rcParams['axes.labelsize'] = 10
    matplotlib.rcParams['axes.titlesize'] = 11
    matplotlib.rcParams['xtick.labelsize'] = 9
    matplotlib.rcParams['ytick.labelsize'] = 9
    matplotlib.rcParams['legend.fontsize'] = 9
    matplotlib.rcParams['figure.titlesize'] = 12
    
    # Setup Chinese font support
    # Try to find an available Chinese font on the system
    system = platform.system()
    
    # List of potential Chinese fonts, ordered by preference
    chinese_fonts = []
    if system == 'Darwin':  # macOS
        chinese_fonts = [
            'PingFang TC',      # 繁體中文
            'PingFang SC',      # 簡體中文
            'STHeiti',          # 華文黑體
            'Heiti TC',         # 黑體-繁體中文
            'Heiti SC',         # 黑體-簡體中文
            'Arial Unicode MS', # Arial Unicode (if installed)
        ]
    elif system == 'Linux':
        chinese_fonts = [
            'WenQuanYi Micro Hei',
            'WenQuanYi Zen Hei',
            'Noto Sans CJK TC',
            'Noto Sans CJK SC',
            'Droid Sans Fallback',
        ]
    elif system == 'Windows':
        chinese_fonts = [
            'Microsoft YaHei',
            'Microsoft JhengHei',
            'SimHei',
            'SimSun',
        ]
    
    # Get available fonts
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # Find the first available Chinese font
    chinese_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            chinese_font = font
            break
    
    # Set font family: use Chinese font if available, otherwise keep default
    if chinese_font:
        # Prepend Chinese font to sans-serif font list
        current_sans_serif = matplotlib.rcParams.get('font.sans-serif', ['Arial', 'DejaVu Sans'])
        if isinstance(current_sans_serif, str):
            current_sans_serif = [current_sans_serif]
        # Remove duplicates and ensure Chinese font is first
        font_list = [chinese_font] + [f for f in current_sans_serif if f != chinese_font]
        matplotlib.rcParams['font.sans-serif'] = font_list
    else:
        # If no Chinese font found, at least try to use a font that supports Unicode
        # This won't fix the issue but won't break anything either
        pass
    
    # Fix negative sign rendering issue with some fonts
    matplotlib.rcParams['axes.unicode_minus'] = False


def plot_with_uncertainty(
    ax,
    x: 'np.ndarray',
    mean: 'np.ndarray',
    std: 'np.ndarray',
    n_std: int = 2,
    color: str = 'C0',
    alpha_fill: float = 0.2,
    label_mean: Optional[str] = None,
    **kwargs
):
    """
    Plot mean with uncertainty band.
    
    Args:
        ax: Matplotlib axis
        x: X values
        mean: Mean values
        std: Standard deviation values
        n_std: Number of standard deviations for band
        color: Line color
        alpha_fill: Fill alpha
        label_mean: Label for mean line
        **kwargs: Additional arguments for plot
    """
    x = x.flatten()
    mean = mean.flatten()
    std = std.flatten()
    
    # Plot mean
    ax.plot(x, mean, color=color, label=label_mean, **kwargs)
    
    # Fill uncertainty band
    ax.fill_between(
        x,
        mean - n_std * std,
        mean + n_std * std,
        color=color,
        alpha=alpha_fill,
    )


def get_figure_size(
    style: Literal['wide', 'square', 'tall'] = 'wide',
    scale: float = 1.0
) -> Tuple[float, float]:
    """
    Get figure size based on style.
    
    Args:
        style: Figure style ('wide', 'square', 'tall')
        scale: Scale factor to multiply dimensions
    
    Returns:
        (width, height) tuple
    """
    base_sizes = {
        'wide': (12, 4),
        'square': (6, 6),
        'tall': (6, 8),
    }
    w, h = base_sizes.get(style, base_sizes['wide'])
    return (w * scale, h * scale)


def format_axes(
    ax,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend: bool = True,
    grid: bool = False,
):
    """
    Format axes with common settings.
    
    Args:
        ax: Matplotlib axis
        title: Title text
        xlabel: X-axis label
        ylabel: Y-axis label
        legend: Whether to show legend
        grid: Whether to show grid
    """
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if legend:
        ax.legend()
    if grid:
        ax.grid(True, alpha=0.3)


def create_custom_colormap(
    name: Literal['probability', 'uncertainty', 'diverging'] = 'probability'
) -> LinearSegmentedColormap:
    """
    Create custom colormap for specific use cases.
    
    Args:
        name: Colormap name
        
    Returns:
        Matplotlib colormap
    """
    if name == 'probability':
        # Blue to white colormap for probabilities
        colors = ['#ffffff', '#e3f2fd', '#90caf9', '#2196f3', '#1565c0']
        return LinearSegmentedColormap.from_list('probability', colors, N=256)
    elif name == 'uncertainty':
        # Yellow to red colormap for uncertainty
        colors = ['#fff9c4', '#fff176', '#ffeb3b', '#fbc02d', '#f57c00']
        return LinearSegmentedColormap.from_list('uncertainty', colors, N=256)
    elif name == 'diverging':
        # Blue-white-red diverging colormap
        colors = ['#1565c0', '#42a5f5', '#ffffff', '#ef5350', '#c62828']
        return LinearSegmentedColormap.from_list('diverging', colors, N=256)
    else:
        return plt.cm.viridis
