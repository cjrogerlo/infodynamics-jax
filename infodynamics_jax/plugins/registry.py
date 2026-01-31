"""
Plugin registry and discovery system.

Supports two discovery mechanisms:
1. Entry points (recommended): Plugins register via pyproject.toml/setup.py
2. Explicit registration: Plugins call register_plugin() at import time

Entry point example (in private package's pyproject.toml):

    [project.entry-points."infodynamics_jax.plugins"]
    pro_data_stream = "hedge_fund_private.data:LOBDataStream"
    pro_obs_energy = "hedge_fund_private.likelihoods:StudentTObsEnergy"

Explicit registration example (in private package):

    from infodynamics_jax.plugins import register_plugin
    from .my_plugin import MyDataStream

    register_plugin('data_stream', 'proprietary', MyDataStream)
"""

import importlib.metadata
import warnings
from typing import Dict, Any, Type, Optional, Callable
from collections import defaultdict


class PluginRegistry:
    """
    Central registry for all plugins.

    Singleton pattern ensures consistency across the application.
    """

    _instance = None
    _plugins: Dict[str, Dict[str, Any]]  # {category: {name: plugin_class}}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._plugins = defaultdict(dict)
            cls._instance._discovered = False
        return cls._instance

    def register(
        self,
        category: str,
        name: str,
        plugin: Any,
        override: bool = False,
    ) -> None:
        """
        Register a plugin.

        Args:
            category: Plugin category (e.g., 'data_stream', 'obs_energy')
            name: Plugin name (e.g., 'proprietary_lob', 'public_csv')
            plugin: Plugin class or factory function
            override: If True, allow overriding existing plugins

        Raises:
            ValueError: If plugin already registered and override=False
        """
        if name in self._plugins[category] and not override:
            raise ValueError(
                f"Plugin '{name}' already registered in category '{category}'. "
                f"Use override=True to replace."
            )

        self._plugins[category][name] = plugin

    def get(self, category: str, name: str) -> Optional[Any]:
        """
        Get a specific plugin.

        Args:
            category: Plugin category
            name: Plugin name

        Returns:
            Plugin class/factory, or None if not found
        """
        return self._plugins[category].get(name)

    def get_all(self, category: str) -> Dict[str, Any]:
        """
        Get all plugins in a category.

        Returns:
            Dictionary of {name: plugin} for the category
        """
        return dict(self._plugins[category])

    def list_categories(self) -> list[str]:
        """List all plugin categories."""
        return list(self._plugins.keys())

    def list_plugins(self, category: str) -> list[str]:
        """List all plugin names in a category."""
        return list(self._plugins[category].keys())

    def discover_entry_points(self, group: str = "infodynamics_jax.plugins") -> int:
        """
        Discover plugins via entry points.

        This scans installed packages for entry points in the specified group.

        Args:
            group: Entry point group name

        Returns:
            Number of plugins discovered

        Example entry point in private package's pyproject.toml:
            [project.entry-points."infodynamics_jax.plugins"]
            data_stream.proprietary = "my_package.data:ProprietaryStream"
            obs_energy.student_t = "my_package.energy:StudentTEnergy"
        """
        count = 0

        try:
            # Python 3.10+ API
            entry_points = importlib.metadata.entry_points()
            if hasattr(entry_points, 'select'):
                # Python 3.10+
                plugins = entry_points.select(group=group)
            else:
                # Python 3.9 fallback
                plugins = entry_points.get(group, [])

            for ep in plugins:
                try:
                    # Parse entry point name: "category.name" or "category_name"
                    if '.' in ep.name:
                        category, name = ep.name.split('.', 1)
                    else:
                        # Default to using entry point name as both category and name
                        category = "misc"
                        name = ep.name

                    # Load the plugin
                    plugin_class = ep.load()

                    # Register it
                    self.register(category, name, plugin_class, override=False)
                    count += 1

                except Exception as e:
                    warnings.warn(
                        f"Failed to load plugin '{ep.name}' from entry point: {e}",
                        RuntimeWarning,
                    )

        except Exception as e:
            warnings.warn(
                f"Failed to discover entry points: {e}",
                RuntimeWarning,
            )

        self._discovered = True
        return count

    def discover_module(self, module_path: str) -> int:
        """
        Discover plugins by importing a module.

        The module should call register_plugin() at import time.

        Args:
            module_path: Dotted module path (e.g., 'my_package.plugins')

        Returns:
            Number of plugins registered during import

        Example usage:
            >>> registry.discover_module('hedge_fund_private.plugins')
        """
        before_count = sum(len(plugins) for plugins in self._plugins.values())

        try:
            importlib.import_module(module_path)
        except ImportError as e:
            warnings.warn(
                f"Could not import plugin module '{module_path}': {e}",
                RuntimeWarning,
            )
            return 0

        after_count = sum(len(plugins) for plugins in self._plugins.values())
        return after_count - before_count


# Global registry instance
_registry = PluginRegistry()


def get_registry() -> PluginRegistry:
    """Get the global plugin registry."""
    return _registry


def register_plugin(
    category: str,
    name: str,
    plugin: Any,
    override: bool = False,
) -> None:
    """
    Register a plugin (convenience function).

    Args:
        category: Plugin category (e.g., 'data_stream', 'obs_energy')
        name: Plugin name
        plugin: Plugin class or factory
        override: Allow overriding existing plugins
    """
    _registry.register(category, name, plugin, override=override)


def discover_plugins(
    entry_points: bool = True,
    modules: Optional[list[str]] = None,
) -> Dict[str, int]:
    """
    Discover all available plugins.

    Args:
        entry_points: If True, scan for entry point plugins
        modules: Optional list of module paths to import

    Returns:
        Dictionary of discovery results: {'entry_points': count, 'modules': count}

    Example:
        >>> from infodynamics_jax.plugins import discover_plugins
        >>> results = discover_plugins(
        ...     entry_points=True,
        ...     modules=['my_private_package.plugins']
        ... )
        >>> print(f"Found {results['entry_points']} entry point plugins")
    """
    results = {'entry_points': 0, 'modules': 0}

    if entry_points and not _registry._discovered:
        results['entry_points'] = _registry.discover_entry_points()

    if modules:
        for module in modules:
            results['modules'] += _registry.discover_module(module)

    return results


def load_plugin(
    category: str,
    name: str,
    fallback: Optional[str] = None,
    auto_discover: bool = True,
) -> Optional[Any]:
    """
    Load a plugin by category and name, with optional fallback.

    Args:
        category: Plugin category
        name: Plugin name
        fallback: Fallback plugin name if first choice not found
        auto_discover: If True, auto-discover plugins before loading

    Returns:
        Plugin class/factory, or None if not found

    Example:
        >>> # Try to load proprietary data stream, fallback to public CSV
        >>> StreamClass = load_plugin(
        ...     'data_stream',
        ...     'proprietary_lob',
        ...     fallback='public_csv'
        ... )
    """
    if auto_discover and not _registry._discovered:
        discover_plugins(entry_points=True)

    plugin = _registry.get(category, name)

    if plugin is None and fallback is not None:
        plugin = _registry.get(category, fallback)

    return plugin


# Auto-discover on first import (can be disabled via environment variable)
import os
if os.getenv('INFODYNAMICS_NO_AUTO_DISCOVER') != '1':
    try:
        discover_plugins(entry_points=True, modules=None)
    except Exception:
        # Silently fail - plugins are optional
        pass
