"""
Plugin system for infodynamics-jax.

This module provides a plugin architecture that allows:
1. Public core library to define abstract interfaces
2. Private/proprietary implementations to extend functionality
3. Runtime discovery of available plugins
4. Clean separation between open-source and proprietary code

Design principles:
- Public core defines ONLY interfaces (abstract base classes)
- Private implementations register via entry points or explicit imports
- No hard dependencies on private code in public core
- Plugins can be discovered at runtime without breaking core functionality

Example usage:
    >>> from infodynamics_jax.plugins import registry
    >>> # Discover all available data stream implementations
    >>> streams = registry.get_plugins('data_stream')
    >>> # Use proprietary implementation if available, fallback to public
    >>> StreamClass = streams.get('proprietary', streams['public_default'])
"""

from .registry import (
    PluginRegistry,
    get_registry,
    register_plugin,
    discover_plugins,
    load_plugin,
)
from .interfaces import (
    DataStreamPlugin,
    ObservationEnergyPlugin,
    DynamicsEnergyPlugin,
    RiskEvaluatorPlugin,
)

__all__ = [
    # Registry functions
    'PluginRegistry',
    'get_registry',
    'register_plugin',
    'discover_plugins',
    'load_plugin',
    # Base interfaces
    'DataStreamPlugin',
    'ObservationEnergyPlugin',
    'DynamicsEnergyPlugin',
    'RiskEvaluatorPlugin',
]
