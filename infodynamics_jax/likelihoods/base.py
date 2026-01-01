# infodynamics_jax/likelihoods/base.py

_LIKELIHOOD_REGISTRY = {}


def register(name, likelihood):
    """
    Register a likelihood object or factory under a string key.
    """
    if name in _LIKELIHOOD_REGISTRY:
        raise KeyError(f"Likelihood '{name}' already registered.")
    _LIKELIHOOD_REGISTRY[name] = likelihood


def get(name):
    """
    Retrieve a likelihood by name.
    """
    try:
        return _LIKELIHOOD_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown likelihood '{name}'. "
            f"Available: {list(_LIKELIHOOD_REGISTRY.keys())}"
        )
