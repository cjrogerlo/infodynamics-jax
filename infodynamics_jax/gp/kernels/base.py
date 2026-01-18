# infodynamics_jax/gp/kernels/base.py

_LIKELIHOOD_REGISTRY = {}

def register(name: str, obj):
    if name in _LIKELIHOOD_REGISTRY:
        raise KeyError(f"Likelihood '{name}' already registered.")
    _LIKELIHOOD_REGISTRY[name] = obj

def get(name: str):
    try:
        return _LIKELIHOOD_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown likelihood '{name}'. "
            f"Available: {list(_LIKELIHOOD_REGISTRY.keys())}"
        )
