_REGISTRY = {}

def register(name, fn):
    _REGISTRY[name] = fn

def get(name):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown kernel '{name}'.")
    return _REGISTRY[name]