# infodynamics_jax/kernels/composite.py
def sum_kernel(k1, k2):
    def k(X, Z, params):
        return k1(X, Z, params) + k2(X, Z, params)
    return k

def product_kernel(k1, k2):
    def k(X, Z, params):
        return k1(X, Z, params) * k2(X, Z, params)
    return k

def scale_kernel(k, scale):
    def k_scaled(X, Z, params):
        return scale * k(X, Z, params)
    return k_scaled
