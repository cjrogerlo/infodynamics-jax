def sum_kernel(k1, k2):
    def k(params, X, Z):
        return k1(params["k1"], X, Z) + k2(params["k2"], X, Z)
    return k

def product_kernel(k1, k2):
    def k(params, X, Z):
        return k1(params["k1"], X, Z) * k2(params["k2"], X, Z)
    return k

def scale_kernel(k, scale_key="scale"):
    def ks(params, X, Z):
        return params[scale_key] * k(params, X, Z)
    return ks