def test_imports():
    import infodynamics_jax

    from infodynamics_jax.core.phi import Phi
    from infodynamics_jax.kernels.params import KernelParams
    from infodynamics_jax.energy import InertialEnergy
    from infodynamics_jax.energy.expected import VariationalState

    # kernels
    from infodynamics_jax.kernels import get as get_kernel
    get_kernel("rbf")

    # likelihoods
    from infodynamics_jax.likelihoods import get as get_likelihood
    get_likelihood("gaussian")
    get_likelihood("bernoulli")
