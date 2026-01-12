def test_imports():
    import infodynamics_jax

    from infodynamics_jax.core.phi import Phi
    from infodynamics_jax.gp.kernels.params import KernelParams
    from infodynamics_jax.energy import InertialEnergy
    from infodynamics_jax.gp.ansatz import VariationalState

    # kernels
    from infodynamics_jax.gp.kernels import get as get_kernel
    get_kernel("rbf")

    # likelihoods
    from infodynamics_jax.gp.likelihoods import get as get_likelihood
    get_likelihood("gaussian")
    get_likelihood("bernoulli")
