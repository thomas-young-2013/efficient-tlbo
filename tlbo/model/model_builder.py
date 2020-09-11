import numpy as np
from .gp_mcmc import GaussianProcessMCMC
from .gp import GaussianProcess
from .rf_with_instances import RandomForestWithInstances
from .gp_kernels import ConstantKernel, Matern, HammingKernel, WhiteKernel
from .gp_base_prior import HorseshoePrior, LognormalPrior
from .util_funcs import get_rng, get_types


def build_model(model_str, config_space, types, bounds, seed, rng):
    if model_str == 'rf':
        model = RandomForestWithInstances(configspace=config_space,
                                          types=types, bounds=bounds,
                                          seed=seed)
    elif model_str == 'gp_mcmc':
        model = create_gp_model(config_space=config_space,
                                types=types,
                                bounds=bounds,
                                rng=rng)
    else:
        raise ValueError("Invalid model str %s!" % model_str)

    return model


def create_gp_model(config_space, types, bounds, rng=None):
    """
        Construct the Gaussian process model that is capable of dealing with categorical hyperparameters.
    """
    if rng is None:
        _, rng = get_rng(rng)

    cov_amp = ConstantKernel(
        2.0,
        constant_value_bounds=(np.exp(-10), np.exp(2)),
        prior=LognormalPrior(mean=0.0, sigma=1.0, rng=rng),
    )

    cont_dims = np.nonzero(types == 0)[0]
    cat_dims = np.nonzero(types != 0)[0]

    if len(cont_dims) > 0:
        exp_kernel = Matern(
            np.ones([len(cont_dims)]),
            [(np.exp(-6.754111155189306), np.exp(0.0858637988771976)) for _ in range(len(cont_dims))],
            nu=2.5,
            operate_on=cont_dims,
        )

    if len(cat_dims) > 0:
        ham_kernel = HammingKernel(
            np.ones([len(cat_dims)]),
            [(np.exp(-6.754111155189306), np.exp(0.0858637988771976)) for _ in range(len(cat_dims))],
            operate_on=cat_dims,
        )

    noise_kernel = WhiteKernel(
        noise_level=1e-8,
        noise_level_bounds=(np.exp(-25), np.exp(2)),
        prior=HorseshoePrior(scale=0.1, rng=rng),
    )

    if len(cont_dims) > 0 and len(cat_dims) > 0:
        # both
        kernel = cov_amp * (exp_kernel * ham_kernel) + noise_kernel
    elif len(cont_dims) > 0 and len(cat_dims) == 0:
        # only cont
        kernel = cov_amp * exp_kernel + noise_kernel
    elif len(cont_dims) == 0 and len(cat_dims) > 0:
        # only cont
        kernel = cov_amp * ham_kernel + noise_kernel
    else:
        raise ValueError()

    seed = rng.randint(0, 2 ** 20)
    model = GaussianProcessMCMC(config_space, types, bounds, seed, kernel, normalize_y=True)
    return model
