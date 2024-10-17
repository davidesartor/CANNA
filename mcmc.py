from typing import Callable
import numpy as np
import emcee


def emcee_sample(
    log_prob: Callable,
    x0: np.ndarray,
    x0_noise: float = 1e-3,
    walkers: int = 64,
    burn: int = 300,
    steps: int = 1024,
    flatten: bool = True,
    verbose: bool = True,
):
    if verbose:
        print("Running MCMC sampling")
    x0 = x0 * np.random.normal(1, x0_noise, size=(walkers, *x0.shape))
    sampler = emcee.EnsembleSampler(walkers, x0.shape[-1], log_prob)
    sampler.run_mcmc(x0, burn, progress=verbose, tune=True)
    state = sampler.get_chain()
    if state is None:
        raise ValueError("Initial run failed, try again")
    state = state[-1, :, :]
    sampler.reset()
    if verbose:
        print("Finished initial run, burn-in dropped and starting real run")
    sampler.run_mcmc(state, steps, progress=verbose, tune=True)
    x = sampler.get_chain()
    if x is None:
        raise ValueError("Real run failed, try again")
    return x if not flatten else x.reshape(-1, x.shape[-1])
