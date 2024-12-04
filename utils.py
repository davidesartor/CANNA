from tqdm.auto import tqdm
import corner
from mcmc import emcee_sample
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr


def corner_plot(
    dataset,
    model,
    verbose=False,
    samples=32 * 1024,
    examples=1,
    ode_steps=8,
    plot_prior=False,
):
    @jax.jit
    def sample(rng):
        rng_x0, rng_x1, rng_y = jr.split(rng, 3)
        x0 = jax.vmap(dataset.sample_params)(jr.split(rng_x0, samples))
        x1 = dataset.sample_params(rng_x1)
        y = dataset.sample_observation(rng_y, x1)

        x0_triv = jnp.concat(jax.vmap(dataset.as_trivial)(x0), axis=-1)
        return x0, x1, y, x0_triv

    for i in tqdm(range(examples)):
        x0, x1, y, x0_triv = sample(jr.key(i))
        truths = np.array(jnp.concat(x1, axis=-1))

        # sample using CNF
        x_cnf = np.array(model.push(x0_triv, y, n_steps=ode_steps))

        # # sample using MCMC
        # x_mcmc = emcee_sample(
        #     log_prob=lambda x: dataset.log_posterior(x, y),
        #     x0=x1,
        #     walkers=64,
        #     steps=len(x_cnf) // 64,
        #     burn=300,
        #     verbose=verbose,
        # )

        # plot corner
        corner_kwargs: dict = dict(
            labels=dataset.param_names, show_titles=True, truths=truths
        )
        fig = None
        if plot_prior:
            fig = corner.corner(x0, color="black", **corner_kwargs)
        fig = corner.corner(x_cnf, color="blue", fig=fig, **corner_kwargs)
        # fig = corner.corner(x_mcmc, color="green", fig=fig, **corner_kwargs)
