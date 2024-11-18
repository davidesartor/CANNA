import numpy as np
import torch
from tqdm.auto import tqdm
import corner
from mcmc import emcee_sample


def corner_plot(
    dataset,
    model,
    verbose=False,
    samples=32 * 1024,
    examples=1,
    ode_steps=8,
    plot_prior=False,
):
    dataset.dataloader(batch_size=samples, batches=examples)
    for i in tqdm(range(examples)):
        x0 = dataset.sample_params(samples)
        x1 = dataset.sample_params(1)
        y = dataset.sample_observation(x1).squeeze()

        # hack to flatten #TODO: add flatten params
        x0 = dataset.conditional_map(np.zeros(samples), x0, x0)
        x1 = dataset.conditional_map(np.ones(1), x1, x1).squeeze()

        # sample using CNF
        x_cnf = model.push(
            torch.as_tensor(x0), torch.as_tensor(y), verbose=verbose, n_steps=ode_steps
        )
        x_cnf = x_cnf.cpu().numpy()

        # sample using MCMC
        x_mcmc = emcee_sample(
            log_prob=lambda x: dataset.log_posterior(x, y),
            x0=x1,
            walkers=64,
            steps=len(x_cnf) // 64,
            burn=300,
            verbose=verbose,
        )

        # plot corner
        corner_kwargs: dict = dict(
            labels=dataset.parameter_names, show_titles=True, truths=x1
        )
        fig = None
        if plot_prior:
            fig = corner.corner(x0, color="black", **corner_kwargs)
        fig = corner.corner(x_cnf, color="blue", fig=fig, **corner_kwargs)
        fig = corner.corner(x_mcmc, color="green", fig=fig, **corner_kwargs)
