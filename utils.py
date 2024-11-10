from tqdm.auto import tqdm
import corner
from mcmc import emcee_sample


def corner_plot(
    dataset,
    model,
    verbose=False,
    samples=32 * 1024,
    examples=1,
    ode_steps=4,
    plot_prior=False,
):
    for t, x0, x1, y in tqdm(dataset.dataloader(batch_size=samples, batches=examples)):
        y, x_true = y[0], x1[0]

        # sample using CNF
        x_cnf = model.push(x0, y, verbose=verbose, n_steps=ode_steps).cpu().numpy()

        # sample using MCMC
        x_mcmc = emcee_sample(
            log_prob=lambda x: dataset.log_posterior(x, y.numpy()),
            x0=x_true.numpy(),
            walkers=64,
            steps=len(x_cnf) // 64,
            burn=300,
            verbose=verbose,
        )

        # plot corner
        corner_kwargs: dict = dict(
            labels=dataset.parameter_names, show_titles=True, truths=x_true.numpy()
        )
        fig = None
        if plot_prior:
            fig = corner.corner(x0.numpy(), color="black", **corner_kwargs)
        fig = corner.corner(x_cnf, color="blue", fig=fig, **corner_kwargs)
        fig = corner.corner(x_mcmc, color="green", fig=fig, **corner_kwargs)
