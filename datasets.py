from typing import Callable
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.stats import multivariate_normal
from priors import *


class PosteriorFlowDataset[Observation](nnx.Module):
    def __init__(
        self,
        priors: dict[str, ManifoldDistribution],
        coupling_jitter: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.rngs = rngs
        self.priors = list(priors.values())
        self.param_names = list(priors.keys())
        self.coupling_jitter = coupling_jitter

    def sample_params(self, rng: Key) -> list[Param]:
        rngs = jr.split(rng, len(self.priors))
        return [p.sample(rng) for p, rng in zip(self.priors, rngs)]

    def sample_observation(self, rng: Key, params: list[jax.Array]) -> jax.Array:
        raise NotImplementedError

    def get_train_sample(self, rng: Key):
        rng_t, rng_x0, rng_x1, rng_y, rng_jitter = jr.split(rng, 5)
        t = jr.uniform(rng_t)
        x0 = self.sample_params(rng_x0)
        x1 = self.sample_params(rng_x1)
        y = self.sample_observation(rng_y, x1)

        def flat_cond_map(t):
            xt = self.conditional_map(t, x0, x1)
            xt = jnp.concat([x for x in self.as_trivial(xt)], axis=-1)
            return xt

        xt = flat_cond_map(t)
        xt = xt + self.coupling_jitter * jr.normal(rng_jitter, xt.shape)

        dx = jax.jacobian(flat_cond_map)(t)
        return t, xt, dx, y

    def as_trivial(self, params: list[Param]) -> list[Trivial]:
        return [
            prior.trivialization_pinv(x).flatten()
            for prior, x in zip(self.priors, params)
        ]

    def as_params(self, trivializations: list[Trivial]) -> list[Param]:
        return [
            prior.trivialization(x).flatten()
            for prior, x in zip(self.priors, trivializations)
        ]

    def conditional_map(
        self, t: Scalar, x0: list[Param], x1: list[Param]
    ) -> list[Param]:
        return [prior.geodesic(t, x0, x1) for prior, x0, x1 in zip(self.priors, x0, x1)]

    def flow_input(self, t: Scalar, x0: list[Param], x1: list[Param]) -> jax.Array:
        xt = self.conditional_map(t, x0, x1)
        return jnp.concatenate([x for x in self.as_trivial(xt)], axis=-1)

    def log_likelihood(self, params: list[Param], observation: Observation):
        raise NotImplementedError

    def log_prior(self, params: list[Param]):
        return sum(p.log_pdf(x) for p, x in zip(self.priors, params))

    def log_posterior(self, params: list[Param], observation: Observation):
        return self.log_prior(params) + self.log_likelihood(params, observation)


class PointDataset(PosteriorFlowDataset):
    def __init__(
        self, dim, prior_mean=0.0, prior_std=1.0, noise_std=0.1, *, rngs: nnx.Rngs
    ):
        super().__init__(
            priors={f"x{i}": Normal(prior_mean, prior_std) for i in range(dim)},
            rngs=rngs,
        )
        self.noise_mean = nnx.Param(jnp.zeros(dim))
        noise_cov_sqrt = noise_std * jr.uniform(self.rngs(), (dim, dim))
        self.noise_cov = nnx.Param(noise_cov_sqrt @ noise_cov_sqrt.T)

    def sample_observation(self, rng: Key, params: list[Param]) -> jax.Array:
        x = jnp.concatenate(params, axis=-1)
        noise = jr.multivariate_normal(rng, self.noise_mean.value, self.noise_cov.value)
        return x + noise

    def log_likelihood(self, params: list[Param], observation: jax.Array):
        res = observation - jnp.concatenate(params, axis=-1)
        return multivariate_normal.logpdf(
            res, self.noise_mean.value, self.noise_cov.value
        )


class SinusoidDataset(PosteriorFlowDataset):
    def __init__(
        self,
        amp_range=(0.1, 10.0),
        omg_range=(0.01 * jnp.pi, 0.1 * jnp.pi),
        sample_rate=1.0,
        observation_time=16.0,
        noise_std=0.1,
        *,
        rngs: nnx.Rngs,
    ):
        priors = {
            "A": LogUniform(*amp_range),
            "ω": LogUniform(*omg_range),
            "φ": PeriodicUniform(2 * jnp.pi),
        }
        super().__init__(priors=priors, rngs=rngs)
        self.observation_times = nnx.Param(
            jnp.arange(0.0, observation_time, sample_rate)
        )
        self.noise_mean = nnx.Param(jnp.zeros(len(self.observation_times)))
        self.noise_cov = nnx.Param(
            noise_std**2 * jnp.eye(len(self.observation_times))
        )

    def clean_signal(self, params: list[Param]):
        amp, omg, phi = params
        t = self.observation_times
        return amp * jnp.sin(omg * t + phi)

    def sample_observation(self, rng: Key, params: list[Param]) -> jax.Array:
        noise = jr.multivariate_normal(rng, self.noise_mean.value, self.noise_cov.value)
        return self.clean_signal(params) + noise

    def log_likelihood(self, params: list[Param], observation: jax.Array):
        res = observation - self.clean_signal(params)
        return multivariate_normal.logpdf(
            res, self.noise_mean.value, self.noise_cov.value
        )
