from priors import *


class PosteriorFlowDataset[Observation](eqx.Module):
    priors: list[Prior]
    param_names: list[str]
    coupling_jitter: float

    def sample_params(self, rng: Key) -> list[Param]:
        rngs = jr.split(rng, len(self.priors))
        return [prior.sample(rng) for prior, rng in zip(self.priors, rngs)]

    def sample_observation(self, rng: Key, params: list[Param]) -> Observation:
        raise NotImplementedError

    def train_sample(self, rng: Key):
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

    def train_batch(self, rng: Key, batch_size: int):
        return jax.vmap(self.train_sample)(jr.split(rng, batch_size))

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

    def log_likelihood(self, params: list[Param], observation: Observation) -> Scalar:
        raise NotImplementedError

    def log_prior(self, params: list[Param]):
        return sum(p.log_pdf(x) for p, x in zip(self.priors, params))

    def log_posterior(self, params: list[Param], observation: Observation) -> Scalar:
        return self.log_prior(params) + self.log_likelihood(params, observation)


class PointDataset(PosteriorFlowDataset):
    priors: list[Prior]
    param_names: list[str]
    coupling_jitter: float

    noise_mean: Array
    noise_cov: Array

    def __init__(
        self,
        dim,
        prior_mean=0.0,
        prior_std=1.0,
        noise_std=0.1,
        coupling_jitter=0.0,
        seed=0,
    ):
        self.priors = [Normal(prior_mean, prior_std) for _ in range(dim)]
        self.param_names = [f"x{i}" for i in range(dim)]
        self.coupling_jitter = coupling_jitter

        self.noise_mean = jnp.zeros(dim)
        noise_cov_sqrt = noise_std * jr.uniform(jr.key(seed), (dim, dim))
        self.noise_cov = noise_cov_sqrt @ noise_cov_sqrt.T

    def sample_observation(self, rng: Key, params: list[Param]) -> Array:
        x = jnp.concatenate(params, axis=-1)
        noise = jr.multivariate_normal(rng, self.noise_mean, self.noise_cov)
        return x + noise

    def log_likelihood(self, params: list[Param], observation: Array):
        res = observation - jnp.concatenate(params, axis=-1)
        return stats.multivariate_normal.logpdf(res, self.noise_mean, self.noise_cov)


class SinusoidDataset(PosteriorFlowDataset):
    priors: list[Prior]
    param_names: list[str]
    coupling_jitter: float

    observation_times: Array
    noise_mean: Array
    noise_cov: Array

    def __init__(
        self,
        amp_prior=LogUniform(0.1, 10.0),
        omg_prior=LogUniform(0.01 * jnp.pi, 0.1 * jnp.pi),
        phi_prior=PeriodicUniform(2 * jnp.pi),
        sample_rate=1.0,
        observation_time=16.0,
        noise_std=0.1,
    ):
        self.param_names = ["A", "ω", "φ"]
        self.priors = [amp_prior, omg_prior, phi_prior]

        self.observation_times = jnp.arange(0.0, observation_time, sample_rate)
        self.noise_mean = jnp.zeros(len(self.observation_times))
        self.noise_cov = noise_std**2 * jnp.eye(len(self.observation_times))

    def clean_signal(self, params: list[Param]):
        amp, omg, phi = params
        t = self.observation_times
        return amp * jnp.sin(omg * t + phi)

    def sample_observation(self, rng: Key, params: list[Param]) -> Array:
        noise = jr.multivariate_normal(rng, self.noise_mean, self.noise_cov)
        return self.clean_signal(params) + noise

    def log_likelihood(self, params: list[Param], observation: Array):
        res = observation - self.clean_signal(params)
        return stats.multivariate_normal.logpdf(res, self.noise_mean, self.noise_cov)
