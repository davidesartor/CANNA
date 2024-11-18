from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.stats import multivariate_normal
from priors import Prior, Normal, Uniform, LogUniform, Periodic


@dataclass
class PosteriorFlowDataset(Dataset):
    priors: list[Prior]
    coupling_jitter: float = 0.0

    def __len__(self):
        return 1

    def dataloader(self, batch_size, batches=256, **kwargs):
        sampler = [batch_size] * batches
        return DataLoader(self, batch_size=None, sampler=sampler, **kwargs)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(f"x_{i}" for i, _ in enumerate(self.priors))

    def __getitem__(self, batch_size):
        t = np.random.rand(batch_size)
        x0 = self.sample_params(batch_size)
        x1 = self.sample_params(batch_size)
        y = self.sample_observation(x1)

        xt, dx = self.conditional_map(t, x0, x1, get_derivative=True)
        xt += self.coupling_jitter * np.random.randn(*xt.shape)
        return t, xt, dx, y

    def conditional_map(self, t, x0, x1, get_derivative=False):
        xt, dx = [], []
        for prior, a, b in zip(self.priors, x0, x1):
            map, derivative = prior.geodesic(t, a, b)
            xt.append(map)
            dx.append(derivative)
        xt = np.stack(xt, axis=-1)
        dx = np.stack(dx, axis=-1)
        return (xt, dx) if get_derivative else xt

    def sample_params(self, batch_size):
        return tuple(prior.sample(batch_size) for prior in self.priors)

    def log_prior(self, params):
        return sum(distr.log_pdf(x) for distr, x in zip(self.priors, params))

    def sample_observation(self, params):
        raise NotImplementedError

    def log_likelihood(self, params, observation):
        raise NotImplementedError

    def log_posterior(self, params, observation):
        return self.log_prior(params) + self.log_likelihood(params, observation)


class PointDataset(PosteriorFlowDataset):
    def __init__(self, dim, prior_mean=0.0, prior_std=1.0, noise_cov=0.1):
        self.priors = [Normal(prior_mean, prior_std) for _ in range(dim)]
        noise_cov = np.sqrt(noise_cov) * np.random.rand(dim, dim)
        noise_cov = noise_cov @ noise_cov.T
        self.noise = multivariate_normal(cov=noise_cov)

    def sample_observation(self, x):
        x = np.stack(x, axis=-1)
        noise = self.noise.rvs(len(x))
        return x + noise

    def log_likelihood(self, x, y):
        x = np.stack(x, axis=-1)
        return self.noise.logpdf(y - x)


class SinusoidDataset(PosteriorFlowDataset):
    def __init__(
        self,
        amp_prior: Prior = LogUniform(0.1, 10),
        omg_prior: Prior = LogUniform(0.01 * np.pi, 0.1 * np.pi),
        phi_prior: Prior = Periodic(0.0, 2 * np.pi),
        sample_rate=1,
        observation_time=16.0,
        noise_cov=0.1,
    ):
        self.observation_times = np.arange(0.0, observation_time, sample_rate)
        self.priors = [amp_prior, omg_prior, phi_prior]
        noise_cov = noise_cov * np.eye(len(self.observation_times))
        self.noise = multivariate_normal(cov=noise_cov)  # type: ignore

    @property
    def parameter_names(self):
        return "A", "ω", "φ"

    def clean_signal(self, x):
        amp, omg, phi = np.split(x, 3, axis=-1)
        t = self.observation_times
        return amp * np.sin(omg * t + phi)

    def sample_observation(self, x):
        x = np.stack(x, axis=-1)
        return self.clean_signal(x) + self.noise.rvs(len(x))

    def log_likelihood(self, x, y):
        return self.noise.logpdf(y - self.clean_signal(x))
