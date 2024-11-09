from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.stats import multivariate_normal, loguniform, uniform


class PosteriorFlowDataset(Dataset):
    def __len__(self):
        return 1

    def dataloader(self, batch_size, batches=256, **kwargs):
        sampler = [batch_size] * batches
        return DataLoader(self, batch_size=None, sampler=sampler, **kwargs)

    def __getitem__(self, batch_size):
        t = np.random.rand(batch_size)
        x0 = self.sample_params(batch_size)
        x1 = self.sample_params(batch_size)
        y = self.sample_observation(x1)
        to_float = lambda x: x.astype(np.float32)
        return tuple(map(to_float, (t, x0, x1, y)))

    def log_posterior(self, params, observation):
        log_prior = self.log_prior(params)
        log_likelihood = self.log_likelihood(params, observation)
        return log_prior + log_likelihood

    @property
    def parameter_names(self) -> tuple[str, ...]:
        raise NotImplementedError

    def sample_params(self, batch_size):
        raise NotImplementedError

    def sample_observation(self, params):
        raise NotImplementedError

    def log_prior(self, params):
        raise NotImplementedError

    def log_likelihood(self, params, observation):
        raise NotImplementedError


class PointDataset(PosteriorFlowDataset):
    def __init__(self, dim, prior_mean=0.0, prior_cov=1.0, noise_cov=0.1):
        prior_mean = prior_mean * np.ones(dim)
        prior_cov = prior_cov * np.eye(dim)
        noise_cov = np.sqrt(noise_cov) * np.random.rand(dim, dim)
        noise_cov = noise_cov @ noise_cov.T
        self.dim = dim
        self.prior_distr = multivariate_normal(prior_mean, prior_cov)
        self.noise_distr = multivariate_normal(cov=noise_cov)

    @property
    def parameter_names(self):
        return tuple(f"x{i}" for i in range(self.dim))

    def sample_params(self, batch_size):
        x = self.prior_distr.rvs(batch_size)
        return x

    def sample_observation(self, x):
        noise = self.noise_distr.rvs(len(x))
        return x + noise

    def log_prior(self, x):
        return self.prior_distr.logpdf(x)

    def log_likelihood(self, x, y):
        return self.noise_distr.logpdf(y - x)


class SinusoidDataset(PosteriorFlowDataset):
    def __init__(
        self,
        sample_rate=1.0,
        observation_time=256.0,
        noise_cov=1.0,
        amp_range=(0.1, 10),
        omg_range=(0.01 * np.pi, 0.1 * np.pi),
        phi_range=(-np.pi, np.pi),
    ):
        self.observation_times = np.arange(0.0, observation_time, sample_rate)
        self.amp_prior = loguniform(*amp_range)
        self.omg_prior = loguniform(*omg_range)
        self.phi_prior = uniform(*phi_range)
        noise_cov = noise_cov * np.eye(len(self.observation_times))
        self.noise_distr = multivariate_normal(cov=noise_cov)

    @property
    def parameter_names(self):
        return "A", "ω", "φ"

    def sample_params(self, batch_size):
        amp = self.amp_prior.rvs(batch_size)
        omg = self.omg_prior.rvs(batch_size)
        phi = self.phi_prior.rvs(batch_size)
        return np.stack([amp, omg, phi], axis=-1)

    def clean_signal(self, x):
        amp, omg, phi = np.split(x, 3, axis=-1)
        t = self.observation_times
        y = amp * np.sin(omg * t + phi)
        return y

    def sample_observation(self, x):
        noise = self.noise_distr.rvs(len(x))
        y = self.clean_signal(x) + noise
        return y[..., None]

    def log_prior(self, x):
        amp, omg, phi = np.split(x, 3, axis=-1)
        log_prior_amp = self.amp_prior.logpdf(amp)
        log_prior_omg = self.omg_prior.logpdf(omg)
        log_prior_phi = self.phi_prior.logpdf(phi)
        return log_prior_amp + log_prior_omg + log_prior_phi

    def log_likelihood(self, x, y):
        return self.noise_distr.logpdf(y[..., 0] - self.clean_signal(x))
