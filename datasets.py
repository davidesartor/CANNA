from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.stats import multivariate_normal


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
        if isinstance(prior_mean, (int, float)):
            prior_mean = prior_mean * np.ones(dim)
        if isinstance(prior_cov, (int, float)):
            prior_cov = prior_cov * np.eye(dim)
        if isinstance(noise_cov, (int, float)):
            noise_cov = np.sqrt(noise_cov) * np.random.rand(dim, dim)
            noise_cov = noise_cov @ noise_cov.T
        self.dim = dim
        self.prior_distr = multivariate_normal(prior_mean, prior_cov)  # type: ignore
        self.noise_distr = multivariate_normal(cov=noise_cov)

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
