import torch
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
        # Return t, x_t, d_x, y
        t = np.random.rand(batch_size)
        x0 = self.sample_params(batch_size)
        x1 = self.sample_params(batch_size)
        y = self.sample_observation(x1)
        x_t, dx = self.conditional_map(t, x0, x1)

        to_float = lambda x: x.astype(np.float32)
        return tuple(map(to_float, (t, x_t, dx, y, x0, x1)))

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
    
    def conditional_map(self, t, x0, x1):
        #! Precompute oracoli in dataset
        # Return x_t, dx.
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
        sample_rate=1,
        observation_time=16.0,
        noise_cov=0.1,
        amp_range=(0.1, 10),
        omg_range=(0.01 * np.pi, 0.1 * np.pi),
        phi_range=(-np.pi, np.pi),
        coupling_jitter=0.0
    ):
        self.observation_times = np.arange(0.0, observation_time, sample_rate)
        self.amp_prior = loguniform(*amp_range)
        self.omg_prior = loguniform(*omg_range)
        self.phi_prior = uniform(*phi_range)
        noise_cov = noise_cov * np.eye(len(self.observation_times))
        self.noise_distr = multivariate_normal(cov=noise_cov)
        self.coupling_jitter = coupling_jitter

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
        return amp * np.sin(omg * t + phi)

    def sample_observation(self, x):
        noise = self.noise_distr.rvs(len(x))
        return self.clean_signal(x) + noise

    def log_prior(self, x):
        amp, omg, phi = np.split(x, 3, axis=-1)
        log_prior_amp = self.amp_prior.logpdf(amp)
        log_prior_omg = self.omg_prior.logpdf(omg)
        log_prior_phi = self.phi_prior.logpdf(phi)
        return log_prior_amp + log_prior_omg + log_prior_phi

    def log_likelihood(self, x, y):
        return self.noise_distr.logpdf(y - self.clean_signal(x))
    
    def conditional_map_derivative(self, t, x0, x1):
        # Vogliamo una lista [topo_param_1, ..., topo_param_n]
        #? Forse meglio fare una sola funzione conditional map
        #? since we have to transport depending on the topology
        return x1 - x0

    def conditional_map(self, t, x0, x1):
        #! Precompute oracoli in dataset
        x_jitter = self.coupling_jitter * np.random.randn(*x0.shape)
        while t.ndim < x0.ndim:
            t = np.expand_dims(t, axis=-1)
        d_x = self.conditional_map_derivative(t, x0, x1)
        xt = x0 + t * d_x + x_jitter
        return xt, d_x
    


class RiemmannianSinusoidDataset(PosteriorFlowDataset):
    def __init__(
        self,
        sample_rate=1,
        observation_time=16.0,
        noise_cov=0.1,
        amp_range=(0.1, 10),
        omg_range=(0.01 * np.pi, 0.1 * np.pi),
        phi_range=(-np.pi, np.pi),
        coupling_jitter=0.0
    ):
        self.observation_times = np.arange(0.0, observation_time, sample_rate)
        self.amp_prior = loguniform(*amp_range)
        self.omg_prior = loguniform(*omg_range)
        self.phi_prior = uniform(*phi_range)
        noise_cov = noise_cov * np.eye(len(self.observation_times))
        self.noise_distr = multivariate_normal(cov=noise_cov)
        self.coupling_jitter = coupling_jitter

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
        return amp * np.sin(omg * t + phi)

    def sample_observation(self, x):
        noise = self.noise_distr.rvs(len(x))
        return self.clean_signal(x) + noise

    def log_prior(self, x):
        amp, omg, phi = np.split(x, 3, axis=-1)
        log_prior_amp = self.amp_prior.logpdf(amp)
        log_prior_omg = self.omg_prior.logpdf(omg)
        log_prior_phi = self.phi_prior.logpdf(phi)
        return log_prior_amp + log_prior_omg + log_prior_phi

    def log_likelihood(self, x, y):
        return self.noise_distr.logpdf(y - self.clean_signal(x))

    def exponential_map(self, x, u):
        """
        exp_x: T_x M -> M
        """
        # Sphere
        # norm_u = np.linalg.norm(u)
        # return x * np.cos(norm_u) + (u/norm_u) * np.sin(norm_u)
        # Flat torus https://github.com/facebookresearch/riemannian-fm/blob/main/manifm/manifolds/torus.py#L4
        return (x + u) % (2 * np.pi)

    def log_map(self, x, y):
        """
        log_x: M -> T_x M
        """
        # Sphere
        # theta = np.arccos(np.inner(x,y))
        # return (theta / np.sin(theta)) * (y - np.cos(theta)*x)
        # Flat torus https://github.com/facebookresearch/riemannian-fm/blob/main/manifm/manifolds/torus.py#L4
        return np.arctan2(np.sin(y - x), np.cos(y - x))


    def conditional_map_derivative(self, t, x0, x1):
        amp_0, omg_0, phi_0 = np.split(x0, 3, axis=-1)
        amp_1, omg_1, phi_1 = np.split(x1, 3, axis=-1)
        d_amp = amp_1 - amp_0
        d_omg = omg_1 - omg_0
        d_phi = self.log_map(phi_0, phi_1) #% (2 * np.pi)

        return d_amp, d_omg, d_phi

    def conditional_map(self, t, x0, x1):
        #! Precompute oracoles in dataset
        x_jitter = self.coupling_jitter * np.random.randn(*x0.shape)
        while t.ndim < x0.ndim:
            t = np.expand_dims(t, axis=-1)

        # Note that it is important to compute phi_t before d_phi
        amp_0, omg_0, phi_0 = np.split(x0, 3, axis=-1)
        amp_1, omg_1, phi_1 = np.split(x1, 3, axis=-1)

        #? In the paper phi_0 and phi_1 are reversed?
        phi_t = self.exponential_map(phi_0, t*self.log_map(phi_0, phi_1))

        d_amp, d_omg, d_phi = self.conditional_map_derivative(t, x0, x1)
        amp_t = amp_0 + t * d_amp# + x_jitter
        omg_t = omg_0 + t * d_omg# + x_jitter


        x_t = np.stack([amp_t, omg_t, phi_t], axis=-1).squeeze(1)
        d_x = np.stack([d_amp, d_omg, d_phi], axis=-1).squeeze(1)

        return x_t, d_x
    

