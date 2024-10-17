from typing import Callable
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


def log_uniform(key, range):
    min, max = (jnp.log(el) for el in range)
    return jnp.exp(jr.uniform(key, minval=min, maxval=max))


def uniform(key, range):
    min, max = range
    return jr.uniform(key, minval=min, maxval=max)


class SingleSinusoidDataset(eqx.Module):
    amp_range: tuple[float, float] = (1e0, 1e1)
    omg_range: tuple[float, float] = (2 * jnp.pi * 1e0, 2 * jnp.pi * 1e1)
    phi_range: tuple[float, float] = (-jnp.pi, jnp.pi)

    total_observation_time: float = 1.0
    sampling_rate: float = 1 / 1024
    noise_psd: Callable = lambda f: jnp.ones_like(f)

    def observation_times(self, ndim_batch=0):
        T, dt = self.total_observation_time, self.sampling_rate
        t = jnp.arange(0.0, T, dt)
        return jnp.expand_dims(t, range(ndim_batch))

    def observation_frequencies(self, ndim_batch=0):
        T, dt = self.total_observation_time, self.sampling_rate
        f = jnp.abs(jnp.fft.fftfreq(int(T / dt), dt)[: int(T / 2) + 1])
        return jnp.expand_dims(f, range(ndim_batch))

    def sample_params(self, key):
        key_amp, key_omg, key_phi = jr.split(key, 3)
        amp = log_uniform(key_amp, self.amp_range)
        omg = log_uniform(key_omg, self.omg_range)
        phi = uniform(key_phi, self.phi_range)
        return jnp.array([amp, omg, phi])

    def sample_noise(self, key, freq_domain=False):
        f = self.observation_frequencies()
        freq_scaling = jnp.sqrt(self.noise_psd(f))
        white_noise_time = jr.normal(key, (len(self.observation_times()),))
        white_noise_freq = jnp.fft.rfft(white_noise_time)
        colored_noise_freq = (white_noise_freq * freq_scaling)[..., None]
        colored_noise_time = jnp.fft.irfft(colored_noise_freq, axis=-2)
        return colored_noise_freq if freq_domain else colored_noise_time

    def clean_signal(self, params, freq_domain=False):
        amp, omg, phi = jnp.split(params, 3, axis=-1)
        t = self.observation_times(ndim_batch=params.ndim - 1)
        h = amp * jnp.sin(omg * t + phi)
        if freq_domain:
            h = jnp.fft.rfft(h, axis=-1)
        return h[..., None]

    def datastream(self, key, clean_signal):
        noise = self.sample_noise(key)
        return clean_signal + noise

    def get_item(self, key):
        key_params, key_data = jr.split(key)
        params = self.sample_params(key_params)
        clean = self.clean_signal(params)
        noisy = self.datastream(key_data, clean)
        return params, clean, noisy

    @eqx.filter_jit
    def get_batch(self, key, batch_size):
        return jax.vmap(self.get_item)(jr.split(key, batch_size))

    def log_likelihood(self, params, datastream):
        theoretical = self.clean_signal(params)
        residuals = jnp.abs(datastream - theoretical) ** 2
        log_lk = -0.5 * residuals.sum(-1).sum(-1)
        return log_lk

        datastream_freq = jnp.fft.rfft(datastream, axis=-2)
        theoretical_freq = self.clean_signal(params, freq_domain=True)
        residuals_squared = jnp.abs(datastream_freq - theoretical_freq) ** 2

        f = self.observation_frequencies(params.ndim - 1)
        freq_scaling = self.noise_psd(f)[..., None]
        residuals_scaled = residuals_squared / freq_scaling
        log_lk = -0.5 * jnp.mean(residuals_scaled.sum(-1)[..., 1:], axis=-1)
        return log_lk

    def log_prior(self, params):
        is_in_range = lambda x, r: (r[0] < x) * (x < r[1])
        amp, omg, phi = jnp.split(params, 3, axis=-1)
        log_lk = -jnp.log(amp) - jnp.log(omg) + 0.0 * phi

        valid = (
            is_in_range(amp, self.amp_range)
            * is_in_range(omg, self.omg_range)
            * is_in_range(phi, (-2 * jnp.pi, 2 * jnp.pi))
        )
        log_lk = jnp.where(valid, log_lk, -jnp.inf)
        return log_lk
