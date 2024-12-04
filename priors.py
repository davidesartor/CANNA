from jaxtyping import Array, Float, Scalar, Key
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy import stats
from flax import nnx

Param = Float[Array, "P"]
Trivial = Float[Array, "T"]


class ManifoldDistribution(nnx.Module):
    """Base class for prior distributions on manifolds
    Samples are of type P, representing a point on the manifold
    and allow a trivialized representation of type T

    example:
    a phase in S1 may be represented as a float in [0, 2pi],
    and allow a trivialized representation as a 2d vector of sin and cos
    in this case, P = float, T = 2d vector
    """

    def sample(self, rng: Key) -> Param:
        """Sample a value from the prior"""
        raise NotImplementedError

    def log_pdf(self, x: Param) -> Scalar:
        """Compute the log probability density of a given value"""
        raise NotImplementedError

    def geodesic(self, t: Scalar, x0: Param, x1: Param) -> Param:
        """Evaluate the geodesic connecting x0 and x1 at time t
        by default, this is a linear interpolation (euclidean metric)"""
        raise NotImplementedError

    def trivialization(self, x: Trivial) -> Param:
        """Get the parameter value corresponding to a trivialized representation
        i.e. go from a R^n representation to the manifold representation
        this map is surjective but not necessarily invertible"""
        raise NotImplementedError

    def trivialization_pinv(self, x: Param) -> Trivial:
        """Convert a parameter value to a trivialized representation that represents it
        i.e. go from the manifold representation to one of its R^n representation
        this is a pseudo-inverse, as the trivialization map may not be invertible"""
        raise NotImplementedError


class Normal(ManifoldDistribution):
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def sample(self, rng: Key) -> Param:
        return self.mean + self.std * jr.normal(rng, shape=(1,))

    def log_pdf(self, x: Param) -> Scalar:
        return stats.norm.logpdf(x, self.mean, self.std).squeeze(-1)

    def geodesic(self, t: Scalar, x0: Param, x1: Param) -> Param:
        return (1 - t) * x0 + t * x1

    def trivialization(self, x: Trivial) -> Param:
        """map from N(0,1) to N(mean, std)"""
        return self.mean + self.std * x

    def trivialization_pinv(self, x: Param) -> Trivial:
        """map from N(mean, std) to N(0,1)"""
        return (x - self.mean) / self.std


class Uniform(ManifoldDistribution):
    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

    def sample(self, rng: Key) -> Param:
        return jr.uniform(rng, shape=(1,), minval=self.low, maxval=self.high)

    def log_pdf(self, x: Param) -> Scalar:
        return stats.uniform.logpdf(x, self.low, self.high - self.low).squeeze(-1)

    def geodesic(self, t: Scalar, x0: Param, x1: Param) -> Param:
        return (1 - t) * x0 + t * x1

    def trivialization(self, x: Trivial) -> Param:
        """map from [-1, 1] to [low, high]"""
        return self.low + (x + 1.0) * (self.high - self.low) / 2

    def trivialization_pinv(self, x: Param) -> Trivial:
        """map from [low, high] to [-1, 1]"""
        return 2 * (x - self.low) / (self.high - self.low) - 1


class LogUniform(ManifoldDistribution):
    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

    def sample(self, rng: Key) -> Param:
        log_low, log_high = jnp.log(self.low), jnp.log(self.high)
        return jnp.exp(jr.uniform(rng, shape=(1,), minval=log_low, maxval=log_high))

    def log_pdf(self, x: Param) -> Scalar:
        log_low, log_high = jnp.log(self.low), jnp.log(self.high)
        p = stats.uniform.logpdf(jnp.log(x), log_low, log_high - log_low)
        return jnp.where(x < 0, -jnp.inf, p - jnp.log(x)).squeeze(-1)

    def geodesic(self, t: Scalar, x0: Param, x1: Param) -> Param:
        return jnp.exp((1 - t) * jnp.log(x0) + t * jnp.log(x1))

    def trivialization(self, x: Trivial) -> Param:
        """map from [-1, 1] to [low, high]"""
        log_low, log_high = jnp.log(self.low), jnp.log(self.high)
        return jnp.exp(log_low + (x + 1.0) * (log_high - log_low) / 2)

    def trivialization_pinv(self, x: Param) -> Trivial:
        """map from [low, high] to [-1, 1]"""
        log_low, log_high = jnp.log(self.low), jnp.log(self.high)
        return 2 * (jnp.log(x) - log_low) / (log_high - log_low) - 1


class PeriodicUniform(ManifoldDistribution):
    def __init__(self, scale: float):
        self.scale = scale

    def sample(self, rng: Key) -> Param:
        return jr.uniform(rng, shape=(1,), minval=0.0, maxval=self.scale * 2 * jnp.pi)

    def log_pdf(self, x: Param):
        return stats.uniform.logpdf(x, scale=self.scale * 2 * jnp.pi)

    def geodesic(self, t: Scalar, x0: Param, x1: Param) -> Param:
        logmap = jnp.arctan2(
            jnp.sin((x1 - x0) / self.scale), jnp.cos((x1 - x0) / self.scale)
        )
        xt = (x0 + t * logmap) % self.scale  # exponential map
        return xt

    def trivialization(self, x: Trivial) -> Param:
        """map from sin and cos to [0, scale]"""
        sin, cos = jnp.split(x, 2, axis=-1)
        return jnp.arctan2(sin, cos) * self.scale

    def trivialization_pinv(self, x: Param) -> Trivial:
        """map from [0, scale] to sin and cos"""
        return jnp.concatenate(
            [jnp.sin(x / self.scale), jnp.cos(x / self.scale)], axis=-1
        )
