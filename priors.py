from jaxtyping import Array, Float, Key
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy import stats
import equinox as eqx

Scalar = Float[Array, ""]
Param = Float[Array, "P"]


class Prior(eqx.Module):
    """Base class for prior distributions on manifolds"""

    def sample(self, rng: Key) -> Param:
        """Sample a value from the prior"""
        raise NotImplementedError

    def logpdf(self, x: Param) -> Scalar:
        """Compute the log probability density of a given value"""
        raise NotImplementedError

    def geodesic(self, t: Scalar, x0: Param, x1: Param) -> Param:
        """Evaluate the geodesic connecting x0 and x1 at time t"""
        raise NotImplementedError

    def step(self, x: Param, dx: Param, t: Scalar) -> Param:
        """Take a step in the direction of dx from x0"""
        raise NotImplementedError


class Normal(Prior):
    mean: float
    std: float

    def sample(self, rng: Key) -> Param:
        return self.mean + self.std * jr.normal(rng, shape=(1,))

    def logpdf(self, x: Param) -> Scalar:
        return stats.norm.logpdf(x, self.mean, self.std).squeeze(-1)

    def geodesic(self, t: Scalar, x0: Param, x1: Param) -> Param:
        return (1 - t) * x0 + t * x1

    def step(self, x: Param, dx: Array, t: Scalar) -> Param:
        return x + t * dx


class Uniform(Prior):
    low: float
    high: float

    def sample(self, rng: Key) -> Param:
        return jr.uniform(rng, shape=(1,), minval=self.low, maxval=self.high)

    def logpdf(self, x: Param) -> Scalar:
        return stats.uniform.logpdf(x, self.low, self.high - self.low).squeeze(-1)

    def geodesic(self, t: Scalar, x0: Param, x1: Param) -> Param:
        return (1 - t) * x0 + t * x1

    def step(self, x: Param, dx: Array, t: Scalar) -> Param:
        return x + t * dx


class LogUniform(Prior):
    low: float
    high: float

    def sample(self, rng: Key) -> Param:
        log_low, log_high = jnp.log(self.low), jnp.log(self.high)
        return jnp.exp(jr.uniform(rng, shape=(1,), minval=log_low, maxval=log_high))

    def logpdf(self, x: Param) -> Scalar:
        log_low, log_high = jnp.log(self.low), jnp.log(self.high)
        p = stats.uniform.logpdf(jnp.log(x), log_low, log_high - log_low)
        return jnp.where(x < 0, -jnp.inf, p - jnp.log(x)).squeeze(-1)

    def geodesic(self, t: Scalar, x0: Param, x1: Param) -> Param:
        return jnp.exp((1 - t) * jnp.log(x0) + t * jnp.log(x1))

    def step(self, x: Param, dx: Array, t: Scalar) -> Param:
        # This is probably not the correct step for a log-uniform distribution
        return x + t * dx


class PeriodicUniform(Prior):
    high: float

    def sample(self, rng: Key) -> Param:
        return jr.uniform(rng, shape=(1,), minval=0.0, maxval=self.high)

    def logpdf(self, x: Param):
        return stats.uniform.logpdf(x, scale=self.high).squeeze(-1)

    def geodesic(self, t: Scalar, x0: Param, x1: Param) -> Param:
        s = 2 * jnp.pi / self.high
        logmap = jnp.arctan2(jnp.sin((x1 - x0) * s), jnp.cos((x1 - x0) * s))
        xt = (x0 + t * logmap) % self.high  # exponential map
        return xt

    def step(self, x: Param, dx: Array, t: Scalar) -> Param:
        return (x + t * dx) % self.high
