from dataclasses import dataclass
from typing import Protocol
import numpy as np
from numpy.typing import ArrayLike, NDArray
from numpy.random import uniform, normal


class Prior(Protocol):
    def sample(self, n: int) -> NDArray:
        ...

    def log_pdf(self, x: NDArray) -> NDArray:
        ...

    def geodesic(self, t: NDArray, x0: NDArray, x1: NDArray) -> tuple[NDArray, NDArray]:
        xt = (1 - t) * x0 + t * x1
        dx = x1 - x0
        return xt, dx


@dataclass
class Normal(Prior):
    mean: float
    std: float

    def sample(self, n):
        return normal(self.mean, self.std, n)

    def log_pdf(self, x):
        return (
            -0.5 * np.log(2 * np.pi)
            + np.log(self.std)
            - 0.5 * ((x - self.mean) / self.std) ** 2
        )


@dataclass
class Uniform(Prior):
    low: float
    high: float

    def sample(self, n):
        return uniform(self.low, self.high, n)

    def log_pdf(self, x):
        return np.where(
            (self.low <= x) & (x <= self.high),
            -np.log(self.high - self.low),
            -np.inf,
        )


@dataclass
class Periodic(Uniform):
    def geodesic(self, t, x0, x1):
        assert (self.low, self.high) == (0.0, 2 * np.pi), "Custom Range Not implemented"
        dx = np.arctan2(np.sin(x1 - x0), np.cos(x1 - x0))  # log map
        xt = (x0 + t * dx) % (2 * np.pi)  # exponential map
        return xt, dx


@dataclass
class LogUniform(Prior):
    low: float
    high: float

    def sample(self, n):
        return np.exp(uniform(np.log(self.low), np.log(self.high), n))

    def log_pdf(self, x):
        return np.where(
            (self.low <= x) & (x <= self.high),
            -np.log(x * np.log(self.high / self.low)),
            -np.inf,
        )

    # def geodesic(self, t, x0, x1):
    #     xt = np.exp(np.log(x0) * (1 - t) + np.log(x1) * t)
    #     dx = xt * (np.log(x1) - np.log(x0))
    #     return xt, dx
