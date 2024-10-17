import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn


class FCBlock(nn.Module):
    expand_factor: int = 4
    depth: int = 1

    @nn.compact
    def __call__(self, x):
        *B, C = x.shape
        for _ in range(self.depth):
            x = nn.Dense(C * self.expand_factor)(x)
            x = nn.silu(x)
        x = nn.Dense(C)(x)
        return x


class ZeroInitDense(nn.Dense):
    kernel_init: nn.initializers.Initializer = nn.initializers.zeros
    bias_init: nn.initializers.Initializer = nn.initializers.zeros


class AdaLayerNorm(nn.Module):
    @nn.compact
    def __call__(self, x, scale, shift):
        x = nn.LayerNorm(use_scale=False, use_bias=False)(x)
        return x * (1 + scale) + shift


class Modulation(nn.Module):
    dim: int
    n: int = 1

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.dim)(x)
        x = nn.silu(x)
        x = ZeroInitDense(self.dim * self.n)(x)
        return jnp.split(x, self.n, axis=-1)


class DiTBlock(nn.Module):
    heads: int

    @nn.compact
    def __call__(self, x, c):
        scale1, shift1, gate1, scale2, shift2, gate2 = (
            m[..., None, :] for m in Modulation(dim=x.shape[-1], n=6)(c)
        )
        attention = nn.MultiHeadAttention(self.heads)
        x = x + gate1 * attention(AdaLayerNorm()(x, scale1, shift1))
        x = x + gate2 * FCBlock()(AdaLayerNorm()(x, scale2, shift2))
        return x


class DownSample(nn.Module):
    factor: int = 2

    @nn.compact
    def __call__(self, x):
        return nn.Conv(x.shape[-1], self.factor, self.factor)(x)


class Patchify(nn.Module):
    dim: int
    patch_size: int = 16

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.dim, self.patch_size // 2, self.patch_size // 2)(x)
        x = nn.silu(x)
        x = nn.Conv(self.dim, 4, 2)(x)
        return x


class DePatchify(nn.Module):
    out_channels: int
    patch_size: int = 16

    @nn.compact
    def __call__(self, x):
        *B, T, D = x.shape
        x = nn.Conv(D, 1)(x)
        x = nn.silu(x)
        x = nn.Conv(self.out_channels * self.patch_size, 1)(x)
        return x.reshape((*B, -1, self.out_channels))


class PosEmbed(nn.Module):
    embed_dim: int
    max_period: int = 10000

    def positional_embedding(self, k):
        log_freqs = -np.log(self.max_period) * np.arange(0, 1, 2 / self.embed_dim)
        x = k[:, None] * np.exp(log_freqs)[None, :]
        x = jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1)
        return x

    @nn.compact
    def __call__(self, x):
        *B, L, C = x.shape
        x = self.positional_embedding(jnp.arange(L))
        if len(B) > 0:
            x = jnp.expand_dims(x, list(range(len(B))))
        return x


class TimeEmbed(PosEmbed):
    embed_dim: int
    max_period: int = 100

    @nn.compact
    def __call__(self, t):
        x = self.positional_embedding(t * self.max_period)
        x = FCBlock()(x)
        return x


class ParamEmbed(nn.Module):
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.embed_dim)(x)
        x = nn.silu(x)
        x = FCBlock()(x)
        return x
