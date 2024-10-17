from typing import Callable, NamedTuple
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import flax.linen as nn
import optax
from tqdm import tqdm


class Batch(NamedTuple):
    t: jax.Array
    x0: jax.Array
    x1: jax.Array
    d: jax.Array


class CNF(nn.Module):
    @nn.compact
    def __call__(self, t, x, d):
        raise NotImplementedError

    def target_flow(self, x0, x1):
        return x1 - x0

    def phi_t(self, t, x0, x1):
        delta_x = self.target_flow(x0, x1)
        t = jnp.expand_dims(t, axis=range(1, delta_x.ndim))
        xt = x0 + delta_x * t
        return xt

    def loss(self, key, batch, coupling_sigma=1e-8):
        t, x0, x1, d = batch
        target_flow = self.target_flow(x0, x1)
        xt_mean = self.phi_t(t, x0, x1)
        xt = xt_mean + coupling_sigma * jr.normal(key, x0.shape)
        flow = self(t, xt, d)
        loss_flow = jnp.mean(optax.l2_loss(flow, target_flow))
        return loss_flow

    def push(self, x0, d, n_steps=16):
        xt, dt = [x0], 1 / n_steps
        d = jnp.tile(d, (x0.shape[0], *(1 for _ in d.shape)))
        for t in jnp.arange(0, 1, 1 / n_steps):
            t = t * jnp.ones((x0.shape[0],))
            flow = self(t, xt[-1], d)
            xt.append(xt[-1] + dt * flow)
        return jnp.stack(xt, axis=0)


class CLIPCNF(CNF):
    @nn.compact
    def __call__(self, t, x, d, x1=None):
        raise NotImplementedError

    def loss(self, key, batch, coupling_sigma=1e-3):
        t, x0, x1, d = batch
        target_flow = self.target_flow(x0, x1)
        xt_mean = self.phi_t(t, x0, x1)
        xt = xt_mean + coupling_sigma * jr.normal(key, x0.shape)
        (flow, x1_emb, d_emb, temp) = self(t, xt, d, x1)
        loss_flow = jnp.mean(optax.l2_loss(flow, target_flow))
        loss_clip = self.clip_loss(x1_emb, d_emb, temp)
        return loss_flow + 0.01 * loss_clip

    def clip_loss(self, x_emb, d_emb, temp):
        # make unitaty L2 norm embeddings
        x_emb = x_emb / jnp.linalg.norm(x_emb, axis=-1, keepdims=True)
        d_emb = d_emb / jnp.linalg.norm(d_emb, axis=-1, keepdims=True)
        similarity = jnp.dot(x_emb, d_emb.T) * temp
        assert similarity.shape[-2] == similarity.shape[-1]

        cross_entropy_fn = optax.softmax_cross_entropy_with_integer_labels
        targets = jnp.arange(similarity.shape[-1])
        loss_x = cross_entropy_fn(similarity, targets)
        loss_d = cross_entropy_fn(similarity.T, targets)
        return jnp.mean((loss_d + loss_x) / 2)


class Trainer(eqx.Module):
    get_batch_fn: Callable
    optimizer: optax.GradientTransformation
    epochs: int
    epoch_steps: int
    batch_size: int
    tabulate: bool = False

    def init_state(self, key, model: nn.Module):
        batch = self.get_batch_fn(key, self.batch_size)
        params = model.init(key, key, batch, method="loss")
        opt_state = self.optimizer.init(params)
        if self.tabulate:
            print(model.tabulate(key, key, batch, method="loss", depth=1))
        return (params, opt_state)

    def fit(self, model: CNF, *, key=jr.key(0), params=None, opt_state=None):
        def train_step(state, key):
            params, opt_state = state
            key_batch, key_loss = jr.split(key)
            batch = self.get_batch_fn(key_batch, self.batch_size)

            loss_fn = lambda par: model.apply(par, key_loss, batch, method="loss")
            loss, grads = jax.value_and_grad(loss_fn)(params)

            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss

        @eqx.filter_jit
        def train_epoch(state, key):
            keys = jr.split(key, self.epoch_steps)
            state, losses = jax.lax.scan(train_step, state, keys)
            return state, losses

        key_init, *keys_epochs = jr.split(key, self.epochs + 1)
        state = self.init_state(key_init, model)
        state = (params or state[0], opt_state or state[1])

        log = []
        for key in (pbar := tqdm(keys_epochs)):
            state, losses = train_epoch(state, key)
            log.append(losses)
            pbar.set_postfix(loss=(log[-1].mean()))

        params, opt_state = state
        log = jnp.stack(log, axis=0)
        return params, opt_state, log
