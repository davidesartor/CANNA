from datasets import *
import optax
from flax import linen as nn
from tqdm import tqdm


class MLPCNF(nn.Module):
    dim: int = 512
    depth: int = 4
    norm: bool = True

    @nn.compact
    def __call__(
        self, t: Scalar, xt: Float[Array, "xt_flat"], y: Float[Array, "y_flat"]
    ) -> Float[Array, "xt_flat"]:
        h = jnp.concatenate([t[..., None], xt, y], axis=-1)
        for _ in range(self.depth - 1):
            h = nn.Dense(self.dim)(h)
            h = nn.silu(h)
            if self.norm:
                h = nn.RMSNorm()(h)
        h = nn.Dense(xt.shape[-1])(h)
        return h

    def loss(self, batch):
        t, xt, dx, y = batch
        pred = self(t, xt, y)
        return optax.l2_loss(pred, dx).mean()

    def push(self, x0: Float[Array, "n x"], y: Float[Array, "y"], n_steps=4):
        def runje_kutta_step(flow, x, t):
            k1 = flow(t, x, y)
            k2 = flow(t + dt / 2, x + k1 * dt / 2, y)
            k3 = flow(t + dt / 2, x + k2 * dt / 2, y)
            k4 = flow(t + dt, x + k3 * dt, y)
            next_x = x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
            return next_x, x

        dt = 1 / n_steps
        ts = dt * jnp.arange(n_steps)
        x1, xt = nn.scan(runje_kutta_step, variable_broadcast="params")(self, x0, ts)
        return x1, xt

    @nn.nowrap
    def fit(
        self,
        dataset: PosteriorFlowDataset,
        epochs=10,
        batch_size=1024,
        steps_per_epoch=256,
        learning_rate=3e-4,
        weight_decay=1e-4,
        seed=0,
    ):
        rng = jr.key(seed)
        rng, rng_init = jr.split(rng)
        params = self.init(rng_init, dataset.train_sample(jr.key(0)), method="loss")
        optimizer = optax.adamw(learning_rate, weight_decay=weight_decay)
        opt_state = optimizer.init(params)

        @jax.jit
        def generate_batches(rng):
            get_batch = lambda rng: dataset.train_batch(rng, batch_size)
            return jax.vmap(get_batch)(jr.split(rng, steps_per_epoch))

        @jax.jit
        def train_epoch(params, opt_state, batches):
            def optimization_step(carry, batch):
                @jax.value_and_grad
                def loss_fn(params, batch):
                    return self.apply(params, batch, method="loss")

                params, opt_state = carry
                loss, grads = loss_fn(params, batch)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                carry = (params, opt_state)
                return carry, loss

            (params, opt_state), losses = jax.lax.scan(
                optimization_step, (params, opt_state), batches
            )
            return params, opt_state, losses

        loss_log = []
        for rng in (pbar := tqdm(jr.split(rng, epochs))):
            batches = generate_batches(rng)
            params, opt_state, losses = train_epoch(params, opt_state, batches)
            pbar.set_postfix(loss=losses.mean())
            loss_log.append(losses)
        return params, jnp.array(loss_log)
