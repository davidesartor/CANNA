from datasets import *
import optax
from tqdm import tqdm


class MLPCNF(eqx.Module):
    linear_layers: list[eqx.nn.Linear]
    norm_layers: list[eqx.nn.RMSNorm | eqx.nn.Identity]

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden: int = 512,
        depth: int = 4,
        norm: bool = True,
        *,
        rng: Key,
    ):
        dims = [x_dim + y_dim + 1] + [hidden for _ in range(depth - 1)] + [x_dim]
        self.norm_layers = [
            eqx.nn.RMSNorm(hidden) if norm else eqx.nn.Identity()
            for _ in range(depth - 1)
        ]
        self.linear_layers = [
            eqx.nn.Linear(in_dim, out_dim, key=k)
            for in_dim, out_dim, k in zip(dims[:-1], dims[1:], jr.split(rng, depth))
        ]

    def __call__(
        self, t: Scalar, xt: Float[Array, "xt_flat"], y: Float[Array, "y_flat"]
    ) -> Float[Array, "xt_flat"]:
        h = jnp.concatenate([t[..., None], xt, y], axis=-1)
        for linear, norm in zip(self.linear_layers[:-1], self.norm_layers):
            h = linear(h)
            h = jax.nn.silu(h)
            h = norm(h)
        h = self.linear_layers[-1](h)
        return h

    def push(self, x0: Float[Array, "x"], y: Float[Array, "y"], n_steps=4):
        def runje_kutta_step(x, t):
            k1 = self(t, x, y)
            k2 = self(t + dt / 2, x + k1 * dt / 2, y)
            k3 = self(t + dt / 2, x + k2 * dt / 2, y)
            k4 = self(t + dt, x + k3 * dt, y)
            next_x = x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
            return next_x, x

        dt = 1 / n_steps
        ts = dt * jnp.arange(n_steps)
        x1, xt = jax.lax.scan(runje_kutta_step, x0, ts)
        return x1, xt

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
        optimizer = optax.adamw(learning_rate, weight_decay=weight_decay)
        opt_state = optimizer.init(eqx.filter(self, eqx.is_array))

        @jax.jit
        def generate_batches(rng):
            get_batch = lambda rng: dataset.train_batch(rng, batch_size)
            return jax.vmap(get_batch)(jr.split(rng, steps_per_epoch))

        @jax.value_and_grad
        def loss_fn(flow, batch):
            t, xt, dx, y = batch
            pred = jax.vmap(flow)(t, xt, y)
            return optax.l2_loss(pred, dx).mean()

        @eqx.filter_jit
        def train_epoch(flow, opt_state, batches):
            def optimization_step(carry, batch):
                flow, opt_state = carry
                loss, grads = loss_fn(flow, batch)
                updates, opt_state = optimizer.update(grads, opt_state, flow)
                flow = eqx.apply_updates(flow, updates)
                carry = (flow, opt_state)
                return carry, loss

            (flow, opt_state), losses = jax.lax.scan(
                optimization_step, (flow, opt_state), batches
            )
            return flow, opt_state, losses

        loss_log = []
        for rng in (pbar := tqdm(jr.split(rng, epochs))):
            batches = generate_batches(rng)
            self, opt_state, losses = train_epoch(self, opt_state, batches)
            pbar.set_postfix(loss=losses.mean())
            loss_log.append(losses)
        return self, jnp.array(loss_log)
