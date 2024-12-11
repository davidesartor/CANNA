from datasets import *
import optax
from tqdm import tqdm


class TimestepEmbedder(eqx.Module):
    lin1: eqx.nn.Linear
    lin2: eqx.nn.Linear
    freqs: Array

    def __init__(self, dim: int, frequencies: int = 256, max_period=1000, *, rng: Key):
        rng_lin1, rng_lin2 = jr.split(rng, 2)
        self.lin1 = eqx.nn.Linear(frequencies, dim, key=rng_lin1)
        self.lin2 = eqx.nn.Linear(dim, dim, key=rng_lin2)
        self.freqs = jnp.exp(
            -jnp.log(max_period) * jnp.linspace(0, 1, frequencies // 2)
        )

    def __call__(self, t: Scalar):
        x = t[..., None] * self.freqs
        x = jnp.concat([jnp.cos(x), jnp.sin(x)], axis=-1)
        x = self.lin2(jax.nn.silu(self.lin1(x)))
        return x


class ObsEmbedder(eqx.Module):
    lin1: eqx.nn.Linear
    lin2: eqx.nn.Linear

    def __init__(self, y_dim, dim: int, *, rng: Key):
        rng_lin1, rng_lin2 = jr.split(rng, 2)
        self.lin1 = eqx.nn.Linear(y_dim, dim, key=rng_lin1)
        self.lin2 = eqx.nn.Linear(dim, dim, key=rng_lin2)

    def __call__(self, y: Observation):
        x = self.lin2(jax.nn.silu(self.lin1(y)))
        return x


class ResBlock(eqx.Module):
    norm: eqx.nn.LayerNorm
    adaLN: eqx.nn.Linear
    lin1: eqx.nn.Linear
    lin2: eqx.nn.Linear

    def __init__(self, dim: int, expand: int = 2, *, rng: Key):
        rng_norm, rng_lin1, rng_lin2 = jr.split(rng, 3)
        self.norm = eqx.nn.LayerNorm(dim, use_weight=False, use_bias=False)
        self.adaLN = eqx.nn.Linear(dim, 3 * dim, key=rng_norm)
        self.adaLN = eqx.tree_at(
            lambda x: x.weight, self.adaLN, replace_fn=lambda x: x * 0.0
        )
        self.lin1 = eqx.nn.Linear(dim, expand * dim, key=rng_lin1)
        self.lin2 = eqx.nn.Linear(expand * dim, dim, key=rng_lin2)

    def __call__(self, x, y):
        shift, scale, gate = jnp.split(self.adaLN(y), 3, axis=-1)
        h = shift + (1 + scale) * self.norm(x)
        h = self.lin2(jax.nn.silu(self.lin1(h)))
        return x + gate * h


class MLPCNF(eqx.Module):
    t_emb: TimestepEmbedder
    y_emb: ObsEmbedder
    in_proj: eqx.nn.Linear
    res_blocks: list[ResBlock]
    out_proj: eqx.nn.Linear

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden: int = 256,
        depth: int = 2,
        *,
        rng: Key,
    ):
        rng, rng_t, rng_y = jr.split(rng, 3)
        self.t_emb = TimestepEmbedder(hidden, rng=rng_t)
        self.y_emb = ObsEmbedder(y_dim, hidden, rng=rng_y)
        rng, rng_in, rng_out = jr.split(rng, 3)
        self.in_proj = eqx.nn.Linear(x_dim, hidden, key=rng_in)
        self.res_blocks = [ResBlock(hidden, rng=rng) for rng in jr.split(rng, depth)]
        self.out_proj = eqx.nn.Linear(hidden, x_dim, key=rng_out)

    def __call__(self, t: Scalar, xt: list[Param], y: Observation) -> list[Param]:
        c = jax.nn.silu(self.t_emb(t) + self.y_emb(y))
        h = self.in_proj(jnp.concat(xt, axis=-1))
        for block in self.res_blocks:
            h = block(h, c)
        h = self.out_proj(h)
        return jnp.split(h, h.shape[-1], axis=-1)

    def push(self, priors: list[Prior], x0: list[Param], y: Observation, n_steps=16):
        # def runje_kutta_step(x, t):
        #     k1 = self(t, x, y)
        #     k2 = self(t + dt / 2, x + k1 * dt / 2, y)
        #     k3 = self(t + dt / 2, x + k2 * dt / 2, y)
        #     k4 = self(t + dt, x + k3 * dt, y)
        #     next_x = x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
        #     return next_x, x

        def runje_kutta_step(x: list[Param], t: Scalar):
            k1 = self(t, x, y)
            x_k1 = [p.step(x, k, dt / 2) for p, x, k in zip(priors, x, k1)]
            k2 = self(t + dt / 2, x_k1, y)
            x_k2 = [p.step(x, k, dt / 2) for p, x, k in zip(priors, x, k2)]
            k3 = self(t + dt / 2, x_k2, y)
            x_k3 = [p.step(x, k, dt) for p, x, k in zip(priors, x, k3)]
            k4 = self(t + dt, x_k3, y)

            k_tot = map(lambda a, b, c, d: (a + 2 * b + 2 * c + d), k1, k2, k3, k4)
            next_x = [p.step(x, k, dt / 6) for p, x, k in zip(priors, x, k_tot)]
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
        def loss_fn(model, batch):
            t, xt, dx, y = batch
            pred = jax.vmap(model)(t, xt, y)
            balanced_loss = lambda p, t: optax.l2_loss(p, t).mean() / t.std()
            return sum(jax.tree.map(balanced_loss, pred, dx))

        @eqx.filter_jit
        def train_epoch(model, opt_state, batches):
            def optimization_step(carry, batch):
                model, opt_state = carry
                loss, grads = loss_fn(model, batch)
                updates, opt_state = optimizer.update(grads, opt_state, model)
                model = eqx.apply_updates(model, updates)
                carry = (model, opt_state)
                return carry, loss

            (model, opt_state), losses = jax.lax.scan(
                optimization_step, (model, opt_state), batches
            )
            return model, opt_state, losses

        loss_log = []
        for rng in (pbar := tqdm(jr.split(rng, epochs))):
            batches = generate_batches(rng)
            self, opt_state, losses = train_epoch(self, opt_state, batches)
            pbar.set_postfix(loss=losses.mean())
            loss_log.append(losses)
        return self, jnp.array(loss_log)
