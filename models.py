import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx
from jaxtyping import Key, Float, Array, Scalar
import optax
from tqdm import tqdm


class MLPCNF(nnx.Sequential):
    def __init__(self, x_dim, y_dim, hidden_dim=512, depth=4, *, rngs: nnx.Rngs):
        layers = []
        layers.append(nnx.Linear(1 + x_dim + y_dim, hidden_dim, rngs=rngs))
        for _ in range(depth):
            layers.append(nnx.RMSNorm(hidden_dim, rngs=rngs))
            layers.append(nnx.Linear(hidden_dim, hidden_dim, rngs=rngs))
            layers.append(jax.nn.gelu)
        layers.append(nnx.Linear(hidden_dim, x_dim, rngs=rngs))
        super().__init__(*layers)

    def __call__(
        self, t: Scalar, xt: Float[Array, "x"], y: Float[Array, "y"]
    ) -> Float[Array, "x"]:
        h = jnp.concatenate([t[..., None], xt, y], axis=-1)
        return super().__call__(h)

    @nnx.jit(static_argnames="n_steps")
    def push(self, x0: Float[Array, "n x"], y: Float[Array, "y"], n_steps=4):
        @nnx.scan(in_axes=(None, nnx.Carry, 0))
        @nnx.vmap(in_axes=(None, 0, None))
        def runje_kutta(flow, x, t):
            k1 = flow(t, x, y)
            k2 = flow(t + dt / 2, x + k1 * dt / 2, y)
            k3 = flow(t + dt / 2, x + k2 * dt / 2, y)
            k4 = flow(t + dt, x + k3 * dt, y)
            next_x = x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
            return next_x, x

        dt = 1 / n_steps
        ts = dt * jnp.arange(n_steps)
        x1, xt = runje_kutta(self, x0, ts)
        return x1


class Trainer(nnx.Module):
    def __init__(
        self,
        dataset,
        model,
        batch_size=64,
        epochs=100,
        steps_per_epoch=64,
        learning_rate=3e-4,
        *,
        rngs: nnx.Rngs,
    ):
        self.dataset = dataset
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.optimizer = nnx.Optimizer(self.model, optax.adam(learning_rate))
        self.rngs = rngs

    @nnx.jit
    def train_epoch(self):
        @nnx.scan
        def optimization_step(trainer, batch):
            @nnx.value_and_grad
            def loss_fn(flow, batch):
                t, xt, dx, y = batch
                pred = jax.vmap(flow)(t, xt, y)
                return optax.l2_loss(pred, dx).mean()

            loss, grads = loss_fn(trainer.model, batch)
            trainer.optimizer.update(grads)
            return trainer, loss

        @jax.vmap
        def get_batches(rng):
            return jax.vmap(self.dataset.get_train_sample)(
                jr.split(rng, self.batch_size)
            )

        batches = get_batches(jr.split(self.rngs(), self.steps_per_epoch))
        self, losses = optimization_step(self, batches)
        return losses

    def fit(self):
        for _ in (pbar := tqdm(range(self.epochs))):
            losses = self.train_epoch()
            pbar.set_postfix(loss=losses.mean())
