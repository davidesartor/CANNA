from tqdm import tqdm
import torch
import torch.nn as nn
from lightning import LightningModule


class ConditionalFlowMatching(LightningModule):
    def __init__(self, coupling_jitter=0.0, lr=3e-4):
        super().__init__()
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])

    def push(self, x, y, n_steps=4, verbose=False):
        if verbose:
            print("Pushing data through flow")
        dt = 1 / n_steps
        ts = torch.arange(0, 1, dt, device=self.device).expand(x.shape[0], -1)
        x, y = x.to(self.device), y.to(self.device).expand((x.shape[0], *y.shape))
        with torch.no_grad():
            for t in ts.T:
                k1 = self(t, x, y)
                k2 = self(t + dt / 2, x + dt / 2 * k1, y)
                k3 = self(t + dt / 2, x + dt / 2 * k2, y)
                k4 = self(t + dt, x + dt * k3, y)
                x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x

    def conditional_map_derivative(self, t, x0, x1):
        return x1 - x0

    def conditional_map(self, t, x0, x1):
        x_jitter = self.hparams["coupling_jitter"] * torch.randn_like(x0)
        while t.ndim < x0.ndim:
            t = t.unsqueeze(-1)
        d_x = self.conditional_map_derivative(t, x0, x1)
        xt = x0 + t * d_x + x_jitter
        return xt, d_x

    def training_step(self, batch, batch_idx):
        t, x0, x1, y = batch
        x, d_x = self.conditional_map(t, x0, x1)
        flow = self(t, x, y)
        loss = nn.functional.mse_loss(flow, d_x)
        self.log("flow_loss", loss, prog_bar=True)
        return loss

    def __call__(self, t, x, y):
        raise NotImplementedError


class MLP(nn.Sequential):
    def __init__(self, in_dim, out_dim, hidden_dim=512, depth=1, norm=True):
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.GELU())
        for _ in range(depth - 1):
            if norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        if norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Linear(hidden_dim, out_dim))
        super().__init__(*layers)


class MLPCNF(ConditionalFlowMatching):
    def __init__(self, dim, obs_dim=None, hidden_dim=512, depth=1, norm=True, **kwargs):
        super().__init__(**kwargs)
        self.flow = MLP(
            in_dim=dim + (obs_dim or dim) + 1,
            out_dim=dim,
            hidden_dim=hidden_dim,
            depth=depth,
            norm=norm,
        )

    def __call__(self, t, x, y):
        t = t.unsqueeze(-1)
        y = y.flatten(1)
        h = torch.cat([t, x, y], dim=-1)
        return self.flow(h)
