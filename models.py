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

    def push(self, x, y, n_steps=16, verbose=False):
        if verbose:
            print("Pushing data through flow")
        with torch.no_grad():
            y = y.expand((x.shape[0], *y.shape))
            ts = torch.arange(0, 1, 1 / n_steps).expand(x.shape[0], -1)
            for t in tqdm(ts.T, disable=not verbose):
                x = x + self(t, x, y) / n_steps
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
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            if norm:
                layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Linear(hidden_dim, out_dim))
        super().__init__(*layers)


class MLPCNF(ConditionalFlowMatching):
    def __init__(self, dim, hidden_dim=512, depth=1, norm=True, **kwargs):
        super().__init__(**kwargs)
        self.flow = MLP(
            in_dim=2 * dim + 1,
            out_dim=dim,
            hidden_dim=hidden_dim,
            depth=depth,
            norm=norm,
        )

    def __call__(self, t, x, y):
        h = torch.cat([t.unsqueeze(-1), x, y], dim=-1)
        return self.flow(h)
