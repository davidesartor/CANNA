from tqdm import tqdm
import torch
import torch.nn as nn
from lightning import LightningModule


class ConditionalFlowMatching(LightningModule):
    def __init__(self, lr=3e-4):
        super().__init__()
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])

    def push(self, x, y, n_steps=16, verbose=False):
        if verbose:
            print("Pushing data through flow")
        ts = torch.arange(0, 1, 1 / n_steps, device=self.device).expand(x.shape[0], -1)
        x = torch.as_tensor(x).to(self.device)
        y = torch.as_tensor(y).to(self.device).expand((x.shape[0], *y.shape))
        with torch.no_grad():
            for t in ts.T:
                x = x + self(t, x, y) / n_steps
        return x

    def training_step(self, batch, batch_idx):
        t, xt, dx, y = batch
        flow = self(t, xt, y)
        loss = nn.functional.mse_loss(flow, dx)
        self.log("flow_loss", loss, prog_bar=True)
        return loss

    def __call__(self, t, xt, y):
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
    def __init__(self, dim, obs_dim=None, hidden_dim=512, depth=1, norm=True, **kwargs):
        super().__init__(**kwargs)
        self.flow = MLP(
            in_dim=dim + (obs_dim or dim) + 1,
            out_dim=dim,
            hidden_dim=hidden_dim,
            depth=depth,
            norm=norm,
        )

    def __call__(self, t, xt, y):
        h = torch.cat([t.unsqueeze(-1), xt, y], dim=-1)
        return self.flow(h)
