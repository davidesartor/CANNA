from datasets import SinusoidDataset
from models import MLPCNF
from lightning import Trainer
from lightning.pytorch.callbacks import RichModelSummary


if __name__ == "__main__":
    """
    Auxiliary file to run training from a python file instead of notebooks
    """
    dataset = SinusoidDataset()

    model = MLPCNF(
        dim=3,
        obs_dim=len(dataset.observation_times),
        hidden_dim=128,
        depth=4,
        lr=1e-4,
    )
    # model = torch.compile(model)

    train_dataloader = dataset.dataloader(
        batch_size=1024,
        batches=1024,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
    )

    trainer = Trainer(
        accelerator="gpu",
        max_epochs=10,
        callbacks=[RichModelSummary()],
    )

    trainer.fit(model, train_dataloader)
