"""Train or evaluate EvoField on the prepared Navier-Stokes T20 MAT file."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from scipy.io import loadmat
from torch.utils.data import DataLoader, TensorDataset

from openstl.datasets.base_data import BaseDataModule
from openstl.methods.tau import TAU


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Evaluate this checkpoint")
    args = parser.parse_args()

    config = Path(__file__).resolve().parents[1] / "configs/navier_stokes_t20/evofield.json"
    cfg = json.loads(config.read_text())
    seed_everything(cfg["seed"], workers=True)
    u = loadmat(args.data, variable_names=["u"])["u"]
    if u.shape != (1200, 64, 64, 20) or u.dtype != np.float32 or not np.isfinite(u).all():
        raise ValueError("Expected finite float32 u with shape (1200, 64, 64, 20)")

    loaders = []
    for start, stop, batch_size in ((0, 864, 2), (864, 960, 4), (960, 1200, 4)):
        sequence = u[start:stop].transpose(0, 3, 1, 2)[:, :, None]
        dataset = TensorDataset(torch.from_numpy(sequence[:, :10].copy()),
                                torch.from_numpy(sequence[:, 10:].copy()))
        dataset.mean, dataset.std, dataset.data_name = 0., 1., cfg["dataname"]
        loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=start == 0,
                                  num_workers=cfg["num_workers"], pin_memory=True))

    args.output.mkdir(parents=True, exist_ok=False)
    data = BaseDataModule(*loaders)
    cfg.update(save_dir=str(args.output), steps_per_epoch=len(loaders[0]),
               test_mean=0., test_std=1.)
    (args.output / "effective_config.json").write_text(json.dumps(cfg, indent=2))
    model = TAU(**cfg)
    checkpoint = ModelCheckpoint(dirpath=args.output / "checkpoints", filename="best",
                                 monitor="val_loss", mode="min", save_top_k=1)
    trainer = Trainer(accelerator="gpu", devices=1, precision="32-true",
                      max_epochs=cfg["epoch"], default_root_dir=str(args.output),
                      accumulate_grad_batches=cfg["accumulate_grad_batches"],
                      deterministic=cfg["deterministic"], callbacks=[checkpoint])
    if args.checkpoint is None:
        trainer.fit(model, datamodule=data)
    else:
        trainer.test(model, datamodule=data, ckpt_path=str(args.checkpoint))


if __name__ == "__main__":
    main()
