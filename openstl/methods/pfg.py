import torch
from openstl.models.pfg_model import PFG_Model
from .base_method import Base_method


class PFG(Base_method):
    """Official PFG core with the canonical project training/evaluation base."""

    def _build_model(self, **args):
        return PFG_Model(**args)

    def forward(self, batch_x, batch_y=None, **kwargs):
        if self.hparams.pre_seq_length != 4 or self.hparams.aft_seq_length != 4:
            raise ValueError("This registered adapter is for TaxiBJ 4-to-4.")
        return self.model(batch_x)

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x)
        loss = self.criterion(pred_y, batch_y)
        self.log(
            'train_loss', loss,
            on_step=True, on_epoch=True, prog_bar=True
        )
        return loss
